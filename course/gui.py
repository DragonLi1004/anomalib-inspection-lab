"""Local-only inspection UI. No extra framework or cloud inference required."""
import argparse
import base64
import csv
import io
import json
import mimetypes
import os
import secrets
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit
from catalog import DATASETS, METHODS, catalog, describe_run

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "course/outputs"
STATIC = ROOT / "course/web"
TOKEN = secrets.token_urlsafe(32)
LOCK = threading.Lock()
JOB = {}
PROCESS = None
PREDICTION_THREAD = None
from prediction_worker import PredictionWorker
PREDICTOR = PredictionWorker()
CATEGORIES = {"bottle": (209, 83, 63), "cable": (224, 150, 92), "capsule": (219, 132, 109)}


def read_json(path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def runs():
    result = []
    for path in sorted(OUTPUT.glob("*/metrics.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        data = read_json(path)
        if data:
            data.update(id=path.parent.name, dataset=data.get("dataset", "hazelnut_toy"), method=data.get("method", "baseline"))
            data.update(describe_run(data))
            data.pop("latency_samples_ms", None)
            result.append(data)
    return result


def local_run(name):
    if not isinstance(name, str) or not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("Invalid experiment ID")
    path = (OUTPUT / name).resolve()
    if not path.is_relative_to(OUTPUT.resolve()) or not path.is_dir():
        raise ValueError("Experiment not found")
    return path


def images(path):
    return [p.relative_to(ROOT).as_posix() for p in sorted((path / "images").rglob("*")) if p.suffix.lower() in (".jpg", ".png", ".jpeg")]


def package_path(model_id="bundle"):
    report = read_json(ROOT / "course/results/lossless_metrics.json", {})
    if not report.get("complete"):
        raise ValueError("无损模型包尚未生成")
    bundle = (ROOT / report["package"]).resolve()
    if not bundle.is_relative_to(OUTPUT.resolve()) or bundle.suffix != ".alpack":
        raise ValueError("Invalid package location")
    if model_id == "bundle":
        return bundle
    if model_id not in {r["source_run"] for r in report["runs"]}:
        raise ValueError("Unknown packaged model")
    return bundle.parent / "single" / (model_id + ".alpack")


def job_state():
    global JOB
    with LOCK:
        if not JOB:
            return None
        if PROCESS is not None and PROCESS.poll() is not None:
            JOB["status"] = "completed" if PROCESS.returncode == 0 else "failed"
            JOB["exit_code"] = PROCESS.returncode
        state = dict(JOB)
    log = OUTPUT / "jobs" / (state["id"] + ".log")
    try:
        with log.open("rb") as stream:
            stream.seek(max(0, log.stat().st_size - 12000))
            state["log"] = stream.read().decode("utf-8", errors="replace")
    except OSError:
        state["log"] = ""
    return state


def datasets():
    result = [{"id": "hazelnut_toy", "label": "榛子 · 小样本演示", "expected": 56,
               "images": sum(1 for p in (ROOT / "course/data/hazelnut_toy").rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg") and "mask" not in p.parts),
               "ready": (ROOT / "course/data/hazelnut_toy/good").is_dir()}]
    for name, expected in CATEGORIES.items():
        root = ROOT / "course/data/MVTecAD" / name
        counts = tuple(len(list((root / split).rglob("*.png"))) for split in ("train", "test", "ground_truth"))
        result.append({"id": name, "label": DATASETS[name]["label"] + " · " + name, "images": counts[0] + counts[1],
                       "expected": expected[0] + expected[1], "ready": counts == expected, "counts": counts})
    return result


def launch(body):
    global JOB, PROCESS, PREDICTION_THREAD
    task = body.get("task")
    device = body.get("device", "cuda")
    if device not in ("cpu", "cuda"):
        raise ValueError("Invalid device")
    if task == "train":
        dataset, method = body.get("dataset"), body.get("method")
        if dataset not in ["hazelnut_toy", *CATEGORIES] or method not in ("baseline", "pca96", "pca192"):
            raise ValueError("Invalid experiment configuration")
        if not next(d for d in datasets() if d["id"] == dataset)["ready"]:
            raise ValueError("数据集尚未下载完整")
        seed = int(body.get("seed", 42))
        if not 0 <= seed <= 1000000:
            raise ValueError("Seed outside supported range")
        command = [sys.executable, "-u", str(ROOT / "course/experiment.py"), "--dataset", dataset, "--method", method, "--device", device, "--seed", str(seed)]
        label = f"{DATASETS[dataset]['label']} / {METHODS[method]['name']} / 种子 {seed}"
    elif task == "predict":
        run = local_run(body.get("run"))
        if not (run / "metrics.json").is_file() or not (run / "baseline.ckpt").is_file():
            raise ValueError("没有可用的本地模型")
        image_id = body.get("image", "")
        if not isinstance(image_id, str) or not image_id.endswith(".png") or Path(image_id).name != image_id:
            raise ValueError("Invalid uploaded image")
        path = (OUTPUT / "uploads" / image_id).resolve()
        if not path.is_relative_to((OUTPUT / "uploads").resolve()) or not path.is_file():
            raise ValueError("Uploaded image missing")
        command = [sys.executable, "-u", str(ROOT / "course/predict.py"), "--run", run.name, "--image", str(path), "--device", device]
        label = "单图检测 / " + describe_run({**read_json(run / "metrics.json", {}), "id": run.name})["display_name"]
    elif task == "download":
        command = [sys.executable, "-u", str(ROOT / "course/download_mvtec.py")]
        label = "下载 MVTec AD 三个完整类别"
    else:
        raise ValueError("Unknown task")
    with LOCK:
        if (PROCESS is not None and PROCESS.poll() is None) or (PREDICTION_THREAD is not None and PREDICTION_THREAD.is_alive()):
            raise ValueError("已有任务运行中，请等待完成")
        job_id = time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)
        directory = OUTPUT / "jobs"
        directory.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update(PYTHONUTF8="1", PYTHONUNBUFFERED="1", TEMP=str(ROOT / ".runtime/temp"), TMP=str(ROOT / ".runtime/temp"), MPLCONFIGDIR=str(ROOT / ".cache/matplotlib"))
        JOB = {"id": job_id, "label": label, "task": task, "status": "running", "started": time.time()}
        if task == "predict":
            PROCESS = None
            request = {"run": run.name, "image": str(path), "device": device}
            def detect():
                started = time.perf_counter()
                try:
                    with (directory / (job_id + ".log")).open("w", encoding="utf-8") as log:
                        result = PREDICTOR.predict(request, log, env)
                    with LOCK:
                        JOB.update(status="completed", exit_code=0, cache_hit=result["timing"]["cache_hit"],
                                   elapsed_seconds=time.perf_counter()-started)
                except Exception as error:
                    with (directory / (job_id + ".log")).open("a", encoding="utf-8") as log:
                        log.write(str(error) + "\n")
                    with LOCK:
                        JOB.update(status="failed", exit_code=1)
            PREDICTION_THREAD = threading.Thread(target=detect, daemon=True)
            PREDICTION_THREAD.start()
        else:
            # Only one model may occupy the 4 GB GPU; release the worker before training.
            PREDICTOR.close()
            with (directory / (job_id + ".log")).open("wb") as log:
                try:
                    PROCESS = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env,
                                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                except OSError:
                    JOB.update(status="failed", exit_code=1)
                    PROCESS = None
                    raise
        return dict(JOB)


class Handler(BaseHTTPRequestHandler):
    def send(self, data, content_type="application/json; charset=utf-8", status=200):
        payload = json.dumps(data, ensure_ascii=False).encode() if isinstance(data, (dict, list)) else data
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' blob: data:; style-src 'self' 'unsafe-inline'; script-src 'self'; frame-ancestors 'none'; base-uri 'none'")
        self.end_headers()
        self.wfile.write(payload)

    def host_ok(self):
        return self.headers.get("Host") in (f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}")

    def do_GET(self):
        if not self.host_ok():
            return self.send({"error": "Invalid host"}, status=403)
        parsed = urlsplit(self.path)
        params = parse_qs(parsed.query)
        try:
            if parsed.path == "/api/state":
                prediction = []
                for file in sorted(OUTPUT.glob("prediction-*/predictions.json"), reverse=True)[:5]:
                    prediction.append({"id": file.parent.name, "records": read_json(file, []), "images": images(file.parent), "timing": read_json(file.parent / "timing.json", {})})
                return self.send({"token": TOKEN, "catalog": catalog(), "runs": runs(), "datasets": datasets(), "job": job_state(), "predictions": prediction})
            if parsed.path == "/api/summary":
                from summarize_experiments import collect_records, summarize
                device = params.get("device", ["cuda"])[0]
                if device not in ("cuda", "cpu"):
                    raise ValueError("Invalid device")
                try:
                    return self.send(summarize(collect_records(device)))
                except ValueError as error:
                    return self.send({"error": str(error)}, status=409)
            if parsed.path == "/api/storage":
                report = read_json(ROOT / "course/results/lossless_metrics.json", {"complete": False})
                if report.get("complete"):
                    report["package_available"] = package_path().is_file()
                    report["package_local_path"] = str(package_path())
                return self.send(report)
            if parsed.path == "/api/package":
                return self.send(package_path(params.get("run", ["bundle"])[0]).read_bytes(), "application/octet-stream")
            if parsed.path == "/api/run":
                path = local_run(params.get("id", [""])[0])
                return self.send({"metrics": read_json(path / "metrics.json", {}), "images": images(path), "splits": read_json(path / "split_manifest.json", {})})
            if parsed.path == "/api/export":
                buf = io.StringIO()
                fields = ["id", "display_name", "model_name", "dataset_label", "method_label", "dataset", "method", "seed", "image_AUROC", "image_F1Score", "pixel_AUROC", "pixel_F1Score", "memory_bank_bytes", "projection_bytes", "search_state_bytes", "latency_median_ms", "split_sha256", "protocol_version", "coreset_sha256"]
                writer = csv.DictWriter(buf, fields, extrasaction="ignore")
                writer.writeheader()
                for row in runs():
                    writer.writerow({**row, **row["metrics"][0]})
                return self.send(buf.getvalue().encode("utf-8-sig"), "text/csv; charset=utf-8")
            if parsed.path == "/api/file":
                path = (ROOT / params.get("path", [""])[0]).resolve()
                if not path.is_relative_to(OUTPUT.resolve()) or path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                    raise ValueError("Image path not allowed")
            else:
                name = unquote(parsed.path).lstrip("/") or "index.html"
                path = (STATIC / name).resolve()
                if not path.is_relative_to(STATIC.resolve()) or path.suffix not in (".html", ".js", ".css", ".svg"):
                    raise ValueError("Invalid asset")
            return self.send(path.read_bytes(), mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        except (ValueError, OSError) as error:
            return self.send({"error": str(error)}, status=404)

    def do_POST(self):
        if not self.host_ok() or self.headers.get("X-Lab-Token") != TOKEN:
            return self.send({"error": "Invalid local session"}, status=403)
        origin = self.headers.get("Origin")
        if origin and origin not in (f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}"):
            return self.send({"error": "Invalid origin"}, status=403)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 16_000_000:
                raise ValueError("请求过大：图片请小于 10 MB")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("Expected object")
            if self.path == "/api/job":
                return self.send(launch(body))
            if self.path == "/api/upload":
                from PIL import Image
                data = base64.b64decode(body["data"], validate=True)
                if len(data) > 10_000_000:
                    raise ValueError("图片请小于 10 MB")
                with Image.open(io.BytesIO(data)) as image:
                    if image.width * image.height > 20_000_000:
                        raise ValueError("图片请小于 2000 万像素")
                    image.load()
                    folder = OUTPUT / "uploads"
                    folder.mkdir(parents=True, exist_ok=True)
                    name = secrets.token_hex(12) + ".png"
                    image.convert("RGB").save(folder / name)
                return self.send({"id": name})
            raise ValueError("Unknown operation")
        except (ValueError, KeyError, TypeError, OSError) as error:
            return self.send({"error": str(error)}, status=400)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Inspection Lab: http://127.0.0.1:{args.port}", flush=True)
    if args.open_browser:
        threading.Thread(target=webbrowser.open, args=(f"http://127.0.0.1:{args.port}",), daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        PREDICTOR.close()
        server.server_close()


if __name__ == "__main__":
    main()
