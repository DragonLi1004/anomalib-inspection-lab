"""Original anomalib prediction path with an optional one-model session cache."""
import argparse
import gc
import hashlib
import json
import queue
import secrets
import sys
import threading
import time
from pathlib import Path

from baseline import ROOT
import torch
from anomalib.engine import Engine
from anomalib.models import Patchcore
from anomalib.visualization.image import ImageVisualizer

RESPONSE_PREFIX = "INSPECTION_RESULT:"


def tensor_digest(value):
    if value is None:
        return None
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(str((array.shape, str(array.dtype))).encode() + array.tobytes()).hexdigest()


class PredictionSession:
    """Cache only one checkpoint; retain upstream preprocessing and postprocessing."""
    def __init__(self):
        self.model = None
        self.key = None

    def clear(self):
        self.model = None
        self.key = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load(self, run_name, device):
        if device not in ("cuda", "cpu"):
            raise ValueError("Invalid device")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; explicitly use --device cpu if needed.")
        outputs = (ROOT / "course/outputs").resolve()
        if run_name is None:
            run = Path((outputs / "latest_baseline.txt").read_text(encoding="utf-8").strip()).resolve()
        else:
            if Path(run_name).name != run_name or run_name in (".", ".."):
                raise ValueError("Invalid experiment ID")
            run = (outputs / run_name).resolve()
        checkpoint = (run / "baseline.ckpt").resolve()
        metadata_path = (run / "metrics.json").resolve()
        if not checkpoint.is_relative_to(outputs) or not metadata_path.is_relative_to(outputs):
            raise ValueError("Checkpoint must belong to local outputs")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        dependencies = [checkpoint, metadata_path]
        auxiliary_path = (run / "auxiliary.ckpt").resolve()
        if metadata.get("method") == "fine_guard":
            if not auxiliary_path.is_relative_to(run):
                raise ValueError("Auxiliary checkpoint must belong to model directory")
            dependencies.append(auxiliary_path)
        # Reload after either branch or its method metadata changes.
        key = (str(checkpoint), device, *((p.stat().st_size, p.stat().st_mtime_ns) for p in dependencies))
        hit = key == self.key and self.model is not None
        if not hit:
            self.clear()
            from compact_patchcore import CompactPatchcore
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            cls = CompactPatchcore if metadata.get("method", "baseline").startswith("pca") else Patchcore
            self.model = cls.load_from_checkpoint(checkpoint, map_location="cpu", weights_only=False)
            if metadata.get("method") == "peak_mask":
                from peak_refinement import install_peak
                install_peak(self.model, float(metadata["peak_alpha"]))
            if metadata.get("method") == "fine_guard":
                from fine_guard import install_fine_guard
                for path, field in [(checkpoint, "source_checkpoint_sha256"), (auxiliary_path, "auxiliary_checkpoint_sha256")]:
                    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata[field]:
                        raise ValueError("FineGuard checkpoint integrity mismatch")
                auxiliary = Patchcore.load_from_checkpoint(auxiliary_path, map_location="cpu", weights_only=False)
                install_fine_guard(self.model, auxiliary, float(metadata["fine_threshold"]))
            self.key = key
        return self.model, run, hit

    def predict(self, image, run_name=None, device="cuda", audit=False):
        image = Path(image).resolve()
        if not image.is_file():
            raise FileNotFoundError(image)
        start = time.perf_counter()
        model, run, hit = self.load(run_name, device)
        loaded = time.perf_counter()
        output = ROOT / "course/outputs" / ("prediction-" + time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(4))
        output.mkdir(parents=True, exist_ok=False)
        model.visualizer = ImageVisualizer(fields=["image"], output_dir=output / "images")
        # A fresh trainer isolates callbacks, output directories and per-request state.
        engine = Engine(default_root_dir=output / "engine", accelerator="gpu" if device == "cuda" else "cpu",
                        devices=1, logger=False, enable_progress_bar=False, enable_model_summary=False)
        predictions = engine.predict(model=model, data_path=image, return_predictions=True)
        inferred = time.perf_counter()
        records, signatures = [], []
        for batch in predictions:
            for item in batch:
                records.append({"image": str(item.image_path), "anomaly_score": float(item.pred_score),
                                "anomalous": bool(item.pred_label), "model_run": run.name})
                if audit:
                    signatures.append({name: tensor_digest(getattr(item, name, None))
                                       for name in ("pred_score", "pred_label", "anomaly_map", "pred_mask")})
        (output / "predictions.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
        timing = {"cache_hit": hit, "load_seconds": loaded-start,
                  "predict_and_render_seconds": inferred-loaded, "request_seconds": time.perf_counter()-start}
        (output / "timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
        return {"records": records, "output": str(output), "timing": timing, "signatures": signatures}


def worker(idle_seconds):
    requests = queue.Queue()
    def read_requests():
        for line in sys.stdin:
            requests.put(line)
        requests.put(None)
    threading.Thread(target=read_requests, daemon=True).start()
    session = PredictionSession()
    try:
        while True:
            try:
                line = requests.get(timeout=idle_seconds)
            except queue.Empty:
                return
            if line is None:
                return
            try:
                request = json.loads(line)
                result = session.predict(request["image"], request["run"], request["device"], request.get("audit", False))
                response = {"ok": True, **result}
            except Exception as error:
                # A failed prediction cannot leave a partially initialized cached model.
                session.clear()
                response = {"ok": False, "error": f"{type(error).__name__}: {error}"}
            print(RESPONSE_PREFIX + json.dumps(response, ensure_ascii=False), flush=True)
    finally:
        session.clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--run")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--idle-seconds", type=float, default=120)
    args = parser.parse_args()
    if args.worker:
        if args.idle_seconds <= 0:
            parser.error("idle-seconds must be positive")
        worker(args.idle_seconds)
    else:
        if args.image is None:
            parser.error("--image is required")
        result = PredictionSession().predict(args.image, args.run, args.device)
        print(json.dumps(result["records"], indent=2, ensure_ascii=False))
        print(f"Results: {result['output']}")


if __name__ == "__main__":
    main()
