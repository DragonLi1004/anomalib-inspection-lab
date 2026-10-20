"""Resume a fixed, sequential matrix of independent experiment processes."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "course/outputs"
CATEGORIES = ("bottle", "cable", "capsule")
METHODS = ("baseline", "pca96", "pca192")
SEEDS = (42, 43, 44)


def code_hash():
    return hashlib.sha256(b"".join((ROOT / "course" / name).read_bytes() for name in ("experiment.py", "compact_patchcore.py"))).hexdigest()


def completed_runs(device):
    found = {}
    for path in sorted(OUTPUT.glob("*/metrics.json"), key=lambda p: p.stat().st_mtime):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            if row.get("protocol_version") == "paired-coreset-v2" and row.get("course_code_sha256") == code_hash() and row.get("device") == device and (path.parent / "baseline.ckpt").is_file():
                found[(row.get("dataset"), row.get("seed"), row.get("method"))] = path.parent.name
        except (ValueError, OSError):
            continue
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    plan = [(category, seed, method) for seed in SEEDS for category in CATEGORIES for method in METHODS]
    existing = completed_runs(args.device)
    pending = [item for item in plan if item not in existing]
    print(f"Matrix: {len(plan)} experiments; reusable: {len(plan)-len(pending)}; pending: {len(pending)}", flush=True)
    if args.dry_run:
        for item in pending:
            print(*item)
        return
    folder = OUTPUT / "suites" / time.strftime("%Y%m%d-%H%M%S")
    folder.mkdir(parents=True, exist_ok=False)
    state = {"protocol_version": "paired-coreset-v2", "course_code_sha256": code_hash(), "device": args.device,
             "categories": CATEGORIES, "seeds": SEEDS, "methods": METHODS, "total": len(plan), "completed": 0,
             "status": "running", "results": []}
    def save():
        temporary = folder / "status.tmp"
        temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
        temporary.replace(folder / "status.json")
    env = os.environ.copy()
    env.update(PYTHONUTF8="1", TEMP=str(ROOT / ".runtime/temp"), TMP=str(ROOT / ".runtime/temp"),
               MPLCONFIGDIR=str(ROOT / ".cache/matplotlib"), XDG_CACHE_HOME=str(ROOT / ".cache"))
    try:
        for index, (category, seed, method) in enumerate(plan, 1):
            key = (category, seed, method)
            state["current"] = {"dataset": category, "seed": seed, "method": method}
            save()
            if key in existing:
                run_id = existing[key]
                print(f"[{index}/{len(plan)}] REUSE {run_id}", flush=True)
            else:
                print(f"[{index}/{len(plan)}] RUN {category} / {method} / seed {seed}", flush=True)
                log = folder / f"{category}-{method}-{seed}.log"
                with log.open("wb") as stream:
                    subprocess.run([sys.executable, "-u", str(ROOT / "course/experiment.py"), "--dataset", category,
                                    "--method", method, "--seed", str(seed), "--device", args.device],
                                   cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, env=env, check=True)
                run_id = completed_runs(args.device)[key]
            state["results"].append({"dataset": category, "seed": seed, "method": method, "run_id": run_id, "reused": key in existing})
            state["completed"] = index
            save()
        state["status"] = "completed"
        print("All 27 experiments completed. Run course/summarize_experiments.py to export the verified report.", flush=True)
    except Exception as error:
        state["status"] = "failed"
        state["error"] = str(error)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
