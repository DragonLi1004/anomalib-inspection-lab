"""Reload the latest local baseline checkpoint and detect one image."""
import argparse
import json
import time
from pathlib import Path

# Configure local caches before importing timm/torch.
from baseline import ROOT
import torch
from anomalib.engine import Engine
from anomalib.models import Patchcore
from anomalib.visualization.image import ImageVisualizer

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--run", help="Local experiment folder name; defaults to latest run")
    args = parser.parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(args.image)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; explicitly use --device cpu if needed.")
    latest = ROOT / "course/outputs/latest_baseline.txt"
    run = ROOT / "course/outputs" / args.run if args.run else Path(latest.read_text(encoding="utf-8").strip())
    checkpoint = run / "baseline.ckpt"
    if not checkpoint.resolve().is_relative_to((ROOT / "course/outputs").resolve()):
        raise ValueError("Checkpoint must belong to this project's local outputs.")
    # Only load our own locally generated checkpoint; never an external pickle.
    from compact_patchcore import CompactPatchcore
    metadata = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    model_class = CompactPatchcore if metadata.get("method", "baseline").startswith("pca") else Patchcore
    model = model_class.load_from_checkpoint(checkpoint, map_location="cpu", weights_only=False)
    output = ROOT / "course/outputs" / ("prediction-" + time.strftime("%Y%m%d-%H%M%S"))
    output.mkdir(parents=True, exist_ok=False)
    model.visualizer = ImageVisualizer(fields=["image"], output_dir=output / "images")
    engine = Engine(default_root_dir=output / "engine", accelerator="gpu" if args.device == "cuda" else "cpu",
                    devices=1, logger=False, enable_progress_bar=False, enable_model_summary=False)
    predictions = engine.predict(model=model, data_path=args.image.resolve(), return_predictions=True)
    records = []
    for batch in predictions:
        for item in batch:
            records.append({"image": str(item.image_path), "anomaly_score": float(item.pred_score),
                            "anomalous": bool(item.pred_label), "model_run": run.name})
    (output / "predictions.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(records, indent=2, ensure_ascii=False))
    print(f"Results: {output}")

if __name__ == "__main__":
    main()
