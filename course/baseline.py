"""Run upstream PatchCore unchanged on the official small demo dataset.

This is a deployment check, NOT a full MVTec AD benchmark.
"""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["TORCH_HOME"] = str(ROOT / "course/models/torch")
os.environ["HF_HOME"] = str(ROOT / "course/models/huggingface")
os.environ["TIMM_USE_OLD_CACHE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["MPLBACKEND"] = "Agg"

import torch
from lightning import seed_everything
from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore
from anomalib.visualization.image import ImageVisualizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Check the environment or explicitly select --device cpu.")
    if not (ROOT / "course/models/torch/hub/checkpoints/resnet18-f37072fd.pth").is_file():
        raise FileNotFoundError("Run course/prepare_assets.py first.")
    seed_everything(args.seed, workers=True)
    torch.set_num_threads(4)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    output = ROOT / "course/outputs" / f"baseline-{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    data = Folder(
        name="hazelnut_toy", root=ROOT / "course/data/hazelnut_toy",
        normal_dir="good", abnormal_dir=["colour", "crack"], mask_dir="mask",
        train_batch_size=1, eval_batch_size=1, num_workers=0,
        test_split_mode="from_dir", test_split_ratio=0.2,
        val_split_mode="from_test", val_split_ratio=0.5, seed=args.seed,
    )
    data.setup()
    splits = {}
    for split in ["train", "val", "test"]:
        dataset = getattr(data, f"{split}_data")
        rows = []
        for row in dataset.samples.to_dict(orient="records"):
            image_path = Path(row["image_path"])
            rows.append({"image": image_path.relative_to(ROOT).as_posix(),
                         "label": int(row["label_index"]),
                         "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest()})
        splits[split] = rows
    sets = [set(row["image"] for row in splits[s]) for s in ["train", "val", "test"]]
    assert all(not sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3))
    (output / "split_manifest.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    model = Patchcore(
        backbone="resnet18.tv_in1k", layers=["layer2", "layer3"],
        coreset_sampling_ratio=0.01, num_neighbors=9,
        pre_processor=Patchcore.configure_pre_processor(image_size=(224, 224)),
        visualizer=ImageVisualizer(output_dir=output / "images"),
    )
    engine = Engine(
        default_root_dir=output / "engine", accelerator="gpu" if args.device == "cuda" else "cpu",
        devices=1, logger=False, enable_progress_bar=False, enable_model_summary=False,
    )
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    engine.fit(model=model, datamodule=data)
    fit_seconds = time.perf_counter() - started
    metrics = engine.test(model=model, datamodule=data)
    engine.trainer.save_checkpoint(output / "baseline.ckpt")
    bank = model.model.memory_bank
    report = {
        "purpose": "deployment smoke baseline; not a full MVTec AD result",
        "upstream_commit": "fcf9ffa7382c1bfdd85e65bf7b2ecae0f6a9c1af",
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "seed": args.seed, "device": args.device, "backbone": "resnet18.tv_in1k",
        "image_size": [224, 224], "batch_size": 1, "coreset_sampling_ratio": 0.01,
        "num_neighbors": 9, "split_counts": {k: len(v) for k, v in splits.items()},
        "metrics": metrics, "fit_wall_seconds": fit_seconds,
        "memory_bank_shape": list(bank.shape),
        "memory_bank_bytes": bank.numel() * bank.element_size(),
        "peak_torch_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
        "timing_note": "fit timing includes feature extraction and coreset; no speedup claim",
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (ROOT / "course/outputs/latest_baseline.txt").write_text(str(output), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
