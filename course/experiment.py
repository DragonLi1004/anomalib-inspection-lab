"""Reproducible upstream/PCA experiments; independent calibration/test splits."""
import argparse
import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path

# This sets offline weights/cache locations before torch/timm imports.
from baseline import ROOT
import torch
from lightning import seed_everything
from anomalib.data import Folder, MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore
from anomalib.visualization.image import ImageVisualizer
from compact_patchcore import CompactPatchcore
from download_mvtec import EXPECTED


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["hazelnut_toy", *EXPECTED], default="hazelnut_toy")
    parser.add_argument("--method", choices=["baseline", "pca96", "pca192"], default="baseline")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed_everything(args.seed, workers=True)
    torch.set_num_threads(4)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    output = ROOT / "course/outputs" / (f"{args.dataset}-{args.method}-" + time.strftime("%Y%m%d-%H%M%S"))
    output.mkdir(parents=True, exist_ok=False)
    common = dict(train_batch_size=1, eval_batch_size=1, num_workers=0,
                  val_split_mode="from_test", val_split_ratio=0.5, seed=args.seed)
    if args.dataset == "hazelnut_toy":
        data = Folder(name=args.dataset, root=ROOT / "course/data/hazelnut_toy", normal_dir="good",
                      abnormal_dir=["colour", "crack"], mask_dir="mask",
                      test_split_mode="from_dir", test_split_ratio=0.2, **common)
    else:
        root = ROOT / "course/data/MVTecAD"
        counts = tuple(len(list((root / args.dataset / split).rglob("*.png"))) for split in ("train", "test", "ground_truth"))
        if counts != EXPECTED[args.dataset]:
            raise RuntimeError(f"Dataset incomplete: {counts}. Run download_mvtec.py first.")
        data = MVTecAD(root=root, category=args.dataset, **common)
    data.setup()
    splits = {}
    for split in ("train", "val", "test"):
        rows = []
        for row in getattr(data, split + "_data").samples.to_dict(orient="records"):
            path = Path(row["image_path"])
            rows.append({"image": path.relative_to(ROOT).as_posix(), "label": int(row["label_index"]),
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        splits[split] = rows
    sets = [{r["image"] for r in splits[s]} for s in ("train", "val", "test")]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Data split overlap")
    manifest = json.dumps(splits, sort_keys=True, indent=2)
    (output / "split_manifest.json").write_text(manifest, encoding="utf-8")
    kwargs = dict(backbone="resnet18.tv_in1k", layers=["layer2", "layer3"], coreset_sampling_ratio=0.01,
                  num_neighbors=9, pre_processor=Patchcore.configure_pre_processor(image_size=(224, 224)),
                  visualizer=ImageVisualizer(output_dir=output / "images"))
    model = Patchcore(**kwargs) if args.method == "baseline" else CompactPatchcore(projection_dim=int(args.method[3:]), **kwargs)
    engine = Engine(default_root_dir=output / "engine", accelerator="gpu" if args.device == "cuda" else "cpu",
                    devices=1, logger=False, enable_progress_bar=False, enable_model_summary=False)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    engine.fit(model=model, datamodule=data)
    fit_seconds = time.perf_counter() - start
    metrics = engine.test(model=model, datamodule=data)
    checkpoint = output / "baseline.ckpt"
    engine.trainer.save_checkpoint(checkpoint)
    bank = model.model.memory_bank
    projection_bytes = sum(x.numel() * x.element_size() for n, x in model.model.named_buffers() if n.startswith("projection_"))
    # Fixed real test tensor; no IO, visualization, preprocessing or postprocessing in timing.
    sample = next(iter(data.test_dataloader())).image[:1].to(model.device)
    sample = model.pre_processor(sample)
    model.model.eval()
    def sync():
        if args.device == "cuda":
            torch.cuda.synchronize()
    timings = []
    with torch.inference_mode():
        for _ in range(10):
            model.model(sample)
        sync()
        for _ in range(30):
            start = time.perf_counter()
            model.model(sample)
            sync()
            timings.append((time.perf_counter() - start) * 1000)
    report = {
        "purpose": "course experiment; custom disjoint validation/test protocol, not official full-test benchmark",
        "protocol_version": "paired-coreset-v2",
        "course_code_sha256": hashlib.sha256(b"".join((ROOT / "course" / name).read_bytes() for name in ("experiment.py", "compact_patchcore.py"))).hexdigest(),
        "coreset_sha256": getattr(model.model, "coreset_sha256", hashlib.sha256(bank.detach().cpu().contiguous().numpy().tobytes()).hexdigest()),
        "dataset": args.dataset, "method": args.method, "seed": args.seed, "device": args.device,
        "upstream_commit": "fcf9ffa7382c1bfdd85e65bf7b2ecae0f6a9c1af",
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "backbone": "resnet18.tv_in1k", "image_size": [224, 224], "batch_size": 1,
        "coreset_sampling_ratio": 0.01, "num_neighbors": 9, "val_split_ratio": 0.5,
        "split_counts": {k: len(v) for k, v in splits.items()},
        "split_sha256": hashlib.sha256(manifest.encode()).hexdigest(), "metrics": metrics,
        "fit_wall_seconds": fit_seconds, "memory_bank_shape": list(bank.shape),
        "memory_bank_bytes": bank.numel() * bank.element_size(), "projection_bytes": projection_bytes,
        "search_state_bytes": bank.numel() * bank.element_size() + projection_bytes,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "latency_median_ms": statistics.median(timings), "latency_samples_ms": timings,
        "timing_note": "batch=1; 10 warmups + 30 synchronized forwards of first test tensor; includes backbone, projection, kNN, raw map; excludes IO, preprocessing, calibrated postprocessing and rendering",
        "peak_torch_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
        "pca_fit_data": "normal training coreset only" if args.method != "baseline" else None,
        "retained_variance": float(model.model.retained_variance) if args.method != "baseline" else None,
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (ROOT / "course/outputs/latest_baseline.txt").write_text(str(output), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
