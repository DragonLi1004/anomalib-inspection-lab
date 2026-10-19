"""Verify imports and actual CUDA computation, then record the environment."""

import importlib.metadata
import json
import platform
from pathlib import Path

import torch
import anomalib
from anomalib.models import Patchcore
from anomalib.engine import Engine


def main():
    report = {
        "python": platform.python_version(),
        "anomalib_source": anomalib.__file__,
        "packages": {name: importlib.metadata.version(name) for name in
                     ["anomalib", "torch", "torchvision", "lightning", "timm", "numpy", "scikit-learn"]},
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
    }
    if report["cuda_available"]:
        report["gpu"] = torch.cuda.get_device_name(0)
        report["gpu_total_bytes"] = torch.cuda.get_device_properties(0).total_memory
        x = torch.randn(256, 256, device="cuda")
        result = x @ x.T
        torch.cuda.synchronize()
        report["cuda_matmul_finite"] = bool(result.isfinite().all().item())
        assert report["cuda_matmul_finite"]
    output = Path(__file__).resolve().parent / "outputs/environment.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
