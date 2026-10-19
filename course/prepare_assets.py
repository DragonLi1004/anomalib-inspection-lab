"""Download pinned public demo data and ImageNet weights into this project."""

import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = [
    (
        "https://github.com/open-edge-platform/anomalib/releases/download/hazelnut_toy_dataset/hazelnut_toy.zip",
        "course/data/hazelnut_toy.zip",
        "c286c4c3e3924442bfe5ed549699b398f910531ec318c729bb6c25917cfad56e",
    ),
    (
        "https://download.pytorch.org/models/resnet18-f37072fd.pth",
        "course/models/torch/hub/checkpoints/resnet18-f37072fd.pth",
        "f37072fd",
    ),
]


def prepare():
    records = []
    for url, relative, expected in ASSETS:
        path = ROOT / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".partial")
            print(f"Downloading {url}", flush=True)
            with urllib.request.urlopen(url, timeout=180) as response, temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            with temporary.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if not digest.startswith(expected):
                raise RuntimeError(f"Checksum mismatch: {temporary}")
            temporary.replace(path)
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if not digest.startswith(expected):
            raise RuntimeError(f"Checksum mismatch: {path}")
        records.append({"url": url, "path": relative, "sha256": digest, "bytes": path.stat().st_size})
    destination = (ROOT / "course/data").resolve()
    with zipfile.ZipFile(ROOT / ASSETS[0][1]) as archive:
        for member in archive.infolist():
            if not (destination / member.filename).resolve().is_relative_to(destination):
                raise ValueError("Archive member is outside the data directory")
        archive.extractall(destination)
    (destination / "assets.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print("Demo data and pretrained weights are ready.")


if __name__ == "__main__":
    prepare()
