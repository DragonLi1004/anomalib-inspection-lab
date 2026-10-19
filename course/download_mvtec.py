"""Download selected complete MVTec AD categories from a public Kaggle mirror.

Range requests avoid downloading the entire 5 GB ZIP. ZIP CRC and PNG decoding
validate transport; this repackaged mirror has no official archive SHA256.
"""
import argparse
import concurrent.futures
import hashlib
import io
import json
import struct
import threading
import time
import urllib.request
import zipfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "https://www.kaggle.com/api/v1/datasets/download/ipythonx/mvtec-ad"
CATEGORIES = ("bottle", "cable", "capsule")
EXPECTED = {"bottle": (209, 83, 63), "cable": (224, 150, 92), "capsule": (219, 132, 109)}


def request(url, start, end):
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=90) as response:
                if response.status != 206:
                    raise ValueError("Server must support byte ranges")
                data = response.read()
                if len(data) != end - start + 1:
                    raise ValueError("Incomplete range")
                return data
        except Exception:
            if attempt == 4:
                raise
            time.sleep(attempt + 1)


class RemoteZip(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.pos = url, size, 0

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def tell(self):
        return self.pos

    def read(self, count=-1):
        end = self.size - 1 if count < 0 else min(self.size - 1, self.pos + count - 1)
        if end < self.pos:
            return b""
        data = request(self.url, self.pos, end)
        self.pos += len(data)
        return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", nargs="+", choices=CATEGORIES, default=list(CATEGORIES))
    args = parser.parse_args()
    target = ROOT / "course/data/MVTecAD"
    target.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(SOURCE, headers={"Range": "bytes=0-0"}), timeout=60) as response:
        url = response.geturl()
        size = int(response.headers["Content-Range"].split("/")[-1])
    with zipfile.ZipFile(RemoteZip(url, size)) as archive:
        entries = [i for i in archive.infolist() if i.filename.split("/")[0] in args.categories or i.filename in ("license.txt", "readme.txt")]
    lock = threading.Lock()
    done = 0

    def fetch(info):
        nonlocal done
        dest = (target / info.filename).resolve()
        if not dest.is_relative_to(target.resolve()):
            raise ValueError("Unsafe archive path")
        if info.is_dir():
            return None
        if dest.exists() and dest.stat().st_size == info.file_size and zlib.crc32(dest.read_bytes()) == info.CRC:
            data = dest.read_bytes()
        else:
            header = request(url, info.header_offset, info.header_offset + 29)
            fields = struct.unpack("<4s5H3I2H", header)
            if fields[0] != b"PK\x03\x04" or fields[3] not in (0, 8):
                raise ValueError("Unsupported ZIP entry")
            start = info.header_offset + 30 + fields[-2] + fields[-1]
            packed = request(url, start, start + info.compress_size - 1)
            data = zlib.decompress(packed, -15) if info.compress_type == 8 else packed
            if len(data) != info.file_size or zlib.crc32(data) != info.CRC:
                raise ValueError("ZIP CRC mismatch: " + info.filename)
            if dest.suffix.lower() == ".png":
                from PIL import Image
                with Image.open(io.BytesIO(data)) as image:
                    image.verify()
            dest.parent.mkdir(parents=True, exist_ok=True)
            temp = dest.with_suffix(dest.suffix + ".partial")
            temp.write_bytes(data)
            temp.replace(dest)
        with lock:
            done += 1
            if done % 25 == 0 or done == len(entries):
                print(f"Downloaded/verified {done}/{len(entries)}", flush=True)
        return {"path": info.filename, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        files = list(pool.map(fetch, entries))
    counts = {}
    for category in args.categories:
        counts[category] = {split: len(list((target / category / split).rglob("*.png"))) for split in ("train", "test", "ground_truth")}
        if tuple(counts[category].values()) != EXPECTED[category]:
            raise ValueError(f"Unexpected category counts: {category}: {counts[category]}")
    manifest = {"source": SOURCE, "source_type": "third-party repackaged public mirror", "official": "https://www.mvtec.com/company/research/datasets/mvtec-ad", "license": "CC BY-NC-SA 4.0; academic non-commercial use", "counts": counts, "files": files}
    (target / "download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(counts, indent=2), flush=True)


if __name__ == "__main__":
    main()
