"""Resumable ranged download from a public URL, verified with SHA-256."""
import argparse, concurrent.futures, hashlib, math, time, urllib.request
from pathlib import Path

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("output", type=Path)
    parser.add_argument("sha256")
    args=parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():
        with args.output.open("rb") as stream:
            if hashlib.file_digest(stream,"sha256").hexdigest()==args.sha256:
                print("Already verified",flush=True)
                return
        raise RuntimeError("Existing output has unexpected checksum")
    with urllib.request.urlopen(urllib.request.Request(args.url,headers={"Range":"bytes=0-0"}),timeout=60) as response:
        if response.status!=206:
            raise RuntimeError("Server does not support range requests")
        total=int(response.headers["Content-Range"].split("/")[-1])
    block=8*1024*1024
    chunks=math.ceil(total/block)
    parts=args.output.parent/(args.output.name+".parts")
    parts.mkdir(exist_ok=True)
    def fetch(i):
        start=i*block
        end=min(total,(i+1)*block)-1
        path=parts/f"{i:05d}"
        if path.exists() and path.stat().st_size==end-start+1:
            return i
        for attempt in range(4):
            try:
                request=urllib.request.Request(args.url,headers={"Range":f"bytes={start}-{end}"})
                with urllib.request.urlopen(request,timeout=90) as response:
                    if response.status!=206 or response.headers["Content-Range"]!=f"bytes {start}-{end}/{total}":
                        raise RuntimeError("Unexpected response range")
                    with path.open("wb") as output:
                        while data:=response.read(1024*1024):
                            output.write(data)
                if path.stat().st_size!=end-start+1:
                    raise RuntimeError("Incomplete part")
                return i
            except Exception:
                if attempt==3: raise
                time.sleep(attempt+1)
    print(f"Downloading {total/1024**3:.2f} GiB in {chunks} parts (12 connections)",flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        futures=[executor.submit(fetch,i) for i in range(chunks)]
        for done,_ in enumerate(concurrent.futures.as_completed(futures),1):
            _.result()
            if done%16==0 or done==chunks:
                print(f"{done}/{chunks} parts ({100*done/chunks:.1f}%)",flush=True)
    temporary=args.output.with_suffix(args.output.suffix+".partial")
    digest=hashlib.sha256()
    with temporary.open("wb") as output:
        for i in range(chunks):
            with (parts/f"{i:05d}").open("rb") as source:
                while data:=source.read(1024*1024):
                    digest.update(data)
                    output.write(data)
    if digest.hexdigest()!=args.sha256:
        raise RuntimeError(f"Checksum mismatch: {digest.hexdigest()}")
    temporary.replace(args.output)
    print(f"Verified: {args.output} SHA256={digest.hexdigest()}",flush=True)

if __name__=="__main__":
    main()
