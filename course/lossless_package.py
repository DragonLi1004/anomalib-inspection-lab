"""Lossless checkpoint packages with cross-model ZIP-payload deduplication.

The package is a storage/deployment artifact. Restoration reconstructs original
checkpoint bytes, including original ZIP headers, padding and central directory.
No torch/pickle deserialization is used for packing or restoring.
"""
import argparse,hashlib,json,re,struct,zipfile
from pathlib import Path

FORMAT='anomalib-lossless-checkpoints-v1'

def sha(data):return hashlib.sha256(data).hexdigest()

def checkpoint_chunks(path):
    """Split at original ZIP payload boundaries, keeping every source byte."""
    raw=path.read_bytes()
    with zipfile.ZipFile(path) as archive:
        entries=sorted(archive.infolist(),key=lambda info:info.header_offset)
        cursor=0
        for info in entries:
            offset=info.header_offset
            if raw[offset:offset+4]!=b'PK\x03\x04':raise ValueError('Invalid local ZIP header')
            name_length,extra_length=struct.unpack_from('<HH',raw,offset+26)
            start=offset+30+name_length+extra_length
            end=start+info.compress_size
            if start<cursor or end>len(raw):raise ValueError('Overlapping or truncated ZIP payload')
            if start>cursor:yield raw[cursor:start]
            if end>start:yield raw[start:end]
            cursor=end
        if cursor<len(raw):yield raw[cursor:]

def pack_checkpoints(models,destination):
    """models maps safe model IDs to trusted, local .ckpt paths."""
    destination=Path(destination)
    if destination.exists():raise FileExistsError(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    manifest={'format':FORMAT,'models':{},'blob_sizes':{}}
    temporary=destination.with_name(destination.name+'.partial')
    if temporary.exists():raise FileExistsError(temporary)
    try:
        with zipfile.ZipFile(temporary,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
            for model_id,path in models.items():
                if not re.fullmatch(r'[A-Za-z0-9_-]+',model_id):raise ValueError('Invalid model ID')
                path=Path(path);parts=[]
                for chunk in checkpoint_chunks(path):
                    digest=sha(chunk);parts.append(digest)
                    if digest not in manifest['blob_sizes']:
                        manifest['blob_sizes'][digest]=len(chunk)
                        archive.writestr('blobs/'+digest,chunk)
                manifest['models'][model_id]={'sha256':sha(path.read_bytes()),'bytes':path.stat().st_size,'parts':parts}
            archive.writestr('manifest.json',json.dumps(manifest,sort_keys=True,separators=(',',':')))
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    original=sum(m['bytes'] for m in manifest['models'].values())
    return {'models':len(models),'original_bytes':original,'package_bytes':destination.stat().st_size,
            'reduction_percent':(1-destination.stat().st_size/original)*100,
            'unique_payload_bytes':sum(manifest['blob_sizes'].values()),'package_sha256':sha(destination.read_bytes())}

def restore_checkpoint(package,model_id,destination):
    """Restore one model to a caller-chosen path; never overwrite different bytes."""
    destination=Path(destination)
    with zipfile.ZipFile(package) as archive:
        manifest=json.loads(archive.read('manifest.json'))
        if manifest.get('format')!=FORMAT:raise ValueError('Unsupported package format')
        model=manifest['models'][model_id]
        if destination.exists():
            if sha(destination.read_bytes())==model['sha256']:return destination
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True,exist_ok=True)
        temporary=destination.with_name(destination.name+'.partial')
        if temporary.exists():raise FileExistsError(temporary)
        digest=hashlib.sha256();total=0
        try:
            with temporary.open('xb') as stream:
                for part in model['parts']:
                    if not re.fullmatch(r'[0-9a-f]{64}',part):raise ValueError('Invalid blob identifier')
                    size=manifest['blob_sizes'][part]
                    if type(size) is not int or not 0<=size<=model['bytes']:raise ValueError('Invalid blob size')
                    with archive.open('blobs/'+part) as blob:
                        data=blob.read(size+1)
                    if len(data)!=size or sha(data)!=part:raise ValueError('Corrupt package blob')
                    total+=len(data)
                    if total>model['bytes']:raise ValueError('Restored model exceeds recorded size')
                    stream.write(data);digest.update(data)
            if total!=model['bytes'] or digest.hexdigest()!=model['sha256']:raise ValueError('Restored checkpoint checksum mismatch')
            temporary.replace(destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    return destination

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    pack=sub.add_parser('pack');pack.add_argument('--run',action='append',required=True);pack.add_argument('--output',type=Path,required=True)
    restore=sub.add_parser('restore');restore.add_argument('--package',type=Path,required=True);restore.add_argument('--run',required=True);restore.add_argument('--output',type=Path,required=True)
    args=p.parse_args();root=Path(__file__).resolve().parents[1]
    if args.action=='pack':
        models={}
        for run in args.run:
            if not re.fullmatch(r'[A-Za-z0-9_-]+',run):raise ValueError('Invalid run ID')
            models[run]=root/'course/outputs'/run/'baseline.ckpt'
        print(json.dumps(pack_checkpoints(models,args.output),indent=2))
    else:print(restore_checkpoint(args.package,args.run,args.output))

if __name__=='__main__':main()
