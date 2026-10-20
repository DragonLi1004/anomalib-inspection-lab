"""Validation-only selection, followed by locked paired exact-inference audit."""
import argparse, gc, hashlib, json, random, statistics, time
from pathlib import Path
from baseline import ROOT
import torch
from lightning import seed_everything
from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine
from exact_search import configure_exact

OUT=ROOT/'course/outputs/exact-optimization'
CANDIDATES={'upstream':None,'lifetime':None,'chunk196':196,'chunk392':392,'chunk512':512,'chunk784':784}

def apply(model,name):
    configure_exact(model.model,CANDIDATES[name],enabled=name!='upstream')

def sources():
    return [r for r in json.loads((ROOT/'course/results/multiseed_metrics.json').read_text())['runs'] if r['method']=='baseline']

def setup(row):
    gc.collect(); torch.cuda.empty_cache()
    seed_everything(row['seed'],workers=True)
    run=ROOT/'course/outputs'/row['run_id']
    data=MVTecAD(root=ROOT/'course/data/MVTecAD',category=row['dataset'],train_batch_size=1,eval_batch_size=1,num_workers=0,val_split_mode='from_test',val_split_ratio=.5,seed=row['seed'])
    data.setup()
    manifest=json.loads((run/'split_manifest.json').read_text())
    for split in ['train','val','test']:
        paths=[Path(p).relative_to(ROOT).as_posix() for p in getattr(data,split+'_data').samples.image_path.tolist()]
        assert paths==[r['image'] for r in manifest[split]],f'Split mismatch: {split}'
        for item in manifest[split]:
            assert hashlib.sha256((ROOT/item['image']).read_bytes()).hexdigest()==item['sha256']
    m=Patchcore.load_from_checkpoint(run/'baseline.ckpt',map_location='cpu',weights_only=False).eval().cuda()
    m.visualizer=None
    return m,data

def compare(m,loader,names):
    stats={name:{'images':0,'unequal_scores':0,'unequal_maps':0,'max_score_error':0.,'max_map_error':0.} for name in names}
    samples=[]
    with torch.inference_mode():
        for batch in loader:
            x=m.pre_processor(batch.image.cuda())
            if len(samples)<3: samples.append(x)
            apply(m,'upstream'); ref=m.model(x)
            for name in names:
                apply(m,name); out=m.model(x); s=stats[name];s['images']+=len(x)
                s['unequal_scores']+=int(not torch.equal(ref.pred_score,out.pred_score))
                s['unequal_maps']+=int(not torch.equal(ref.anomaly_map,out.anomaly_map))
                s['max_score_error']=max(s['max_score_error'],float((ref.pred_score-out.pred_score).abs().max()))
                s['max_map_error']=max(s['max_map_error'],float((ref.anomaly_map-out.anomaly_map).abs().max()))
            del ref,out,x
    return stats,samples

def benchmark(m,samples,names,rounds=9,repetitions=12):
    result={n:{'block_median_ms':[],'peak_allocated_bytes':[],'peak_extra_bytes':[]} for n in names}
    rng=random.Random(20261020)
    with torch.inference_mode():
        for n in names:
            apply(m,n)
            for _ in range(12):
                out=m.model(samples[0]); del out
        torch.cuda.synchronize()
        for r in range(rounds):
            order=list(names);rng.shuffle(order)
            for n in order:
                apply(m,n);times=[]
                for i in range(repetitions):
                    x=samples[i%len(samples)]
                    torch.cuda.synchronize();t=time.perf_counter()
                    out=m.model(x);torch.cuda.synchronize()
                    times.append((time.perf_counter()-t)*1000);del out
                result[n]['block_median_ms'].append(statistics.median(times))
        for n in names:
            apply(m,n)
            for x in samples:
                torch.cuda.synchronize();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
                out=m.model(x);torch.cuda.synchronize()
                peak=torch.cuda.max_memory_allocated()
                result[n]['peak_allocated_bytes'].append(peak)
                result[n]['peak_extra_bytes'].append(peak-base)
                del out
    for n,d in result.items():
        d['latency_median_ms']=statistics.median(d['block_median_ms'])
        d['peak_allocated_bytes_max']=max(d['peak_allocated_bytes'])
        d['peak_extra_bytes_max']=max(d['peak_extra_bytes'])
    return result

def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2),encoding='utf-8')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['select','test']);args=parser.parse_args()
    torch.set_num_threads(4)
    if args.phase=='select':
        selection={'phase':'validation_only','candidates':CANDIDATES,'runs':[], 'source_sha256':hashlib.sha256((ROOT/'course/exact_search.py').read_bytes()).hexdigest()}
        for row in [r for r in sources() if r['seed']==42]:
            m,data=setup(row)
            stats,samples=compare(m,data.val_dataloader(),list(CANDIDATES)[1:])
            timing=benchmark(m,samples,list(CANDIDATES),rounds=7)
            selection['runs'].append({'dataset':row['dataset'],'source_run':row['run_id'],'parity':stats,'benchmark':timing})
            print(json.dumps(selection['runs'][-1]),flush=True)
            del m,data,samples;torch.cuda.empty_cache()
        eligible=[n for n in list(CANDIDATES)[1:] if all(r['parity'][n]['unequal_scores']==0 and r['parity'][n]['unequal_maps']==0 and r['benchmark'][n]['peak_extra_bytes_max']<r['benchmark']['upstream']['peak_extra_bytes_max'] for r in selection['runs'])]
        if not eligible:
            save(OUT/'selection_failed.json',selection);raise RuntimeError('No candidate satisfies exact validation parity and memory reduction on every category')
        # Prefer the lowest mean workspace ratio; break equal-memory ties by latency.
        selected=min(eligible,key=lambda n:(statistics.mean(r['benchmark'][n]['peak_extra_bytes_max']/r['benchmark']['upstream']['peak_extra_bytes_max'] for r in selection['runs']), statistics.mean(r['benchmark'][n]['latency_median_ms']/r['benchmark']['upstream']['latency_median_ms'] for r in selection['runs'])))
        selection.update(selected=selected,eligible=eligible,selection_rule='Exact raw output equality and lower peak extra GPU allocation in every validation category; then minimum mean workspace ratio, using latency only to break ties',frozen_at=time.strftime('%Y-%m-%dT%H:%M:%S'))
        save(OUT/'selection.json',selection);print('LOCKED: '+selected,flush=True)
    else:
        plan=json.loads((OUT/'selection.json').read_text());name=plan['selected']
        assert plan['source_sha256']==hashlib.sha256((ROOT/'course/exact_search.py').read_bytes()).hexdigest(),'Source changed after selection'
        results={'phase':'locked_test','selection':plan,'runs':[]}
        for row in sources():
            m,data=setup(row)
            stats,samples=compare(m,data.test_dataloader(),[name])
            parity=stats[name]
            if parity['unequal_scores'] or parity['unequal_maps']:
                save(OUT/'test_failed.json',{'run':row['run_id'],'parity':parity});raise RuntimeError('Test raw outputs not identical; do not claim acceptance')
            measurements=benchmark(m,samples,['upstream',name],rounds=15)
            metrics={}
            for mode in ['upstream',name]:
                apply(m,mode)
                engine=Engine(default_root_dir=OUT/'engine'/row['run_id']/mode,accelerator='gpu',devices=1,logger=False,enable_progress_bar=False,enable_model_summary=False)
                metrics[mode]=engine.test(model=m,datamodule=data)[0]
            assert metrics['upstream']==metrics[name],(row['run_id'],metrics)
            record={'dataset':row['dataset'],'seed':row['seed'],'source_run':row['run_id'],'split_sha256':row['split_sha256'],'coreset_sha256':row['coreset_sha256'],'parity':parity,'benchmark':measurements,'metrics':metrics,'historical_metrics':row['metrics'][0]}
            results['runs'].append(record);save(OUT/'test_progress.json',results)
            print('TEST VERIFIED '+row['run_id'],flush=True)
            del m,data,samples,engine;torch.cuda.empty_cache()
        results['complete']=len(results['runs'])==9;save(OUT/'verified.json',results)
        print('ALL 9 BASELINE MODELS VERIFIED',flush=True)

if __name__=='__main__':main()
