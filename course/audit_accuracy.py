"""Evaluate frozen validation choices exactly once, paired with the source checkpoint."""
import gc,hashlib,json,random,shutil,statistics,time
from pathlib import Path
import numpy as np
from baseline import ROOT
import torch
from anomalib.engine import Engine
from anomalib.post_processing import PostProcessor
from optimize_exact import sources,setup
from peak_refinement import install_peak
from select_peak import f1

OUT=ROOT/'course/outputs/accuracy-20261025'


def processor_original(model):
    p=model.post_processor
    original=PostProcessor(enable_normalization=p.enable_normalization,enable_thresholding=p.enable_thresholding,
        enable_threshold_matching=p.enable_threshold_matching,image_sensitivity=p.image_sensitivity,pixel_sensitivity=p.pixel_sensitivity)
    original.load_state_dict(p.state_dict())
    return original.to(model.device)


def timing(model,sample,old,new):
    measurements={name:[] for name in ('baseline','peak')}
    peaks={}
    model.eval()
    with torch.inference_mode():
        for name,p in [('baseline',old),('peak',new)]:
            for _ in range(10): p(model.model(sample))
        rng=random.Random(20261025)
        for _ in range(40):
            order=[('baseline',old),('peak',new)];rng.shuffle(order)
            for name,p in order:
                torch.cuda.synchronize();t=time.perf_counter()
                result=p(model.model(sample));torch.cuda.synchronize()
                measurements[name].append((time.perf_counter()-t)*1000)
                del result
        for name,p in [('baseline',old),('peak',new)]:
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            result=p(model.model(sample));torch.cuda.synchronize()
            peaks[name]=torch.cuda.max_memory_allocated();del result
    return {'median_ms':{k:statistics.median(v) for k,v in measurements.items()},'samples_ms':measurements,
            'peak_allocated_bytes':peaks,'scope':'preprocessed batch=1; original backbone + nearest-neighbor + postprocessor; excludes image I/O/preprocessing/rendering; 10 warmups/40 interleaved repeats'}


def main():
    torch.set_num_threads(4)
    selection_path=OUT/'peak-selection.json'
    selection=json.loads(selection_path.read_text())
    assert selection['complete'] and len(selection['runs'])==9
    assert selection['source_sha256']==hashlib.sha256((ROOT/'course/peak_refinement.py').read_bytes()).hexdigest()
    frozen=OUT/'frozen-peak-selection.json'
    if frozen.exists(): raise RuntimeError('Frozen evaluation already started; inspect existing artifacts before rerunning')
    frozen.write_bytes(selection_path.read_bytes())
    report={'complete':False,'selection_sha256':hashlib.sha256(frozen.read_bytes()).hexdigest(),
            'runs':[],'purpose':'Retrospective paired accuracy regression on existing fixed splits; no new unseen-data generalization claim'}
    for row,choice in zip(sources(),selection['runs'],strict=True):
        assert row['run_id']==choice['source_run']
        m,data=setup(row)
        source=ROOT/'course/outputs'/row['run_id']
        engine=Engine(default_root_dir=OUT/'engine'/row['run_id']/'baseline',accelerator='gpu',devices=1,
            logger=False,enable_progress_bar=False,enable_model_summary=False)
        before=engine.test(model=m,datamodule=data)[0]
        assert before==row['metrics'][0],('Historical baseline mismatch',before,row['metrics'][0])
        install_peak(m,choice['alpha'])
        engine=Engine(default_root_dir=OUT/'engine'/row['run_id']/'peak',accelerator='gpu',devices=1,
            logger=False,enable_progress_bar=False,enable_model_summary=False)
        after=engine.test(model=m,datamodule=data)[0]
        # Do not stop/hide a failed candidate: record every model's held-out result.
        for key in ('image_AUROC','image_F1Score','pixel_AUROC'):
            assert before[key]==after[key],f'Invariant changed: {key}'
        m=m.eval().cuda()
        old=processor_original(m);new=m.post_processor
        counts=[];changed_pixels=0;sample=None;sample_path=None
        with torch.inference_mode():
            for batch in data.test_dataloader():
                m.pre_processor.on_test_batch_start(None,None,batch,0)
                x=batch.image.cuda();raw=m.model(x);a=old(raw);b=new(raw)
                assert torch.equal(a.pred_score,b.pred_score) and torch.equal(a.pred_label,b.pred_label) and torch.equal(a.anomaly_map,b.anomaly_map)
                gt=batch.gt_mask.cuda().bool().reshape(a.pred_mask.shape)
                counts.append([[int((p & gt).sum()),int((p & ~gt).sum()),int((~p & gt).sum())] for p in (a.pred_mask,b.pred_mask)])
                changed_pixels+=int((a.pred_mask!=b.pred_mask).sum())
                if sample is None: sample=x;sample_path=str(batch.image_path[0])
        counts=np.array(counts,dtype=np.int64)
        computed=f1(counts.sum(0))
        assert abs(computed[0]-before['pixel_F1Score'])<1e-6 and abs(computed[1]-after['pixel_F1Score'])<1e-6
        rng=np.random.default_rng(20261025);idx=rng.integers(0,len(counts),size=(2000,len(counts)))
        boot=f1(counts[idx].sum(1));ci=np.quantile(boot[:,1]-boot[:,0],[.025,.975]).tolist()
        performance=timing(m,sample,old,new)
        item={'dataset':row['dataset'],'seed':row['seed'],'source_run':row['run_id'],
            'alpha':choice['alpha'],'before':before,'after':after,
            'delta_pp':{k:100*(after[k]-before[k]) for k in before},
            'all_metrics_non_decreasing':all(after[k]>=before[k] for k in before),
            'test_evaluations':len(counts),'changed_mask_pixels':changed_pixels,
            'pixel_counts':{'before':counts.sum(0)[0].tolist(),'after':counts.sum(0)[1].tolist()},
            'pixel_f1_gain_bootstrap_ci95':ci,'counts_per_image':counts.tolist(),
            'score_label_map_bit_exact':True,'timing':performance,
            'checkpoint_sha256':hashlib.sha256((source/'baseline.ckpt').read_bytes()).hexdigest()}
        report['runs'].append(item)
        (OUT/'test-results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in item.items() if k not in ('counts_per_image','timing')}),flush=True)
        del m,data,engine,old,new,sample,x,raw,a,b,gt
        gc.collect();torch.cuda.empty_cache()
    report['complete']=True
    report['all_metrics_non_decreasing']=all(r['all_metrics_non_decreasing'] for r in report['runs'])
    report['test_evaluations']=sum(r['test_evaluations'] for r in report['runs'])
    (OUT/'test-results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')

if __name__=='__main__':main()
