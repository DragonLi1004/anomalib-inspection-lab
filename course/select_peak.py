"""Freeze peak-aware calibration before test: per-image paired validation bootstrap."""
import hashlib,json
import numpy as np
from baseline import ROOT
import torch
from optimize_exact import sources,setup
from peak_refinement import ALPHAS,peak_mask
OUT=ROOT/'course/outputs/accuracy-20261025'


def f1(counts):
    tp,fp,fn=np.moveaxis(np.asarray(counts),-1,0)
    return 2*tp/np.maximum(1,2*tp+fp+fn)


def main():
    torch.set_num_threads(4)
    report={'complete':False,'phase':'validation-only','alphas':ALPHAS,'runs':[],
        'selection_rule':'Eligible only if validation pixel F1 gain >= 0.005 and paired image-bootstrap 95% lower bound > 0. Maximize lower bound, tie prefer smaller alpha. Otherwise alpha=0 (upstream). 2000 draws, RNG 20261025. Each run uses only its own validation split.',
        'test_policy':'Freeze once; evaluate all 9 runs once. No replacing individual test regressions with baseline after seeing test labels.',
        'limitations':'Retrospective existing data; validation/test splits are disjoint within run but images overlap across seeds and historical tests were viewed.'}
    for row in sources():
        m,data=setup(row)
        all_counts=[]
        with torch.inference_mode():
            for batch in data.val_dataloader():
                m.pre_processor.on_validation_batch_start(None,None,batch,0)
                raw=m.model(batch.image.cuda())
                pred=m.post_processor(raw)
                gt=batch.gt_mask.cuda().bool().reshape(raw.anomaly_map.shape)
                counts=[]
                for alpha in ALPHAS:
                    mask=pred.pred_mask if alpha==0 else peak_mask(raw.anomaly_map,m.post_processor.pixel_threshold,alpha)
                    counts.append([int((mask & gt).sum()),int((mask & ~gt).sum()),int((~mask & gt).sum())])
                all_counts.append(counts)
        counts=np.array(all_counts,dtype=np.int64)
        values=f1(counts.sum(0))
        rng=np.random.default_rng(20261025)
        samples=rng.integers(0,len(counts),size=(2000,len(counts)))
        boot=f1(counts[samples].sum(1))
        ci=np.quantile(boot-boot[:,0,None],[.025,.975],axis=0)
        eligible=[i for i in range(1,len(ALPHAS)) if values[i]-values[0]>=.005 and ci[0,i]>0]
        best=max(eligible,key=lambda i:(ci[0,i],-ALPHAS[i])) if eligible else 0
        item={'source_run':row['run_id'],'dataset':row['dataset'],'seed':row['seed'],
              'alpha':ALPHAS[best],'baseline_f1':float(values[0]),'selected_f1':float(values[best]),
              'selected_gain_ci95':ci[:,best].tolist(),'candidate_f1':values.tolist(),
              'candidate_gain_ci95':ci.T.tolist(),'validation_images':len(counts),'counts_per_image':counts.tolist()}
        report['runs'].append(item)
        (OUT/'peak-selection.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in item.items() if k!='counts_per_image'}),flush=True)
        del m,data
    report['complete']=True
    report['source_sha256']=hashlib.sha256((ROOT/'course/peak_refinement.py').read_bytes()).hexdigest()
    (OUT/'peak-selection.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
if __name__=='__main__':main()
