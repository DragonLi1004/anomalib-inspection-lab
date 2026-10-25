"""Select pixel-mask refinement using each model's validation split only."""
import hashlib,json,time
from baseline import ROOT
import torch
from optimize_exact import sources,setup
from mask_refinement import CANDIDATES,refine_mask

OUT=ROOT/'course/outputs/accuracy-20261025'

def f1(c):
    tp,fp,fn=c
    return 2*tp/max(1,2*tp+fp+fn)

def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    report={'phase':'validation-only','candidates':CANDIDATES,'runs':[],'complete':False,
        'selection_rule':'Maximum validation micro pixel F1; prefer identity then smaller kernels on exact ties. Per-run selection; no test metrics used.',
        'limitations':'Historical test sets were already evaluated in earlier work; this is retrospective development, not fresh external validation.'}
    for row in sources():
        m,data=setup(row)
        counts=[[0,0,0] for _ in CANDIDATES]
        with torch.inference_mode():
            for batch in data.val_dataloader():
                m.pre_processor.on_validation_batch_start(None,None,batch,0)
                pred=m.post_processor(m.model(batch.image.cuda()))
                gt=batch.gt_mask.cuda().bool().reshape(pred.pred_mask.shape)
                for i,config in enumerate(CANDIDATES):
                    mask=refine_mask(pred.pred_mask,pred.pred_score,**config)
                    values=[(mask&gt).sum(),(mask&~gt).sum(),(~mask&gt).sum()]
                    for j,v in enumerate(values): counts[i][j]+=int(v)
        scores=[f1(c) for c in counts]
        best=max(range(len(scores)),key=lambda i:(scores[i],-i))
        item={'source_run':row['run_id'],'dataset':row['dataset'],'seed':row['seed'],
              'selected':CANDIDATES[best],'baseline_f1':scores[0],'selected_f1':scores[best],
              'candidate_f1':scores,'counts':counts,'validation_images':len(data.val_data)}
        report['runs'].append(item)
        (OUT/'selection.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(item),flush=True)
        del m,data
    report['complete']=True
    report['source_sha256']=hashlib.sha256((ROOT/'course/mask_refinement.py').read_bytes()).hexdigest()
    (OUT/'selection.json').write_text(json.dumps(report,indent=2),encoding='utf-8')

if __name__=='__main__':main()
