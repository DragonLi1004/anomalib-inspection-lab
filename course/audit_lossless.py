"""Reproduce package size savings and verify restored upstream model metrics."""
import gc,hashlib,json,time
from baseline import ROOT
import torch
from anomalib.models import Patchcore
from anomalib.engine import Engine
from optimize_exact import setup
from lossless_package import pack_checkpoints,restore_checkpoint


def main():
    torch.set_num_threads(4)
    sources=[r for r in json.loads((ROOT/'course/results/multiseed_metrics.json').read_text())['runs'] if r['method']=='baseline']
    out=ROOT/'course/outputs'/('lossless-'+time.strftime('%Y%m%d-%H%M%S'));out.mkdir(parents=True)
    models={r['run_id']:ROOT/'course/outputs'/r['run_id']/'baseline.ckpt' for r in sources}
    package=out/'baseline-9-models.alpack'
    start=time.perf_counter();bundle=pack_checkpoints(models,package);bundle['pack_seconds']=time.perf_counter()-start
    result={'format':'lossless-course-audit-v1','package':package.relative_to(ROOT).as_posix(),'bundle':bundle,'runs':[],
      'method':'Deduplicate original checkpoint ZIP payloads by SHA-256, then ZIP Deflate level 6; restore exact original bytes before ordinary inference.',
      'device_name':torch.cuda.get_device_name(),'torch_version':torch.__version__}
    print(json.dumps(bundle),flush=True)
    for row in sources:
        run=row['run_id'];original=models[run]
        single=out/'single'/f'{run}.alpack';single_stats=pack_checkpoints({run:original},single)
        start=time.perf_counter();restored=restore_checkpoint(package,run,out/'restored'/run/'baseline.ckpt');restore_seconds=time.perf_counter()-start
        original_hash=hashlib.sha256(original.read_bytes()).hexdigest()
        assert original_hash==hashlib.sha256(restored.read_bytes()).hexdigest()
        # Same verified split construction as the previous audit, but never enable
        # memory-efficient execution: this is the original upstream inference path.
        reference,data=setup(row)
        model=Patchcore.load_from_checkpoint(restored,map_location='cpu',weights_only=False).eval()
        for key,value in reference.state_dict().items():assert torch.equal(value.cpu(),model.state_dict()[key]),key
        del reference;gc.collect();torch.cuda.empty_cache();model.visualizer=None
        engine=Engine(default_root_dir=out/'engine'/run,accelerator='gpu',devices=1,logger=False,enable_progress_bar=False,enable_model_summary=False)
        metrics=engine.test(model=model,datamodule=data)[0]
        assert metrics==row['metrics'][0],(run,metrics,row['metrics'][0])
        record={'dataset':row['dataset'],'seed':row['seed'],'source_run':run,'checkpoint_sha256':original_hash,
          'byte_identical':True,'state_dict_identical':True,'metrics_identical':True,'original_metrics':row['metrics'][0],
          'restored_metrics':metrics,'single_package':single_stats,'restore_seconds':restore_seconds,
          'split_sha256':row['split_sha256'],'test_images':row['split_counts']['test']}
        result['runs'].append(record)
        (out/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print('RESTORE VERIFIED '+run,flush=True)
        del model,data,engine;gc.collect();torch.cuda.empty_cache()
    result['complete']=True
    (out/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (ROOT/'course/outputs/latest_lossless.txt').write_text(str(out),encoding='utf-8')
    print('LOSSLESS AUDIT COMPLETE '+str(out),flush=True)

if __name__=='__main__':main()
