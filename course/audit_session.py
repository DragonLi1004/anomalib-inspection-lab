"""Audit model reuse against locked test splits; measure cold versus warm service latency."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'course/outputs/session-audit'


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/name).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def sources():
    return [r for r in json.loads((ROOT/'course/results/multiseed_metrics.json').read_text())['runs'] if r['method']=='baseline']


def verify():
    from baseline import ROOT
    import torch
    from anomalib.engine import Engine
    from predict import PredictionSession
    from optimize_exact import setup
    torch.set_num_threads(4)
    report = {'complete': False, 'runs': [], 'protocol': 'same-checkpoint-session-reuse-v1'}
    session = PredictionSession()
    for row in sources():
        session.clear()
        reference, data = setup(row)  # Verifies all image SHA256 and locked splits; no search modification.
        run = ROOT/'course/outputs'/row['run_id']
        manifest = json.loads((run/'split_manifest.json').read_text())
        # Exercise the actual GUI path before inspecting the cached model.
        cold = session.predict(ROOT/manifest['test'][0]['image'], run.name, 'cuda', audit=True)
        warm = session.predict(ROOT/manifest['test'][0]['image'], run.name, 'cuda', audit=True)
        assert not cold['timing']['cache_hit'] and warm['timing']['cache_hit']
        assert cold['records'] == warm['records'] and cold['signatures'] == warm['signatures']
        cached = session.model.eval().cuda()
        count = 0
        with torch.inference_mode():
            for batch in data.test_dataloader():
                x = batch.image.cuda()
                expected = reference.model(reference.pre_processor(x))
                actual = cached.model(cached.pre_processor(x))
                assert torch.equal(expected.pred_score, actual.pred_score), 'Raw score changed'
                assert torch.equal(expected.anomaly_map, actual.anomaly_map), 'Anomaly map changed'
                count += len(x)
        cached.visualizer = None
        engine = Engine(default_root_dir=OUT/'evaluation'/run.name, accelerator='gpu', devices=1,
                        logger=False, enable_progress_bar=False, enable_model_summary=False)
        metrics = engine.test(model=cached, datamodule=data)
        assert metrics == row['metrics'], (metrics, row['metrics'])
        record = {'source_run': run.name, 'dataset': row['dataset'], 'seed': row['seed'],
                  'test_evaluations': count, 'raw_scores_and_maps_bit_exact': True,
                  'single_image_postprocessed_outputs_bit_exact': True,
                  'four_metrics_identical': True, 'metrics': metrics,
                  'checkpoint_sha256': hashlib.sha256((run/'baseline.ckpt').read_bytes()).hexdigest()}
        report['runs'].append(record)
        save('verification.json', report)
        print(json.dumps(record), flush=True)
        del reference, cached, data, engine, expected, actual, x
        session.clear()
        gc.collect()
        torch.cuda.empty_cache()
    report['complete'] = True
    report['test_evaluations'] = sum(r['test_evaluations'] for r in report['runs'])
    save('verification.json', report)


def artifact_hashes(folder):
    return {p.relative_to(folder/'images').as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (folder/'images').rglob('*.png')}


def benchmark():
    from prediction_worker import PredictionWorker
    env = os.environ.copy()
    env.update(PYTHONUTF8='1', PYTHONUNBUFFERED='1', PYTHONPATH=str(ROOT/'course'),
               TEMP=str(ROOT/'.runtime/temp'), TMP=str(ROOT/'.runtime/temp'), MPLCONFIGDIR=str(ROOT/'.cache/matplotlib'))
    # Keep a reproducible, fixed pre-optimization reference outside the inference implementation.
    original = OUT/'original_predict.py'
    if not original.is_file():
        original.write_bytes(subprocess.check_output(['git','show','ce49ccfa7a26393aaeb8393ad9684e557ecaa52f:course/predict.py'], cwd=ROOT))
    expected = subprocess.check_output(['git','show','ce49ccfa7a26393aaeb8393ad9684e557ecaa52f:course/predict.py'], cwd=ROOT).decode('utf-8-sig').replace('\r\n','\n')
    assert original.read_text(encoding='utf-8-sig') == expected, 'Original reference differs from frozen commit'
    report = {'complete': False, 'baseline_commit': 'ce49ccfa7a26393aaeb8393ad9684e557ecaa52f',
              'original_predict_sha256': hashlib.sha256(original.read_bytes()).hexdigest(), 'rounds_per_category': 3,
              'timing_scope': 'client request through saved prediction and rendered PNG; excludes GUI polling and upload', 'categories': []}
    worker = PredictionWorker()
    try:
        for row in [r for r in sources() if r['seed']==42]:
            manifest = json.loads((ROOT/'course/outputs'/row['run_id']/'split_manifest.json').read_text())
            samples = [manifest['test'][i]['image'] for i in (0,len(manifest['test'])//2,len(manifest['test'])-1)]
            worker.close()
            with (OUT/'benchmark-worker.log').open('a',encoding='utf-8') as log:
                request = {'run': row['run_id'], 'image': str(ROOT/samples[0]), 'device': 'cuda'}
                started = time.perf_counter()
                first = worker.predict(request, log, env)
                first_seconds = time.perf_counter()-started
                results=[]
                for index,image in enumerate(samples):
                    request['image'] = str(ROOT/image)
                    pair = {}
                    # Alternate order, on the same image/device, while the warm service stays resident.
                    for mode in (['original','warm'] if index%2==0 else ['warm','original']):
                        started=time.perf_counter()
                        if mode=='original':
                            output=subprocess.run([sys.executable,'-u',str(original),'--run',row['run_id'],'--device','cuda','--image',str(ROOT/image)],
                                                  cwd=ROOT,env=env,capture_output=True,text=True,encoding='utf-8',check=True)
                            elapsed=time.perf_counter()-started
                            (OUT/f"original-{row['dataset']}-{index}.log").write_text(output.stdout+output.stderr,encoding='utf-8')
                            folder=Path(next(line[len('Results: '):] for line in output.stdout.splitlines() if line.startswith('Results: ')))
                            records=json.loads((folder/'predictions.json').read_text(encoding='utf-8'))
                            pair[mode]={'seconds':elapsed,'records':records,'artifacts':artifact_hashes(folder)}
                        else:
                            result=worker.predict(request,log,env)
                            elapsed=time.perf_counter()-started
                            assert result['timing']['cache_hit']
                            pair[mode]={'seconds':elapsed,'records':result['records'],'artifacts':artifact_hashes(Path(result['output']))}
                    assert pair['original']['records']==pair['warm']['records']
                    assert pair['original']['artifacts'] and pair['original']['artifacts']==pair['warm']['artifacts']
                    results.append({'image':image,'original_seconds':pair['original']['seconds'],
                                    'warm_seconds':pair['warm']['seconds'],'records_and_png_bytes_identical':True})
                    print(json.dumps({'category':row['dataset'],**results[-1]}),flush=True)
            cold_median=statistics.median(r['original_seconds'] for r in results)
            warm_median=statistics.median(r['warm_seconds'] for r in results)
            report['categories'].append({'dataset':row['dataset'],'source_run':row['run_id'],
                'worker_first_request_seconds':first_seconds,'original_median_seconds':cold_median,
                'warm_median_seconds':warm_median,'latency_reduction_percent':100*(1-warm_median/cold_median),
                'speedup':cold_median/warm_median,'samples':results})
            save('benchmark.json',report)
        report['complete']=True
        save('benchmark.json',report)
    finally:
        worker.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['verify','benchmark'])
    args=parser.parse_args()
    verify() if args.phase=='verify' else benchmark()
