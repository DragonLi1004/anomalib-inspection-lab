"""Register only validation-enabled, fully audited refinements as new local model entries."""
import hashlib,json,shutil,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'course/outputs/accuracy-20261025'


def main():
    report=json.loads((OUT/'test-results.json').read_text())
    assert report['complete'] and report['all_metrics_non_decreasing'] and report['test_evaluations']==552
    assert report['selection_sha256']==hashlib.sha256((OUT/'frozen-peak-selection.json').read_bytes()).hexdigest()
    registry=OUT/'registered.json'
    if registry.exists():
        print(registry.read_text());return
    entries=[]
    for row in report['runs']:
        if row['alpha']==0:continue
        source=ROOT/'course/outputs'/row['source_run']
        assert hashlib.sha256((source/'baseline.ckpt').read_bytes()).hexdigest()==row['checkpoint_sha256']
        run=ROOT/'course/outputs'/(f"{row['dataset']}-peak_mask-"+time.strftime('%Y%m%d-%H%M%S'))
        run.mkdir(exist_ok=False)
        shutil.copy2(source/'baseline.ckpt',run/'baseline.ckpt')
        shutil.copy2(source/'split_manifest.json',run/'split_manifest.json')
        metadata=json.loads((source/'metrics.json').read_text())
        metadata.update(method='peak_mask',protocol_version='accuracy-mask-v1',source_run=row['source_run'],
            source_checkpoint_sha256=row['checkpoint_sha256'],peak_alpha=row['alpha'],metrics=[row['after']],
            baseline_metrics=[row['before']],accuracy_selection_sha256=report['selection_sha256'],accuracy_timing=row['timing'],
            accuracy_scope='Validation-enabled pixel mask refinement; retrospective regression; pixel recall tradeoff disclosed in report',
            source_fit_wall_seconds=metadata['fit_wall_seconds'],fit_wall_seconds=0.)
        (run/'metrics.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        entries.append({'id':run.name,'source_run':row['source_run'],'alpha':row['alpha'],
                        'checkpoint_byte_identical':hashlib.sha256((run/'baseline.ckpt').read_bytes()).hexdigest()==row['checkpoint_sha256']})
    registry.write_text(json.dumps(entries,indent=2),encoding='utf-8')
    print(registry.read_text())
if __name__=='__main__':main()
