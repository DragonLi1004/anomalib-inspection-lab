"""Publish local FineGuard model entries only after all frozen tests pass."""
import hashlib
import json
import shutil
import time
from baseline import ROOT
from study_fine import OUT
from study_precision import save


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    result = json.loads((OUT / 'test-results.json').read_text(encoding='utf-8'))
    assert result['complete'] and result['accepted'] and result['meets_one_pp_target']
    assert result['selection_sha256'] == digest(OUT / 'frozen-selection.json')
    registered = OUT / 'registered.json'
    if registered.exists():
        print(registered.read_text(encoding='utf-8'))
        return
    records = []
    for row in result['runs']:
        source = ROOT / 'course/outputs' / row['source_run']
        aux = OUT / f'{result["selection"]["auxiliary"]}-s{row["seed"]}'
        assert digest(source / 'baseline.ckpt') == row['source_checkpoint_sha256']
        assert digest(aux / 'candidate.ckpt') == row['auxiliary_checkpoint_sha256']
        output = ROOT / 'course/outputs' / f'capsule-fine_guard-s{row["seed"]}-{time.strftime("%Y%m%d-%H%M%S")}'
        output.mkdir(exist_ok=False)
        shutil.copy2(source / 'baseline.ckpt', output / 'baseline.ckpt')
        shutil.copy2(source / 'split_manifest.json', output / 'split_manifest.json')
        shutil.copy2(aux / 'candidate.ckpt', output / 'auxiliary.ckpt')
        metadata = json.loads((source / 'metrics.json').read_text(encoding='utf-8'))
        training = json.loads((aux / 'validation-cache.json').read_text(encoding='utf-8'))
        metadata.update(method='fine_guard', protocol_version='fine-mask-v1', source_run=row['source_run'],
            source_checkpoint_sha256=row['source_checkpoint_sha256'],
            auxiliary_checkpoint_sha256=row['auxiliary_checkpoint_sha256'],
            fine_threshold=row['threshold'], fine_config=training['config'],
            fine_selection_sha256=result['selection_sha256'], metrics=[row['after']], baseline_metrics=[row['before']],
            pixel_rates=row['pixel_rates'], fine_timing=row['timing'],
            source_fit_wall_seconds=metadata['fit_wall_seconds'], fit_wall_seconds=training['fit_seconds'],
            checkpoint_bytes=(output / 'baseline.ckpt').stat().st_size + (output / 'auxiliary.ckpt').stat().st_size,
            auxiliary_memory_bank_shape=training['memory_bank_shape'],
            auxiliary_memory_bank_bytes=4 * training['memory_bank_shape'][0] * training['memory_bank_shape'][1],
            latency_median_ms=None,
            timing_note='See fine_timing: original input preprocessing + both branches + postprocessing; not comparable to old model-only timing',
            accuracy_scope='High-resolution detail-mask union; original score, label and heatmap unchanged. Extra compute and model storage.')
        metadata['memory_bank_bytes'] += metadata['auxiliary_memory_bank_bytes']
        metadata['search_state_bytes'] += metadata['auxiliary_memory_bank_bytes']
        save(output / 'metrics.json', metadata)
        records.append({'id': output.name, 'seed': row['seed'], 'source_run': row['source_run']})
    save(registered, records)
    print(json.dumps(records, ensure_ascii=False))


if __name__ == '__main__':
    main()
