"""Validation-only selection of recall-preserving detail masks across three seeds."""
import hashlib
import json
import numpy as np
from baseline import ROOT
import torch
from anomalib.models import Patchcore
from optimize_exact import setup, sources
from study_precision import OUT as STUDY, save, rates
from detail_guard import union_mask

OUT = ROOT / 'course/outputs/detail-guard-v1'
MULTIPLIERS = (.8, 1., 1.2, 1.5, 2.)
CANDIDATES = [{'name': 'original', 'auxiliary': None, 'multiplier': 1.}] + [
    {'name': f'{name}-t{value:g}', 'auxiliary': name, 'multiplier': value}
    for name in ['coverage03', 'coverage05', 'layer2'] for value in MULTIPLIERS
] + [{'name': f'baseline-threshold-t{value:g}', 'auxiliary': 'baseline', 'multiplier': value}
     for value in [.8, .9]]


def arrays_rates(counts):
    tp, fp, fn = np.moveaxis(counts, -1, 0)
    return np.stack([2 * tp / np.maximum(1, 2 * tp + fp + fn),
                     tp / np.maximum(1, tp + fp), tp / np.maximum(1, tp + fn)], axis=-1)


def main():
    torch.set_num_threads(4)
    if (OUT / 'selection.json').exists():
        raise RuntimeError('Completed selection exists; do not overwrite')
    plan = {'candidates': CANDIDATES, 'seeds': [42, 43, 44], 'dataset': 'capsule',
        'rule': 'A common candidate must preserve validation pixel F1 and pixel precision in every seed, improve mean pixel F1 by >=0.01, and have positive paired image bootstrap 95% F1 gain lower bound in every seed. Recall is monotonic by union. Choose largest mean gain lower bound, ties prefer smaller auxiliary bank. Image scores, labels and continuous maps remain original.',
        'test_policy': 'Freeze one common configuration before any test; report all seeds without post-test reselection. Existing historical data is retrospective; overlapping cross-seed images prevent an independent replication claim.',
        'bootstrap': {'draws': 2000, 'seed': 20261026}}
    save(OUT / 'plan.json', plan)
    rows = []
    for source in [r for r in sources() if r['dataset'] == 'capsule']:
        seed = source['seed']
        baseline, data = setup(source)
        auxiliaries = {}
        for name in ['coverage03', 'coverage05', 'layer2']:
            checkpoint = STUDY / f'{name}-s{seed}' / 'candidate.ckpt'
            assert checkpoint.exists(), checkpoint
            auxiliaries[name] = Patchcore.load_from_checkpoint(checkpoint, map_location='cpu', weights_only=False).eval().cuda()
        counts, images = [], []
        with torch.inference_mode():
            for batch in data.val_dataloader():
                baseline.pre_processor.on_validation_batch_start(None, None, batch, 0)
                x = batch.image.cuda()
                raw = baseline.model(x)
                reference = baseline.post_processor(raw)
                truth = batch.gt_mask.cuda().bool().reshape(reference.pred_mask.shape)
                maps = {name: model.model(x).anomaly_map for name, model in auxiliaries.items()}
                maps['baseline'] = raw.anomaly_map
                thresholds = {name: float(model.post_processor.pixel_threshold) for name, model in auxiliaries.items()}
                thresholds['baseline'] = float(baseline.post_processor.pixel_threshold)
                masks = [reference.pred_mask] + [union_mask(reference.pred_mask, maps[c['auxiliary']],
                    thresholds[c['auxiliary']] * c['multiplier'], reference.pred_label) for c in CANDIDATES[1:]]
                counts.append([[int((mask & truth).sum()), int((mask & ~truth).sum()), int((~mask & truth).sum())] for mask in masks])
                images.append(str(batch.image_path[0]))
        a = np.array(counts, dtype=np.int64)
        measured = arrays_rates(a.sum(0))
        rng = np.random.default_rng(20261026)
        samples = rng.integers(0, len(a), (2000, len(a)))
        bootstrap = arrays_rates(a[samples].sum(1))[:, :, 0]
        interval = np.quantile(bootstrap - bootstrap[:, 0, None], [.025, .975], axis=0).T
        row = {'seed': seed, 'source_run': source['run_id'], 'images': images,
               'counts_per_image': counts, 'candidates': [
                   {**c, **rates(a.sum(0)[i]), 'delta_f1': float(measured[i, 0] - measured[0, 0]),
                    'delta_precision': float(measured[i, 1] - measured[0, 1]),
                    'delta_recall': float(measured[i, 2] - measured[0, 2]),
                    'gain_ci95': interval[i].tolist()} for i, c in enumerate(CANDIDATES)],
               'auxiliary_checkpoint_sha256': {name: hashlib.sha256((STUDY / f'{name}-s{seed}' / 'candidate.ckpt').read_bytes()).hexdigest() for name in auxiliaries}}
        rows.append(row)
        save(OUT / 'validation-progress.json', {'runs': rows})
        print(json.dumps({'seed': seed, 'candidates': row['candidates']}, ensure_ascii=False), flush=True)
        del baseline, data, auxiliaries, maps, masks, reference, raw, x, truth
    eligible = []
    for i in range(1, len(CANDIDATES)):
        candidates = [r['candidates'][i] for r in rows]
        if (all(c['delta_f1'] >= 0 and c['delta_precision'] >= 0 and c['gain_ci95'][0] > 0 for c in candidates)
                and np.mean([c['delta_f1'] for c in candidates]) >= .01):
            eligible.append(i)
    selected = max(eligible, key=lambda i: np.mean([r['candidates'][i]['gain_ci95'][0] for r in rows])) if eligible else None
    report = {'complete': True, 'phase': 'validation-only', 'plan': plan, 'runs': rows,
              'eligible': [CANDIDATES[i]['name'] for i in eligible],
              'selected': CANDIDATES[selected] if selected is not None else None,
              'source_sha256': hashlib.sha256((ROOT / 'course/detail_guard.py').read_bytes()).hexdigest()}
    save(OUT / 'selection.json', report)
    print('SELECTED ' + json.dumps(report['selected']), flush=True)


if __name__ == '__main__':
    main()
