"""Select a common fine-detail policy using validation only, with paired image intervals."""
import hashlib
import json
import numpy as np
from baseline import ROOT
from study_precision import save, rates
from study_fine import OUT, CANDIDATES
from select_detail_guard import arrays_rates

MULTIPLIERS = (.8, .9, 1., 1.05, 1.1, 1.2, 1.4)


def main():
    destination = OUT / 'selection.json'
    if destination.exists():
        raise RuntimeError('Selection exists; preserve completed experiments')
    excluded = json.loads((OUT / 'resource-exclusion.json').read_text(encoding='utf-8-sig'))
    active = {name: cfg for name, cfg in CANDIDATES.items() if name != excluded['candidate']}
    candidates = [{'name': 'original', 'auxiliary': None, 'multiplier': 1.}] + [
        {'name': f'{name}-t{v:g}', 'auxiliary': name, 'multiplier': v}
        for name in active for v in MULTIPLIERS]
    plan = {'candidates': candidates, 'seeds': [42, 43, 44], 'grid': [224, 224],
        'rule': 'Common policy across seeds. Every seed validation pixel F1 and precision must not decrease, and paired image bootstrap 95% F1 gain lower bound must be positive. Mean F1 gain >=0.01. Maximize mean lower bound; exact ties prefer smaller input resolution. Union guarantees recall monotonicity; original scores/maps unchanged.',
        'limitations': 'Historical retrospective data, overlapping images across seeds, descriptive intervals without multiple-comparison correction. No test-based selection.'}
    save(OUT / 'selection-plan.json', plan)
    runs = []
    for seed in [42, 43, 44]:
        base = np.load(OUT / f'baseline-s{seed}.npz')
        truth, original = base['truth'], base['masks']
        gate = base['labels'].reshape(-1, 1, 1, 1)
        metadata, auxiliary = {}, {}
        for name in active:
            folder = OUT / f'{name}-s{seed}'
            metadata[name] = json.loads((folder / 'validation-cache.json').read_text(encoding='utf-8'))
            cache = np.load(folder / 'validation-maps.npz')
            assert np.array_equal(cache['paths'], base['paths'])
            auxiliary[name] = cache['maps']
        counts = []
        for candidate in candidates:
            if candidate['auxiliary'] is None:
                mask = original
            else:
                name = candidate['auxiliary']
                threshold = metadata[name]['native_pixel_threshold'] * candidate['multiplier']
                mask = original | ((auxiliary[name] > threshold) & gate)
            axes = tuple(range(1, mask.ndim))
            counts.append(np.stack([(mask & truth).sum(axes), (mask & ~truth).sum(axes), (~mask & truth).sum(axes)], -1))
        a = np.stack(counts, axis=1)
        point = arrays_rates(a.sum(0))
        rng = np.random.default_rng(20261026)
        samples = rng.integers(0, len(a), (2000, len(a)))
        boot = arrays_rates(a[samples].sum(1))[:, :, 0]
        intervals = np.quantile(boot - boot[:, 0, None], [.025, .975], axis=0).T
        row = {'seed': seed, 'counts_per_image': a.tolist(), 'images': base['paths'].tolist(),
            'auxiliary_metadata': metadata, 'candidates': [{**c, **rates(a.sum(0)[i]),
                'gain_ci95': intervals[i].tolist(), 'delta_f1': float(point[i, 0] - point[0, 0]),
                'delta_precision': float(point[i, 1] - point[0, 1]), 'delta_recall': float(point[i, 2] - point[0, 2])}
                for i, c in enumerate(candidates)]}
        runs.append(row)
        print(json.dumps({'seed': seed, 'candidates': row['candidates']}), flush=True)
    eligible = [i for i in range(1, len(candidates)) if all(r['candidates'][i]['delta_f1'] >= 0
        and r['candidates'][i]['delta_precision'] >= 0 and r['candidates'][i]['gain_ci95'][0] > 0 for r in runs)
        and np.mean([r['candidates'][i]['delta_f1'] for r in runs]) >= .01]
    chosen = max(eligible, key=lambda i: (np.mean([r['candidates'][i]['gain_ci95'][0] for r in runs]),
        -CANDIDATES[candidates[i]['auxiliary']]['size'])) if eligible else None
    report = {'complete': True, 'phase': 'validation-only', 'plan': plan, 'runs': runs, 'resource_exclusion': excluded,
        'selected': candidates[chosen] if chosen is not None else None,
        'eligible': [candidates[i]['name'] for i in eligible],
        'source_sha256': {name: hashlib.sha256((ROOT / 'course' / name).read_bytes()).hexdigest()
                          for name in ['fine_guard.py', 'detail_guard.py']}}
    save(destination, report)
    print('SELECTED ' + json.dumps(report['selected']), flush=True)


if __name__ == '__main__':
    main()
