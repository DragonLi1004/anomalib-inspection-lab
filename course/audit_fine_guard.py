"""One frozen test audit of a validation-selected fine-detail policy; never reselect on tests."""
import argparse
import gc
import hashlib
import json
import random
import statistics
import time
from pathlib import Path
import numpy as np
from baseline import ROOT
import torch
from anomalib.engine import Engine
from anomalib.models import Patchcore
from optimize_exact import sources, setup
from fine_guard import install_fine_guard, native_branch_input
from study_fine import OUT
from study_precision import save, rates
from select_detail_guard import arrays_rates


def benchmark(base_core, base_pre, base_post, fine, sample):
    def original():
        return base_post(base_core(native_branch_input(base_pre, sample)))
    def optimized():
        return fine.post_processor(fine.model(sample))
    methods = {'baseline': original, 'fine_guard': optimized}
    times, peaks = {key: [] for key in methods}, {}
    with torch.inference_mode():
        for func in methods.values():
            for _ in range(10):
                func()
        rng = random.Random(20261026)
        for _ in range(40):
            order = list(methods)
            rng.shuffle(order)
            for name in order:
                torch.cuda.synchronize()
                start = time.perf_counter()
                result = methods[name]()
                torch.cuda.synchronize()
                times[name].append((time.perf_counter() - start) * 1000)
                del result
        for name, func in methods.items():
            start_bytes = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            result = func()
            torch.cuda.synchronize()
            peaks[name] = {'total_allocated_bytes': torch.cuda.max_memory_allocated(),
                           'incremental_bytes': torch.cuda.max_memory_allocated() - start_bytes}
            del result
    return {'median_ms': {key: statistics.median(value) for key, value in times.items()},
            'p95_ms': {key: float(np.quantile(value, .95)) for key, value in times.items()},
            'samples_ms': times, 'peak': peaks,
            'scope': 'same original-size GPU input, preprocessing + model(s) + postprocessing, batch=1; excludes IO, rendering and startup. Both branches resident for both measurements; total allocations are NOT standalone deployment VRAM. 10 warmups +40 interleaved synchronized runs.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    selected_path = OUT / 'selection.json'
    selection = json.loads(selected_path.read_text(encoding='utf-8'))
    assert selection['complete'] and selection['selected'] is not None, 'No eligible validation candidate'
    for name, digest in selection['source_sha256'].items():
        actual = hashlib.sha256((ROOT / 'course' / name).read_bytes()).hexdigest()
        if actual != digest:
            amendment = json.loads((OUT / 'implementation-amendment.json').read_text(encoding='utf-8'))
            assert name == amendment['file'] and digest == amendment['old_sha256'] and actual == amendment['new_sha256']
    frozen = OUT / 'frozen-selection.json'
    result_path = OUT / 'test-results.json'
    if frozen.exists():
        assert args.resume, 'Frozen audit already started; use --resume to retain completed runs'
        assert frozen.read_bytes() == selected_path.read_bytes()
        report = json.loads(result_path.read_text(encoding='utf-8')) if result_path.exists() else {'runs': []}
    else:
        frozen.write_bytes(selected_path.read_bytes())
        report = {'runs': []}
    report.update(implementation_amendment=json.loads((OUT / 'implementation-amendment.json').read_text(encoding='utf-8')), complete=False, phase='frozen-test', selection=selection['selected'],
        selection_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),
        limitations='Retrospective fixed historical splits; overlaps across seeds, not independent unseen-data confirmation')
    save(result_path, report)
    policy = selection['selected']
    for source in [row for row in sources() if row['dataset'] == 'capsule']:
        seed = source['seed']
        if any(r['seed'] == seed for r in report['runs']):
            continue
        original, data = setup(source)
        def engine(label):
            return Engine(default_root_dir=OUT / 'audit' / f's{seed}' / label, accelerator='gpu', devices=1,
                logger=False, enable_progress_bar=False, enable_model_summary=False)
        before = engine('baseline').test(model=original, datamodule=data)[0]
        assert before == source['metrics'][0], ('Historical mismatch', before, source['metrics'][0])
        folder = OUT / f'{policy["auxiliary"]}-s{seed}'
        evidence = next(r for r in selection['runs'] if r['seed'] == seed)['auxiliary_metadata'][policy['auxiliary']]
        checkpoint = folder / 'candidate.ckpt'
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == evidence['checkpoint_sha256']
        auxiliary = Patchcore.load_from_checkpoint(checkpoint, map_location='cpu', weights_only=False).eval().cuda()
        threshold = evidence['native_pixel_threshold'] * policy['multiplier']
        base_core, base_pre, base_post = original.model, original.pre_processor, original.post_processor
        fine = install_fine_guard(original, auxiliary, threshold).cuda().eval()
        after = engine('fine-guard').test(model=fine, datamodule=data)[0]
        for key in ('image_AUROC', 'image_F1Score', 'pixel_AUROC'):
            assert before[key] == after[key], ('Invariant metric changed', key)
        fine.cuda().eval()  # Engine.test teardown returns the model to CPU.
        base_post.cuda()  # Detached reference scalar buffers must use the same GPU arithmetic.
        counts, paths, exact, sample = [], [], True, None
        with torch.inference_mode():
            for batch in data.test_dataloader():
                # Compare both actual deployment branches on the same original-size GPU input.
                image = batch.image.cuda()
                _, gt = base_pre.transform(batch.image, batch.gt_mask)
                a = base_post(base_core(native_branch_input(base_pre, image)))
                b = fine.post_processor(fine.model(image))
                gt = gt.cuda().bool().reshape(a.pred_mask.shape)
                exact = exact and all(torch.equal(getattr(a, key), getattr(b, key)) for key in ['pred_score', 'pred_label', 'anomaly_map'])
                assert not (a.pred_mask & ~b.pred_mask).any(), 'Original positive pixel removed'
                counts.append([[int((p & gt).sum()), int((p & ~gt).sum()), int((~p & gt).sum())] for p in [a.pred_mask, b.pred_mask]])
                paths.append(Path(batch.image_path[0]).relative_to(ROOT).as_posix())
                if sample is None:
                    sample = image
        assert exact, 'Original scores/maps changed'
        a = np.array(counts, dtype=np.int64)
        totals = a.sum(0)
        for index, metrics in enumerate([before, after]):
            assert abs(rates(totals[index])['pixel_F1Score'] - metrics['pixel_F1Score']) < 1e-6, 'Manual/Engine mismatch'
        rng = np.random.default_rng(20261026)
        indices = rng.integers(0, len(a), (2000, len(a)))
        bootstrap = arrays_rates(a[indices].sum(1))[:, :, 0]
        interval = np.quantile(bootstrap[:, 1] - bootstrap[:, 0], [.025, .975]).tolist()
        item = {'seed': seed, 'dataset': 'capsule', 'source_run': source['run_id'], 'threshold': threshold,
            'before': before, 'after': after, 'delta_pp': {key: 100 * (after[key] - before[key]) for key in before},
            'pixel_rates': {'before': rates(totals[0]), 'after': rates(totals[1])},
            'pixel_counts': {'before': totals[0].tolist(), 'after': totals[1].tolist()},
            'four_metrics_non_decreasing': all(after[key] >= before[key] - 1e-7 for key in before),
            'recall_non_decreasing': bool(rates(totals[1])['pixel_recall'] >= rates(totals[0])['pixel_recall']),
            'precision_non_decreasing': bool(rates(totals[1])['pixel_precision'] >= rates(totals[0])['pixel_precision']),
            'original_scores_labels_maps_bit_exact': exact, 'original_mask_subset': True,
            'test_evaluations': len(a), 'counts_per_image': counts, 'images': paths,
            'gain_ci95': interval, 'timing': benchmark(base_core, base_pre, base_post, fine, sample),
            'auxiliary_checkpoint_sha256': evidence['checkpoint_sha256'],
            'source_checkpoint_sha256': hashlib.sha256((ROOT / 'course/outputs' / source['run_id'] / 'baseline.ckpt').read_bytes()).hexdigest()}
        report['runs'].append(item)
        save(result_path, report)
        print(json.dumps({key: value for key, value in item.items() if key not in ['counts_per_image', 'images', 'timing']}), flush=True)
        del original, auxiliary, fine, data, base_core, base_pre, base_post, sample, image, b
        gc.collect()
        torch.cuda.empty_cache()
    report['complete'] = len(report['runs']) == 3
    report['accepted'] = all(r['four_metrics_non_decreasing'] and r['recall_non_decreasing'] for r in report['runs'])
    report['mean_pixel_f1_gain_pp'] = statistics.mean(r['delta_pp']['pixel_F1Score'] for r in report['runs'])
    report['meets_one_pp_target'] = report['mean_pixel_f1_gain_pp'] >= 1.
    save(result_path, report)


if __name__ == '__main__':
    main()
