"""Resumable capsule accuracy study. Validation only; never reads test predictions.

Each invocation evaluates one candidate/seed and saves its checkpoint and evidence.
Original split manifests and input/mask resolution stay unchanged.
"""
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

from baseline import ROOT
import torch
from lightning import seed_everything
from torchmetrics.classification import BinaryAUROC
from anomalib.models import Patchcore
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from optimize_exact import sources, setup

OUT = ROOT / 'course/outputs/precision-study-v2'
CANDIDATES = {
    'coverage03': {'ratio': .03, 'layers': ['layer2', 'layer3']},
    'coverage05': {'ratio': .05, 'layers': ['layer2', 'layer3']},
    'layer2': {'ratio': .01, 'layers': ['layer2']},
}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def rates(counts):
    tp, fp, fn = counts
    return {'pixel_F1Score': 2 * tp / max(1, 2 * tp + fp + fn),
            'pixel_precision': tp / max(1, tp + fp),
            'pixel_recall': tp / max(1, tp + fn)}


def evaluate_validation(model, data):
    model.eval().cuda()
    pixel_auc = BinaryAUROC(thresholds=None)
    image_auc = BinaryAUROC(thresholds=None)
    rows, image_counts = [], [0, 0, 0]
    with torch.inference_mode():
        for batch in data.val_dataloader():
            model.pre_processor.on_validation_batch_start(None, None, batch, 0)
            pred = model.post_processor(model.model(batch.image.cuda()))
            truth = batch.gt_mask.bool().reshape(pred.pred_mask.shape).cpu()
            mask = pred.pred_mask.cpu()
            assert tuple(mask.shape[-2:]) == (224, 224)
            counts = [int((mask & truth).sum()), int((mask & ~truth).sum()), int((~mask & truth).sum())]
            labels = batch.gt_label.cpu().long().flatten()
            predicted = pred.pred_label.cpu().bool().flatten()
            positive = labels.bool()
            image_counts = [a + b for a, b in zip(image_counts, [int((predicted & positive).sum()),
                int((predicted & ~positive).sum()), int((~predicted & positive).sum())])]
            pixel_auc.update(pred.anomaly_map.cpu().flatten(), truth.long().flatten())
            image_auc.update(pred.pred_score.cpu().flatten(), labels)
            path = Path(batch.image_path[0])
            rows.append({'image': path.relative_to(ROOT).as_posix(), 'defect': path.parent.name,
                         'counts': counts, 'gt_pixels': int(truth.sum()), 'image_label': int(labels[0]),
                         'image_prediction': int(predicted[0]), 'image_score': float(pred.pred_score[0])})
    total = [sum(row['counts'][i] for row in rows) for i in range(3)]
    metrics = {**rates(total), 'image_AUROC': float(image_auc.compute()),
               'image_F1Score': rates(image_counts)['pixel_F1Score'], 'pixel_AUROC': float(pixel_auc.compute())}
    defects = {}
    for name in sorted({row['defect'] for row in rows}):
        group = [row for row in rows if row['defect'] == name]
        counts = [sum(row['counts'][i] for row in group) for i in range(3)]
        defects[name] = {'images': len(group), 'counts': counts, **rates(counts)}
    return {'metrics': metrics, 'counts': total, 'images': len(rows), 'by_defect': defects, 'per_image': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', choices=CANDIDATES, required=True)
    parser.add_argument('--seed', type=int, choices=[42, 43, 44], required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    row = next(r for r in sources() if r['dataset'] == 'capsule' and r['seed'] == args.seed)
    destination = OUT / f'{args.candidate}-s{args.seed}'
    destination.mkdir(parents=True, exist_ok=True)
    result_path = destination / 'validation.json'
    if result_path.exists():
        print('ALREADY COMPLETE ' + str(result_path), flush=True)
        return
    plan = {'protocol': 'precision-study-v2', 'phase': 'validation-only', 'dataset': 'capsule',
            'seeds': [42, 43, 44], 'candidates': CANDIDATES, 'input_size': [224, 224],
            'selection_rule': 'Compare a single common candidate across all three seeds. Require each seed image AUROC, image F1, pixel AUROC and pixel recall non-decreasing on validation; maximize mean pixel F1 with target gain >= 0.01. Freeze before any test evaluation. A failed candidate remains recorded; never select based on test.',
            'limitations': 'Historical data, retrospective study; validation/test disjoint per seed but overlap across seeds. Validation improvements are not test improvements.'}
    plan_path = OUT / 'plan.json'
    if plan_path.exists():
        assert json.loads(plan_path.read_text(encoding='utf-8')) == plan
    else:
        save(plan_path, plan)
    baseline_path = OUT / f'baseline-s{args.seed}.json'
    baseline, data = setup(row)
    if baseline_path.exists():
        reference = json.loads(baseline_path.read_text(encoding='utf-8'))
    else:
        reference = evaluate_validation(baseline, data)
        reference.update(source_run=row['run_id'], split_sha256=row['split_sha256'])
        save(baseline_path, reference)
    del baseline, data
    gc.collect()
    torch.cuda.empty_cache()
    seed_everything(args.seed, workers=True)
    data = MVTecAD(root=ROOT / 'course/data/MVTecAD', category='capsule', train_batch_size=1,
                  eval_batch_size=1, num_workers=0, val_split_mode='from_test', val_split_ratio=.5, seed=args.seed)
    data.setup()
    manifest = json.loads((ROOT / 'course/outputs' / row['run_id'] / 'split_manifest.json').read_text())
    for split in ['train', 'val', 'test']:
        assert [Path(p).relative_to(ROOT).as_posix() for p in getattr(data, split + '_data').samples.image_path.tolist()] == [r['image'] for r in manifest[split]]
    config = CANDIDATES[args.candidate]
    model = Patchcore(backbone='resnet18.tv_in1k', layers=config['layers'],
        coreset_sampling_ratio=config['ratio'], num_neighbors=9,
        pre_processor=Patchcore.configure_pre_processor(image_size=(224, 224)), visualizer=False)
    engine = Engine(default_root_dir=destination / 'engine', accelerator='gpu', devices=1,
                    logger=False, enable_progress_bar=False, enable_model_summary=False)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    engine.fit(model=model, datamodule=data)
    fit_seconds = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated()
    checkpoint = destination / 'candidate.ckpt'
    engine.trainer.save_checkpoint(checkpoint)
    result = evaluate_validation(model, data)
    result.update(candidate=args.candidate, seed=args.seed, phase='validation-only',
        config=config, source_run=row['run_id'], split_sha256=row['split_sha256'],
        baseline_metrics=reference['metrics'],
        delta={key: result['metrics'][key] - value for key, value in reference['metrics'].items()},
        fit_seconds=fit_seconds, fit_peak_allocated_bytes=peak,
        memory_bank_shape=list(model.model.memory_bank.shape), checkpoint_bytes=checkpoint.stat().st_size,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    guards = ['image_AUROC', 'image_F1Score', 'pixel_AUROC', 'pixel_recall']
    result['validation_guards_pass'] = all(result['delta'][key] >= -1e-7 for key in guards)
    save(result_path, result)
    print(json.dumps({key: value for key, value in result.items() if key not in ['per_image', 'by_defect']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
