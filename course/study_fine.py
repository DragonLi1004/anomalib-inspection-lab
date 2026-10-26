"""Train detail branches; cache validation maps on the unchanged 224px reference grid."""
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from baseline import ROOT
import torch
from torch.nn import functional as F
from lightning import seed_everything
from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine
from optimize_exact import setup, sources
from study_precision import save

OUT = ROOT / 'course/outputs/fine-study-v3'
CANDIDATES = {
    'r320-l23': {'size': 320, 'layers': ['layer2', 'layer3'], 'ratio': .01},
    'r384-l2': {'size': 384, 'layers': ['layer2'], 'ratio': .01},
    'r224-l12': {'size': 224, 'layers': ['layer1', 'layer2'], 'ratio': .01},
}


def baseline_cache(source):
    baseline, data = setup(source)
    path = OUT / f'baseline-s{source["seed"]}.npz'
    if not path.exists():
        maps, masks, truths, labels, scores, paths = [], [], [], [], [], []
        with torch.inference_mode():
            for batch in data.val_dataloader():
                baseline.pre_processor.on_validation_batch_start(None, None, batch, 0)
                result = baseline.post_processor(baseline.model(batch.image.cuda()))
                maps.append(result.anomaly_map.cpu().numpy())
                masks.append(result.pred_mask.cpu().numpy())
                truths.append(batch.gt_mask.bool().reshape(result.pred_mask.shape).numpy())
                labels.append(result.pred_label.cpu().numpy().reshape(-1))
                scores.append(result.pred_score.cpu().numpy().reshape(-1))
                paths.append(Path(batch.image_path[0]).relative_to(ROOT).as_posix())
        np.savez_compressed(path, maps=np.concatenate(maps), masks=np.concatenate(masks),
            truth=np.concatenate(truths), labels=np.concatenate(labels), scores=np.concatenate(scores), paths=np.array(paths))
    del baseline, data
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', choices=CANDIDATES, required=True)
    parser.add_argument('--seed', type=int, choices=[42, 43, 44], required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    out = OUT / f'{args.candidate}-s{args.seed}'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'validation-cache.json').exists():
        print('ALREADY COMPLETE', flush=True)
        return
    save(OUT / 'plan.json', {'phase': 'validation-only', 'candidates': CANDIDATES,
        'seeds': [42, 43, 44], 'evaluation_grid': [224, 224],
        'alignment': 'Auxiliary raw map resized bilinearly with align_corners=False; original GT resized only by original preprocessing. Input branch resizes directly from original image, never from the 224px input.',
        'test_policy': 'No test predictions until common candidate and mask policy have been frozen.'})
    source = next(r for r in sources() if r['dataset'] == 'capsule' and r['seed'] == args.seed)
    baseline_cache(source)
    seed_everything(args.seed, workers=True)
    data = MVTecAD(root=ROOT / 'course/data/MVTecAD', category='capsule', train_batch_size=1,
        eval_batch_size=1, num_workers=0, val_split_mode='from_test', val_split_ratio=.5, seed=args.seed)
    data.setup()
    cfg = CANDIDATES[args.candidate]
    model = Patchcore(backbone='resnet18.tv_in1k', layers=cfg['layers'], coreset_sampling_ratio=cfg['ratio'],
        num_neighbors=9, pre_processor=Patchcore.configure_pre_processor(image_size=(cfg['size'], cfg['size'])), visualizer=False)
    engine = Engine(default_root_dir=out / 'engine', accelerator='gpu', devices=1,
        logger=False, enable_progress_bar=False, enable_model_summary=False)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    engine.fit(model=model, datamodule=data)
    fit_seconds = time.perf_counter() - start
    fit_peak = torch.cuda.max_memory_allocated()
    checkpoint = out / 'candidate.ckpt'
    engine.trainer.save_checkpoint(checkpoint)
    model.eval().cuda()
    maps, paths = [], []
    with torch.inference_mode():
        for batch in data.val_dataloader():
            x = model.pre_processor.transform(batch.image).cuda()
            raw = model.model(x).anomaly_map
            aligned = raw if raw.shape[-2:] == (224, 224) else F.interpolate(raw, (224, 224), mode='bilinear', align_corners=False)
            maps.append(aligned.cpu().numpy())
            paths.append(Path(batch.image_path[0]).relative_to(ROOT).as_posix())
    reference = np.load(OUT / f'baseline-s{args.seed}.npz')
    assert paths == reference['paths'].tolist()
    np.savez_compressed(out / 'validation-maps.npz', maps=np.concatenate(maps), paths=np.array(paths))
    result = {'candidate': args.candidate, 'seed': args.seed, 'config': cfg, 'phase': 'validation-only',
        'source_run': source['run_id'], 'split_sha256': source['split_sha256'], 'images': len(paths),
        'native_pixel_threshold': float(model.post_processor.pixel_threshold),
        'fit_seconds': fit_seconds, 'fit_peak_allocated_bytes': fit_peak,
        'checkpoint_bytes': checkpoint.stat().st_size, 'memory_bank_shape': list(model.model.memory_bank.shape),
        'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    save(out / 'validation-cache.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
