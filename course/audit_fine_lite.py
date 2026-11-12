"""Export a shared-backbone checkpoint and verify the frozen FineGuard test splits."""
import gc
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import tempfile
import time

import torch
from baseline import ROOT
from fine_guard import install_fine_guard
from fine_guard_lite import export_checkpoint, load_lite
from optimize_exact import sources, setup
from anomalib.models import Patchcore

OUT = ROOT / "course/outputs"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def run(row, folder, old):
    base, data = setup(row)
    meta = read(folder / "metrics.json")
    aux = Patchcore.load_from_checkpoint(folder / "auxiliary.ckpt", weights_only=False, map_location="cpu")
    ref = install_fine_guard(base, aux, meta["fine_threshold"]).cuda().eval()
    name = f"capsule-fine_lite-s{row['seed']}-shared-v1"
    target = OUT / name
    with tempfile.TemporaryDirectory(dir=OUT) as temp:
        stage = Path(temp) / name
        stage.mkdir()
        checkpoint = stage / "baseline.ckpt"
        export_checkpoint(folder, checkpoint, meta["fine_threshold"])
        lite = load_lite(checkpoint).cuda().eval()
        models = [ref, lite]
        params = [sum(p.numel() for p in m.parameters()) for m in models]
        assert params[1] < params[0]
        timings = [[], []]
        skipped = 0
        total = torch.zeros(3, device="cuda", dtype=torch.long)
        paths = []
        def infer(model, x):
            return model.post_processor(model.model(x))
        with torch.inference_mode():
            for i, batch in enumerate(data.test_dataloader()):
                x = batch.image.cuda()
                if i == 0:
                    for m in models:
                        for _ in range(6):
                            infer(m, x)
                paths.append(Path(batch.image_path[0]).relative_to(ROOT).as_posix())
                predictions = {}
                for j in ((0,1) if i % 2 == 0 else (1,0)):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    predictions[j] = infer(models[j], x)
                    torch.cuda.synchronize()
                    timings[j].append(1000*(time.perf_counter()-start))
                first, second = predictions[0], predictions[1]
                for field in ("pred_label", "pred_score", "anomaly_map", "pred_mask"):
                    assert torch.equal(getattr(first,field),getattr(second,field)), (i,field)
                skipped += int(not second.pred_label.item())
                _, gt = ref.model.original_pre.transform(batch.image, batch.gt_mask)
                gt = gt.cuda().bool().reshape_as(second.pred_mask)
                mask = second.pred_mask
                total += torch.stack([(mask & gt).sum(),(mask & ~gt).sum(),(~mask & gt).sum()])
        assert paths == old["images"] and total.tolist() == old["pixel_counts"]["after"]
        sizes = [(folder/"auxiliary.ckpt").stat().st_size+(folder/"baseline.ckpt").stat().st_size,checkpoint.stat().st_size]
        result = dict(seed=row["seed"], unique_parameters=params, checkpoint_bytes=sizes, test_evaluations=len(paths),
                      skipped_detail=skipped, identical_outputs=True, counts=total.tolist(), metrics=old["after"],
                      latency_ms=timings, mean_ms=list(map(statistics.mean,timings)), median_ms=list(map(statistics.median,timings)))
        meta.update(method="fine_lite",checkpoint_bytes=sizes[1],lite_parent=folder.name,
                    lite_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),lite_evaluation=result,
                    timing_note="See lite_evaluation: batch-one paired pre/postprocessing and inference, no IO.")
        (stage/"metrics.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
        shutil.copy2(folder/"split_manifest.json",stage/"split_manifest.json")
        if not target.exists():
            stage.rename(target)
        else:
            assert read(target/"metrics.json")["lite_parent"] == folder.name
            assert hashlib.sha256((target/"baseline.ckpt").read_bytes()).hexdigest()==read(target/"metrics.json")["lite_sha256"]
        return result


def main():
    torch.set_num_threads(4)
    study = OUT / "fine-study-v3"
    original = read(study/"test-results.json")["runs"]
    records = []
    for entry in read(study/"registered.json"):
        seed = entry["seed"]
        row = next(r for r in sources() if r["dataset"]=="capsule" and r["seed"]==seed)
        records.append(run(row, OUT/entry["id"], next(r for r in original if r["seed"]==seed)))
        print({k:v for k,v in records[-1].items() if k!="latency_ms"},flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    (ROOT/"course/results/fine_lite_metrics.json").write_text(json.dumps(records,indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
