"""Paired multi-seed summary with explicit missing-data and provenance checks."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import statistics
from datetime import datetime
from pathlib import Path

from experiment_suite import ROOT, OUTPUT, CATEGORIES, METHODS, SEEDS, code_hash

MATCH_KEYS = ("dataset", "seed", "device", "backbone", "image_size", "batch_size", "coreset_sampling_ratio",
              "num_neighbors", "val_split_ratio", "split_sha256", "coreset_sha256", "protocol_version", "course_code_sha256")
METRICS = ("image_AUROC", "image_F1Score", "pixel_AUROC", "pixel_F1Score")
MEASURES = (*METRICS, "search_state_bytes", "memory_bank_bytes", "projection_bytes", "checkpoint_bytes", "latency_median_ms")


def collect_records(device="cuda", output=OUTPUT, source_hash=None):
    selected = {}
    expected_hash = source_hash or code_hash()
    for path in sorted(output.glob("*/metrics.json"), key=lambda p: p.stat().st_mtime):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if (row.get("protocol_version") == "paired-coreset-v2" and row.get("course_code_sha256") == expected_hash
                and row.get("device") == device and row.get("dataset") in CATEGORIES
                and row.get("seed") in SEEDS and row.get("method") in METHODS):
            selected[(row["dataset"], row["seed"], row["method"])] = {"run_id": path.parent.name, **row}
    return list(selected.values())


def mean_std(values):
    return {"mean": statistics.mean(values), "std": statistics.stdev(values) if len(values) > 1 else None}


def summarize(records, require_complete=False):
    selected = {(r["dataset"], r["seed"], r["method"]): r for r in records}
    verified = {}
    missing = []
    for category in CATEGORIES:
        for seed in SEEDS:
            group = [selected.get((category, seed, method)) for method in METHODS]
            missing.extend({"dataset": category, "seed": seed, "method": method} for method, row in zip(METHODS, group) if row is None)
            if any(row is None for row in group):
                continue
            base = group[0]
            for row in group[1:]:
                mismatch = [k for k in MATCH_KEYS if row.get(k) != base.get(k) or k not in row or k not in base]
                if mismatch:
                    raise ValueError(f"Unpaired experiment {category}/{seed}/{row['method']}: {mismatch}")
            verified[(category, seed)] = group
    if require_complete and missing:
        raise ValueError(f"Matrix incomplete: {len(missing)} of 27 runs missing")
    rows = []
    for category in CATEGORIES:
        groups = [verified[(category, seed)] for seed in SEEDS if (category, seed) in verified]
        if not groups:
            continue
        for index, method in enumerate(METHODS):
            measurements = [{**group[index], **group[index]["metrics"][0]} for group in groups]
            row = {"dataset": category, "method": method, "n": len(groups),
                   "seeds": [group[0]["seed"] for group in groups],
                   "statistics": {key: mean_std([m[key] for m in measurements]) for key in MEASURES}}
            row["paired_changes"] = {
                "search_state_reduction_percent": mean_std([(1-g[index]["search_state_bytes"]/g[0]["search_state_bytes"])*100 for g in groups]),
                **{key + "_delta_pp": mean_std([(g[index]["metrics"][0][key]-g[0]["metrics"][0][key])*100 for g in groups]) for key in METRICS},
            }
            rows.append(row)
    return {"complete": not missing, "matched_groups": len(verified), "expected_groups": 9,
            "available_runs": len(selected), "expected_runs": 27, "missing": missing, "rows": rows,
            "note": "Sample SD across seeds 42/43/44; overlapping data splits, not independent datasets or confidence intervals."}


def format_stat(stat, scale=1, digits=4):
    deviation = "—" if stat["std"] is None else f"{stat['std']*scale:.{digits}f}"
    return f"{stat['mean']*scale:.{digits}f} ± {deviation}"


def export_report(device="cuda"):
    records = collect_records(device)
    summary = summarize(records, require_complete=True)
    folder = ROOT / "course/results"
    folder.mkdir(parents=True, exist_ok=True)
    paths = [ROOT / "course" / name for name in ("experiment.py", "compact_patchcore.py")]
    report = {"protocol_version": "paired-coreset-v2", "created_at": datetime.now().astimezone().isoformat(),
              "course_source_lf_sha256": hashlib.sha256(b"".join(p.read_text(encoding="utf-8").encode() for p in paths)).hexdigest(),
              "source_files_in_hash_order": [p.relative_to(ROOT).as_posix() for p in paths],
              "environment": {p: importlib.metadata.version(p) for p in ("torch", "torchvision", "lightning", "timm", "numpy")},
              "summary": summary, "runs": records}
    (folder / "multiseed_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    fields = ["dataset", "method", "seed", *MEASURES, "split_sha256", "coreset_sha256", "run_id"]
    with (folder / "multiseed_metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({**row, **row["metrics"][0]} for row in records)
    fields = ["dataset", "method", "n"] + [key + suffix for key in MEASURES for suffix in ("_mean", "_std")]
    with (folder / "multiseed_summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        for row in summary["rows"]:
            writer.writerow({"dataset": row["dataset"], "method": row["method"], "n": row["n"], **{key+"_"+stat: value for key, stats in row["statistics"].items() for stat, value in stats.items()}})
    lines = ["# 多随机种子实验：96 / 192 维 PCA 对比", "", "本报告完成三个类别 × 三种方法 × 三个种子，共 27 组实验。旧的单种子报告保留为历史记录。", "",
             "## 实验设置与复现", "", "固定种子 42、43、44，类别 bottle、cable、capsule，方法 baseline、pca96、pca192。复用同一版本已完成实验，未删除或替换不利结果。每个类别/种子下，三种方法的数据划分、骨干配置与压缩前 coreset 哈希均相同。", "",
             "PCA 仅拟合正常训练特征库，验证集校准阈值，测试集评估。种子同时改变数据划分和 coreset 随机过程，所以波动包含这两种因素。只使用 3 类共 1,017 张图像；各随机划分共享同一批数据，不能当成 3 个独立外部测试集。", "",
             "执行 `python course/experiment_suite.py` 可续跑固定矩阵，`python course/summarize_experiments.py` 重新生成报告。环境仍由 uv 管理。", "",
             "![质量与存储对比](results/multiseed_comparison.png)", "", "## 质量：均值 ± 样本标准差（n=3）", "", "| 类别 | 方法 | 图像 AUROC | 图像 F1 | 像素 AUROC | 像素 F1 |", "|---|---|---:|---:|---:|---:|"]
    for row in summary["rows"]:
        lines.append(f"| {row['dataset']} | {row['method']} | " + " | ".join(format_stat(row["statistics"][key]) for key in METRICS) + " |")
    lines += ["", "标准差为 ddof=1，不是置信区间，也不是显著性检验。AUROC 极小的浮点舍入误差保留在原始 JSON 中。", "", "## 相对于原始基线的配对变化", "", "| 类别 | 压缩方法 | 检索状态减少 % | 图像 AUROC 变化（百分点） | 图像 F1 变化（百分点） | 像素 AUROC 变化（百分点） |", "|---|---|---:|---:|---:|---:|"]
    for row in summary["rows"]:
        if row["method"] == "baseline":
            continue
        changes = row["paired_changes"]
        lines.append(f"| {row['dataset']} | {row['method']} | " + " | ".join(format_stat(changes[key], digits=3) for key in ("search_state_reduction_percent", "image_AUROC_delta_pp", "image_F1Score_delta_pp", "pixel_AUROC_delta_pp")) + " |")
    lines += ["", "检索状态包含特征库、投影均值和矩阵。96 维特征库本身减少 75%，192 维减少 50%，但加上投影开销后，实际总检索状态收益更小。该指标不是模型文件大小或总显存。", "", "## 模型大小与耗时", "", "| 类别 | 方法 | 检索状态 KiB | 模型文件 MiB | 前向中位 ms（各实验的均值 ± SD） |", "|---|---|---:|---:|---:|"]
    for row in summary["rows"]:
        stats = row["statistics"]
        lines.append(f"| {row['dataset']} | {row['method']} | {format_stat(stats['search_state_bytes'], 1/1024, 1)} | {format_stat(stats['checkpoint_bytes'], 1/1048576, 2)} | {format_stat(stats['latency_median_ms'], digits=2)} |")
    lines += ["", "每次实验预热 10 次，对第一张测试图进行 30 次 CUDA 同步前向计时；表中汇总三个实验各自的中位数。计时包含骨干、投影、检索与原始异常图，不含读图、预处理、阈值后处理与显示。部分种子 42 数据复用先前运行，测量时段和笔记本负载不受控；这里只作描述，不能证明稳定加速。", "", "## 解释边界", "", "- 可以据此量化存储减少以及各类别的质量权衡；没有预设压缩必定提高精度。", "- 96/192 维比较属于探索性实验。看到测试结果后选择维度，不能再把同一测试集当成完全未参与方案选择的最终验收数据。", "- 若把先前示例目标“存储至少减少 50%、每类别平均图像 AUROC 下降不超过 1 个百分点”作为验收要求，必须逐项核验；192 维含投影后的存储收益可能不足 50%。不能只因精度更好就改称已达到原目标。", "- 后续正式验收应先固定方法和阈值，再使用额外未参与方案选择的类别或独立数据。", "", "[27 组逐次指标](results/multiseed_metrics.csv) · [均值和标准差](results/multiseed_summary.csv) · [原始 JSON 与哈希](results/multiseed_metrics.json)", ""]
    lines += ["## 本轮结论", "", "以下判断基于已观察的测试结果，属于探索性结论。各项收益均按类别统计，不能推广为所有工业场景的保证。", ""]
    for method in METHODS[1:]:
        rows = [row for row in summary["rows"] if row["method"] == method]
        reductions = [row["paired_changes"]["search_state_reduction_percent"]["mean"] for row in rows]
        worst_drop = max(0, *[-row["paired_changes"]["image_AUROC_delta_pp"]["mean"] for row in rows])
        passed = min(reductions) >= 50 and worst_drop <= 1
        outcome = "满足" if passed else "未同时满足"
        lines.append(f"- {method}：检索状态减少 {min(reductions):.2f}%–{max(reductions):.2f}%；各类别平均图像 AUROC 的最大下降为 {worst_drop:.3f} 个百分点。{outcome}先前示例中的“存储至少减少 50%、每类别平均图像 AUROC 下降不超过 1 个百分点”两项条件。")
    lines += ["", "平均值满足某项限制，并不代表每个种子都满足。可以结合上表选择精度和存储之间的折中，但正式验收应使用预先固定的方案与独立数据。", ""]
    (ROOT / "course/多种子实验报告.md").write_text("\n".join(lines), encoding="utf-8")
    from plot_results import plot_report
    plot_report(report)
    print("Exported 27 verified experiments, 9 summary rows and comparison figure")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    export_report(parser.parse_args().device)
