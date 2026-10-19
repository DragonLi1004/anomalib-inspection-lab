"""Export verified paired experiments without machine paths or dataset images."""
import csv
import importlib.metadata
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEYS = ("dataset", "seed", "device", "backbone", "image_size", "batch_size", "coreset_sampling_ratio",
        "num_neighbors", "val_split_ratio", "split_sha256", "coreset_sha256", "protocol_version", "course_code_sha256")


def main():
    runs = []
    for path in sorted((ROOT / "course/outputs").glob("*/metrics.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("protocol_version") == "paired-coreset-v2" and data.get("seed") == 42:
            runs.append({"run_id": path.parent.name, **data})
    pairs = []
    for category in ("bottle", "cable", "capsule"):
        compact = next(r for r in runs if r["dataset"] == category and r["method"] == "pca96")
        baseline = next(r for r in runs if r["method"] == "baseline" and all(r[key] == compact[key] for key in KEYS))
        pairs.append((baseline, compact))
    folder = ROOT / "course/results"
    folder.mkdir(parents=True, exist_ok=True)
    source_paths = [ROOT / "course" / n for n in ("experiment.py", "compact_patchcore.py")]
    raw_hash = hashlib.sha256(b"".join(p.read_bytes() for p in source_paths)).hexdigest()
    lf_hash = hashlib.sha256(b"".join(p.read_text(encoding="utf-8").encode("utf-8") for p in source_paths)).hexdigest()
    if any(r["course_code_sha256"] != raw_hash for pair in pairs for r in pair):
        raise ValueError("Course source changed since these runs; preserve their source before exporting")
    metadata = {"protocol": "paired-coreset-v2", "seed": 42,
                "source_files_in_hash_order": ["course/experiment.py", "course/compact_patchcore.py"],
                "course_source_lf_sha256": lf_hash,
                "hash_note": "Run hashes include original Windows line endings. The LF-normalized hash identifies the Git sources.",
                "environment": {p: importlib.metadata.version(p) for p in ("torch", "torchvision", "lightning", "timm", "numpy")},
                "pairs": [{"baseline": a, "pca96": b} for a, b in pairs]}
    (folder / "pilot_metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metrics = ("image_AUROC", "image_F1Score", "pixel_AUROC", "pixel_F1Score")
    with (folder / "pilot_metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, ["dataset", "method", "seed", *metrics, "memory_bank_bytes", "projection_bytes", "search_state_bytes", "checkpoint_bytes", "latency_median_ms", "split_sha256", "coreset_sha256"], extrasaction="ignore")
        writer.writeheader()
        for pair in pairs:
            for run in pair:
                writer.writerow({**run, **run["metrics"][0]})
    lines = ["# 初步实验报告：PatchCore 特征压缩", "", "日期：2026-10-19（Asia/Shanghai）。RTX 3050 Laptop GPU，4 GB。", "",
             "**本报告只使用修正随机状态后的 v2 配对实验。** 旧版结果保留在本地，但不用于以下结论。每组已验证压缩前 coreset 和数据划分的 SHA256 一致。", "",
             "## 用户原始运行结果", "", "2026-10-18 的 hazelnut_toy 运行：训练/验证/测试 = 28/14/14，图像 AUROC 1.0000、图像 F1 0.7059、像素 AUROC 0.9312、像素 F1 0.2875。重复运行指标一致，说明可以复现；样本太少，且 F1 显示实际分类与分割还有不足。", "",
             "## 数据与配置", "", "已下载 bottle、cable、capsule 共 1,017 张图像和 264 张标注。训练使用全部官方正常训练图，原始测试部分按 seed=42 分为独立验证/测试。输入 224×224、ResNet18、batch=1、coreset=1%、9 邻居。基线采用未修改的上游 PatchCore，PCA 仅拟合正常训练 coreset。", "",
             "| 类别 | 训练/验证/测试 | 方法 | 图像 AUROC | 图像 F1 | 像素 AUROC | 像素 F1 |", "|---|---|---|---:|---:|---:|---:|"]
    for pair in pairs:
        for r in pair:
            c = r["split_counts"]
            lines.append(f'| {r["dataset"]} | {c["train"]}/{c["val"]}/{c["test"]} | {r["method"]} | ' + " | ".join(f'{r["metrics"][0][k]:.4f}' for k in metrics) + " |")
    lines += ["", "## 存储与计时", "", "检索状态 = 特征库 + 投影均值/矩阵，不等于模型总大小或总显存。所有类别的特征库本身减少 75%，以下同时计入投影开销。", "", "| 类别 | 原始检索状态 KiB | PCA96 检索状态 KiB | 减少比例 | 原始/PCA 模型文件 MiB | 原始/PCA 前向中位 ms |", "|---|---:|---:|---:|---:|---:|"]
    for a, b in pairs:
        lines.append(f'| {a["dataset"]} | {a["search_state_bytes"]/1024:.1f} | {b["search_state_bytes"]/1024:.1f} | {(1-b["search_state_bytes"]/a["search_state_bytes"])*100:.2f}% | {a["checkpoint_bytes"]/1048576:.2f} / {b["checkpoint_bytes"]/1048576:.2f} | {a["latency_median_ms"]:.2f} / {b["latency_median_ms"]:.2f} |')
    lines += ["", "前向计时预热 10 次，再对固定第一张测试图同步运行 30 次，包含骨干/投影/检索/原始异常图，不含读图、预处理、后处理与可视化。只有一轮笔记本测量，不能据此声称稳定加速。原始 30 次样本保存在 JSON 中。", "", "## 结论与限制", "", "- 已取得可衡量的存储减少；精度变化必须按类别报告，不应只挑选表现最好的类别。", "- PCA 不是无损压缩，个别类别可能出现 AUROC 或 F1 下降。项目价值在于测量并展示这个权衡，而不是保证精度提高。", "- 当前只覆盖 3 类、1 个随机种子，属于初步结果；没有做置信区间或显著性检验。", "- 不能与完整 MVTec AD 测试协议或 WideResNet 论文结果直接比较。", "- 在已观察的测试结果上选择 96/192 维属于探索性调参；最终验收应固定方案，并使用未参与选择的数据。", "", "## 接下来如何继续", "", "1. 预先确定存储目标和精度容忍范围，例如检索状态下降至少 50%，各类别图像 AUROC 下降不超过 1 个百分点；这只是待验收目标，不是已满足的承诺。", "2. 固定方案后，追加 seed=43、44 的配对实验，报告均值和标准差；必要时扩大类别。", "3. 对不同维度做预先约定的对比。若 96 维质量下降过多，考虑 192 维，但同时重新计算含投影的存储开销。", "4. 增加固定负载下的重复计时，分别报告模型前向与读图到显示的端到端耗时。", "", "## 可追溯材料", "", "[原始汇总 JSON](results/pilot_metrics.json) / [对比 CSV](results/pilot_metrics.csv)。每条记录包含划分与 coreset 哈希、课程代码 SHA256、上游提交、原始指标及计时样本。完整 checkpoint、图片和逐图清单在本机对应的 course/outputs/ 实验目录，不上传 GitHub。", ""]
    (ROOT / "course/实验报告.md").write_text("\n".join(lines), encoding="utf-8")
    print("Verified and exported", len(pairs), "paired experiments")


if __name__ == "__main__":
    main()
