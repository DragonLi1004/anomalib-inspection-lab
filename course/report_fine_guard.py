"""Publish the full FineGuard paired regression, including failed validation explorations."""
import csv
import hashlib
import json
import statistics
from pathlib import Path
from baseline import ROOT
from study_fine import OUT
from study_precision import save


def main():
    audit = json.loads((OUT / 'test-results.json').read_text(encoding='utf-8'))
    selection = json.loads((OUT / 'frozen-selection.json').read_text(encoding='utf-8'))
    assert audit['complete']
    runs = [{k: v for k, v in row.items() if k not in ['counts_per_image', 'images']} for row in audit['runs']]
    means = {mode: {key: statistics.mean(r[mode][key] for r in runs) for key in runs[0]['before']} for mode in ['before', 'after']}
    mean_rates = {mode: {key: statistics.mean(r['pixel_rates'][mode][key] for r in runs) for key in ['pixel_precision', 'pixel_recall']} for mode in ['before', 'after']}
    ablations = [json.loads(path.read_text(encoding='utf-8')) for path in sorted((ROOT / 'course/outputs/precision-study-v2').glob('*/validation.json'))]
    report = {'complete': True, 'accepted': audit['accepted'], 'name': 'FineGuard v1 · 高分辨率细节补检',
        'date': '2026-10-26', 'dataset': 'capsule', 'selection': audit['selection'],
        'mean_pixel_f1_gain_pp': audit['mean_pixel_f1_gain_pp'], 'means': means, 'mean_rates': mean_rates,
        'test_evaluations': sum(r['test_evaluations'] for r in runs), 'runs': runs,
        'selection_sha256': audit['selection_sha256'], 'implementation_amendment': audit['implementation_amendment'],
        'resource_exclusion': selection['resource_exclusion'],
        'validation': [{'seed': r['seed'], 'candidates': r['candidates']} for r in selection['runs']],
        'single_branch_ablations': [{k: r[k] for k in ['candidate', 'seed', 'metrics', 'delta', 'validation_guards_pass']} for r in ablations],
        'limitations': ['Retrospective historical data; 3 seeds overlap in images; evaluations are not unique images.',
            'Classification and ranking heatmap remain original; gains concern binary defect localization.',
            'Auxiliary model increases storage and inference latency. Precision/recall are reported separately.',
            'Descriptive image bootstrap intervals; no multiple-comparison correction.',
            'First audit interrupted for CPU/GPU resize mismatch; same frozen parameters rerun after parity fix, then audit-only device/mask fixes. All logs retained.']}
    results = ROOT / 'course/results'
    save(results / 'fine_metrics.json', report)
    with (results / 'fine_metrics.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['seed', 'metric', 'before', 'after', 'delta_pp'])
        for r in runs:
            for k in r['before']:
                writer.writerow([r['seed'], k, r['before'][k], r['after'][k], r['delta_pp'][k]])
            for k in ['pixel_precision', 'pixel_recall']:
                a, b = r['pixel_rates']['before'][k], r['pixel_rates']['after'][k]
                writer.writerow([r['seed'], k, a, b, 100*(b-a)])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    x = list(range(3))
    for mode, shift, color in [('before', -.18, '#455c75'), ('after', .18, '#087d6c')]:
        bars = axes[0].bar([i+shift for i in x], [100*r[mode]['pixel_F1Score'] for r in runs], .36, label=mode, color=color)
        axes[0].bar_label(bars, fmt='%.2f', padding=3, fontsize=9)
        bars = axes[1].bar([i+shift for i in [0, 1]], [100*mean_rates[mode][k] for k in ['pixel_precision', 'pixel_recall']], .36, label=mode, color=color)
        axes[1].bar_label(bars, fmt='%.2f', padding=3, fontsize=9)
    axes[0].set_xticks(x, [f'Seed {r["seed"]}' for r in runs])
    axes[0].set_title(f'Capsule pixel F1: mean +{audit["mean_pixel_f1_gain_pp"]:.3f} pp')
    axes[1].set_xticks([0, 1], ['Pixel precision', 'Pixel recall'])
    axes[1].set_title('Mean over three seeds')
    for axis in axes:
        axis.set_ylim(0, 100); axis.set_ylabel('Percent'); axis.legend(); axis.spines[['top', 'right']].set_visible(False)
    fig.savefig(results / 'fine_comparison.png', dpi=170, bbox_inches='tight'); plt.close(fig)
    lines = ['# 精度优化第二轮对比报告：FineGuard v1', '', '日期：2026-10-26。基于 anomalib / PatchCore 的胶囊缺陷定位改进。', '',
        f'胶囊三个种子的平均像素 F1：**{means["before"]["pixel_F1Score"]:.6f} → {means["after"]["pixel_F1Score"]:.6f}，提升 {audit["mean_pixel_f1_gain_pp"]:.3f} 个百分点**。', '',
        f'冻结配置：`{audit["selection"]["name"]}`。本次完成 {report["test_evaluations"]} 次测试图像评估，含跨种子重复图像。原有四项检测指标与召回率逐种子非退化验收：**{'通过' if audit['accepted'] else '未通过'}**。', '',
        '![对比图](results/fine_comparison.png)', '', '## 全部测试结果', '',
        '| 种子 | 原像素 F1 | 新像素 F1 | 提升（百分点） | 像素精确率（前→后） | 像素召回率（前→后） |', '|---|---:|---:|---:|---|---|']
    for r in runs:
        p = r['pixel_rates']
        lines.append(f'| {r["seed"]} | {r["before"]["pixel_F1Score"]:.6f} | {r["after"]["pixel_F1Score"]:.6f} | {r["delta_pp"]["pixel_F1Score"]:+.3f} | {p["before"]["pixel_precision"]:.2%} → {p["after"]["pixel_precision"]:.2%} | {p["before"]["pixel_recall"]:.2%} → {p["after"]["pixel_recall"]:.2%} |')
    lines += ['', '| 三种子均值 | 原始 | 优化后 |', '|---|---:|---:|']
    for k in means['before']:
        lines.append(f'| {k} | {means["before"][k]:.6f} | {means["after"][k]:.6f} |')
    lines += ['', '图像 AUROC、图像 F1、像素 AUROC 逐种子完全一致。保留原评分和连续热力图，改进发生在最终二值缺陷掩码；不把这一结果描述为分类准确率提高。瓶子与电缆的部署模型本轮未改动，也不宣称获得新精度收益。', '',
        '## 方法、筛选与复现口径', '',
        '原分支采用 224×224 输入、ResNet18 layer2+layer3。辅助分支采用 384×384 输入、ResNet18 layer2，正常训练特征的 1% coreset。辅助异常图以双线性插值映射到原 224×224 坐标。仅在原模型判为异常的图像中补充高置信度像素，最终掩码为原掩码与补充掩码的并集。', '',
        '补检阈值为各辅助模型原验证校准阈值的 1.2 倍。并集保留所有原阳性像素，所以召回率不会因删除旧区域而下降；F1 和精确率仍须实测，无法由并集结构保证。', '',
        '候选仅在各自验证集比较。三个种子使用同一分支配置和阈值倍数，要求每个种子的验证 F1、精确率均不下降，F1 增益的按图片配对 bootstrap 95% 区间下限为正，平均 F1 至少增加 0.01，再按平均区间下限选择。参数与权重在测试前冻结。', '',
        '两组高分辨率候选共六次训练完成；layer1+layer2 候选因 4 GB 笔记本训练开销过高，在测试前停止，保留部分日志。较早的 3%/5% 特征库和仅 layer2 单分支共九组验证结果，以及较早的掩码并集失败筛选，都保留在本地；单分支结果也纳入 JSON，避免只报告成功尝试。', '',
        f'冻结配置 SHA256：`{audit["selection_sha256"]}`。首次测试发现 CPU/GPU 缩放造成微小像素 AUROC 差异。修复为上游数据模块的 CPU 缩放、设备侧归一化路径后，用相同权重和阈值重跑；后续中断修复的是审计脚本的设备迁移、掩码类型、参考标量所在设备及 NumPy 布尔序列化处理。没有根据测试指标重新调参。原始日志与实现变更哈希均保留。', '',
        '## 不确定性与代价', '',
        '本数据集已用于此前研究，各种子的图片有重叠。这是历史数据上的回归对比，不是独立外部数据上的一次性验证。以下区间仅作描述，未进行多候选比较校正：']
    for r in runs:
        lines.append(f'- S{r["seed"]} 像素 F1 增益区间：[{100*r["gain_ci95"][0]:.3f}, {100*r["gain_ci95"][1]:.3f}] 个百分点。')
    lines += ['', '| 种子 | 原预处理+推理中位耗时 ms | 新中位耗时 ms | 原/新增量张量峰值 MiB |', '|---|---:|---:|---:|']
    for r in runs:
        t = r['timing']
        lines.append(f'| {r["seed"]} | {t["median_ms"]["baseline"]:.3f} | {t["median_ms"]["fine_guard"]:.3f} | {t["peak"]["baseline"]["incremental_bytes"]/1048576:.2f} / {t["peak"]["fine_guard"]["incremental_bytes"]/1048576:.2f} |')
    lines += ['', '计时使用同一原尺寸 GPU 输入、10 次预热和 40 次交替同步测量，包括 CPU 缩放/传输、模型及后处理，不含读图、绘图与启动。测量时两分支均常驻，因此增量张量峰值仅表示执行工作空间，不代表独立部署总显存；不能与之前只测模型前向的耗时混比。额外辅助模型会增加存储、加载与推理成本，不能声称速度或模型大小同步改善。', '',
        '之前的推理会话复用仍可使用。旧无损组合包的 75.63% 缩减只针对旧九模型包，不能套用于新增辅助权重。', '',
        '## 使用与复现', '', '已完成 39 项自动化测试；真实预测成功生成图像，重复预测签名一致且命中模型缓存。GUI 报告页、三组模型选项及报告下载接口已核对。', '', '图像检测 → 胶囊 → `PatchCore-R18-384 + FineGuard v1`。保留原基线作为默认；必须同时保留新模型目录中的 baseline.ckpt、auxiliary.ckpt 和 metrics.json。', '',
        '实验目录：`course/outputs/fine-study-v3/`。训练用 `study_fine.py`，验证筛选用 `select_fine_guard.py`，冻结审计用 `audit_fine_guard.py`，注册用 `register_fine_guard.py`，本报告用 `report_fine_guard.py` 生成。既有冻结文件不得覆盖；新研究应使用新目录。辅助脚本和资源排除记录体现本次实验过程，未完成候选不能算作精度失败。', '',
        '明细：`course/results/fine_metrics.json`；表格：`course/results/fine_metrics.csv`。']
    (ROOT / 'course/精度优化第二轮对比报告.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'accepted': report['accepted'], 'means': means, 'mean_rates': mean_rates, 'gain_pp': report['mean_pixel_f1_gain_pp']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
