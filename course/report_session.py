"""Publish measured session latency and unchanged detection metrics."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'course/outputs/session-audit'
RESULTS=ROOT/'course/results'


def main():
    verification=json.loads((OUT/'verification.json').read_text(encoding='utf-8'))
    benchmark=json.loads((OUT/'benchmark.json').read_text(encoding='utf-8'))
    assert verification['complete'] and benchmark['complete']
    assert len(verification['runs'])==9 and verification['test_evaluations']==552
    assert all(r['four_metrics_identical'] and r['raw_scores_and_maps_bit_exact'] for r in verification['runs'])
    assert all(r['warm_median_seconds']<r['original_median_seconds'] for r in benchmark['categories'])
    files=['predict.py','prediction_worker.py','audit_session.py']
    report={'complete':True,'verification':verification,'benchmark':benchmark,
            'source_lf_sha256':{name:hashlib.sha256((ROOT/'course'/name).read_text(encoding='utf-8-sig').replace('\r\n','\n').encode()).hexdigest() for name in files},
            'limits':['Same RTX 3050 Laptop 4 GB, CUDA, batch size 1; 3 timings per category, seed 42.',
                      'Warm requests reuse one process and checkpoint; first request remains cold.',
                      'Latency includes rendering and JSON output, excludes browser upload and polling.',
                      '552 evaluations contain repeated images across seeds, not 552 distinct images.',
                      'Resident process retains RAM/CUDA resources until 120 s idle, training, or GUI shutdown.',
                      'No claim of faster neural-network forward pass or reduced inference memory.']}
    (RESULTS/'session_metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    fields=['dataset','original_median_seconds','warm_median_seconds','latency_reduction_percent','speedup','worker_first_request_seconds']
    with (RESULTS/'session_metrics.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(benchmark['categories'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=benchmark['categories']
    fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained')
    x=list(range(len(rows)))
    before=ax.bar([n-.19 for n in x],[r['original_median_seconds'] for r in rows],.36,label='Original: new process per image',color='#384a63')
    after=ax.bar([n+.19 for n in x],[r['warm_median_seconds'] for r in rows],.36,label='Optimized: reused session',color='#12856c')
    ax.bar_label(before,fmt='%.2f s',padding=4);ax.bar_label(after,fmt='%.3f s',padding=4)
    ax.set_xticks(x,[r['dataset'].capitalize() for r in rows]);ax.set_ylabel('Request latency (seconds; lower is better)')
    ax.set_ylim(0,max(r['original_median_seconds'] for r in rows)*1.25)
    ax.set_title('Repeated single-image inspection: same predictions, less waiting')
    ax.legend(loc='upper right');ax.spines[['top','right']].set_visible(False)
    fig.text(.5,-.035,'Median of 3 paired images per category; includes rendered output. First request still needs startup.',ha='center',fontsize=9)
    fig.savefig(RESULTS/'session_comparison.png',dpi=170,bbox_inches='tight');plt.close(fig)
    table='\n'.join(f"| {r['dataset']} | {r['original_median_seconds']:.3f} | {r['warm_median_seconds']:.3f} | {r['latency_reduction_percent']:.2f}% | {r['speedup']:.1f}× | {r['worker_first_request_seconds']:.2f} |" for r in rows)
    metric_names=['image_AUROC','image_F1Score','pixel_AUROC','pixel_F1Score']
    metric_rows=[]
    for category in ('bottle','cable','capsule'):
        items=[r for r in verification['runs'] if r['dataset']==category]
        metric_rows.append('| '+category+' | '+' | '.join(f"{statistics.mean(r['metrics'][0][m] for r in items):.6f}" for m in metric_names)+' | 全部 0 |')
    text=f'''# 单图检测会话复用优化报告

在原有无损存储优化之外，本轮增加 **连续检测响应时间优化**。相同模型的后续请求复用 Python 进程和已加载权重，保持 anomalib 原始预处理、模型前向、后处理与可视化路径。每次请求创建独立 Engine，隔离回调及输出目录。

## 实测结果

RTX 3050 Laptop 4 GB、CUDA、batch=1。每类别使用 seed 42 的 3 张固定测试图，同一图片交替运行两种方式。下表为每类别 3 次请求的中位数。

| 类别 | 原流程秒 | 复用后秒 | 延迟减少 | 比值 | 新服务首次请求秒 |
|---|---:|---:|---:|---:|---:|
{table}

![同图配对计时](results/session_comparison.png)

计时从客户端发起请求到结果 JSON 和可视化 PNG 落盘完成。原流程包含进程退出；优化流程返回响应后保持服务。两者都是实际完成请求的时间，**不包含浏览器上传、页面轮询或用户操作**。GUI 在单图任务期间每 0.5 秒检查一次完成状态。原实现快照来自提交 `{benchmark['baseline_commit']}` 的 `course/predict.py`，SHA256 为 `{benchmark['original_predict_sha256']}`。

首张图、空闲超时后的首张图仍有进程启动成本；切换模型需重新加载权重。这里的比值不是神经网络前向加速比，也不能用于声称大批量吞吐提升。9 次计时只反映本机当前负载，原始逐次测量保存在 JSON。

## 检测质量不退化

固定原来的 bottle、cable、capsule × seed 42/43/44，共 9 个 checkpoint。逐文件检查已有数据划分与图像 SHA256，复用前后各运行一次真实单图接口，再核对全部最终测试图的模型原始输出。

- **552 次测试图像评估**的原始异常得分与异常图逐位一致；这些是跨种子的评估次数，包含重复图片。
- 9 个模型各复核一次原始/缓存单图请求，后处理得分、标签、异常图、缺陷掩码哈希完全一致。
- 对复用模型再次执行上游 `Engine.test`，9 组的图像 AUROC、图像 F1、像素 AUROC、像素 F1 与冻结的原始实验记录逐项严格相等。
- 计时使用的 9 对原流程/复用请求，其结果记录和可视化 PNG 字节也全部相同。

以下为各类别 3 种子的均值（复用前后相同）。

| 类别 | 图像 AUROC | 图像 F1 | 像素 AUROC | 像素 F1 | 四指标变化 |
|---|---:|---:|---:|---:|---|
{chr(10).join(metric_rows)}

## 生命周期与资源代价

GUI 仅缓存一个模型，模型路径、设备、checkpoint 或元数据文件大小/修改时间改变时重新加载。空闲 120 秒、开始训练或退出 GUI 时释放工作进程；异常响应清除缓存，退出失联可重新启动。所有任务保持串行。每次输出目录加随机后缀，连续快速请求也不会覆盖。

复用需要暂时保留 Python、模型及 CUDA 运行时资源，**并不宣称空闲内存/显存不增加**。收益是检测质量保持不变时减少连续交互的等待；原有 75.63% 组合部署包缩减仍然成立。若要求每个时刻所有资源指标都不增加，应使用原单次 CLI 模式。

GUI 已核验模型选择、真实历史结果、复用状态和处理耗时展示。浏览器文件上传后的完整操作链未重新测试；推理验证通过现有本地数据集完成。

## 复现

在项目根目录运行：

```powershell
.\\.venv\\Scripts\\python.exe course/audit_session.py verify
.\\.venv\\Scripts\\python.exe course/audit_session.py benchmark
.\\.venv\\Scripts\\python.exe course/report_session.py
.\\.venv\\Scripts\\python.exe -m unittest discover -s course -p "test_*.py"
```

GUI 启动后自动使用复用服务，`course/predict.py --image ... --run ...` 仍提供单次独立进程模式。实验数据和模型在本地 `course/outputs`；GitHub 仅提交源码、汇总及报告。
'''
    (ROOT/'course/会话复用优化报告.md').write_text(text,encoding='utf-8')
    print(json.dumps({f['dataset']:f['latency_reduction_percent'] for f in rows}))


if __name__=='__main__':
    main()
