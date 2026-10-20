"""Generate the accepted lossless-storage report and retain rejected search data."""
import csv,hashlib,json,statistics
from pathlib import Path
from baseline import ROOT

def main():
    folder=Path((ROOT/'course/outputs/latest_lossless.txt').read_text().strip())
    result=json.loads((folder/'audit.json').read_text())
    assert result['complete'] and len(result['runs'])==9
    for row in result['runs']:
        assert row['byte_identical'] and row['state_dict_identical'] and row['metrics_identical']
        assert row['original_metrics']==row['restored_metrics']
        assert row['single_package']['reduction_percent']>0
    categories=[]
    for category in ['bottle','cable','capsule']:
        rows=[r for r in result['runs'] if r['dataset']==category]
        categories.append({'dataset':category,'n':3,
            'original_bytes_mean':statistics.mean(r['single_package']['original_bytes'] for r in rows),
            'single_package_bytes_mean':statistics.mean(r['single_package']['package_bytes'] for r in rows),
            'single_reduction_percent_mean':statistics.mean(r['single_package']['reduction_percent'] for r in rows),
            'metrics':{k:statistics.mean(r['original_metrics'][k] for r in rows) for k in rows[0]['original_metrics']}})
    result['summary']=categories
    result['test_image_evaluations']=sum(r['test_images'] for r in result['runs'])
    result['source_lf_sha256']={p:hashlib.sha256((ROOT/'course'/p).read_text(encoding='utf-8-sig').encode()).hexdigest() for p in ['lossless_package.py','audit_lossless.py']}
    result['acceptance']='Accepted for model-package storage: all 9 restored checkpoint files byte-identical, all state tensors identical, all four test detection metrics identical. Original upstream inference path retained. Packing/restoration are offline deployment operations.'
    rejected=json.loads((ROOT/'course/outputs/exact-optimization/verified.json').read_text())
    rejected['acceptance']='Not selected: exact detection quality, but some runs had no forward peak-memory benefit and inference latency increased. Never enabled in the production prediction path.'
    result['rejected_search_summary']=[]
    name=rejected['selection']['selected']
    for r in rejected['runs']:
        a,b=r['benchmark']['upstream'],r['benchmark'][name]
        result['rejected_search_summary'].append({'dataset':r['dataset'],'seed':r['seed'],
            'raw_outputs_identical':r['parity']['unequal_scores']==r['parity']['unequal_maps']==0,
            'temporary_memory_reduction_percent':(1-b['peak_extra_bytes_max']/a['peak_extra_bytes_max'])*100,
            'latency_change_percent':(b['latency_median_ms']/a['latency_median_ms']-1)*100})
    outputs=ROOT/'course/results'
    (outputs/'lossless_metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (outputs/'search_exploration.json').write_text(json.dumps(rejected,indent=2),encoding='utf-8')
    fields=['dataset','seed','source_run','original_bytes','package_bytes','reduction_percent','byte_identical','metrics_identical','restore_seconds']
    with (outputs/'lossless_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader();w.writerows({**r,**r['single_package']} for r in result['runs'])
    b=result['bundle']
    lines=['# 保持原始检测指标的无损模型存储优化','',
      f"**最终采用：无损压缩与跨模型去重。** 9 个原始 PatchCore 模型从 {b['original_bytes']/1048576:.2f} MiB 压缩到 {b['package_bytes']/1048576:.2f} MiB，部署包减少 **{b['reduction_percent']:.2f}%**。单模型无需跨模型去重也有收益，见下表。",'',
      '![模型包存储对比](results/lossless_comparison.png)','',
      '## 为什么不会损失精度','',
      '不同类别和随机种子的 checkpoint 重复保存了相同的预训练骨干等张量。新增打包工具按原 ZIP 数据区边界拆分文件，对字节内容做 SHA-256 去重，再用 Deflate 6 无损压缩。每个模型保存恢复配方，所有原 ZIP 头、填充和目录字节也完整保留。','',
      '恢复过程逐块验证 SHA-256，最终再核验整个 checkpoint 的字节数与 SHA-256。恢复文件与原文件逐字节相同；不是量化、PCA、剪枝，也不改变骨干、特征库、数据划分、阈值或推理代码。','',
      f"已恢复并验证全部 9 个原始模型，逐项比对 state_dict，再用原始 Engine.test 完成 {result['test_image_evaluations']} 次测试图像评估。9 组的图像 AUROC、图像 F1、像素 AUROC、像素 F1 与原始记录全部完全一致。多个种子复用同一数据集，评估次数不等于独立图片数量。",'',
      '## 单个模型也能节省存储','',
      '| 类别 | 原始模型 MiB，三种子均值 | 单模型包 MiB | 体积减少 | 四项检测指标 |','|---|---:|---:|---:|---|']
    for r in categories:lines.append(f"| {r['dataset']} | {r['original_bytes_mean']/1048576:.2f} | {r['single_package_bytes_mean']/1048576:.2f} | {r['single_reduction_percent_mean']:.2f}% | 全部不变 |")
    lines+=['','## 原始与恢复后的指标（三种子均值相同）','','| 类别 | 图像 AUROC | 图像 F1 | 像素 AUROC | 像素 F1 |','|---|---:|---:|---:|---:|']
    for r in categories:lines.append('| '+r['dataset']+' | '+' | '.join(f"{r['metrics'][k]:.4f}" for k in ['image_AUROC','image_F1Score','pixel_AUROC','pixel_F1Score'])+' |')
    times=[r['restore_seconds'] for r in result['runs']]
    lines+=['','## 指标边界与使用成本','',
      '- 改进指标是**部署包字节数**。原始基线是已有的原生 PyTorch checkpoint，不是另一个专门压缩后的归档方案。多模型收益取决于共享内容，单模型收益单独报告。',
      '- 恢复之后使用完全相同的原始模型和推理路径，不新增推理缓存或压缩步骤；原始训练与检测代码未改动。本轮没有声称推理加速、训练加速或 GPU 显存下降。',
      f"- 打包 9 个模型实测约 {b['pack_seconds']:.2f} 秒；从组合包恢复一个模型约 {min(times):.3f}–{max(times):.3f} 秒。这是额外的一次性部署操作，不计入模型前向耗时。",
      '- 包不能直接交给 torch.load，需要先恢复。可只恢复要部署的模型。打包与恢复都保留原文件，当前本地磁盘因为还保留备份和验证产物，不会自动空出 75.63% 的空间。',
      '- 格式具备校验但不具备签名认证；只加载自己生成、来源可信的 checkpoint。SHA-256 一致性不意味着任意来源的 pickle 都可安全加载。','',
      '## 本地使用','',f"组合包：`{result['package']}`。GUI 新增“无损部署”页，可查看体积和一致性验证，也可下载已生成的组合包或单个模型包。",'',
      '在项目根目录执行（模型 ID 可从 GUI 或 CSV 复制）：','','```powershell',
      r'.\.venv\Scripts\python.exe course\lossless_package.py pack --run bottle-baseline-20261019-092413 --output course\outputs\my-model.alpack',
      f".\\.venv\\Scripts\\python.exe course\\lossless_package.py restore --package {result['package']} --run bottle-baseline-20261019-092413 --output course/outputs/restored-bottle/baseline.ckpt",
      '```','',
      '重复恢复到相同内容的文件会直接复用；若目标已有不同内容则拒绝覆盖。恢复不依赖原来的 checkpoint 文件。参数选择不涉及检测集调参。','',
      '## 未采用的方案','',
      'PCA96/PCA192 的质量损失仍保留在此前报告。本轮还测试了中间张量提前释放与近邻分块：在验证集筛选、冻结 196 分块后，完整测试的分数与热力图一致，但部分组没有显存收益且耗时上升，因此没有接入默认检测流程，也不计为最终优化成果。','',
      '[无损逐次验证 JSON](results/lossless_metrics.json) · [CSV](results/lossless_metrics.csv) · [未采用的显存实验原始数据](results/search_exploration.json)','']
    (ROOT/'course/无损存储优化报告.md').write_text('\n'.join(lines),encoding='utf-8')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
    colors=['#45697b','#087e69']
    vals=[b['original_bytes']/1048576,b['package_bytes']/1048576]
    bars=axes[0].bar(['9 original checkpoints','One deduplicated package'],vals,color=colors,width=.55)
    axes[0].bar_label(bars,fmt='%.2f MiB',padding=4);axes[0].set_ylim(0,max(vals)*1.2)
    axes[0].set_ylabel('Deployment artifact size (MiB)');axes[0].set_title(f"Nine-model bundle: {b['reduction_percent']:.2f}% smaller")
    xs=list(range(3))
    for delta,key,label,color in [(-.18,'original_bytes_mean','Original checkpoint',colors[0]),(.18,'single_package_bytes_mean','Single-model package',colors[1])]:
        bars=axes[1].bar([x+delta for x in xs],[r[key]/1048576 for r in categories],width=.34,label=label,color=color)
        axes[1].bar_label(bars,fmt='%.2f',padding=3,fontsize=9)
    axes[1].set_xticks(xs,[r['dataset'] for r in categories]);axes[1].set_ylim(0,16);axes[1].set_ylabel('Mean size across 3 seeds (MiB)');axes[1].legend(loc='upper right',fontsize=9)
    axes[1].set_title('Single-model packages also reduce size')
    fig.suptitle('Lossless storage: byte-identical restored checkpoints\nAll 4 detection metrics unchanged across 9 paired evaluations',fontweight='bold')
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15)
    fig.savefig(outputs/'lossless_comparison.png',dpi=180);plt.close(fig)
    print(f"Exported accepted lossless report: {b['reduction_percent']:.2f}% bundle reduction")

if __name__=='__main__':main()
