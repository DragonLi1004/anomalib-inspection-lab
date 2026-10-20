"""Generate a shareable scientific figure from the verified summary only."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))


def plot_report(report):
    if not report['summary']['complete']:
        raise ValueError('Refusing to plot an incomplete matrix as a final result')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import PercentFormatter

    categories = ['bottle', 'cable', 'capsule']
    methods = ['baseline', 'pca96', 'pca192']
    labels = ['Original PatchCore (384D)', 'PCA 96D', 'PCA 192D']
    colors = ['#456478', '#127f6a', '#c08840']
    rows = {(r['dataset'], r['method']):r for r in report['summary']['rows']}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    fig.suptitle('PatchCore: quality and storage across three seeds\nSeeds 42 / 43 / 44 · matched splits and coresets · exploratory results, not confidence intervals', fontsize=13, fontweight='bold')
    for ax, metric, title in zip(axes.flat[:3], ['image_AUROC', 'image_F1Score', 'pixel_AUROC'],
                                ['Image AUROC', 'Image F1', 'Pixel AUROC']):
        for index, (method, color, label) in enumerate(zip(methods, colors, labels)):
            x = np.arange(3)+(index-1)*0.18
            means = [rows[(c,method)]['statistics'][metric]['mean'] for c in categories]
            errors = [rows[(c,method)]['statistics'][metric]['std'] for c in categories]
            ax.errorbar(x, means, yerr=errors, fmt='o', markersize=6, capsize=4, color=color, label=label)
        ax.set_xticks(range(3), categories)
        ax.set_ylabel(title + ' (0–1)')
        ax.set_title(title+' · mean ± sample SD', fontsize=11)
        ax.grid(axis='y', alpha=.2)
        # Point/error-bar charts use data-based axes; not zero-baseline bars.
        lower=min(rows[(c,m)]['statistics'][metric]['mean']-rows[(c,m)]['statistics'][metric]['std'] for c in categories for m in methods)
        ax.set_ylim(max(0, lower-.025),1.02)
    ax=axes[1,1]
    for index,(method,color,label) in enumerate(zip(methods,colors,labels)):
        values=[rows[(c,method)]['statistics']['search_state_bytes']['mean']/rows[(c,'baseline')]['statistics']['search_state_bytes']['mean'] for c in categories]
        bars=ax.bar(np.arange(3)+(index-1)*.24, values, width=.22, color=color, label=label)
        for bar,value in zip(bars,values):
            ax.text(bar.get_x()+bar.get_width()/2, value+.02, f'{value*100:.1f}%',ha='center',fontsize=8)
    ax.set_xticks(range(3),categories)
    ax.set_ylim(0,1.15)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylabel('Search state / original')
    ax.set_title('Memory bank + projection overhead',fontsize=11)
    ax.grid(axis='y',alpha=.2)
    for axis in axes.flat:
        axis.spines[['top','right']].set_visible(False)
    handles, legend_labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,legend_labels,loc='outside lower center',ncol=3,frameon=False)
    target=ROOT/'course/results/multiseed_comparison.png'
    fig.savefig(target,dpi=180,facecolor='white')
    plt.close(fig)
    return target


if __name__=='__main__':
    report=json.loads((ROOT/'course/results/multiseed_metrics.json').read_text(encoding='utf-8'))
    print(plot_report(report))
