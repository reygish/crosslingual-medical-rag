import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

results_rag = pd.read_csv('experiments/rag_eval.csv')
print(f"Loaded results_rag from experiments/rag_eval.csv (rows={len(results_rag)})")

results_norag = pd.read_csv('experiments/norag_eval.csv')
print(f"Loaded results_norag from experiments/norag_eval.csv (rows={len(results_norag)})")

# =========================================
# GLOBAL STYLE
# =========================================

sns.set_theme(
    style="whitegrid",
    context="talk"
)

plt.rcParams['figure.dpi'] = 120
plt.rcParams['axes.titleweight'] = 'bold'

# Soft academic palette
rag_palette = sns.color_palette("Blues", 5)
norag_palette = sns.color_palette("Oranges", 2)

# =========================================
# RAG VIOLIN PLOT
# =========================================

rag_cols = [
    'faithfulness',
    'answer_relevancy',
    'context_precision',
    'context_recall',
    'answer_correctness'
]

rag_labels = [
    'Faithfulness',
    'Answer\nRelevancy',
    'Context\nPrecision',
    'Context\nRecall',
    'Answer\nCorrectness'
]

rag_long = results_rag[rag_cols].melt(
    var_name='Metric',
    value_name='Score'
)

plt.figure(figsize=(15, 7))

ax = sns.violinplot(
    data=rag_long,
    x='Metric',
    y='Score',
    palette=rag_palette,
    inner=None,
    linewidth=1.2,
    cut=0,
    saturation=0.9
)

# Overlay boxplot for quartiles
sns.boxplot(
    data=rag_long,
    x='Metric',
    y='Score',
    width=0.05,
    showcaps=True,
    boxprops={
        'facecolor': 'white',
        'zorder': 3
    },
    whiskerprops={'linewidth': 1.5},
    medianprops={
        'color': 'black',
        'linewidth': 2
    },
    showfliers=False
)

# Mean markers
means = results_rag[rag_cols].mean().values

for i, mean in enumerate(means):
    plt.scatter(
        i,
        mean,
        marker='D',
        s=90,
        edgecolor='black',
        linewidth=1,
        zorder=10,
        label='Mean Score' if i == 0 else ""
    )

# Mean value text
for i, mean in enumerate(means):
    plt.text(
        i + 0.20,
        mean + 0.05,
        f"{mean:.3f}",
        ha='center',
        fontsize=10,
        fontweight='bold'
    )

ax.set_xticklabels(rag_labels)

plt.title(
    'Distribution of RAG Evaluation Metrics',
    fontsize=20,
    pad=20
)

plt.xlabel('')
plt.ylabel('Score', fontsize=14)

plt.ylim(-0.02, 1.02)

plt.grid(
    axis='y',
    linestyle='--',
    alpha=0.3
)

plt.legend(
    frameon=True,
    fancybox=True
)

sns.despine(left=False, bottom=False)

plt.tight_layout()

plt.savefig(
    'experiments/rag_scores_violin_plot.jpeg',
    dpi=300,
    bbox_inches='tight'
)

plt.show()

# =========================================
# NON-RAG VIOLIN PLOT
# =========================================

norag_cols = [
    'answer_relevancy',
    'answer_correctness'
]

norag_labels = [
    'Answer\nRelevancy',
    'Answer\nCorrectness'
]

norag_long = results_norag[norag_cols].melt(
    var_name='Metric',
    value_name='Score'
)

plt.figure(figsize=(9, 7))

ax = sns.violinplot(
    data=norag_long,
    x='Metric',
    y='Score',
    palette=norag_palette,
    inner=None,
    linewidth=1.2,
    cut=0,
    saturation=0.9
)

sns.boxplot(
    data=norag_long,
    x='Metric',
    y='Score',
    width=0.05,
    showcaps=True,
    boxprops={
        'facecolor': 'white',
        'zorder': 3
    },
    whiskerprops={'linewidth': 1.5},
    medianprops={
        'color': 'black',
        'linewidth': 2
    },
    showfliers=False
)

means = results_norag[norag_cols].mean().values

for i, mean in enumerate(means):
    plt.scatter(
        i,
        mean,
        marker='D',
        s=90,
        edgecolor='black',
        linewidth=1,
        zorder=10,
        label='Mean Score' if i == 0 else ""
    )

for i, mean in enumerate(means):
    plt.text(
        i + 0.1,
        mean + 0.01,
        f"{mean:.3f}",
        ha='center',
        fontsize=10,
        fontweight='bold'
    )

ax.set_xticklabels(norag_labels)

plt.title(
    'Distribution of Non-RAG Evaluation Metrics',
    fontsize=20,
    pad=20
)

plt.xlabel('')
plt.ylabel('Score', fontsize=14)

plt.ylim(-0.02, 1.02)

plt.grid(
    axis='y',
    linestyle='--',
    alpha=0.3
)

plt.legend(
    frameon=True,
    fancybox=True
)

sns.despine(left=False, bottom=False)

plt.tight_layout()

plt.savefig(
    'experiments/norag_scores_violin_plot.jpeg',
    dpi=300,
    bbox_inches='tight'
)

plt.show()

print("Violin plots saved to experiments/")