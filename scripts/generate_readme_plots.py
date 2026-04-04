"""
Generate the updated Quantum Advantage comparison chart including all market models (2022-2025).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUT = Path("evaluation/latest/plots")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',
    'text.color': '#e6edf3',
    'axes.labelcolor': '#e6edf3',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'grid.color': '#21262d',
    'grid.alpha': 0.8,
})

# ── MARKET COMPARISON CHART ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 8))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#161b22')

models = [
    "Classical\nBaseline\n(Our Ablation)",
    "Quantum Kernel\nLearning\n(NIH, 2024)",
    "QLSTM Indic\nLow-Resource\n(SciELO, 2024)",
    "Quantum\nTransfer\nLearning + BERT\n(2023)",
    "CQLSTM\n(MDPI, 2023)",
    "QFNN\nEnglish Twitter\n(arXiv, 2023)",
    "Hindi QCNN\nQ-VAE\n(IJISAE, 2024)*",
    "Hybrid QCNN\n(OURS)\nIIIT Ranchi",
]

accuracies = [58.2, 75.5, 74.5, 79.0, 86.0, 89.5, 97.6, 79.4]

# Color scheme: grey for others, gold for competitors above us, bright green for ours
colors = [
    '#6e7681',   # classical baseline - grey  
    '#8b949e',   # quantum kernel - grey-blue
    '#388bfd',   # QLSTM - blue
    '#388bfd',   # QTL BERT - blue
    '#d29922',   # CQLSTM - gold (beats us)
    '#d29922',   # QFNN - gold (beats us)
    '#da3633',   # Hindi Q-VAE - red (big claim, small dataset)
    '#238636',   # OUR MODEL - green
]

bars = ax.bar(models, accuracies, color=colors, width=0.55,
              edgecolor='#30363d', linewidth=1.2, zorder=3)

# Accuracy labels on bars
for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f'{acc:.1f}%',
            ha='center', va='bottom', fontsize=10.5,
            fontweight='bold', color='#e6edf3')

# Highlight OUR model bar with a glow border
bars[-1].set_edgecolor('#39d353')
bars[-1].set_linewidth(2.5)

# Dashed lines for reference
ax.axhline(y=79.4, color='#39d353', linestyle='--', lw=1.5, alpha=0.5, zorder=2)
ax.axhline(y=33.3, color='#f78166', linestyle=':', lw=1.2, alpha=0.6, zorder=2,
           label='Random Baseline (33.3%)')

ax.set_ylim(0, 110)
ax.set_ylabel('Accuracy (%)', fontsize=13, labelpad=10)
ax.set_title(
    'Hybrid QCNN vs. Market Models — Sentiment Analysis (2022–2025)\n'
    'IIIT Ranchi Research · Ashutosh Karn · Supervisor: Dr. Roshan Singh',
    fontsize=13, fontweight='bold', color='#e6edf3', pad=20
)
ax.grid(axis='y', zorder=0, alpha=0.6)
ax.spines[['top', 'right']].set_visible(False)

# Legend
legend_patches = [
    mpatches.Patch(color='#6e7681', label='Classical baseline'),
    mpatches.Patch(color='#388bfd', label='Hybrid Quantum models (2023–2024)'),
    mpatches.Patch(color='#d29922', label='High-acc models (English-only, small datasets)'),
    mpatches.Patch(color='#da3633', label='Hindi-only, domain-specific, single seed*'),
    mpatches.Patch(color='#238636', label='Ours — 4-language, n=3 seeds, N=10k'),
]
ax.legend(handles=legend_patches, loc='upper left', fontsize=9,
          facecolor='#1c2128', edgecolor='#30363d', framealpha=0.95)

# Annotation: why our model is competitive
ax.annotate(
    '* 97.64%: Hindi-only, small\ndomain-specific dataset,\nsingle seed — not reproducible',
    xy=(6, 97.6), xytext=(4.8, 103),
    fontsize=8.5, color='#8b949e',
    arrowprops=dict(arrowstyle='->', color='#8b949e', lw=1),
    ha='center',
    bbox=dict(fc='#161b22', ec='#30363d', boxstyle='round,pad=0.3')
)

ax.annotate(
    'OURS: 4 languages\n50k+ samples (planned)\nn=3 seeds, reproducible',
    xy=(7, 79.4), xytext=(5.8, 67),
    fontsize=9, color='#39d353', fontweight='bold',
    arrowprops=dict(arrowstyle='->', color='#39d353', lw=1.5),
    ha='center',
    bbox=dict(fc='#0d1117', ec='#39d353', boxstyle='round,pad=0.4', lw=1.5)
)

plt.tight_layout()
plt.savefig(OUT / 'quantum_advantage_metrics.png', dpi=150,
            bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("✅ Saved: quantum_advantage_metrics.png")
