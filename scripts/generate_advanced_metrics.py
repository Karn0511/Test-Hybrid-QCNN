import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import numpy as np
import os
from pathlib import Path
from sklearn.metrics import classification_report, f1_score

# ==========================================
# PUBLICATION QUALITY CONFIGURATION
# ==========================================
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

# CURATED PALETTE
MODEL_COLORS = {
    "classical": "#457b9d",     # Deep Blue
    "transformer": "#e63946",   # Vivid Red
    "hybrid": "#2a9d8f"         # Teal/Sea Green
}

# ==========================================
# DATA ENRICHMENT ENGINE
# ==========================================
def enrich_experiment_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    enriched_results = []
    
    # Process each experiment in the new dictionary schema
    for m_id, content in data.items():
        metrics = content['metrics_summary']
        config = content['config']
        
        # 1. TAGGING
        use_qcnn = config.get('use_qcnn', False)
        baseline = config.get('baseline')
        
        if baseline == "transformer":
            tag = "transformer"
        elif not use_qcnn:
            tag = "classical"
        else:
            tag = "hybrid"

        # 2. DERIVED METRICS (Using Mean values)
        acc_mean = metrics['accuracy']['mean']
        f1_mean = metrics['f1-score']['mean']
        roc_mean = metrics['roc_auc']['mean']
        
        perf_score = (0.4 * f1_mean + 0.3 * acc_mean + 0.3 * roc_mean)
        
        # Complexity Proxy
        if tag == "classical":
            complexity = 10
        elif tag == "transformer":
            complexity = 100
        else:
            q = config.get('n_qubits', 4)
            l = config.get('layers', 2)
            complexity = (2**q) * l
            
        efficiency = perf_score / np.log(complexity + 1)
        stability = metrics['f1-score']['std'] # Use std dev as stability metric
        
        res = {
            "id": m_id,
            "type": tag,
            "accuracy": acc_mean,
            "f1": f1_mean,
            "roc_auc": roc_mean,
            "performance_score": perf_score,
            "complexity": complexity,
            "efficiency_score": efficiency,
            "stability_score": stability,
            "layers": config.get('layers', 0),
            "n_qubits": config.get('n_qubits', 0)
        }
        enriched_results.append(res)
        
    return pd.DataFrame(enriched_results)

# ==========================================
# VISUALIZATION SUITE
# ==========================================
def plot_ablation_summary(df, output_path):
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='id', y='performance_score', hue='type', palette=MODEL_COLORS, dodge=False)
    plt.title('Scientific Performance Index Across Ablation Matrix', fontweight='bold')
    plt.ylabel('Integrated Performance Score')
    plt.xlabel('Experiment Configuration ID')
    plt.ylim(0.7, 1.0)
    plt.legend(title='Model Class', loc='upper left')
    plt.savefig(output_path / "ablation_summary.png")
    plt.close()

def plot_quantum_vs_classical(df, output_path):
    avg_df = df.groupby('type')[['accuracy', 'f1', 'roc_auc']].mean().reset_index()
    melted = avg_df.melt(id_vars='type', var_name='Metric', value_name='Score')
    
    plt.figure(figsize=(9, 6))
    sns.barplot(data=melted, x='Metric', y='Score', hue='type', palette=MODEL_COLORS)
    plt.title('Aggregate Performance Delta: Quantum-Classical Gap', fontweight='bold')
    plt.ylim(0.7, 1.0)
    plt.grid(axis='y', alpha=0.2)
    plt.savefig(output_path / "quantum_vs_classical.png")
    plt.close()

def plot_complexity_vs_performance(df, output_path):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='complexity', y='performance_score', hue='type', style='type', 
                    palette=MODEL_COLORS, s=200, alpha=0.8)
    
    # Add trend line for hybrid models
    h_df = df[df['type'] == 'hybrid'].sort_values('complexity')
    if not h_df.empty:
        plt.plot(h_df['complexity'], h_df['performance_score'], '--', color=MODEL_COLORS['hybrid'], alpha=0.5)

    plt.title('Complexity Laws: Feature Space Scaling vs Performance', fontweight='bold')
    plt.xlabel(r'Computational Complexity Proxy ($\log Scale$ suggested)')
    plt.ylabel('Performance Score')
    plt.savefig(output_path / "complexity_vs_performance.png")
    plt.close()

def plot_model_efficiency(df, output_path):
    plt.figure(figsize=(10, 5))
    df_sorted = df.sort_values('efficiency_score', ascending=False)
    sns.barplot(data=df_sorted, x='id', y='efficiency_score', hue='type', palette=MODEL_COLORS, dodge=False)
    plt.title('Scientific Efficiency Index (Metric Gain per Parameter Log)', fontweight='bold')
    plt.ylabel('Efficiency Score')
    plt.savefig(output_path / "model_efficiency.png")
    plt.close()

def plot_stability_analysis(df, output_path):
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x='type', y='stability_score', palette=MODEL_COLORS)
    sns.stripplot(data=df, x='type', y='stability_score', color='.3', size=4, alpha=0.5)
    plt.title('Multi-Seed Stability: F1-Score Standard Deviation', fontweight='bold')
    plt.ylabel('Stability (Std Dev of F1)')
    plt.savefig(output_path / "stability_analysis.png")
    plt.close()

def plot_roc_auc_distribution(df, output_path):
    plt.figure(figsize=(10, 6))
    for t in df['type'].unique():
        subset = df[df['type'] == t]
        sns.kdeplot(subset['roc_auc'], label=t, fill=True, color=MODEL_COLORS[t], alpha=0.4)
    
    plt.title('Model Confidence Distribution (ROC-AUC Density)', fontweight='bold')
    plt.xlabel('ROC-AUC Score')
    plt.legend()
    plt.savefig(output_path / "roc_auc_distribution.png")
    plt.close()

def plot_hybrid_gain(df, output_path):
    # Compare Avg Classical vs Avg Hybrid ROC-AUC
    c_avg = df[df['type'] == 'classical']['roc_auc'].mean()
    h_vals = df[df['type'] == 'hybrid']['roc_auc']
    gains = h_vals - c_avg
    
    plt.figure(figsize=(8, 5))
    sns.histplot(gains, color=MODEL_COLORS['hybrid'], kde=True, bins=6)
    plt.axvline(0, color='red', linestyle='--')
    plt.title('Net Quantum Gain Distribution (over Classical Baseline Mean)', fontweight='bold')
    plt.xlabel('ROC-AUC Delta')
    plt.savefig(output_path / "hybrid_gain_plot.png")
    plt.close()

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    metrics_path = Path("evaluation/metrics")
    metrics_path.mkdir(parents=True, exist_ok=True)
    json_path = Path("evaluation/final_metrics.json")
    
    if not json_path.exists():
        print(f"❌ Error: {json_path} not found. Run experiments/runner.py first.")
        # Fallback to old path for backward compatibility if needed, 
        # but the USER requested it "over old".
        json_path = Path("evaluation/ablation_results.json")
        if not json_path.exists():
            exit(1)
        
    print(f"🔬 Loading experiment data from {json_path}...")
    df = enrich_experiment_data(json_path)
    
    # SAVE SUMMARY
    summary_json = df.to_dict(orient='records')
    with open(metrics_path / "metrics_summary.json", 'w') as f:
        json.dump(summary_json, f, indent=2)
    print(f"✅ Summary saved to {metrics_path}/metrics_summary.json")
    
    # GENERATE PLOTS
    print("🎨 Generating publication-grade visualizations...")
    plot_ablation_summary(df, metrics_path)
    plot_quantum_vs_classical(df, metrics_path)
    plot_complexity_vs_performance(df, metrics_path)
    plot_model_efficiency(df, metrics_path)
    plot_stability_analysis(df, metrics_path)
    plot_roc_auc_distribution(df, metrics_path)
    plot_hybrid_gain(df, metrics_path)
    
    print(f"\n🚀 Analysis suite complete. 7 high-fidelity metrics generated in {metrics_path}/")
