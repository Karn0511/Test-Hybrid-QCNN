import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os

def generate_graphs():
    # Read metrics directly from the recent Fast Evaluation pipeline output
    metrics_file = Path("evaluation/latest/metrics/language_metrics.json")
    if not metrics_file.exists():
        print(f"Error: Could not find {metrics_file}")
        return

    with open(metrics_file, "r") as f:
        data = json.load(f)

    results = data.get("results", {})
    if not results:
        print("No results found in metrics.")
        return

    # Preparation for Plot 1: Individual Language Transfer Strengths
    langs = ["english", "hindi", "bhojpuri", "maithili"]
    accuracies = []
    
    for l in langs:
        key = f"STAGE_1_{l}"
        if key in results:
            mean_acc = results[key]["metrics_summary"]["accuracy"]["mean"]
            accuracies.append(mean_acc * 100) # Percentage
        else:
            accuracies.append(0)

    # Plot 1 Formatting
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Custom Gradient-like Colors for Scientific Look
    colors = ['#ff4d4d', '#ffaa00', '#00cccc', '#cc66ff']
    bars = ax.bar(langs, accuracies, color=colors, edgecolor='white', linewidth=1.5)
    
    ax.set_title("Zero-Shot Semantic Transfer Benchmark (Evaluation Mode)", fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel("Absolute Accuracy (%)", fontsize=12)
    ax.set_xlabel("Language Corpus", fontsize=12)
    ax.set_ylim(0, max(50, max(accuracies) * 1.5))
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # Add data labels
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}%", 
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    plots_dir = Path("evaluation/latest/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plots_dir / "multilingual_transfer_benchmark.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("-> Generated Multilingual Bar Chart")

    # Preparation for Plot 2: Absolute Quantum Gain
    q_key = "STAGE_5_hybrid_qcnn"
    t_key = "STAGE_5_baseline_transformer"
    
    q_acc = results[q_key]["metrics_summary"]["accuracy"]["mean"] * 100 if q_key in results else 0
    t_acc = results[t_key]["metrics_summary"]["accuracy"]["mean"] * 100 if t_key in results else 0
    
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    
    models = ["Classical Transformer", "Hybrid QCNN"]
    accs2 = [t_acc, q_acc]
    colors2 = ['#404040', '#00ffcc']
    
    bars2 = ax2.bar(models, accs2, color=colors2, edgecolor='white', linewidth=2)
    
    ax2.set_title("Architectural Comparison: Transformer vs Hybrid QCNN", fontsize=16, fontweight='bold', pad=20)
    ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.set_ylim(0, max(50, max(accs2) * 1.5))
    
    # Gain indicator calculation
    gain = q_acc - t_acc
    if gain != 0:
        gain_text = f"Quantum Gain: {'+' if gain > 0 else ''}{gain:.2f}%"
        ax2.text(0.5, max(accs2) * 1.25, gain_text, ha='center', va='center',
                 fontsize=14, fontweight='bold', color='#00ffcc' if gain > 0 else '#ff4d4d',
                 bbox=dict(facecolor='black', alpha=0.5, edgecolor='gray', pad=10))

    for bar in bars2:
        h = bar.get_height()
        ax2.annotate(f"{h:.2f}%", 
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.savefig(plots_dir / "quantum_advantage_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("-> Generated Quantum Advantage Comparison Chart")
    
    # Write a quick markdown summary in the evaluation folder
    with open("evaluation/latest/reports/final_evaluation_summary.md", "w", encoding='utf-8') as f:
        f.write("# Hybrid QCNN Final Evaluation Results\n\n")
        f.write("Generated natively from the 10-Stage Evaluation Mode matrix.\n\n")
        f.write("## 1. Zero-Shot Multilingual Transfer\n")
        f.write("![Multilingual Transfer Benchmark](../plots/multilingual_transfer_benchmark.png)\n\n")
        f.write("## 2. Quantum Advantage Analysis\n")

        f.write("![Quantum Advantage Comparison](../plots/quantum_advantage_metrics.png)\n\n")
        f.write(f"> **Notice**: Values reflect ultra-fast randomly-initialized evaluation configurations, with Classical at **{t_acc:.2f}%** and Quantum at **{q_acc:.2f}%**.\n")

    print("-> Created evaluation/latest/reports/final_evaluation_summary.md")

if __name__ == "__main__":
    generate_graphs()
