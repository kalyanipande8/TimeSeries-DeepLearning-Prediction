"""
End-to-end pipeline: generate 27 datasets -> train/eval LSTM + Transformer
encoders on all of them -> run suspicious-result detector -> write figures.

Usage: python3 run_all.py
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run_step(name, module):
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    result = subprocess.run([sys.executable, os.path.join(HERE, module)],
                             cwd=HERE)
    if result.returncode != 0:
        sys.exit(result.returncode)


def make_figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    results = json.load(open(os.path.join(HERE, "results", "results.json")))
    ids = [r["dataset_id"] for r in results]
    lstm_r2 = [r["lstm"]["test_r2"] for r in results]
    tf_r2 = [r["transformer"]["test_r2"] for r in results]
    suspicious = [r["lstm"]["detection"]["suspicious"] or r["transformer"]["detection"]["suspicious"]
                  for r in results]

    fig_dir = os.path.join(HERE, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # 1. per-dataset test R^2 for both architectures
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.4
    x = np.arange(len(ids))
    colors_l = ["#d62728" if s else "#1f77b4" for s in suspicious]
    colors_t = ["#d62728" if s else "#ff7f0e" for s in suspicious]
    ax.bar(x - width / 2, lstm_r2, width, label="LSTM (Reservoir) encoder", color=colors_l)
    ax.bar(x + width / 2, tf_r2, width, label="Transformer (attention) encoder", color=colors_t)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(ids, fontsize=7)
    ax.set_xlabel("Dataset ID")
    ax.set_ylabel("Test R$^2$")
    ax.set_title("Per-dataset test R$^2$ across 27 synthetic time-series datasets\n(red = flagged suspicious by detector)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "per_dataset_r2.png"), dpi=130)
    plt.close(fig)

    # 2. train vs val MSE scatter, highlighting flagged overfit models
    fig, ax = plt.subplots(figsize=(6, 6))
    for arch, marker in [("lstm", "o"), ("transformer", "^")]:
        train_mses = [r[arch]["train_mse"] for r in results]
        val_mses = [r[arch]["val_mse"] for r in results]
        flags = [r[arch]["detection"]["suspicious"] for r in results]
        colors = ["#d62728" if f else "#2ca02c" for f in flags]
        ax.scatter(train_mses, val_mses, c=colors, marker=marker, s=60,
                   label=f"{arch}", edgecolor="black", linewidth=0.4)
    lims = [0, max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, "k--", linewidth=0.8, label="train_mse = val_mse")
    ax.set_xlabel("Train MSE")
    ax.set_ylabel("Val MSE")
    ax.set_title("Train vs. validation MSE\n(red = flagged suspicious/overfit by detector)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "train_vs_val_mse.png"), dpi=130)
    plt.close(fig)

    print(f"Figures written to {fig_dir}")


def main():
    run_step("STEP 1/3: generating 27 synthetic multivariate time-series datasets",
              "generate_datasets.py")
    run_step("STEP 2/3: training + evaluating LSTM and Transformer encoders on all datasets",
              "train_models.py")
    print("\n" + "=" * 60)
    print("STEP 3/3: building figures")
    print("=" * 60)
    make_figures()

    summary = json.load(open(os.path.join(HERE, "results", "summary.json")))
    print("\nFinal summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
