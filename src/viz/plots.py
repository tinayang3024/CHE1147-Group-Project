import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import FIG_DIR


def plot_corr_heatmap(df, cols, title, fname, sample_max=5000):
    if not cols:
        return
    sub = df[cols]
    if len(sub) > sample_max:
        sub = sub.sample(sample_max, random_state=42)
    corr = sub.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", vmin=-1, vmax=1, xticklabels=False, yticklabels=False)
    plt.title(title)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, fname)
    plt.savefig(out, dpi=200)
    plt.close()
    print("[saved]", out)


def plot_pca_explained_variance(pca_model, fname="fp_pca_explained_variance.png"):
    if pca_model is None:
        return
    evr = pca_model.explained_variance_ratio_
    cum = np.cumsum(evr)
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(evr) + 1), evr, marker="o", label="Per-component variance")
    plt.plot(range(1, len(evr) + 1), cum, marker=".", label="Cumulative variance")
    plt.axhline(y=0.95, color="red", linestyle="--", linewidth=1, label="95%")
    plt.xlabel("PCA component")
    plt.ylabel("Explained variance (ratio)")
    plt.title("FP PCA: explained variance")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(FIG_DIR, fname)
    plt.savefig(out, dpi=200)
    plt.close()
    print("[saved]", out)
