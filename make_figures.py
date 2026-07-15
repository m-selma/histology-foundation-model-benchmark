"""
make_figures.py
---------------
Run AFTER train_colab.py has produced results/.

Produces:
    fig1_model_comparison.png   - accuracy / balanced acc / macro F1 by model
    fig2_per_class_f1.png       - per-class F1 heatmap
    fig3_confusion.png          - confusion matrix of best model
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, json
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "font.family": "sans-serif",
})
ACCENT="#2563eb"; ACCENT2="#dc2626"; GREY="#94a3b8"

df = pd.read_csv("results/model_comparison.csv")
pc = pd.read_csv("results/per_class_report.csv", index_col=0)
meta = json.load(open("results/meta.json"))
classes = meta["classes"]

# ---- Figure 1: model comparison ----
fig, ax = plt.subplots(figsize=(9,4.8))
d = df.sort_values("Macro F1")
y = np.arange(len(d))
ax.barh(y+0.0, d["Macro F1"], height=0.27, color=ACCENT, label="Macro F1")
ax.barh(y-0.28, d["Balanced Accuracy"], height=0.27, color=ACCENT2, label="Balanced Accuracy")
ax.barh(y+0.28, d["Accuracy"], height=0.27, color=GREY, label="Accuracy")
ax.set_yticks(y); ax.set_yticklabels(d["Model"])
ax.set_xlabel("Score"); ax.set_xlim(0,1.0)
ax.set_title("Colorectal Histology Tissue Classification\n(held-out CRC-VAL-HE-7K test set, different patients)", fontsize=11)
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False); ax.axvline(0, color="k", lw=0.5)
plt.tight_layout(); plt.savefig("figures/fig1_model_comparison.png", bbox_inches="tight"); plt.close()

# ---- Figure 2: per-class F1 heatmap ----
pc_sorted = pc.loc[df.sort_values("Macro F1", ascending=False)["Model"]]
mat = pc_sorted[classes].values
fig, ax = plt.subplots(figsize=(10,4.5))
im = ax.imshow(mat, cmap="RdYlBu", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, fontsize=9)
ax.set_yticks(range(len(pc_sorted))); ax.set_yticklabels(pc_sorted.index, fontsize=9)
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        ax.text(j,i,f"{mat[i,j]:.2f}",ha="center",va="center",fontsize=7,
                color="white" if mat[i,j]<0.5 else "black")
ax.set_title("Per-Class F1 by Model and Tissue Type", fontsize=11)
plt.colorbar(im, label="F1 score", fraction=0.025)
plt.tight_layout(); plt.savefig("figures/fig2_per_class_f1.png", bbox_inches="tight"); plt.close()

# ---- Figure 3: confusion matrix ----
conf = np.load("results/confusion_matrix.npy")
conf_norm = conf / conf.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(7.5,6.5))
im = ax.imshow(conf_norm, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, fontsize=9)
ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes, fontsize=9)
for i in range(len(classes)):
    for j in range(len(classes)):
        if conf_norm[i,j] > 0.01:
            ax.text(j,i,f"{conf_norm[i,j]:.2f}",ha="center",va="center",fontsize=7,
                    color="white" if conf_norm[i,j]>0.5 else "black")
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title(f"Confusion Matrix — {meta['best_model']}\n(row-normalized, held-out test set)", fontsize=11)
plt.colorbar(im, label="Fraction", fraction=0.045)
plt.tight_layout(); plt.savefig("figures/fig3_confusion.png", bbox_inches="tight"); plt.close()

print("Figures saved to figures/")

# ---- Figure 4: label-efficiency curve (if available) ----
import os
if os.path.exists("results/label_efficiency.csv"):
    le = pd.read_csv("results/label_efficiency.csv")
    fig, ax = plt.subplots(figsize=(8,5.5))
    def _color(m):
        if "Phikon" in m: return ACCENT2
        if "ResNet" in m: return ACCENT
        return GREY
    for name, g in le.groupby("Model"):
        g = g.sort_values("label_fraction")
        ax.plot(g["label_fraction"]*100, g["Macro F1"], "o-",
                color=_color(name), linewidth=2, markersize=8, label=name)
    ax.set_xlabel("Training labels used (%)"); ax.set_ylabel("Macro F1 (held-out test)")
    ax.set_title("Label Efficiency: Does Pathology Pretraining Help More\nWhen Labels Are Scarce?", fontsize=11)
    ax.legend(frameon=False, loc="lower right")
    ax.set_xticks([10,25,50,100])
    plt.tight_layout(); plt.savefig("figures/fig4_label_efficiency.png", bbox_inches="tight"); plt.close()
    print("Saved fig4_label_efficiency.png")

# ---- Figure 5: side-by-side UMAP of frozen embeddings ----
import glob
umap_files = sorted(glob.glob("results/umap_*.npz"))
if umap_files:
    n = len(umap_files)
    fig, axes = plt.subplots(1, n, figsize=(7*n, 6))
    if n == 1: axes = [axes]
    classes = json.load(open("results/meta.json"))["classes"]
    palette = plt.cm.tab10(np.linspace(0,1,len(classes)))
    for ax, f in zip(axes, umap_files):
        d = np.load(f); emb = d["emb"]; y = d["y"]
        for i, c in enumerate(classes):
            m = y == i
            if m.sum() > 0:
                ax.scatter(emb[m,0], emb[m,1], s=6, color=palette[i], alpha=0.7,
                           linewidths=0, label=c)
        model_name = os.path.basename(f).replace("umap_","").replace(".npz","")
        ax.set_title(f"{model_name} embedding", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02,0.5), frameon=False, fontsize=7, markerscale=2)
    fig.suptitle("Frozen Embedding Spaces: Cleaner Class Separation Explains Better Classification",
                 fontsize=12, y=1.02)
    plt.tight_layout(); plt.savefig("figures/fig5_embedding_umap.png", bbox_inches="tight"); plt.close()
    print("Saved fig5_embedding_umap.png")
