# Does Pathology-Specific Pretraining Beat ImageNet?

**Benchmarking a pathology foundation model against ImageNet CNNs for colorectal cancer histology classification**

---

## Summary

The computational-pathology field has shifted from ImageNet-pretrained CNNs toward domain-specific *foundation models* pretrained on millions of pathology images. This benchmark asks a practical question: **how much does pathology-specific pretraining actually buy you over generic ImageNet features** for tissue classification, when both are used as frozen feature extractors with a lightweight trained head?

Using the NCT-CRC-HE colorectal cancer dataset and a held-out cohort of **different patients** (CRC-VAL-HE-7K), the benchmark compares:

- **Phikon-v2** — a pathology foundation model (ViT-L, DINOv2, pretrained on 460M public histology tiles across 30+ cancer sites)
- **ResNet50, EfficientNet-B0, MobileNetV3** — ImageNet-pretrained CNNs used as frozen feature extractors
- **ResNet50 (fine-tuned)** — with the final residual block unfrozen

All models share an identical preprocessing and evaluation pipeline, isolating the effect of the feature representation itself.

> **Result:** the pathology foundation model (Phikon-v2) reaches 0.935 macro F1 on the held-out cross-patient test set versus 0.897 for the best ImageNet CNN — a statistically significant +9.4-point accuracy gain (linear probe, 95% CI [+7.9, +10.9]). With only 10% of the training labels it already reaches 0.934, exceeding the ImageNet CNN trained on all labels.

---

## Repository Structure

```
histology-crc-benchmark/
│
├── train_colab.py       # Training + evaluation pipeline (run on Colab GPU)
├── make_figures.py      # Generate figures from results/
├── HOW_TO_RUN.md        # Step-by-step Colab instructions
├── requirements.txt
├── results/             # Metrics (produced by the run)
│   ├── model_comparison.csv
│   ├── per_class_report.csv
│   ├── confusion_matrix.npy
│   └── meta.json
└── figures/             # Generated figures
    ├── fig1_model_comparison.png
    ├── fig2_per_class_f1.png
    └── fig3_confusion.png
```

---

## Dataset

**NCT-CRC-HE** (Kather et al., 2018): 224×224 H&E patches from colorectal cancer and normal tissue, labeled into nine tissue classes — adipose (ADI), background (BACK), debris (DEB), lymphocytes (LYM), mucus (MUC), smooth muscle (MUS), normal colon mucosa (NORM), cancer-associated stroma (STR), and colorectal adenocarcinoma epithelium (TUM).

- **Training:** class-balanced subset of the 100,000-patch NCT-CRC-HE-100K set (86 patients)
- **Test:** class-balanced subset of CRC-VAL-HE-7K (50 **different** patients) — gives an honest cross-patient generalization estimate

Data is streamed from the Hugging Face mirror `1aurent/NCT-CRC-HE`.

---

## How to Run

```python
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
!pip -q install -U datasets huggingface_hub transformers torchvision scikit-learn matplotlib
%run train_colab.py
%run make_figures.py
```

The full run takes roughly 20–30 minutes on a free Colab T4 GPU.

**This will:**

- stream a class-balanced subset of the NCT-CRC-HE colorectal-cancer histology dataset,
- train four ImageNet transfer-learning configurations (ResNet50 frozen, EfficientNet-B0 frozen, MobileNetV3 frozen, and a fine-tuned ResNet50), extract features from Phikon-v2 (Owkin), a ViT-L pathology foundation model pretrained with DINOv2, and train a linear head on top,
- run a 5-seed linear probe with a paired bootstrap significance test comparing the foundation model against the best ImageNet CNN,
- run a label-efficiency sweep (10 / 25 / 50 / 100% of training labels),
- export 2D UMAP embeddings of each frozen feature space,
- evaluate on the held-out CRC-VAL-HE-7K cohort (separate patients),
- write all metrics to results/,
- write all figures to figures/.

**Notes:**
- Reproducibility: a fixed random seed (42) is used throughout.
- Crash recovery: extracted features are cached to results/feature_cache/ as each model completes. Re-running the script reloads the cache and skips re-extraction, so an interrupted run resumes rather than starting over.
- Memory: the run uses a class-balanced subset (800 train / 300 test per class) to fit in free-tier RAM, and frees each model after use. On tighter memory, lower TRAIN_PER_CLASS at the top of train_colab.py.

---

## Methodology

- **Feature extractors:** Phikon-v2 (frozen) and three ImageNet CNNs (frozen), plus a fine-tuned ResNet50. In every case only a linear classification head is trained on top of frozen features (except the fine-tuned variant).
- **Preprocessing:** resize to 224×224, ImageNet normalization, light flip augmentation during training. All models share this pipeline for a controlled comparison (see Limitations regarding Phikon-v2's native preprocessing).
- **Evaluation:** accuracy, balanced accuracy, macro F1, weighted F1, per-class F1, and a confusion matrix, all on the held-out cross-patient test set. Fixed random seed (42).

**Advanced analyses:**
- **Statistical significance:** the pathology foundation model and the best ImageNet CNN are each evaluated with a linear probe across five seeds, and compared with a paired bootstrap test (2,000 resamples) with a 95% confidence interval.
- **Label efficiency:** a linear probe is trained on 10%/25%/50%/100% of labels to test whether pathology pretraining reduces the labeled-data requirement.
- **Embedding analysis:** 2D UMAP projections of each frozen feature space show *why* representations differ in downstream performance.

---

## Limitations

- Single dataset; class-balanced subset for tractable training.
- For pipeline uniformity, all models use ImageNet normalization; Phikon-v2 ships its own preprocessing, so its absolute performance could shift under native preprocessing.
- Foundation models are used as frozen feature extractors; full fine-tuning, and comparison against larger gated models (UNI, Virchow, Prov-GigaPath), are natural extensions.
- Patch classification is an idealized version of the clinical whole-slide task.

---

## References

Chen, R. J., Ding, T., Lu, M. Y., Williamson, D. F. K., et al. (2024). Towards a general-purpose foundation model for computational pathology. *Nature Medicine*, 30(3), 850–862.

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR*, 770–778.

Kather, J. N., Halama, N., & Marx, A. (2018). *100,000 histological images of human colorectal cancer and healthy tissue* [Data set]. Zenodo. https://doi.org/10.5281/zenodo.1214456

Kather, J. N., Krisam, J., Charoentong, P., et al. (2019). Predicting survival from colorectal cancer histology slides using deep learning. *PLOS Medicine*, 16(1), e1002730.

Filiot, A., Jacob, P., Mac Kain, A., & Saillard, C. (2024). Phikon-v2: A large and public feature extractor for biomarker prediction. *arXiv* 2409.09173.

Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *ICML*, 6105–6114.

Vorontsov, E., Bozkurt, A., Casson, A., et al. (2024). A foundation model for clinical-grade computational pathology and rare cancers detection. *Nature Medicine*, 30(10), 2924–2935.

---

## License

MIT License (code). The NCT-CRC-HE dataset is CC-BY; Phikon-v2 is released by Owkin (Filiot et al., 2024).

---

## Citation

```
Marrakchi, S. (2026). Does Pathology-Specific Pretraining Beat ImageNet?
Benchmarking a Pathology Foundation Model against ImageNet CNNs for Colorectal
Cancer Histology Classification. GitHub.
```
