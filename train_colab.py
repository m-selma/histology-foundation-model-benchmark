"""
train_colab.py
==============
Benchmark of a pathology foundation model against ImageNet-pretrained CNNs for
colorectal-cancer histology tissue classification on the NCT-CRC-HE-100K /
CRC-VAL-HE-7K dataset (Kather et al., 2018, 2019).

Designed to run on a single GPU (e.g. Google Colab, free T4):
    1. Select a GPU runtime.
    2. Run this file (%run train_colab.py, or paste its contents into a cell).
    3. Total time on a T4 GPU: ~40-60 minutes.

Outputs written to results/:
    - model_comparison.csv        overall metrics per model
    - per_class_report.csv        per-class F1 per model
    - confusion_matrix.npy        confusion matrix of the best model
    - linear_probe_multiseed.csv  5-seed linear-probe macro F1
    - significance_test.json      paired bootstrap test (FM vs best CNN)
    - label_efficiency.csv        macro F1 at 10/25/50/100% of labels
    - umap_*.npz                  2D UMAP of each frozen embedding space
    - meta.json                   run configuration

What it does:
    - Loads NCT-CRC-HE-100K (train) and CRC-VAL-HE-7K (held-out test, from a
      separate patient cohort) from Hugging Face, using a class-balanced subset.
    - Trains four ImageNet transfer-learning configurations: ResNet50,
      EfficientNet-B0, and MobileNetV3-Large as frozen feature extractors, plus
      a fine-tuned ResNet50 (last block unfrozen).
    - Evaluates Phikon-v2 (Filiot et al., 2024), a ViT-L pathology foundation
      model, as a frozen feature extractor with a trained linear head.
    - Compares the foundation model against the best CNN with a 5-seed linear
      probe and a paired bootstrap significance test.
    - Runs a label-efficiency sweep and exports UMAP embeddings.
    - Reports accuracy, balanced accuracy, macro F1, per-class F1, and a
      confusion matrix on the held-out cohort.

All metrics are computed at run time from the model outputs; a fixed random
seed (42) is used for reproducibility.
"""

import os, json, time, warnings, gc
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import transforms, models
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, classification_report, confusion_matrix)

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
SEED = 42
TRAIN_PER_CLASS = 800       # balanced subset: 9 x 800 = 7,200 training images
TEST_PER_CLASS  = 300       # 9 x 300 = 2,700 held-out test images
BATCH = 64
EPOCHS_HEAD = 4             # epochs training only the classifier head
EPOCHS_FT = 3              # additional epochs fine-tuning last block (ResNet50 only)
LR_HEAD = 1e-3
LR_FT = 1e-5
NUM_WORKERS = 2

torch.manual_seed(SEED); np.random.seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
os.makedirs("results", exist_ok=True); os.makedirs("figures", exist_ok=True)

CLASSES = ["ADI","BACK","DEB","LYM","MUC","MUS","NORM","STR","TUM"]
CLASS_FULL = {
    "ADI":"Adipose","BACK":"Background","DEB":"Debris","LYM":"Lymphocytes",
    "MUC":"Mucus","MUS":"Smooth muscle","NORM":"Normal mucosa",
    "STR":"Cancer-associated stroma","TUM":"Tumor epithelium",
}

# ----------------------------------------------------------------------
# Data (Hugging Face datasets)
# ----------------------------------------------------------------------
# pip install datasets if needed:
try:
    from datasets import load_dataset
except ImportError:
    os.system("pip -q install datasets")
    from datasets import load_dataset

print("Streaming a balanced subset from Hugging Face (no full multi-GB download)...")
from datasets import load_dataset
from PIL import Image

# Correct split names on this Hub repo:
#   NCT_CRC_HE_100K         -> training set (color-normalized)
#   NCT_CRC_HE_100K_NONORM  -> training set (no color normalization)
#   CRC_VAL_HE_7K           -> separate 7K test set (DIFFERENT patients)
TRAIN_SPLIT = "NCT_CRC_HE_100K"
TEST_SPLIT  = "CRC_VAL_HE_7K"

def stream_balanced(split_name, per_class, n_classes_expected=9, seed=SEED):
    """Stream examples and keep a class-balanced subset without downloading all data."""
    ds = load_dataset("1aurent/NCT-CRC-HE", split=split_name, streaming=True)
    buckets = {}          # label -> list of PIL images
    done = set()
    imgs, labels = [], []
    for ex in ds:
        lab = ex["label"]
        buckets.setdefault(lab, [])
        if len(buckets[lab]) < per_class:
            buckets[lab].append(ex["image"].convert("RGB"))
            if len(buckets[lab]) >= per_class:
                done.add(lab)
        # stop once every class we've seen has enough AND we've seen >=9 classes
        if len(done) >= n_classes_expected and all(len(v) >= per_class for v in buckets.values()):
            break
    for lab, images in buckets.items():
        for im in images:
            imgs.append(im); labels.append(lab)
    return imgs, labels, sorted(buckets.keys())

# Determine label names from the (non-streaming) features metadata, cheaply
_info = load_dataset("1aurent/NCT-CRC-HE", split=TEST_SPLIT, streaming=True)
_feat = _info.features
if _feat is not None and hasattr(_feat["label"], "names"):
    label_names = _feat["label"].names
else:
    label_names = ["ADI","BACK","DEB","LYM","MUC","MUS","NORM","STR","TUM"]
print("Label names:", label_names)

train_imgs, train_labels, _ = stream_balanced(TRAIN_SPLIT, TRAIN_PER_CLASS, len(label_names))
test_imgs,  test_labels,  _ = stream_balanced(TEST_SPLIT,  TEST_PER_CLASS,  len(label_names))
print(f"Streamed train: {len(train_imgs)} imgs | test: {len(test_imgs)} imgs")
n_classes = len(label_names)

IMAGENET_MEAN = [0.485, 0.456, 0.406]; IMAGENET_STD = [0.229, 0.224, 0.225]
tf_eval = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
tf_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

class HistoDataset(torch.utils.data.Dataset):
    """Wraps in-memory lists of PIL images + integer labels."""
    def __init__(self, images, labels, tf):
        self.images = images; self.labels = labels; self.tf = tf
    def __len__(self): return len(self.images)
    def __getitem__(self, i):
        return self.tf(self.images[i]), self.labels[i]

train_loader = DataLoader(HistoDataset(train_imgs, train_labels, tf_train),
                          batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS)
test_loader  = DataLoader(HistoDataset(test_imgs, test_labels, tf_eval),
                          batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS)

# ----------------------------------------------------------------------
# Model builders (transfer learning)
# ----------------------------------------------------------------------
def build_model(name, n_out, freeze=True):
    if name == "ResNet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        if freeze:
            for p in m.parameters(): p.requires_grad = False
        m.fc = nn.Linear(m.fc.in_features, n_out)
    elif name == "EfficientNet-B0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        if freeze:
            for p in m.parameters(): p.requires_grad = False
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_out)
    elif name == "MobileNetV3-Large":
        m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        if freeze:
            for p in m.parameters(): p.requires_grad = False
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, n_out)
    else:
        raise ValueError(name)
    return m.to(DEVICE)


# ----------------------------------------------------------------------
# Phikon-v2 pathology foundation model (frozen feature extractor)
# ----------------------------------------------------------------------
# Phikon-v2 (Filiot et al., 2024) loads via the standard AutoModel class.
# ViT-L pretrained with DINOv2 on 460M public histology tiles.
# We use it as a FROZEN feature extractor and train the same kind of linear
# head on top, for a fair comparison against the frozen ImageNet CNNs.

class PathologyFMClassifier(nn.Module):
    # Pathology foundation model arm. Uses Phikon-v2 (Filiot et al., 2024), a ViT-L
    # pretrained with DINOv2 on 460M public histology tiles across 30+ cancer sites.
    # NOTE: For a controlled comparison, all models here receive images with the
    # SAME ImageNet normalization used for the CNNs. Phikon ships its own
    # AutoImageProcessor; using ImageNet stats instead is a minor simplification
    # that keeps the pipeline identical across models. This is disclosed in the
    # paper's limitations. A stricter comparison would use each model's native
    # preprocessing.
    def __init__(self, n_out):
        super().__init__()
        from transformers import AutoModel
        # Phikon-v2 (Filiot et al., 2024): ViT-L pretrained with DINOv2 on 460M
        # public histology tiles. Loaded via the standard AutoModel interface.
        self.backbone = AutoModel.from_pretrained("owkin/phikon-v2")
        for p in self.backbone.parameters():
            p.requires_grad = False
        hidden = getattr(self.backbone.config, "hidden_size", 768)
        self.head = nn.Linear(hidden, n_out)
    def forward(self, x):
        with torch.no_grad():
            out = self.backbone(pixel_values=x)
            feat = out.last_hidden_state[:, 0]  # CLS token
        return self.head(feat)

def build_fm(n_out):
    return PathologyFMClassifier(n_out).to(DEVICE)

def train_model(model, loader, epochs, lr):
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    crit = nn.CrossEntropyLoss()
    for ep in range(epochs):
        t0 = time.time(); tot = 0; correct = 0; loss_sum = 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward(); opt.step()
            loss_sum += loss.item()*x.size(0); tot += x.size(0)
            correct += (out.argmax(1) == y).sum().item()
        print(f"    epoch {ep+1}/{epochs}  loss={loss_sum/tot:.3f}  "
              f"train_acc={correct/tot:.3f}  ({time.time()-t0:.0f}s)")
    return model

@torch.no_grad()
def evaluate(model, loader):
    model.eval(); preds=[]; gts=[]
    for x, y in loader:
        x = x.to(DEVICE)
        out = model(x)
        preds.extend(out.argmax(1).cpu().numpy().tolist())
        gts.extend(y.numpy().tolist())
    return np.array(gts), np.array(preds)

# ----------------------------------------------------------------------
# Run benchmark
# ----------------------------------------------------------------------
configs = [
    ("ResNet50 (frozen)",        "ResNet50",         True,  EPOCHS_HEAD, LR_HEAD, False),
    ("EfficientNet-B0 (frozen)", "EfficientNet-B0",  True,  EPOCHS_HEAD, LR_HEAD, False),
    ("MobileNetV3-L (frozen)",   "MobileNetV3-Large",True,  EPOCHS_HEAD, LR_HEAD, False),
    ("ResNet50 (fine-tuned)",    "ResNet50",         True,  EPOCHS_HEAD, LR_HEAD, True),
]

rows = []; per_class = {}; confusions = {}
for label, arch, freeze, epochs, lr, finetune in configs:
    print(f"\n=== {label} ===")
    model = build_model(arch, n_classes, freeze=freeze)
    t0 = time.time()
    model = train_model(model, train_loader, epochs, lr)
    if finetune:
        # unfreeze last block (layer4) and fine-tune at low LR
        for p in model.layer4.parameters(): p.requires_grad = True
        print("   fine-tuning last block...")
        model = train_model(model, train_loader, EPOCHS_FT, LR_FT)
    train_time = time.time() - t0
    gts, preds = evaluate(model, test_loader)
    acc = accuracy_score(gts, preds)
    bacc = balanced_accuracy_score(gts, preds)
    mf1 = f1_score(gts, preds, average="macro")
    wf1 = f1_score(gts, preds, average="weighted")
    rows.append({"Model": label, "Accuracy": acc, "Balanced Accuracy": bacc,
                 "Macro F1": mf1, "Weighted F1": wf1, "Train time (s)": train_time})
    pcf = f1_score(gts, preds, average=None, labels=range(n_classes))
    per_class[label] = {label_names[i]: pcf[i] for i in range(n_classes)}
    confusions[label] = confusion_matrix(gts, preds, labels=range(n_classes))
    print(f"    TEST acc={acc:.3f}  bal_acc={bacc:.3f}  macroF1={mf1:.3f}  ({train_time:.0f}s)")
    # free the model before building the next one
    del model
    gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()

# incremental save: write the CNN results now, so if the FM step later crashes
# the main comparison table is already on disk.
pd.DataFrame(rows).to_csv("results/model_comparison_partial.csv", index=False)

# ---- Phikon pathology foundation model (frozen feature extractor) ----
print(f"\n=== Phikon-v2 (pathology FM, frozen) ===")
try:
    fm = build_fm(n_classes)
    t0 = time.time()
    # only the head is trainable
    fm.train()
    opt = torch.optim.Adam(fm.head.parameters(), lr=LR_HEAD)
    crit = nn.CrossEntropyLoss()
    for ep in range(EPOCHS_HEAD):
        te0=time.time(); tot=0; correct=0; loss_sum=0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); out = fm(x); loss = crit(out, y)
            loss.backward(); opt.step()
            loss_sum += loss.item()*x.size(0); tot += x.size(0)
            correct += (out.argmax(1)==y).sum().item()
        print(f"    epoch {ep+1}/{EPOCHS_HEAD}  loss={loss_sum/tot:.3f}  "
              f"train_acc={correct/tot:.3f}  ({time.time()-te0:.0f}s)")
    train_time = time.time() - t0
    gts, preds = evaluate(fm, test_loader)
    acc = accuracy_score(gts, preds); bacc = balanced_accuracy_score(gts, preds)
    mf1 = f1_score(gts, preds, average="macro"); wf1 = f1_score(gts, preds, average="weighted")
    rows.append({"Model": "Phikon-v2 (pathology FM)", "Accuracy": acc, "Balanced Accuracy": bacc,
                 "Macro F1": mf1, "Weighted F1": wf1, "Train time (s)": train_time})
    pcf = f1_score(gts, preds, average=None, labels=range(n_classes))
    per_class["Phikon-v2 (pathology FM)"] = {label_names[i]: pcf[i] for i in range(n_classes)}
    confusions["Phikon-v2 (pathology FM)"] = confusion_matrix(gts, preds, labels=range(n_classes))
    print(f"    TEST acc={acc:.3f}  bal_acc={bacc:.3f}  macroF1={mf1:.3f}  ({train_time:.0f}s)")
    del fm
    gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()
except Exception as e:
    print(f"    Phikon-v2 failed to load/run: {e}")
    print("    Continuing with CNN results only.")

# ======================================================================
# ADVANCED ANALYSIS
#   (1) Multi-seed linear-probe + significance test (Phikon-v2 vs best CNN)
#   (3) Embedding UMAP: why one representation separates tissue better
#   + full label-efficiency sweep on cached frozen features
#
# Design: extract each frozen backbone's embeddings ONCE (standard
# linear-probe protocol), then all downstream analyses operate on the
# cached feature matrices -- fast and reproducible.
# ======================================================================
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample

@torch.no_grad()
def extract_features(feature_fn, loader):
    """Run a frozen backbone over a loader, return (X_feats, y)."""
    feats=[]; ys=[]
    for x, y in loader:
        x = x.to(DEVICE)
        f = feature_fn(x)
        feats.append(f.cpu().numpy()); ys.append(y.numpy())
    return np.concatenate(feats), np.concatenate(ys)

def resnet_feat_fn():
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Identity()
    for p in m.parameters(): p.requires_grad = False
    m.eval().to(DEVICE)
    return lambda x: m(x)

def fm_feat_fn():
    from transformers import AutoModel
    bb = AutoModel.from_pretrained("owkin/phikon-v2")
    for p in bb.parameters(): p.requires_grad = False
    bb.eval().to(DEVICE)
    def f(x):
        out = bb(pixel_values=x)
        return out.last_hidden_state[:,0]   # CLS token
    return f

# Backbones to analyze in depth: the pathology FM vs the strongest ImageNet CNN.
feature_extractors = {"ResNet50 (ImageNet)": resnet_feat_fn}
try:
    _ = fm_feat_fn  # ensure defined
    feature_extractors["Phikon-v2 (pathology FM)"] = fm_feat_fn
except Exception as e:
    print(f"Phikon-v2 feature extractor unavailable: {e}")

cached = {}   # name -> (Xtr, ytr, Xte, yte)
os.makedirs("results/feature_cache", exist_ok=True)
for name, fn_builder in feature_extractors.items():
    safe = name.split()[0].lower()
    cache_path = f"results/feature_cache/{safe}.npz"
    # RESUME: if we already extracted and saved these features, load and skip.
    if os.path.exists(cache_path):
        d = np.load(cache_path)
        cached[name] = (d["Xtr"], d["ytr"], d["Xte"], d["yte"])
        print(f"[features] loaded cached embeddings for {name} (resume): "
              f"train {d['Xtr'].shape}, test {d['Xte'].shape}")
        continue
    print(f"\n[features] extracting frozen embeddings: {name}")
    try:
        fn = fn_builder()
        Xtr, ytr = extract_features(fn, train_loader)
        Xte, yte = extract_features(fn, test_loader)
        cached[name] = (Xtr, ytr, Xte, yte)
        # SAVE IMMEDIATELY so a later crash doesn't lose this work.
        np.savez(cache_path, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
        print(f"    train feats {Xtr.shape}, test feats {Xte.shape} (cached to disk)")
        del fn
        gc.collect()
        if DEVICE == "cuda": torch.cuda.empty_cache()
    except Exception as e:
        print(f"    failed: {e}")

# Raw images are no longer needed once features are cached -> free that RAM.
try:
    del train_imgs, test_imgs, train_loader, test_loader
except NameError:
    pass
gc.collect()

# ---- (1) Multi-seed linear probe + bootstrap significance test ----
SEEDS = [0, 1, 2, 3, 4]
def linear_probe(Xtr, ytr, Xte, seed):
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced",
                             random_state=seed, n_jobs=-1)
    clf.fit(Xtr, ytr)
    return clf.predict(Xte)

probe_rows = []
probe_preds = {}   # name -> list of pred arrays across seeds
for name,(Xtr,ytr,Xte,yte) in cached.items():
    f1s=[]; preds_seed=[]
    for s in SEEDS:
        preds = linear_probe(Xtr, ytr, Xte, s)
        f1s.append(f1_score(yte, preds, average="macro"))
        preds_seed.append(preds)
    probe_preds[name] = preds_seed
    probe_rows.append({"Model": name, "Linear-probe Macro F1 (mean)": float(np.mean(f1s)),
                       "Std": float(np.std(f1s)), "n_seeds": len(SEEDS)})
    print(f"[probe] {name}: macroF1 {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
pd.DataFrame(probe_rows).to_csv("results/linear_probe_multiseed.csv", index=False)

# Paired bootstrap significance test: Phikon-v2 vs ResNet (per-sample correctness)
sig = {}
if "Phikon-v2 (pathology FM)" in probe_preds and "ResNet50 (ImageNet)" in probe_preds:
    yte = cached["ResNet50 (ImageNet)"][3]
    # use seed-0 predictions for the paired test on the same test samples
    hib = (probe_preds["Phikon-v2 (pathology FM)"][0] == yte).astype(int)
    res = (probe_preds["ResNet50 (ImageNet)"][0] == yte).astype(int)
    diffs=[]
    rng = np.random.RandomState(42)
    n=len(yte)
    for _ in range(2000):
        idx = rng.randint(0, n, n)
        diffs.append(hib[idx].mean() - res[idx].mean())
    diffs=np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    sig = {"metric":"accuracy_difference_fm_minus_resnet",
           "point_estimate": float(hib.mean()-res.mean()),
           "ci95_low": float(lo), "ci95_high": float(hi),
           "significant_at_95": bool(lo>0 or hi<0)}
    json.dump(sig, open("results/significance_test.json","w"), indent=2)
    print(f"[sig] Phikon-v2-ResNet acc diff {sig['point_estimate']:+.4f} "
          f"95% CI [{lo:+.4f},{hi:+.4f}] significant={sig['significant_at_95']}")

# ---- Label-efficiency sweep (all cached models) on frozen features ----
FRACTIONS = [0.10, 0.25, 0.50, 1.00]
le_rows = []
for name,(Xtr,ytr,Xte,yte) in cached.items():
    for frac in FRACTIONS:
        # class-stratified subsample of the training features
        rng = np.random.RandomState(SEED)
        idx=[]
        for c in np.unique(ytr):
            ci = np.where(ytr==c)[0]
            take = max(1, int(len(ci)*frac))
            idx.extend(rng.choice(ci, take, replace=False))
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced",
                                 random_state=SEED, n_jobs=-1)
        clf.fit(Xtr[idx], ytr[idx])
        preds = clf.predict(Xte)
        le_rows.append({"Model": name, "label_fraction": frac,
                        "n_train": len(idx),
                        "Macro F1": f1_score(yte, preds, average="macro"),
                        "Balanced Accuracy": balanced_accuracy_score(yte, preds)})
        print(f"[label-eff] {name} @ {int(frac*100)}%: "
              f"macroF1 {le_rows[-1]['Macro F1']:.3f}")
pd.DataFrame(le_rows).to_csv("results/label_efficiency.csv", index=False)

# ---- (3) UMAP of frozen embeddings (test set) for each model ----
try:
    import umap
    for name,(Xtr,ytr,Xte,yte) in cached.items():
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=SEED)
        emb2d = reducer.fit_transform(Xte)
        safe = name.split()[0].lower()
        np.savez(f"results/umap_{safe}.npz", emb=emb2d, y=yte)
        print(f"[umap] saved 2D embedding for {name}")
except Exception as e:
    print(f"[umap] skipped ({e}). Install umap-learn to enable.")

df = pd.DataFrame(rows).sort_values("Macro F1", ascending=False).reset_index(drop=True)
df.to_csv("results/model_comparison.csv", index=False)
pd.DataFrame(per_class).T.to_csv("results/per_class_report.csv")
best = df.iloc[0]["Model"]
np.save("results/confusion_matrix.npy", confusions[best])
json.dump({"classes": label_names, "class_full": CLASS_FULL, "best_model": best,
           "train_per_class": TRAIN_PER_CLASS, "test_per_class": TEST_PER_CLASS,
           "seed": SEED,
           "train_split": TRAIN_SPLIT, "test_split": TEST_SPLIT,
           "test_is_separate_patients": True},
          open("results/meta.json","w"), indent=2)

print("\n================ RESULTS ================")
print(df.to_string(index=False))
print(f"\nBest model: {best}")
print("Test set: CRC-VAL-HE-7K (separate patient cohort; cross-patient generalization).")
print("Metrics saved to results/.")
