
# Detection of Anomalous Images

Detecting anomalous images from a large dataset of facial images of human subjects,
primarily using statistical and probabilistic methods.

## Install

```bash
pip install -r requirements.txt
```

## Data Layout

Prepare four image directories:

```
data/
├── good_train/    # normal training images
├── bad_train/     # anomalous training images
├── good_test/     # normal test images  (used to evaluate)
└── bad_test/      # anomalous test images (used to evaluate)
```

Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`

---

## Quick Start (CLI)

### Train

```bash
# Ensemble mode (recommended) – trains SVDD + GLOSH per feature type, stacked with MLP
python train.py \
    --good-train data/good_train/ \
    --bad-train  data/bad_train/ \
    --good-test  data/good_test/ \
    --bad-test   data/bad_test/ \
    --mode       ensemble \
    --output     anomdet_model.pkl

# Single-classifier modes
python train.py ... --mode svdd
python train.py ... --mode glosh
```

### Predict

```bash
python predict.py \
    --input-dir data/unlabelled/ \
    --model     anomdet_model.pkl
```

Output example:
```
Predictions (1 = normal, 0 = anomaly):
  Image                                    Label
  --------------------------------------------------
  img001.jpg                               NORMAL
  img002.jpg                               ANOMALY
  ...
Summary: 120 images → 14 anomalies, 106 normal.
```

---

## Python API

```python
import numpy as np
from anomdet import Anomdet, EnsembleMod, load_features, ALL_FEATURES

# ── Feature extraction ──────────────────────────────────────────────────────
# feature_selector = [edge, haralick, orb, hog, dwt]
train_good = load_features("data/good_train/", [1, 1, 1, 0, 1])
train_bad  = load_features("data/bad_train/",  [1, 1, 1, 0, 1])
train_all  = np.vstack([train_good, train_bad])

test_good = load_features("data/good_test/", [1, 1, 1, 0, 1])
test_bad  = load_features("data/bad_test/",  [1, 1, 1, 0, 1])
test_all  = np.vstack([test_good, test_bad])

# ── SVDD ────────────────────────────────────────────────────────────────────
det = Anomdet()
det.svdtrain(train_all)
preds = det.svdd_predict(test_all)   # list of 0/1

# ── GLOSH ───────────────────────────────────────────────────────────────────
det.glosh(train_all)
preds = det.glosh_predict(test_all)  # list of 0/1

# ── Ensemble ─────────────────────────────────────────────────────────────────
# Stack predictions from multiple feature types / classifiers
ens_features = np.column_stack([pred_svdd_edge, pred_glosh_edge,
                                pred_svdd_haralick, ...])
truth = np.concatenate([np.ones(len(test_good)), np.zeros(len(test_bad))])

ens = EnsembleMod(ens_features, truth)
ens.mlpc()   # MLP (sets ens.clf)
ens.xgbr()   # XGBoost
ens.bagg()   # Bagging

prediction = ens.predict(new_ens_features)
```

---

## Features

| Index | Name      | Description                                   | # features  |
|-------|-----------|-----------------------------------------------|-------------|
| 0     | edge      | Laplacian / Sobel / Prewitt statistics        | 12          |
| 1     | haralick  | Haralick texture (4 directions × 13)          | 52          |
| 2     | orb       | ORB keypoint count                            | 1           |
| 3     | hog       | Normalised HOG descriptor                    | variable    |
| 4     | dwt       | DWT patch energies (db1, 4 subbands)          | variable    |

## Models

| Model   | Class / function         | Notes                            |
|---------|--------------------------|----------------------------------|
| SVDD    | `Anomdet.svdtrain()`     | One-Class SVM, sigmoid kernel    |
| GLOSH   | `Anomdet.glosh()`        | HDBSCAN outlier scores           |
| MLP     | `EnsembleMod.mlpc()`     | Meta-learner over base preds     |
| XGBoost | `EnsembleMod.xgbr()`     | Linear booster                   |
| Bagging | `EnsembleMod.bagg()`     | 500 Decision Tree estimators     |

