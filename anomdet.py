"""
anomdet.py – Core library for AnomDet facial-image anomaly detection.

Classes
-------
Anomdet      : feature extraction + one-class base classifiers (SVDD / GLOSH)
EnsembleMod  : meta-classifier that stacks base-classifier predictions

Standalone helper
-----------------
load_features(path, feature_selector) -> np.ndarray
"""

import os

import cv2
import hdbscan
import mahotas as mt
import numpy as np
import pywt
import skimage.measure
import xgboost as xgb
from skimage.feature import hog
from skimage.filters import laplace, prewitt_h, prewitt_v, sobel
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import OneClassSVM
from sklearn.tree import DecisionTreeClassifier

# Supported image extensions
_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")

# Spatial block size used by HOG and DWT patch reduction
BLOCK_SIZE = (14, 14)

# Feature selector indices
FEAT_EDGE = 0      # Laplacian / Sobel / Prewitt statistics  (12 features)
FEAT_HARALICK = 1  # Haralick texture (4 directions × 13 = 52 features)
FEAT_ORB = 2       # ORB keypoint count                      (1 feature)
FEAT_HOG = 3       # Normalised HOG descriptor               (variable)
FEAT_DWT = 4       # DWT patch energies (4 bands × patches)  (variable)

ALL_FEATURES = [1, 1, 1, 1, 1]
FEATURE_NAMES = ["edge", "haralick", "orb", "hog", "dwt"]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def load_features(path: str, feature_selector: list) -> np.ndarray:
    """Extract features from all images in *path*.

    Parameters
    ----------
    path : str
        Directory containing image files.
    feature_selector : list[int]
        Five-element binary list ``[edge, haralick, orb, hog, dwt]``.
        Set an element to 1 to include that feature group.

    Returns
    -------
    np.ndarray, shape (n_images, n_features)
    """
    image_files = sorted(
        f for f in os.listdir(path) if f.lower().endswith(_IMG_EXTS)
    )
    if not image_files:
        raise ValueError(f"No supported image files found in '{path}'.")

    all_features = []

    for img_name in image_files:
        img_path = os.path.join(path, img_name)
        img = cv2.imread(img_path)  # BGR uint8
        if img is None:
            print(f"  Warning: could not read '{img_path}', skipping.")
            continue

        img = cv2.resize(img, (224, 224))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # uint8 grayscale

        feature = []

        # ── Edge detection features ─────────────────────────────────────────
        if feature_selector[FEAT_EDGE]:
            # Convert to float so filter outputs are consistent
            gray_f = gray.astype(np.float64)
            lap = laplace(gray_f)
            sob = sobel(gray_f)
            hpre = prewitt_h(gray_f)
            vpre = prewitt_v(gray_f)
            feature.extend([
                lap.mean(), lap.var(), np.max(lap),
                sob.mean(), sob.var(), np.max(sob),
                hpre.mean(), hpre.var(), np.max(hpre),
                vpre.mean(), vpre.var(), np.max(vpre),
            ])

        # ── Haralick texture features ────────────────────────────────────────
        if feature_selector[FEAT_HARALICK]:
            # mahotas.haralick requires a 2-D integer array
            textures = mt.features.haralick(gray.astype(np.uint8))  # (4, 13)
            for row in textures:
                feature.extend(row)

        # ── ORB keypoint count ───────────────────────────────────────────────
        if feature_selector[FEAT_ORB]:
            orb = cv2.ORB_create()
            kp = orb.detect(img, None)
            kp, _des = orb.compute(img, kp)
            feature.append(len(kp))

        # ── HOG descriptor ───────────────────────────────────────────────────
        if feature_selector[FEAT_HOG]:
            fd, _ = hog(
                gray,
                orientations=4,
                pixels_per_cell=BLOCK_SIZE,
                cells_per_block=(1, 1),
                visualize=True,
            )
            norm = np.linalg.norm(fd)
            normalized_fd = fd / norm if norm > 0 else fd
            feature.extend(normalized_fd)

        # ── DWT patch energies ───────────────────────────────────────────────
        if feature_selector[FEAT_DWT]:
            coeffs2 = pywt.dwt2(gray.astype(np.float64), "db1")
            LL, (LH, HL, HH) = coeffs2
            for band in (LL, LH, HL, HH):
                reduced = skimage.measure.block_reduce(band, BLOCK_SIZE, np.linalg.norm)
                norm = np.linalg.norm(reduced)
                normalized = reduced / norm if norm > 0 else reduced
                feature.extend(normalized.flatten())

        all_features.append(feature)

    return np.array(all_features, dtype=np.float64)


# ---------------------------------------------------------------------------
# Base classifiers
# ---------------------------------------------------------------------------

class Anomdet:
    """Base anomaly detector.

    Usage
    -----
    >>> det = Anomdet()
    >>> det.svdtrain(train_features)
    >>> preds = det.svdd_predict(test_features)   # list of 0/1 labels

    Or with GLOSH (HDBSCAN):
    >>> det.glosh(train_features)
    >>> preds = det.glosh_predict(test_features)
    """

    def __init__(self):
        self.osvm_model = None
        self.clusters = None

    # ── SVDD ────────────────────────────────────────────────────────────────

    def svdtrain(self, features: np.ndarray) -> None:
        """Fit a One-Class SVM (SVDD) on *features*."""
        self.osvm_model = OneClassSVM(nu=0.001, kernel="sigmoid")
        self.osvm_model.fit(features)

    def svdd_predict(self, features: np.ndarray) -> list:
        """Return binary predictions (1 = normal, 0 = anomaly) from SVDD."""
        if self.osvm_model is None:
            raise RuntimeError("Call svdtrain() before svdd_predict().")
        raw = self.osvm_model.predict(features)
        return [0 if p == -1 else 1 for p in raw]

    # ── GLOSH / HDBSCAN ─────────────────────────────────────────────────────

    def glosh(self, features: np.ndarray) -> None:
        """Fit HDBSCAN on *features* (used for GLOSH outlier scores).

        ``prediction_data=True`` is set so that :meth:`glosh_predict` can
        classify new points without refitting the clusterer.
        """
        self.clusters = hdbscan.HDBSCAN(
            min_cluster_size=100, prediction_data=True
        )
        self.clusters.fit(features)

    def glosh_predict(self, features: np.ndarray) -> list:
        """Return binary predictions (1 = normal, 0 = anomaly) from GLOSH.

        Uses :func:`hdbscan.approximate_predict` so the trained clustering is
        not refitted on new data.
        """
        if self.clusters is None:
            raise RuntimeError("Call glosh() before glosh_predict().")
        labels, _ = hdbscan.approximate_predict(self.clusters, features)
        return [0 if p == -1 else 1 for p in labels]


# ---------------------------------------------------------------------------
# Ensemble meta-classifier
# ---------------------------------------------------------------------------

class EnsembleMod:
    """Ensemble model that stacks predictions from multiple base classifiers.

    Parameters
    ----------
    preds : array-like, shape (n_samples, n_base_classifiers)
        Predictions from each base classifier.
    truths : array-like, shape (n_samples,)
        Ground-truth labels (1 = normal, 0 = anomaly).
    """

    def __init__(self, preds, truths):
        self.preds = np.asarray(preds)
        self.truths = np.asarray(truths)
        self.clf = None       # MLP (default prediction interface)
        self.xgb_model = None
        self.bag_model = None

    # ── MLP ─────────────────────────────────────────────────────────────────

    def mlpc(self) -> None:
        """Train a Multilayer Perceptron ensemble classifier."""
        X_tr, X_te, y_tr, y_te = train_test_split(
            self.preds, self.truths, test_size=0.2, random_state=1
        )
        self.clf = MLPClassifier(
            solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1
        )
        self.clf.fit(X_tr, y_tr)
        preds = self.clf.predict(X_te)
        _print_metrics("MLP", y_te, preds)

    # ── XGBoost ─────────────────────────────────────────────────────────────

    def xgbr(self) -> None:
        """Train an XGBoost ensemble classifier."""
        X_tr, X_te, y_tr, y_te = train_test_split(
            self.preds, self.truths, test_size=0.2, random_state=123
        )
        train_dm = xgb.DMatrix(data=X_tr, label=y_tr)
        test_dm = xgb.DMatrix(data=X_te, label=y_te)
        param = {"booster": "gblinear", "objective": "reg:squarederror"}
        self.xgb_model = xgb.train(
            params=param, dtrain=train_dm, num_boost_round=10
        )
        raw = self.xgb_model.predict(test_dm)
        preds = [1 if p >= 0.5 else 0 for p in raw]
        _print_metrics("XGBoost", y_te, preds)

    # ── Bagging ─────────────────────────────────────────────────────────────

    def bagg(self) -> None:
        """Train a Bagging (Decision-Tree) ensemble classifier."""
        X_tr, X_te, y_tr, y_te = train_test_split(
            self.preds, self.truths, test_size=0.2, random_state=123
        )
        base = DecisionTreeClassifier()
        self.bag_model = BaggingClassifier(
            estimator=base, n_estimators=500, random_state=8
        )
        self.bag_model.fit(X_tr, y_tr)
        preds = self.bag_model.predict(X_te)
        _print_metrics("Bagging", y_te, preds)

    # ── Inference ───────────────────────────────────────────────────────────

    def predict(self, ensemble_features) -> np.ndarray:
        """Predict using the most recently trained classifier (MLP preferred).

        Raises ``RuntimeError`` if no classifier has been trained yet.
        """
        if self.clf is not None:
            return self.clf.predict(ensemble_features)
        if self.bag_model is not None:
            return self.bag_model.predict(ensemble_features)
        raise RuntimeError(
            "No ensemble model trained. Call mlpc(), xgbr(), or bagg() first."
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _print_metrics(name: str, y_true, y_pred) -> None:
    print(f"\n{name} Results")
    print("  Accuracy :", accuracy_score(y_true, y_pred))
    print("  Confusion matrix:\n", confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))
