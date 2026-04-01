#!/usr/bin/env python3
"""
train.py – Training pipeline for AnomDet.

Modes
-----
svdd      Train a single One-Class SVM (SVDD) on all features combined.
glosh     Train a single HDBSCAN (GLOSH) on all features combined.
ensemble  Train per-feature-type SVDD + GLOSH classifiers and stack their
          predictions in an MLP ensemble. (default)

Usage
-----
python train.py \\
    --good-train  data/good_train/ \\
    --bad-train   data/bad_train/ \\
    --good-test   data/good_test/ \\
    --bad-test    data/bad_test/ \\
    --mode        ensemble \\
    --output      anomdet_model.pkl
"""

import argparse
import os
import pickle

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from anomdet import (
    ALL_FEATURES,
    FEATURE_NAMES,
    Anomdet,
    EnsembleMod,
    load_features,
    _print_metrics,
)

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")


def _count_images(path: str) -> int:
    return sum(1 for f in os.listdir(path) if f.lower().endswith(_IMG_EXTS))


def build_ensemble_features(
    good_train: str, bad_train: str, good_test: str, bad_test: str
):
    """Run each single-feature SVDD + GLOSH and return stacked predictions.

    Returns
    -------
    ens_features : np.ndarray, shape (n_test, 2 * n_feature_types)
    truth        : np.ndarray, shape (n_test,)
    detector     : Anomdet  – the last-fitted detector (for inference reuse)
    """
    detector = Anomdet()
    n_good_test = _count_images(good_test)
    n_bad_test = _count_images(bad_test)
    truth = np.concatenate([np.ones(n_good_test), np.zeros(n_bad_test)])

    svdd_preds = []
    glosh_preds = []

    for idx, name in enumerate(FEATURE_NAMES):
        selector = [0] * len(FEATURE_NAMES)
        selector[idx] = 1
        print(f"  [{idx + 1}/{len(FEATURE_NAMES)}] Extracting '{name}' features …")

        train_good = load_features(good_train, selector)
        train_bad = load_features(bad_train, selector)
        train_all = np.vstack([train_good, train_bad])

        test_good = load_features(good_test, selector)
        test_bad = load_features(bad_test, selector)
        test_all = np.vstack([test_good, test_bad])

        detector.svdtrain(train_all)
        svdd_preds.append(detector.svdd_predict(test_all))

        detector.glosh(train_all)
        glosh_preds.append(detector.glosh_predict(test_all))

    # Shape: (n_test, n_feature_types * 2)
    ens_features = np.array(svdd_preds + glosh_preds).T
    return ens_features, truth, detector


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train AnomDet anomaly detection models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--good-train", required=True, help="Good (normal) training images")
    parser.add_argument("--bad-train", required=True, help="Anomalous training images")
    parser.add_argument("--good-test", required=True, help="Good (normal) test images")
    parser.add_argument("--bad-test", required=True, help="Anomalous test images")
    parser.add_argument(
        "--mode",
        choices=["svdd", "glosh", "ensemble"],
        default="ensemble",
        help="Classifier mode",
    )
    parser.add_argument("--output", default="anomdet_model.pkl", help="Path to save model")
    args = parser.parse_args()

    print(f"\n=== AnomDet Training  [mode: {args.mode}] ===\n")

    detector = Anomdet()

    if args.mode in ("svdd", "glosh"):
        print("Extracting features (all types combined) …")
        train_good = load_features(args.good_train, ALL_FEATURES)
        train_bad = load_features(args.bad_train, ALL_FEATURES)
        train_all = np.vstack([train_good, train_bad])

        n_good_test = _count_images(args.good_test)
        n_bad_test = _count_images(args.bad_test)
        test_good = load_features(args.good_test, ALL_FEATURES)
        test_bad = load_features(args.bad_test, ALL_FEATURES)
        test_all = np.vstack([test_good, test_bad])
        truth = np.concatenate([np.ones(n_good_test), np.zeros(n_bad_test)])

        print("Fitting model …")
        if args.mode == "svdd":
            detector.svdtrain(train_all)
            preds = detector.svdd_predict(test_all)
            saved_model = detector.osvm_model
        else:
            detector.glosh(train_all)
            preds = detector.glosh_predict(test_all)
            saved_model = detector.clusters

        _print_metrics(args.mode.upper(), truth, preds)

        with open(args.output, "wb") as fh:
            pickle.dump({"mode": args.mode, "model": saved_model}, fh)

    else:  # ensemble
        print("Building per-feature-type predictions for ensemble …")
        ens_features, truth, detector = build_ensemble_features(
            args.good_train, args.bad_train, args.good_test, args.bad_test
        )

        print("\nTraining ensemble classifiers …")
        ensemble = EnsembleMod(ens_features, truth)
        ensemble.mlpc()

        with open(args.output, "wb") as fh:
            pickle.dump(
                {"mode": "ensemble", "model": ensemble, "detector": detector}, fh
            )

    print(f"\nModel saved to '{args.output}'")


if __name__ == "__main__":
    main()
