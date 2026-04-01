#!/usr/bin/env python3
"""
predict.py – Predict anomalies using a trained AnomDet model.

Usage
-----
python predict.py \\
    --input-dir  data/unlabelled/ \\
    --model      anomdet_model.pkl
"""

import argparse
import os
import pickle

import numpy as np

from anomdet import ALL_FEATURES, FEATURE_NAMES, load_features

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")


def _predict_ensemble(model_data: dict, input_dir: str) -> list:
    """Generate ensemble predictions by re-running per-feature-type classifiers."""
    ensemble = model_data["model"]
    detector = model_data["detector"]

    svdd_preds = []
    glosh_preds = []

    for idx, name in enumerate(FEATURE_NAMES):
        selector = [0] * len(FEATURE_NAMES)
        selector[idx] = 1
        print(f"  [{idx + 1}/{len(FEATURE_NAMES)}] Extracting '{name}' features …")
        features = load_features(input_dir, selector)
        svdd_preds.append(detector.svdd_predict(features))
        glosh_preds.append(detector.glosh_predict(features))

    ens_features = np.array(svdd_preds + glosh_preds).T
    return list(ensemble.predict(ens_features))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict anomalies with a trained AnomDet model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir", required=True, help="Directory of images to classify"
    )
    parser.add_argument(
        "--model", default="anomdet_model.pkl", help="Path to trained model pickle"
    )
    args = parser.parse_args()

    print(f"\n=== AnomDet Prediction ===\n")

    with open(args.model, "rb") as fh:
        model_data = pickle.load(fh)

    mode = model_data["mode"]
    print(f"Model mode : {mode}")
    print(f"Input dir  : {args.input_dir}\n")

    if mode == "ensemble":
        predictions = _predict_ensemble(model_data, args.input_dir)
    elif mode == "svdd":
        model = model_data["model"]
        features = load_features(args.input_dir, ALL_FEATURES)
        raw = model.predict(features)
        predictions = [0 if p == -1 else 1 for p in raw]
    elif mode == "glosh":
        model = model_data["model"]
        features = load_features(args.input_dir, ALL_FEATURES)
        raw = model.fit_predict(features)
        predictions = [0 if p == -1 else 1 for p in raw]
    else:
        raise ValueError(f"Unknown model mode '{mode}'.")

    images = sorted(
        f for f in os.listdir(args.input_dir) if f.lower().endswith(_IMG_EXTS)
    )

    print("Predictions (1 = normal, 0 = anomaly):")
    print(f"  {'Image':<40} Label")
    print("  " + "-" * 50)
    for img, pred in zip(images, predictions):
        label = "NORMAL " if pred == 1 else "ANOMALY"
        print(f"  {img:<40} {label}")

    n_anomalies = sum(1 for p in predictions if p == 0)
    n_total = len(predictions)
    print(f"\nSummary: {n_total} images → {n_anomalies} anomalies, "
          f"{n_total - n_anomalies} normal.")


if __name__ == "__main__":
    main()
