"""
train.py
--------
Command-line training entry point. Works on ANY music dataset CSV --
schema.py auto-detects the genre/artist/track/feature columns.

Usage:
    python -m src.train --data music_dataset.csv --model xgboost
    python -m src.train --data music_dataset.csv --model ensemble --target genre
    python -m src.train --data my_other_dataset.csv --config config.yaml

Outputs (saved to --outdir, default "." i.e. the repo root, matching the
saved-model convention of <algorithm>_<task>_predictor.pkl):
    <model>_genre_predictor.pkl     (if a genre column was detected)
    <model>_artist_predictor.pkl    (if an artist column was detected)
    knn_recommender.pkl             (song recommender)
    schema.pkl                      (detected schema, needed by predict.py / app.py)
    metrics.json                    (accuracy / F1 / top-3 accuracy for each model)
"""
from __future__ import annotations
import argparse
import json
import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.data_loader import load_config, load_dataset, clean_target_column
from src.schema import detect_schema, summarize_schema
from src.models import build_model_pipeline, AVAILABLE_MODELS
from src.recommender import SongRecommender
from src.evaluate import classification_report_dict


def train_classifier(df, target_col, feature_num, feature_cat, model_type, config, label_name):
    df = clean_target_column(df, target_col, config["training"]["min_class_count"])
    if df[target_col].nunique() < 2:
        print(f"[skip] '{label_name}' target has fewer than 2 usable classes after cleaning.")
        return None, None

    X = df[feature_num + feature_cat]
    y_raw = df[target_col].astype(str)
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    stratify = y if np.min(np.bincount(y)) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["training"]["test_size"],
        random_state=config["training"]["random_state"], stratify=stratify,
    )

    pipeline = build_model_pipeline(model_type, feature_num, feature_cat, config)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test) if hasattr(pipeline, "predict_proba") else None
    metrics = classification_report_dict(y_test, y_pred, y_proba, classes=np.unique(y))
    print(f"[{label_name}] model={model_type} -> {metrics}")

    return {"pipeline": pipeline, "encoder": encoder}, metrics


def main():
    parser = argparse.ArgumentParser(description="Train genre/artist classifiers and a song recommender.")
    parser.add_argument("--data", required=True, help="Path to input CSV dataset (any music dataset).")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--model", default="xgboost", choices=AVAILABLE_MODELS,
                         help="Classifier type for genre/artist prediction.")
    parser.add_argument("--target", default="both", choices=["genre", "artist", "both", "none"],
                         help="Which classifier(s) to train.")
    parser.add_argument("--outdir", default=".", help="Directory to save trained .pkl artifacts (default: repo root).")
    args = parser.parse_args()

    config = load_config(args.config)
    df = load_dataset(args.data)
    schema = detect_schema(df, config)
    print(summarize_schema(schema))

    os.makedirs(args.outdir, exist_ok=True)
    metrics_all = {}

    feature_num = schema["numeric_features"]
    feature_cat = schema["categorical_features"]

    if not feature_num and not feature_cat:
        raise ValueError("No usable feature columns were detected in this dataset. "
                          "Check config.yaml's exclude_from_features / cardinality settings.")

    if args.target in ("genre", "both") and schema["genre_col"]:
        artifact, metrics = train_classifier(
            df, schema["genre_col"], feature_num, feature_cat, args.model, config, "genre"
        )
        if artifact:
            joblib.dump(artifact, os.path.join(args.outdir, f"{args.model}_genre_predictor.pkl"))
            metrics_all["genre"] = metrics
    elif args.target in ("genre", "both"):
        print("[skip] No genre-like column detected in this dataset.")

    if args.target in ("artist", "both") and schema["artist_col"]:
        artifact, metrics = train_classifier(
            df, schema["artist_col"], feature_num, feature_cat, args.model, config, "artist"
        )
        if artifact:
            joblib.dump(artifact, os.path.join(args.outdir, f"{args.model}_artist_predictor.pkl"))
            metrics_all["artist"] = metrics
    elif args.target in ("artist", "both"):
        print("[skip] No artist-like column detected in this dataset.")

    # Song recommender (KNN-based) -- trained whenever there are numeric/categorical features,
    # regardless of whether a track column exists (falls back to row index as the song identifier).
    recommender = SongRecommender(
        numeric_features=feature_num,
        categorical_features=feature_cat,
        track_col=schema["track_col"],
        artist_col=schema["artist_col"],
        genre_col=schema["genre_col"],
        config=config,
    )
    recommender.fit(df)
    joblib.dump(recommender, os.path.join(args.outdir, "knn_recommender.pkl"))
    print("[ok] Song recommender trained and saved.")

    joblib.dump(schema, os.path.join(args.outdir, "schema.pkl"))
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(metrics_all, f, indent=2)

    print(f"\nAll artifacts saved to '{args.outdir}/'.")


if __name__ == "__main__":
    main()
