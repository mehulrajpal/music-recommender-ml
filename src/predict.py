"""
predict.py
----------
Load trained .pkl artifacts (saved by train.py, default location: repo root)
and run inference:
  - Predict genre / artist for a new row (or every row of a new CSV)
  - Get top-N song recommendations for a listener profile

Usage:
    python -m src.predict --input new_rows.csv --model xgboost --out predictions.csv --recommend
"""
from __future__ import annotations
import argparse
import glob
import os
import joblib
import pandas as pd

from src.data_loader import load_dataset


def _find_model_file(outdir: str, model_type: str, task: str):
    """Find <model_type>_<task>_predictor.pkl; if model_type is 'auto', pick whichever exists."""
    if model_type != "auto":
        path = os.path.join(outdir, f"{model_type}_{task}_predictor.pkl")
        return path if os.path.exists(path) else None
    matches = glob.glob(os.path.join(outdir, f"*_{task}_predictor.pkl"))
    return matches[0] if matches else None


def load_artifacts(outdir: str = ".", model_type: str = "auto"):
    schema_path = os.path.join(outdir, "schema.pkl")
    if not os.path.exists(schema_path):
        raise FileNotFoundError(
            f"No schema.pkl found in '{outdir}'. Run `python -m src.train --data <your_csv>` first."
        )
    artifacts = {"schema": joblib.load(schema_path)}

    genre_path = _find_model_file(outdir, model_type, "genre")
    artist_path = _find_model_file(outdir, model_type, "artist")
    rec_path = os.path.join(outdir, "knn_recommender.pkl")

    if genre_path:
        artifacts["genre"] = joblib.load(genre_path)
    if artist_path:
        artifacts["artist"] = joblib.load(artist_path)
    if os.path.exists(rec_path):
        artifacts["recommender"] = joblib.load(rec_path)
    return artifacts


def predict_dataframe(df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    schema = artifacts["schema"]
    features = schema["numeric_features"] + schema["categorical_features"]
    X = df[[c for c in features if c in df.columns]]
    out = df.copy()

    if "genre" in artifacts:
        pipeline, encoder = artifacts["genre"]["pipeline"], artifacts["genre"]["encoder"]
        preds = pipeline.predict(X)
        out["predicted_genre"] = encoder.inverse_transform(preds)

    if "artist" in artifacts:
        pipeline, encoder = artifacts["artist"]["pipeline"], artifacts["artist"]["encoder"]
        preds = pipeline.predict(X)
        out["predicted_artist"] = encoder.inverse_transform(preds)

    return out


def main():
    parser = argparse.ArgumentParser(description="Run inference with trained models.")
    parser.add_argument("--input", required=True, help="CSV of new rows to predict on.")
    parser.add_argument("--outdir", default=".", help="Directory containing trained .pkl artifacts.")
    parser.add_argument("--model", default="auto", help="Which trained model prefix to load (e.g. xgboost, ensemble). "
                                                          "'auto' picks whichever *_genre_predictor.pkl / *_artist_predictor.pkl exists.")
    parser.add_argument("--out", default="predictions.csv", help="Where to write predictions.")
    parser.add_argument("--recommend", action="store_true", help="Also print top-5 song recommendations per row.")
    args = parser.parse_args()

    artifacts = load_artifacts(args.outdir, args.model)
    df = load_dataset(args.input)
    result = predict_dataframe(df, artifacts)
    result.to_csv(args.out, index=False)
    print(f"Predictions written to {args.out}")

    if args.recommend and "recommender" in artifacts:
        rec = artifacts["recommender"]
        for i in range(min(3, len(df))):
            print(f"\nTop recommendations similar to row {i}:")
            print(rec.recommend_from_row(df.iloc[[i]], top_n=5).to_string(index=False))


if __name__ == "__main__":
    main()
