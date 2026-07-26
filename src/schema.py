"""
schema.py
---------
Dataset-agnostic schema detection.

Any music dataset has a different set of column names (Spotify exports,
Last.fm dumps, Kaggle "MCRec" style CSVs, custom listening logs, etc).
Instead of hard-coding column names, this module inspects the dataframe's
headers and dtypes and guesses which columns play which role:

    - genre column       (classification target #1)
    - artist column      (classification target #2)
    - track/song column  (identifier for the recommender)
    - user column        (optional, for user-level splits)
    - numeric features    -> used for KNN / recommender similarity
    - categorical features -> one-hot encoded for classifiers

This is what makes the project "plug and play" with a new dataset:
point it at any CSV and it will do its best to find the right columns,
falling back gracefully (e.g. skipping genre prediction entirely if no
genre-like column exists).
"""
from __future__ import annotations
import pandas as pd


def _find_column(columns, candidates):
    """Case-insensitive substring match of candidate names against columns."""
    lower_map = {c.lower(): c for c in columns}
    # exact match first
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    # substring match second
    for col in columns:
        for cand in candidates:
            if cand.lower() in col.lower():
                return col
    return None


def detect_schema(df: pd.DataFrame, config: dict) -> dict:
    """
    Inspect a dataframe and return a dict describing detected roles:

        {
          "genre_col": str | None,
          "artist_col": str | None,
          "track_col": str | None,
          "user_col": str | None,
          "numeric_features": [...],
          "categorical_features": [...],
        }
    """
    sc = config["schema"]
    columns = list(df.columns)

    genre_col = _find_column(columns, sc["genre_candidates"])
    artist_col = _find_column(columns, sc["artist_candidates"])
    track_col = _find_column(columns, sc["track_candidates"])
    user_col = _find_column(columns, sc["user_candidates"])

    reserved = {c for c in [genre_col, artist_col, track_col, user_col] if c}
    exclude_keywords = [k.lower() for k in sc["exclude_from_features"]]

    numeric_features, categorical_features = [], []
    max_card = sc.get("max_categorical_cardinality", 50)

    for col in columns:
        if col in reserved:
            continue
        if any(kw in col.lower() for kw in exclude_keywords):
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_features.append(col)
        elif pd.api.types.is_bool_dtype(df[col]):
            numeric_features.append(col)
        else:
            # candidate categorical column - only keep if low cardinality
            n_unique = df[col].nunique(dropna=True)
            if 1 < n_unique <= max_card:
                categorical_features.append(col)
            # else: treated as free text / identifier -> dropped

    return {
        "genre_col": genre_col,
        "artist_col": artist_col,
        "track_col": track_col,
        "user_col": user_col,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    }


def summarize_schema(schema: dict) -> str:
    lines = ["Detected dataset schema:"]
    lines.append(f"  Genre column      : {schema['genre_col']}")
    lines.append(f"  Artist column     : {schema['artist_col']}")
    lines.append(f"  Track column      : {schema['track_col']}")
    lines.append(f"  User column       : {schema['user_col']}")
    lines.append(f"  Numeric features  : {schema['numeric_features']}")
    lines.append(f"  Categorical feats : {schema['categorical_features']}")
    return "\n".join(lines)
