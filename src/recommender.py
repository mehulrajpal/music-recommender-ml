"""
recommender.py
---------------
Content-based song recommender: predicts WHICH SONGS a listener wants
to hear next, using K-Nearest-Neighbors similarity search over whatever
numeric audio/behavioural features the dataset provides (tempo, energy,
danceability, valence, play_count, rating, etc -- whatever exists).

Because "song" is normally near-unique per row (thousands of classes),
this is treated as a similarity/retrieval problem rather than a
classifier, which is both the standard approach and far more robust
across different datasets.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.preprocessing import build_preprocessor


class SongRecommender:
    def __init__(self, numeric_features, categorical_features, track_col, config: dict,
                 artist_col: str | None = None, genre_col: str | None = None):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.track_col = track_col
        self.artist_col = artist_col
        self.genre_col = genre_col
        self.config = config
        self.preprocessor = build_preprocessor(numeric_features, categorical_features, config)
        self.model = NearestNeighbors(
            n_neighbors=config["recommender"]["n_neighbors"],
            metric=config["recommender"]["metric"],
        )
        self.reference_df: pd.DataFrame | None = None

    def fit(self, df: pd.DataFrame):
        self.reference_df = df.reset_index(drop=True)
        X = self.preprocessor.fit_transform(self.reference_df)
        X = X.toarray() if hasattr(X, "toarray") else X
        self.model.fit(X)
        return self

    def recommend_from_row(self, row: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """Recommend songs similar to a single listener/song profile (as a 1-row dataframe)."""
        X = self.preprocessor.transform(row)
        X = X.toarray() if hasattr(X, "toarray") else X
        n_neighbors = min(top_n, len(self.reference_df))
        distances, indices = self.model.kneighbors(X, n_neighbors=n_neighbors)
        results = self.reference_df.iloc[indices[0]].copy()
        results["similarity_score"] = 1 - distances[0] if self.config["recommender"]["metric"] == "cosine" else -distances[0]
        cols = [c for c in [self.track_col, self.artist_col, self.genre_col] if c] + list(self.numeric_features[:3]) + ["similarity_score"]
        cols = [c for c in cols if c in results.columns]
        return results[cols].sort_values("similarity_score", ascending=False).reset_index(drop=True)

    def recommend_from_preferences(self, preferences: dict, top_n: int = 10) -> pd.DataFrame:
        """
        Recommend songs from a dict of feature preferences, e.g.
        {"energy": 0.8, "danceability": 0.7, "tempo": 120}
        Missing features are filled with the dataset's median/mode.
        """
        template = {}
        for col in self.numeric_features:
            template[col] = preferences.get(col, self.reference_df[col].median())
        for col in self.categorical_features:
            template[col] = preferences.get(col, self.reference_df[col].mode().iloc[0])
        row = pd.DataFrame([template])
        return self.recommend_from_row(row, top_n=top_n)
