"""
test_pipeline.py
-----------------
Covers:
  1. Schema auto-detection on the bundled sample dataset.
  2. Schema auto-detection on a SECOND dataset with completely
     different column names/casing, to prove the pipeline generalizes.
  3. Training + prediction for each classifier type (knn/bagging/boosting/xgboost/ensemble).
  4. The song recommender returning sensible neighbors.

Run with:  pytest -v
"""
import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_config, clean_target_column
from src.schema import detect_schema
from src.models import build_model_pipeline, AVAILABLE_MODELS
from src.recommender import SongRecommender
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


@pytest.fixture(scope="module")
def config():
    return load_config("config.yaml")


@pytest.fixture(scope="module")
def sample_df():
    return pd.read_csv("music_dataset.csv")


@pytest.fixture(scope="module")
def alt_dataset():
    """A second, differently-shaped dataset (different column names/casing,
    a 'Category' instead of 'genre', a 'Performer' instead of 'artist') to
    prove the schema detector isn't hard-coded to one dataset's headers."""
    rng = np.random.default_rng(0)
    n = 300
    return pd.DataFrame({
        "Track Title": [f"T{i}" for i in range(n)],
        "Performer": rng.choice([f"P{i}" for i in range(10)], n),
        "Category": rng.choice(["Lo-fi", "EDM", "Ambient"], n),
        "Tempo_BPM": rng.normal(110, 15, n),
        "Loudness_dB": rng.normal(-8, 3, n),
        "Mood_Score": rng.uniform(0, 1, n),
        "Mode": rng.choice(["Major", "Minor"], n),
    })


def test_schema_detection_sample(sample_df, config):
    schema = detect_schema(sample_df, config)
    assert schema["genre_col"] == "genre"
    assert schema["artist_col"] == "artist_name"
    assert schema["track_col"] == "song_name"
    assert "energy" in schema["numeric_features"]
    assert "explicit" in schema["categorical_features"]
    # identifier / free-text columns must NOT leak into features
    assert "track_id" not in schema["numeric_features"] + schema["categorical_features"]


def test_schema_detection_generalizes_to_new_dataset(alt_dataset, config):
    schema = detect_schema(alt_dataset, config)
    assert schema["genre_col"] == "Category"
    assert schema["artist_col"] == "Performer"
    assert schema["track_col"] == "Track Title"
    assert "Tempo_BPM" in schema["numeric_features"]
    assert "Mode" in schema["categorical_features"]


@pytest.mark.parametrize("model_type", AVAILABLE_MODELS)
def test_classifier_training_and_prediction(sample_df, config, model_type):
    schema = detect_schema(sample_df, config)
    df = clean_target_column(sample_df, schema["genre_col"], config["training"]["min_class_count"])
    feature_num, feature_cat = schema["numeric_features"], schema["categorical_features"]
    X = df[feature_num + feature_cat]
    y = LabelEncoder().fit_transform(df[schema["genre_col"]])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline = build_model_pipeline(model_type, feature_num, feature_cat, config)
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    assert len(preds) == len(y_test)
    accuracy = (preds == y_test).mean()
    # genre has real signal in the synthetic data -- every model should beat random guessing (1/8)
    assert accuracy > 0.3, f"{model_type} accuracy too low: {accuracy}"


def test_recommender_returns_similar_songs(sample_df, config):
    schema = detect_schema(sample_df, config)
    recommender = SongRecommender(
        numeric_features=schema["numeric_features"],
        categorical_features=schema["categorical_features"],
        track_col=schema["track_col"],
        artist_col=schema["artist_col"],
        genre_col=schema["genre_col"],
        config=config,
    )
    recommender.fit(sample_df)
    query_row = sample_df.iloc[[0]]
    recs = recommender.recommend_from_row(query_row, top_n=5)

    assert len(recs) == 5
    # the query's own genre should dominate its nearest neighbors
    top_genre = recs[schema["genre_col"]].mode().iloc[0]
    assert top_genre == query_row[schema["genre_col"]].iloc[0]


def test_recommender_from_preferences_dict(sample_df, config):
    schema = detect_schema(sample_df, config)
    recommender = SongRecommender(
        numeric_features=schema["numeric_features"],
        categorical_features=schema["categorical_features"],
        track_col=schema["track_col"],
        artist_col=schema["artist_col"],
        genre_col=schema["genre_col"],
        config=config,
    )
    recommender.fit(sample_df)
    recs = recommender.recommend_from_preferences({"energy": 0.9, "danceability": 0.8}, top_n=5)
    assert len(recs) == 5
