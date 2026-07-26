"""
data_loader.py
--------------
Loads a CSV dataset and a YAML config, with light generic cleaning that
works regardless of which dataset is supplied.
"""
from __future__ import annotations
import pandas as pd
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load a CSV robustly (handles a couple of common encodings)."""
    try:
        df = pd.read_csv(csv_path)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="latin-1")

    # Drop fully-empty rows/columns and exact duplicate rows -- safe no-ops
    # on a clean dataset, useful on messy real-world exports.
    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="all")
    df = df.drop_duplicates()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def clean_target_column(df: pd.DataFrame, col: str, min_class_count: int = 2) -> pd.DataFrame:
    """
    Drop rows with a missing target, and drop classes so rare that a
    train/test stratified split would fail (e.g. an artist with 1 song).
    """
    df = df[df[col].notna()].copy()
    counts = df[col].value_counts()
    keep_classes = counts[counts >= min_class_count].index
    return df[df[col].isin(keep_classes)].copy()
