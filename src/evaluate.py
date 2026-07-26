"""
evaluate.py
-----------
Shared metric-reporting helpers used by both train.py (CLI) and app.py (Streamlit).
"""
from __future__ import annotations
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
import numpy as np


def classification_report_dict(y_true, y_pred, y_proba=None, classes=None) -> dict:
    report = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "f1_weighted": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "n_test_samples": len(y_true),
        "n_classes": int(len(set(y_true))),
    }
    if y_proba is not None and classes is not None:
        k = min(3, y_proba.shape[1])
        try:
            report["top_3_accuracy"] = round(
                top_k_accuracy_score(y_true, y_proba, k=k, labels=classes), 4
            )
        except ValueError:
            pass
    return report
