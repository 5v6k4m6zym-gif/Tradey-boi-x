"""
Minimal EnsembleModel stub — allows pickle to load X's trained model.pkl
without requiring X's full engine.py to be installed.

The real predict_proba logic (60/40 XGBoost+RF blend) is preserved here.
"""
from __future__ import annotations
from sklearn.pipeline import Pipeline


class EnsembleModel:
    """XGBoost + RandomForest ensemble — 60/40 probability blend."""

    def __init__(self, xgb_pipe: Pipeline, rf_pipe: Pipeline):
        self.xgb = xgb_pipe
        self.rf  = rf_pipe

    def predict_proba(self, X):
        xgb_p = self.xgb.predict_proba(X)
        rf_p  = self.rf.predict_proba(X)
        return 0.60 * xgb_p + 0.40 * rf_p

    def predict(self, X):
        import numpy as np
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
