"""
LightGBM-based reranker.

Trains a LambdaRank model on (user, candidate) pairs where each user is a
"query" and the label is 1 if the candidate is a positive (held-out target),
0 otherwise. The dataframe must be sorted by user_id so each user's rows
are contiguous (group= parameter relies on this).
"""

from __future__ import annotations

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

import config

class LightGBMReranker:
    """LightGBM LambdaRank reranker"""
    def __init__(self):
        self.model = None
        # categories seen at fit time, applied identically at predict time
        # (unseen values become NaN, which LightGBM handles natively)
        self.category_map: dict[str, list] = {}
        # feature column order at fit time (must match at predict time)
        self.feature_cols: list[str] = []

    def _prepare_features(self, df: pd.DataFrame, fitting: bool) -> pd.DataFrame:
        """Select feature columns and apply consistent categorical encoding"""
        # at fit time, snapshot which columns are features
        if fitting:
            self.feature_cols = [c for c in df.columns if c not in config.META_COLS]

        X = df[self.feature_cols].copy()

        # encode categoricals using the fit-time category list
        for col in config.CAT_COLS:
            if col not in X.columns:
                continue
            if fitting:
                self.category_map[col] = (
                    X[col].astype("category").cat.categories.tolist()
                )
            X[col] = pd.Categorical(X[col], categories=self.category_map[col])

        return X

    def fit(self, df: pd.DataFrame) -> None:
        if "label" not in df.columns:
            raise ValueError("fit() requires a 'label' column")

        # group sizes: count of rows per user in dataframe order
        groups = df.groupby("user_id", sort=False).size().tolist()

        X = self._prepare_features(df, fitting=True)
        y = df["label"].values

        self.model = lgb.LGBMRanker(
            objective=config.LGBM_OBJECTIVE,
            metric=config.LGBM_METRIC,
            n_estimators=config.LGBM_N_ESTIMATORS,
            learning_rate=config.LGBM_LR,
            num_leaves=config.LGBM_NUM_LEAVES,
            min_child_samples=config.LGBM_MIN_CHILD_SAMPLES,
            random_state=config.RANDOM_SEED,
            verbose=-1,
        )
        self.model.fit(
            X, y,
            group=groups,
            categorical_feature=[c for c in config.CAT_COLS if c in X.columns],
        )

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("model not fitted; call fit() or load() first")
        X = self._prepare_features(df, fitting=False)
        
        return self.model.predict(X)

    def save(self, path) -> None:
        joblib.dump(
            {
                "model": self.model,
                "category_map": self.category_map,
                "feature_cols": self.feature_cols,
            },
            path,
        )

    def load(self, path) -> None:
        data = joblib.load(path)
        self.model = data["model"]
        self.category_map = data["category_map"]
        self.feature_cols = data["feature_cols"]