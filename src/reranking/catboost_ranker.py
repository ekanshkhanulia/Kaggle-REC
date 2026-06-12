"""
CatBoost-based reranker.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool

import config

class CatBoostReranker:
    """CatBoost YetiRank reranker."""
    def __init__(self):
        self.model = None
        self.feature_cols: list[str] = []

    def prepare_features(self, df: pd.DataFrame, fitting: bool) -> pd.DataFrame:
        """Select feature columns, replace NaN in categorical cols with a token."""
        if fitting:
            self.feature_cols = [c for c in df.columns if c not in config.META_COLS]

        X = df[self.feature_cols].copy()
        # CatBoost categorical cols must be strings; NaN becomes explicit MISSING
        for col in config.CAT_COLS:
            if col in X.columns:
                X[col] = X[col].fillna("MISSING").astype(str)
        return X

    def cat_indices(self, X: pd.DataFrame) -> list[int]:
        """Column indices of categorical features in X."""
        return [X.columns.get_loc(c) for c in config.CAT_COLS if c in X.columns]

    def fit(self, df: pd.DataFrame) -> None:
        if "label" not in df.columns:
            raise ValueError("fit() requires a 'label' column")

        X = self.prepare_features(df, fitting=True)
        y = df["label"].values
        # CatBoost wants per-row group identifier, not per-group sizes
        group_id = df["user_id"].values

        train_pool = Pool(
            data=X,
            label=y,
            group_id=group_id,
            cat_features=self.cat_indices(X),
        )

        self.model = CatBoostRanker(
            loss_function=config.CATBOOST_LOSS,
            iterations=config.CATBOOST_ITER,
            learning_rate=config.CATBOOST_LR,
            depth=config.CATBOOST_DEPTH,
            random_seed=config.RANDOM_SEED,
            verbose=50,
            thread_count= -1
        )
        self.model.fit(train_pool)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Score each row. df does not need a 'label' column."""
        if self.model is None:
            raise RuntimeError("model not fitted; call fit() or load() first")

        X = self.prepare_features(df, fitting=False)
        # Pool at predict time too so cat_features are interpreted correctly
        pool = Pool(data=X, cat_features=self.cat_indices(X))
        return self.model.predict(pool)

    def save(self, path) -> None:
        """Native .cbm for the model, .meta.json for the feature schema."""
        self.model.save_model(str(path))
        meta_path = path.with_suffix(".meta.json")
        meta_path.write_text(
            json.dumps({"feature_cols": self.feature_cols}), encoding="utf-8"
        )

    def load(self, path) -> None:
        self.model = CatBoostRanker()
        self.model.load_model(str(path))
        meta_path = path.with_suffix(".meta.json")
        if meta_path.exists():
            self.feature_cols = json.loads(
                meta_path.read_text(encoding="utf-8")
            )["feature_cols"]