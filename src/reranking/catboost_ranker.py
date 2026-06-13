"""CatBoost-based reranker."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool

import config


class CatBoostReranker:
    def __init__(self):
        self.model = None
        self.feature_cols: list[str] = []

    def prepare_features(self, df: pd.DataFrame, fitting: bool) -> pd.DataFrame:
        if fitting:
            self.feature_cols = [c for c in df.columns if c not in config.META_COLS]

        X = df[self.feature_cols].copy()
        for col in config.CAT_COLS:
            if col in X.columns:
                X[col] = X[col].fillna("MISSING").astype(str)
        return X

    def cat_indices(self, X: pd.DataFrame) -> list[int]:
        return [X.columns.get_loc(c) for c in config.CAT_COLS if c in X.columns]

    def fit(
        self,
        df: pd.DataFrame,
        resume: bool = True,
        model_path: Path | None = None,
    ) -> None:
        if "label" not in df.columns:
            raise ValueError("fit() requires a 'label' column")

        model_path = Path(model_path or config.CATBOOST_MODEL_PATH)
        snapshot_path = config.CATBOOST_SNAPSHOT_PATH

        if resume and model_path.exists():
            print(f"Loading saved CatBoost from {model_path}")
            self.load(model_path)
            return

        X = self.prepare_features(df, fitting=True)
        y = df["label"].values
        group_id = df["user_id"].values
        train_pool = Pool(
            data=X,
            label=y,
            group_id=group_id,
            cat_features=self.cat_indices(X),
        )

        if resume and snapshot_path.exists():
            print(f"Resuming CatBoost from snapshot ({snapshot_path.name})")

        config.CATBOOST_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
        self.model = CatBoostRanker(
            loss_function=config.CATBOOST_LOSS,
            iterations=config.CATBOOST_ITER,
            learning_rate=config.CATBOOST_LR,
            depth=config.CATBOOST_DEPTH,
            random_seed=config.RANDOM_SEED,
            verbose=50,
            thread_count=-1,
            train_dir=str(config.CATBOOST_TRAIN_DIR),
        )
        self.model.fit(
            train_pool,
            save_snapshot=True,
            snapshot_file=config.CATBOOST_SNAPSHOT_FILE,
            snapshot_interval=config.CATBOOST_SNAPSHOT_INTERVAL_SEC,
        )

        self.save(model_path)
        snapshot_path.unlink(missing_ok=True)
        print(f"CatBoost training done, saved {model_path}")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("model not fitted; call fit() or load() first")

        X = self.prepare_features(df, fitting=False)
        pool = Pool(data=X, cat_features=self.cat_indices(X))
        return self.model.predict(pool)

    def save(self, path) -> None:
        self.model.save_model(str(path))
        meta_path = Path(path).with_suffix(".meta.json")
        meta_path.write_text(
            json.dumps({"feature_cols": self.feature_cols}),
            encoding="utf-8",
        )

    def load(self, path) -> None:
        self.model = CatBoostRanker()
        self.model.load_model(str(path))
        meta_path = Path(path).with_suffix(".meta.json")
        if meta_path.exists():
            self.feature_cols = json.loads(
                meta_path.read_text(encoding="utf-8")
            )["feature_cols"]
