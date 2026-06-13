"""
Build training/inference dataset for the reranker.

For each user:
  1. Retrieve top-K candidates from BPR and TF-IDF
  2. Take the union of both
  3. Score every candidate with both retrievers (so candidates from one retriever still get the other's score as a feature)
  4. Assemble a feature vector per (user, candidate) pair

In terms of feature engineering, the features are grouped into:
  - User features: from user's interaction history
  - Item features: properties of the item alone (popularity + metadata)
  - User-item: retrieval scores/ranks + cross features

This is inspired by the general pattern used in recent competition-winning 
learning-to-rank recommender solutions, including the RecSys Challenge 2024 entry by
team FeatureSalad (Yang et al., 2024)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src.retrieval.candidates import retrieve_candidates_for_user

def _compact_reranker_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    float_cols = [
        "bpr_score", "tfidf_score", "cooc_score",
        "item_popularity", "average_rating", "log_rating_number", "price",
        "user_interaction_count", "user_avg_price", "user_avg_item_rating",
        "user_avg_log_rating_number",
        "price_diff_from_user_avg", "rating_diff_from_user_avg",
        "rating_number_diff_from_user_avg",
    ]
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].astype(np.float32)
    int_cols = {
        "user_id": np.int32,
        "item_id": np.int32,
        "bpr_rank": np.int16,
        "tfidf_rank": np.int16,
        "cooc_rank": np.int16,
        "category_match": np.int8,
        "store_match": np.int8,
        "user_category_diversity": np.int16,
        "user_store_diversity": np.int16,
    }
    for col, dtype in int_cols.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)
    if "label" in df.columns:
        df["label"] = df["label"].astype(np.int8)
    return df


def _finalize_reranker_chunk(
    rows: list[dict],
    item_features: pd.DataFrame,
    user_features: pd.DataFrame,
    label_pairs: set[tuple[int, int]] | None,
) -> pd.DataFrame | None:
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df = df.merge(item_features, left_on="item_id", right_index=True, how="left")
    df = df.merge(user_features, left_on="user_id", right_index=True, how="left")
    df["category_match"] = (df["main_category"] == df["user_dominant_category"]).astype(int)
    df["store_match"] = (df["store"] == df["user_dominant_store"]).astype(int)
    df["price_diff_from_user_avg"] = (df["price"] - df["user_avg_price"]).abs()
    df["rating_diff_from_user_avg"] = (df["average_rating"] - df["user_avg_item_rating"]).abs()
    df["rating_number_diff_from_user_avg"] = (
        df["log_rating_number"] - df["user_avg_log_rating_number"]
    ).abs()
    if label_pairs is not None:
        pairs = zip(df["user_id"].values, df["item_id"].values)
        df["label"] = np.fromiter(
            (1 if p in label_pairs else 0 for p in pairs),
            dtype=np.int8,
            count=len(df),
        )
    return _compact_reranker_dtypes(df)

def build_item_features(meta: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-item features, computed once per dataset:
    - item_popularity
    - main_category
    - average_rating
    - log_rating_number
    - price 
    """
    pop_counts = train_df[config.COL_ITEM_ID].value_counts()
    all_items = sorted(set(train_df[config.COL_ITEM_ID].unique()) | set(meta[config.COL_ITEM_ID].unique()))
    item_feats = meta.set_index(config.COL_ITEM_ID).reindex(all_items)
    item_feats["item_popularity"] = np.log1p(item_feats.index.map(pop_counts).fillna(0).astype(float))
    item_feats["log_rating_number"] = np.log1p(item_feats["rating_number"])

    keep = [
        "item_popularity",
        "main_category",
        "store",
        "average_rating",
        "log_rating_number",
        "price",
    ]

    return item_feats[keep]

def build_user_features(train_df: pd.DataFrame, item_features: pd.DataFrame) -> pd.DataFrame:
    """
    Per-user features from interaction history, computed once per dataset:
    - user_interaction_count
    - user_avg_price
    - user_avg_item_rating
    - user_category_diversity
    - user_dominant_category
    """
    train_aug = train_df.merge(
        item_features[[
            "main_category",
            "store",
            "average_rating",
            "log_rating_number",
            "price",
        ]],
        left_on=config.COL_ITEM_ID,
        right_index=True,
        how="left",
    )

    grouped = train_aug.groupby(config.COL_USER_ID)

    user_feats = pd.DataFrame(index=grouped.size().index)
    user_feats["user_interaction_count"] = np.log1p(grouped.size())
    user_feats["user_avg_price"] = grouped["price"].mean()
    user_feats["user_avg_item_rating"] = grouped["average_rating"].mean()
    user_feats["user_avg_log_rating_number"] = grouped["log_rating_number"].mean()
    user_feats["user_category_diversity"] = grouped["main_category"].nunique()
    user_feats["user_store_diversity"] = grouped["store"].nunique()

    def _mode(s: pd.Series) -> str:
        m = s.mode(dropna=True)
        return m.iloc[0] if not m.empty else "UNKNOWN"

    user_feats["user_dominant_category"] = grouped["main_category"].agg(_mode)
    user_feats["user_dominant_store"] = grouped["store"].agg(_mode)
    

    return user_feats

def build_reranker_dataset(
    users: list[int],
    bpr,
    tfidf,
    seen_items: dict[int, set[int]],
    user_timed_histories: dict[int, list[tuple[int, int]]],
    item_features: pd.DataFrame,
    user_features: pd.DataFrame,
    item_item_sim,
    item_to_idx_cooc,
    labels: dict[int, list[int]] | None = None,
    k_retrieval: int = 200,
    cooc_tau: float = config.COOC_TAU,
) -> pd.DataFrame:
    """
    Build the (user, candidate) feature dataframe for the reranker. 
    For each user: union top-K from BPR, TF-IDF, and time-decay co-occurrence,
    score with all retrievers, attach item/user/cross features, optionally label.
    Pass labels=val_targets to train the reranker; pass labels=None at inference.
    Output is sorted by user_id (required for ranker group= parameter).
    """
    users = sorted(users)
    n_users = len(users)
    chunk_users = 1500

    label_pairs: set[tuple[int, int]] | None = None
    if labels is not None:
        label_pairs = set()
        for u, items in labels.items():
            for it in items:
                label_pairs.add((u, it))

    batch_rows: list[dict] = []
    chunk_dfs: list[pd.DataFrame] = []

    for i, user_id in enumerate(users):
        if i > 0 and i % 1000 == 0:
            print(f"  building reranker dataset: {i}/{n_users} users")

        user_seen = seen_items.get(user_id, set())
        history_with_time = user_timed_histories.get(user_id, [])

        pack = retrieve_candidates_for_user(
            user_id,
            user_seen,
            history_with_time,
            bpr,
            tfidf,
            item_item_sim,
            item_to_idx_cooc,
            k_retrieval=k_retrieval,
            cooc_tau=cooc_tau,
        )
        if not pack["rows"]:
            continue
        batch_rows.extend(pack["rows"])

        at_chunk_end = (i + 1) % chunk_users == 0 or i == n_users - 1
        if at_chunk_end and batch_rows:
            chunk_df = _finalize_reranker_chunk(
                batch_rows, item_features, user_features, label_pairs
            )
            if chunk_df is not None:
                chunk_dfs.append(chunk_df)
            batch_rows = []

    if not chunk_dfs:
        return pd.DataFrame()

    if len(chunk_dfs) == 1:
        return chunk_dfs[0]

    return pd.concat(chunk_dfs, ignore_index=True)


def save_df(df: pd.DataFrame, path) -> None:
    """Save a dataframe to parquet, creating parent dirs if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"Cached to {path}  ({len(df):,} rows)")


def load_df(path) -> pd.DataFrame | None:
    """Return cached dataframe if it exists, else None."""
    if not path.exists():
        return None
    print(f"Loading cached dataframe from {path}")
    return pd.read_parquet(path)