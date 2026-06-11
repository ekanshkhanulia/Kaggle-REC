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
from src.reranking.cooccurrence import cooccurrence_scores_for_user

def build_item_features(meta: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-item features, computed once per dataset:
    - item_popularity
    - main_category
    - average_rating
    - log_rating_number
    - price 
    """
    # popularity from interactions
    pop_counts = train_df[config.COL_ITEM_ID].value_counts()

    # cover every item appearing in either train or meta
    all_items = sorted(set(train_df[config.COL_ITEM_ID].unique()) | set(meta[config.COL_ITEM_ID].unique()))

    # for an item that appears in train but not in meta
    # reindex to include this item by creating a row where every meta column is NaN
    item_feats = meta.set_index(config.COL_ITEM_ID).reindex(all_items)
    # popularity = log(1 + #interactions in train), items only in meta get 0
    item_feats["item_popularity"] = np.log1p(item_feats.index.map(pop_counts).fillna(0).astype(float))
    # log-scale rating_number: missing meta stays NaN, tree handles natively
    item_feats["log_rating_number"] = np.log1p(item_feats["rating_number"])

    keep = ["item_popularity", "main_category", "average_rating", "log_rating_number", "price"]

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
    # attach item features to each interaction
    train_aug = train_df.merge(
        item_features[["main_category", "average_rating", "price"]],
        left_on=config.COL_ITEM_ID,
        right_index=True,
        how="left",
    )

    grouped = train_aug.groupby(config.COL_USER_ID)

    user_feats = pd.DataFrame(index=grouped.size().index)
    user_feats["user_interaction_count"] = np.log1p(grouped.size())
    user_feats["user_avg_price"] = grouped["price"].mean() # NaN if none of their items have price
    user_feats["user_avg_item_rating"] = grouped["average_rating"].mean()
    user_feats["user_category_diversity"] = grouped["main_category"].nunique()

    # most-frequent main_category in user's history (UNKNOWN if all NaN)
    def _mode(s: pd.Series) -> str:
        m = s.mode(dropna=True)
        return m.iloc[0] if not m.empty else "UNKNOWN"

    user_feats["user_dominant_category"] = grouped["main_category"].agg(_mode)

    return user_feats

def build_reranker_dataset(
    users: list[int],
    bpr,
    tfidf,
    seen_items: dict[int, set[int]],
    item_features: pd.DataFrame,
    user_features: pd.DataFrame,
    item_item_sim,
    item_to_idx_cooc,
    labels: dict[int, list[int]] | None = None,
    k_retrieval: int = 200,
) -> pd.DataFrame:
    """
    Build the (user, candidate) feature dataframe for the reranker. 
    For each user: union top-K from BPR and TF-IDF, score with both,
    attach item/user/cross features, optionally label.
    Pass labels=val_targets to train the reranker; pass labels=None at inference.
    Output is sorted by user_id (required for ranker group= parameter).
    """
    all_rows: list[dict] = []

    n_users = len(users)
    for i, user_id in enumerate(users):
        if i > 0 and i % 1000 == 0:
            print(f"  building reranker dataset: {i}/{n_users} users")

        user_seen = seen_items.get(user_id, set())

        # retrieve candidates from each retriever
        bpr_recs = bpr.recommend(user_id, user_seen, k=k_retrieval)
        tfidf_recs = tfidf.recommend(user_id, user_seen, k=k_retrieval)

        # rank lookup tables (item_id -> rank within that retriever)
        bpr_ranks = {item: rank for rank, item in enumerate(bpr_recs)}
        tfidf_ranks = {item: rank for rank, item in enumerate(tfidf_recs)}

        # union of candidates
        candidates = list(set(bpr_recs) | set(tfidf_recs))
        if not candidates:
            continue

        # score every candidate with both retrievers
        # (so candidates from TF-IDF still get a BPR score, and vice versa)
        bpr_scores = bpr.score_candidates(user_id, candidates)
        tfidf_profile = tfidf.build_user_profile(user_seen)
        tfidf_scores = tfidf.score_candidates(tfidf_profile, candidates)

        cooc_scores = cooccurrence_scores_for_user(user_seen, candidates, item_item_sim, item_to_idx_cooc)

        # assemble one row per candidate
        for k, item_id in enumerate(candidates):
            all_rows.append({
                "user_id": user_id,
                "item_id": item_id,
                "bpr_score": float(bpr_scores[k]),
                "tfidf_score": float(tfidf_scores[k]),
                "bpr_rank": bpr_ranks.get(item_id, -1),
                "tfidf_rank": tfidf_ranks.get(item_id, -1),
                "cooc_score": float(cooc_scores[k])
            })

    df = pd.DataFrame(all_rows)

    # merge item-level features (one lookup per row)
    df = df.merge(item_features, left_on="item_id", right_index=True, how="left")

    # merge user-level features
    df = df.merge(user_features, left_on="user_id", right_index=True, how="left")

    # cross features
    df["category_match"] = (df["main_category"] == df["user_dominant_category"]).astype(int)
    df["price_diff_from_user_avg"] = (df["price"] - df["user_avg_price"]).abs()

    # labels (training mode only)
    if labels is not None:
        label_pairs = set()
        for u, items in labels.items():
            for it in items:
                label_pairs.add((u, it))

        pairs = list(zip(df["user_id"].values, df["item_id"].values))
        df["label"] = np.fromiter((1 if p in label_pairs else 0 for p in pairs), dtype=np.int8, count=len(pairs))

    # sort by user_id (ranker training requires contiguous groups)
    df = df.sort_values("user_id").reset_index(drop=True)

    return df

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