"""
item-item cosine similarity over user interaction vectors (Sarwar et al. 2001)

Computes a sparse item-item cosine similarity matrix from training interactions.
Then, for each (user, candidate) pair, the feature is the sum of cosine
similarities between the candidate and the items in the user's history.

This explicit pairwise co-occurrence signal complements BPR's latent representation (Liang et al. 2016).

"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

import config


def build_item_item_similarity(train_df: pd.DataFrame):
    """
    Build sparse item-item cosine similarity matrix from interactions.
    """
    # contiguous indexing
    item_ids = sorted(train_df[config.COL_ITEM_ID].unique())
    user_ids = sorted(train_df[config.COL_USER_ID].unique())
    item_to_idx = {it: i for i, it in enumerate(item_ids)}
    user_to_idx = {u: i for i, u in enumerate(user_ids)}

    rows = train_df[config.COL_USER_ID].map(user_to_idx).values
    cols = train_df[config.COL_ITEM_ID].map(item_to_idx).values
    data = np.ones(len(train_df), dtype=np.float32)

    user_item = csr_matrix(
        (data, (rows, cols)),
        shape=(len(user_ids), len(item_ids)),
    )

    # cosine similarity: L2-normalize item vectors (rows of item_user), then dot
    item_user = normalize(user_item.T, norm="l2", axis=1)
    item_item = (item_user @ item_user.T).tocsr()

    # zero out self-similarity (item to itself)
    item_item.setdiag(0)
    item_item.eliminate_zeros()

    return item_item, item_to_idx


def cooccurrence_scores_for_user(user_history: set[int], candidates: list[int], item_item, item_to_idx: dict) -> np.ndarray:
    """
    For one user, score all candidates by summed cosine similarity to user's history.
    Vectorized: one sparse row-sum + indexing operation per user.
    """
    # collect valid history indices (skip items not in training set)
    history_idxs = [item_to_idx[i] for i in user_history if i in item_to_idx]

    # cold user with no in-training history -> all zero
    if not history_idxs:
        return np.zeros(len(candidates), dtype=np.float32)

    # sum the rows of item_item for items in history
    history_sum = np.asarray(item_item[history_idxs].sum(axis=0)).flatten()

    # look up each candidate's score
    scores = np.zeros(len(candidates), dtype=np.float32)
    for k, c in enumerate(candidates):
        if c in item_to_idx:
            scores[k] = history_sum[item_to_idx[c]]

    return scores