from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

import config


def build_item_item_similarity(train_df: pd.DataFrame):
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

    item_user = normalize(user_item.T, norm="l2", axis=1)
    item_item = (item_user @ item_user.T).tocsr()
    item_item.setdiag(0)
    item_item.eliminate_zeros()

    return item_item, item_to_idx


def _time_decay_weights(
    history_with_time: list[tuple[int, int]],
    item_to_idx: dict[int, int],
    tau: float,
) -> tuple[list[int], np.ndarray]:
    if not history_with_time:
        return [], np.array([], dtype=np.float32)

    t_last = history_with_time[-1][1]
    indices: list[int] = []
    weights: list[float] = []
    for item_id, ts in history_with_time:
        if item_id not in item_to_idx:
            continue
        indices.append(item_to_idx[item_id])
        weights.append(float(np.exp(-(t_last - ts) / tau)))

    return indices, np.asarray(weights, dtype=np.float32)


def _weighted_cooc_vector(
    history_with_time: list[tuple[int, int]],
    item_item,
    item_to_idx: dict[int, int],
    tau: float,
) -> np.ndarray | None:
    history_idxs, weights = _time_decay_weights(history_with_time, item_to_idx, tau)
    if not history_idxs:
        return None

    n_items = item_item.shape[0]
    vec = np.zeros(n_items, dtype=np.float64)
    for idx, w in zip(history_idxs, weights):
        vec += w * np.asarray(item_item[idx].todense()).ravel()
    return vec


def cooccurrence_scores_for_user(
    history_with_time: list[tuple[int, int]],
    candidates: list[int],
    item_item,
    item_to_idx: dict[int, int],
    tau: float,
) -> np.ndarray:
    vec = _weighted_cooc_vector(history_with_time, item_item, item_to_idx, tau)
    if vec is None:
        return np.full(len(candidates), np.nan, dtype=np.float32)

    scores = np.full(len(candidates), np.nan, dtype=np.float32)
    for k, item_id in enumerate(candidates):
        if item_id in item_to_idx:
            scores[k] = float(vec[item_to_idx[item_id]])
    return scores


def cooccurrence_recommend(
    history_with_time: list[tuple[int, int]],
    seen: set[int],
    k: int,
    item_item,
    item_to_idx: dict[int, int],
    tau: float,
) -> list[int]:
    vec = _weighted_cooc_vector(history_with_time, item_item, item_to_idx, tau)
    if vec is None:
        return []

    idx_to_item = {i: it for it, i in item_to_idx.items()}

    for item_id in seen:
        if item_id in item_to_idx:
            vec[item_to_idx[item_id]] = -np.inf

    ranked_indices = np.argsort(-vec)
    recommendations: list[int] = []
    for idx in ranked_indices:
        if len(recommendations) >= k:
            break
        if vec[idx] == -np.inf:
            continue
        recommendations.append(idx_to_item[int(idx)])
    return recommendations
