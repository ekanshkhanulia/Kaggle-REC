from __future__ import annotations

import numpy as np

from src.reranking.cooccurrence import cooccurrence_recommend, cooccurrence_scores_for_user


def retrieve_candidates_for_user(
    user_id: int,
    seen: set[int],
    history_with_time: list[tuple[int, int]],
    bpr,
    content_model,
    item_item_sim,
    item_to_idx_cooc,
    k_retrieval: int,
    cooc_tau: float,
) -> dict:
    bpr_recs = bpr.recommend(user_id, seen, k=k_retrieval)
    content_recs = content_model.recommend(user_id, seen, k=k_retrieval)
    cooc_recs = cooccurrence_recommend(
        history_with_time,
        seen,
        k=k_retrieval,
        item_item=item_item_sim,
        item_to_idx=item_to_idx_cooc,
        tau=cooc_tau,
    )

    bpr_ranks = {item: rank for rank, item in enumerate(bpr_recs)}
    content_ranks = {item: rank for rank, item in enumerate(content_recs)}
    cooc_ranks = {item: rank for rank, item in enumerate(cooc_recs)}

    candidates = list(set(bpr_recs) | set(content_recs) | set(cooc_recs))
    if not candidates:
        return {"candidates": [], "rows": []}

    bpr_scores = bpr.score_candidates(user_id, candidates)
    content_profile = content_model.build_user_profile(seen)
    content_scores = content_model.score_candidates(content_profile, candidates)
    cooc_scores = cooccurrence_scores_for_user(
        history_with_time,
        candidates,
        item_item_sim,
        item_to_idx_cooc,
        tau=cooc_tau,
    )

    rows = []
    for k, item_id in enumerate(candidates):
        rows.append({
            "user_id": user_id,
            "item_id": item_id,
            "bpr_score": float(bpr_scores[k]),
            "tfidf_score": float(content_scores[k]),
            "bpr_rank": bpr_ranks.get(item_id, -1),
            "tfidf_rank": content_ranks.get(item_id, -1),
            "cooc_score": float(cooc_scores[k]),
            "cooc_rank": cooc_ranks.get(item_id, -1),
        })

    return {"candidates": candidates, "rows": rows}
