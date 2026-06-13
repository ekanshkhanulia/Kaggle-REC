"""Train BPR/TF-IDF, build reranker data, evaluate, submit."""

from __future__ import annotations

import argparse

import pandas as pd

import config
from src.features import load_all
from src.metrics import candidate_recall, recall_at_k
from src.reranking.catboost_ranker import CatBoostReranker
from src.reranking.cooccurrence import build_item_item_similarity, cooccurrence_recommend
from src.reranking.data_prep import *
from src.reranking.lgbm import LightGBMReranker
from src.retrieval.collaborative.bpr_matrix import BPRRetriever
from src.retrieval.content_based.bm25 import BM25Retriever
from src.retrieval.content_based.tfidf_content import TfidfRetriever


def clear_bpr_artifacts(model_path, checkpoint_path) -> None:
    paths = [
        model_path,
        model_path.with_suffix(".meta.json"),
        model_path.with_suffix(".loss.json"),
        checkpoint_path,
        checkpoint_path.with_suffix(".meta.json"),
    ]
    for path in paths:
        if path.exists():
            path.unlink()


def run_train_bpr(data: dict, full: bool = False, force: bool = False) -> None:
    bpr = BPRRetriever()
    if full:
        model_path = config.BPR_FULL_MODEL_PATH
        checkpoint_path = config.BPR_FULL_CHECKPOINT_PATH
        train_df = data["train"]
        print("Mode train_bpr --full")
        print(f"Training rows {len(train_df)}")
    else:
        model_path = config.BPR_MODEL_PATH
        checkpoint_path = config.BPR_CHECKPOINT_PATH
        train_df = data["train_split"]
        print("Mode train_bpr")
        print(f"Training rows {len(train_df)}")

    if force:
        print("Force retrain")
        clear_bpr_artifacts(model_path, checkpoint_path)

    bpr.fit(
        train_df,
        resume=not force,
        model_path=model_path,
        checkpoint_path=checkpoint_path,
    )
    print(f"Saved {model_path}")


def load_content_retriever(item_text, content: str = "tfidf"):
    if content == "bm25":
        retriever = BM25Retriever()
        retriever.fit(item_text, resume=True)
    else:
        retriever = TfidfRetriever()
        retriever.fit(item_text, resume=True)
    return retriever


def run_train_tfidf(data: dict, force: bool = False) -> None:
    print("Mode train_tfidf")
    print(f"Items {len(data['item_text'])}")
    if force and config.TFIDF_MODEL_PATH.exists():
        print("Force retrain")
        config.TFIDF_MODEL_PATH.unlink()
    tfidf = TfidfRetriever()
    tfidf.fit(data["item_text"], resume=not force)
    print(f"Saved {config.TFIDF_MODEL_PATH}")


def run_train_reranker(
    data: dict,
    reranker_name: str,
    content: str = "tfidf",
    force: bool = False,
    cooc_tau: float = config.COOC_TAU,
) -> None:
    print(f"Mode train_reranker {reranker_name}")

    train_cache_path, _ = reranker_cache_paths(content)
    train_df = None if force else load_df(train_cache_path)

    if train_df is None:
        bpr = BPRRetriever()
        bpr.fit(
            data["train_split"],
            resume=True,
            model_path=config.BPR_MODEL_PATH,
            checkpoint_path=config.BPR_CHECKPOINT_PATH,
        )
        content_model = load_content_retriever(data["item_text"], content=content)
        item_feats = build_item_features(data["meta"], data["train_split"])
        user_feats = build_user_features(data["train_split"], item_feats)

        print("Building co-occurrence matrix...")
        item_item_sim, item_to_idx_cooc = build_item_item_similarity(data["train_split"])

        print("Building reranker training data...")
        train_df = build_reranker_dataset(
            users=list(data["val_targets"].keys()),
            bpr=bpr,
            tfidf=content_model,
            seen_items=data["val_seen_items"],
            user_timed_histories=data["val_timed_histories"],
            item_features=item_feats,
            user_features=user_feats,
            item_item_sim=item_item_sim,
            item_to_idx_cooc=item_to_idx_cooc,
            labels=data["val_targets"],
            k_retrieval=config.K_RETRIEVAL,
            cooc_tau=cooc_tau,
        )
        save_df(train_df, train_cache_path)

    print(f"Training rows {len(train_df)}  positives {train_df['label'].sum()}")

    users_with_positives = (
        train_df.groupby("user_id")["label"].max().pipe(lambda s: s[s == 1].index).tolist()
    )
    print(
        f"Users with positive {len(users_with_positives):,} / "
        f"{train_df['user_id'].nunique():,}"
    )
    train_df = train_df[train_df["user_id"].isin(users_with_positives)].reset_index(drop=True)
    print(f"Rows after filter {len(train_df):,}")

    if reranker_name == "lgbm":
        reranker = LightGBMReranker()
        model_path = config.LGBM_MODEL_PATH
    elif reranker_name == "catboost":
        reranker = CatBoostReranker()
        model_path = config.CATBOOST_MODEL_PATH
    else:
        raise ValueError(f"Unknown reranker {reranker_name}")

    if force:
        for path in (
            model_path,
            model_path.with_suffix(".meta.json"),
            config.CATBOOST_SNAPSHOT_PATH,
        ):
            if path.exists():
                path.unlink()

    if reranker_name == "catboost":
        reranker.fit(train_df, resume=not force, model_path=model_path)
    else:
        reranker.fit(train_df)
        reranker.save(model_path)
        print(f"Saved {model_path}")


def get_inference_df(
    data: dict,
    content: str = "tfidf",
    force: bool = False,
    cooc_tau: float = config.COOC_TAU,
):
    _, infer_cache_path = reranker_cache_paths(content)
    if not force:
        df = load_df(infer_cache_path)
        if df is not None:
            return df

    print(f"Loading retrievers full BPR + {content}")
    bpr = BPRRetriever()
    bpr.fit(
        data["train"],
        resume=True,
        model_path=config.BPR_FULL_MODEL_PATH,
        checkpoint_path=config.BPR_FULL_CHECKPOINT_PATH,
    )
    content_model = load_content_retriever(data["item_text"], content=content)

    print("Building feature lookups...")
    item_feats = build_item_features(data["meta"], data["train"])
    user_feats = build_user_features(data["train"], item_feats)

    print("Building co-occurrence matrix...")
    item_item_sim, item_to_idx_cooc = build_item_item_similarity(data["train"])

    print(f"Building inference data {len(data['test_users']):,} users")
    df = build_reranker_dataset(
        users=data["test_users"],
        bpr=bpr,
        tfidf=content_model,
        seen_items=data["seen_items"],
        user_timed_histories=data["timed_histories"],
        item_features=item_feats,
        user_features=user_feats,
        item_item_sim=item_item_sim,
        item_to_idx_cooc=item_to_idx_cooc,
        labels=None,
        k_retrieval=config.K_RETRIEVAL,
        cooc_tau=cooc_tau,
    )
    save_df(df, infer_cache_path)
    return df


def top10_per_user(df, scores) -> dict[int, list[int]]:
    scored = df[["user_id", "item_id"]].copy()
    scored["score"] = scores
    scored = scored.sort_values(["user_id", "score"], ascending=[True, False])
    return scored.groupby("user_id", sort=False)["item_id"].apply(
        lambda s: s.head(10).tolist()
    ).to_dict()


def run_evaluate(
    data: dict,
    reranker_name: str,
    content: str = "tfidf",
    force: bool = False,
    cooc_tau: float = config.COOC_TAU,
) -> None:
    print(f"Mode evaluate {reranker_name}")

    df = get_inference_df(data, content=content, force=force, cooc_tau=cooc_tau)
    print(f"Inference df {len(df):,} rows  {df['user_id'].nunique():,} users")

    if reranker_name == "lgbm":
        reranker = LightGBMReranker()
        reranker.load(config.LGBM_MODEL_PATH)
    elif reranker_name == "catboost":
        reranker = CatBoostReranker()
        reranker.load(config.CATBOOST_MODEL_PATH)
    else:
        raise ValueError(f"Unknown reranker {reranker_name}")

    scores = reranker.predict(df)
    top10 = top10_per_user(df, scores)
    recall = recall_at_k(top10, data["test_targets"], k=config.TOP_K)
    print(f"Recall@{config.TOP_K} {recall:.4f}")


def run_compare_baselines(data: dict, reranker_name: str, content: str = "tfidf") -> None:
    print("Mode compare_baselines")

    popular = data["train"][config.COL_ITEM_ID].value_counts().head(50).index.tolist()
    pop_recs = {}
    for u in data["test_users"]:
        seen = data["seen_items"].get(u, set())
        pop_recs[u] = [i for i in popular if i not in seen][:10]
    pop_recall = recall_at_k(pop_recs, data["test_targets"], k=10)

    bpr = BPRRetriever()
    bpr.fit(
        data["train"],
        resume=True,
        model_path=config.BPR_FULL_MODEL_PATH,
        checkpoint_path=config.BPR_FULL_CHECKPOINT_PATH,
    )
    bpr_recs = bpr.recommend_all(data["test_users"], data["seen_items"], k=10)
    bpr_recall = recall_at_k(bpr_recs, data["test_targets"], k=10)

    content_model = load_content_retriever(data["item_text"], content=content)
    content_recs = content_model.recommend_all(data["test_users"], data["seen_items"], k=10)
    content_recall = recall_at_k(content_recs, data["test_targets"], k=10)
    content_label = "BM25 alone" if content == "bm25" else "TF-IDF alone"

    _, infer_cache_path = reranker_cache_paths(content)
    df = pd.read_parquet(infer_cache_path)
    if reranker_name == "lgbm":
        reranker = LightGBMReranker()
        reranker.load(config.LGBM_MODEL_PATH)
    elif reranker_name == "catboost":
        reranker = CatBoostReranker()
        reranker.load(config.CATBOOST_MODEL_PATH)

    scores = reranker.predict(df)
    scored = df[["user_id", "item_id"]].assign(score=scores)
    scored = scored.sort_values(["user_id", "score"], ascending=[True, False])
    pipeline_recs = scored.groupby("user_id", sort=False)["item_id"].apply(
        lambda s: s.head(10).tolist()
    ).to_dict()
    pipeline_recall = recall_at_k(pipeline_recs, data["test_targets"], k=10)

    print(f"{'Method':30s} {'Recall@10':>10s}")
    print(f"{'Popularity':30s} {pop_recall:>10.4f}")
    print(f"{content_label:30s} {content_recall:>10.4f}")
    print(f"{'BPR alone':30s} {bpr_recall:>10.4f}")
    print(f"{f'Pipeline ({reranker_name})':30s} {pipeline_recall:>10.4f}")


def run_candidate_recall(
    data: dict,
    content: str = "tfidf",
    k: int = 100,
    cooc_tau: float = config.COOC_TAU,
) -> None:
    print(f"Mode candidate_recall k={k} tau={cooc_tau}")

    bpr = BPRRetriever()
    bpr.fit(
        data["train"],
        resume=True,
        model_path=config.BPR_FULL_MODEL_PATH,
        checkpoint_path=config.BPR_FULL_CHECKPOINT_PATH,
    )
    content_model = load_content_retriever(data["item_text"], content=content)
    content_label = "BM25" if content == "bm25" else "TF-IDF"

    print("Building co-occurrence matrix...")
    item_item_sim, item_to_idx_cooc = build_item_item_similarity(data["train"])

    bpr_pools: dict[int, set[int]] = {}
    content_pools: dict[int, set[int]] = {}
    cooc_pools: dict[int, set[int]] = {}
    union_no_cooc: dict[int, set[int]] = {}
    union_with_cooc: dict[int, set[int]] = {}

    n_users = len(data["test_users"])
    for i, user_id in enumerate(data["test_users"]):
        if i > 0 and i % 500 == 0:
            print(f"  {i}/{n_users} users")
        seen = data["seen_items"].get(user_id, set())
        history_with_time = data["timed_histories"].get(user_id, [])

        bpr_list = bpr.recommend(user_id, seen, k=k)
        content_list = content_model.recommend(user_id, seen, k=k)
        cooc_list = cooccurrence_recommend(
            history_with_time,
            seen,
            k=k,
            item_item=item_item_sim,
            item_to_idx=item_to_idx_cooc,
            tau=cooc_tau,
        )

        bpr_pools[user_id] = set(bpr_list)
        content_pools[user_id] = set(content_list)
        cooc_pools[user_id] = set(cooc_list)
        union_no_cooc[user_id] = bpr_pools[user_id] | content_pools[user_id]
        union_with_cooc[user_id] = union_no_cooc[user_id] | cooc_pools[user_id]

    targets = data["test_targets"]
    bpr_cr = candidate_recall(bpr_pools, targets)
    content_cr = candidate_recall(content_pools, targets)
    cooc_cr = candidate_recall(cooc_pools, targets)
    union_old_cr = candidate_recall(union_no_cooc, targets)
    union_new_cr = candidate_recall(union_with_cooc, targets)

    print(f"\n{'Pool':35s} {'Recall':>10s}")
    print(f"{f'BPR top-{k}':35s} {bpr_cr:>10.4f}")
    print(f"{f'{content_label} top-{k}':35s} {content_cr:>10.4f}")
    print(f"{f'Co-occ top-{k}':35s} {cooc_cr:>10.4f}")
    print(f"{f'Union BPR+{content_label} v1':35s} {union_old_cr:>10.4f}")
    print(f"{f'Union all v2':35s} {union_new_cr:>10.4f}")

    baseline = config.COOC_BASELINE_UNION_RECALL_AT_500 if k == config.K_RETRIEVAL else None
    if baseline is not None:
        delta = union_new_cr - baseline
        print(f"\nBaseline v1 {baseline:.4f}  delta {delta:+.4f}")
        if union_new_cr > baseline + 0.002:
            print("Retrain CatBoost with --force")
        else:
            print("Skip retrain unless delta grows")

    if k != config.K_RETRIEVAL:
        print(f"\nPipeline uses K_RETRIEVAL={config.K_RETRIEVAL}")


def run_submit(
    data: dict,
    reranker_name: str,
    content: str = "tfidf",
    force: bool = False,
    cooc_tau: float = config.COOC_TAU,
) -> None:
    print(f"Mode submit {reranker_name}")

    df = get_inference_df(data, content=content, force=force, cooc_tau=cooc_tau)

    if reranker_name == "lgbm":
        reranker = LightGBMReranker()
        reranker.load(config.LGBM_MODEL_PATH)
    elif reranker_name == "catboost":
        reranker = CatBoostReranker()
        reranker.load(config.CATBOOST_MODEL_PATH)
    else:
        raise ValueError(f"Unknown reranker {reranker_name}")

    scores = reranker.predict(df)
    top10 = top10_per_user(df, scores)

    rows = []
    for user_id in data["test_users"]:
        items = top10.get(user_id, [])
        if len(items) != 10:
            raise ValueError(f"User {user_id} has {len(items)} items, expected 10")
        if len(set(items)) != 10:
            raise ValueError(f"User {user_id} duplicate items {items}")
        rows.append({
            "ID": user_id,
            "user_id": user_id,
            "item_id": ",".join(str(i) for i in items[:10]),
        })

    sub = pd.DataFrame(rows)
    sub.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Wrote {config.SUBMISSION_PATH}  {len(sub):,} users")


def reranker_cache_paths(content: str):
    if content == "bm25":
        train_path = config.ARTIFACTS_DIR / "reranker_train_df_bm25.parquet"
        infer_path = config.ARTIFACTS_DIR / "reranker_inference_df_bm25.parquet"
    else:
        train_path = config.RERANKER_TRAIN_DF_PATH
        infer_path = config.RERANKER_INFERENCE_DF_PATH
    return train_path, infer_path


def run_train_bm25(data: dict, force: bool = False) -> None:
    print("Mode train_bm25")
    print(f"Items {len(data['item_text'])}")
    if force and config.BM25_MODEL_PATH.exists():
        print("Force retrain")
        config.BM25_MODEL_PATH.unlink()
    bm25 = BM25Retriever()
    bm25.fit(data["item_text"], resume=not force)
    print(f"Saved {config.BM25_MODEL_PATH}")


def ensemble_top10(
    data: dict,
    content: str,
    lgbm_weight: float,
    catboost_weight: float,
) -> dict[int, list[int]]:
    df = get_inference_df(data, content=content, force=False)
    print(f"Inference df {len(df):,} rows  {df['user_id'].nunique():,} users")

    lgbm = LightGBMReranker()
    lgbm.load(config.LGBM_MODEL_PATH)
    cat = CatBoostReranker()
    cat.load(config.CATBOOST_MODEL_PATH)

    lgbm_scores = lgbm.predict(df)
    cat_scores = cat.predict(df)

    scored = df[["user_id", "item_id"]].copy()
    scored["lgbm"] = lgbm_scores
    scored["catboost"] = cat_scores

    def _per_user_minmax(g: pd.Series) -> pd.Series:
        lo, hi = g.min(), g.max()
        if hi == lo:
            return g * 0.0
        return (g - lo) / (hi - lo)

    scored["lgbm_norm"] = scored.groupby("user_id")["lgbm"].transform(_per_user_minmax)
    scored["catboost_norm"] = scored.groupby("user_id")["catboost"].transform(_per_user_minmax)
    scored["combined"] = lgbm_weight * scored["lgbm_norm"] + catboost_weight * scored["catboost_norm"]

    scored = scored.sort_values(["user_id", "combined"], ascending=[True, False])
    return scored.groupby("user_id", sort=False)["item_id"].apply(
        lambda s: s.head(10).tolist()
    ).to_dict()


def run_ensemble_evaluate(data: dict, content: str = "tfidf", lgbm_weight: float = 0.5) -> None:
    catboost_weight = 1.0 - lgbm_weight
    print(f"Mode ensemble_evaluate lgbm={lgbm_weight} catboost={catboost_weight}")

    top10 = ensemble_top10(data, content, lgbm_weight, catboost_weight)
    recall = recall_at_k(top10, data["test_targets"], k=config.TOP_K)
    print(f"Recall@{config.TOP_K} {recall:.4f}")


def run_ensemble_submit(data: dict, content: str = "tfidf", lgbm_weight: float = 0.5) -> None:
    catboost_weight = 1.0 - lgbm_weight
    print(f"Mode ensemble_submit lgbm={lgbm_weight} catboost={catboost_weight}")

    top10 = ensemble_top10(data, content, lgbm_weight, catboost_weight)

    rows = []
    for user_id in data["test_users"]:
        items = top10.get(user_id, [])
        if len(items) != 10:
            raise ValueError(f"User {user_id} has {len(items)} items, expected 10")
        if len(set(items)) != 10:
            raise ValueError(f"User {user_id} duplicate items {items}")
        rows.append({
            "ID": user_id,
            "user_id": user_id,
            "item_id": ",".join(str(i) for i in items),
        })

    sub = pd.DataFrame(rows)
    sub.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Wrote {config.SUBMISSION_PATH}  {len(sub):,} users")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid BPR + TF-IDF pipeline")
    parser.add_argument(
        "--mode",
        choices=[
            "train_bpr", "train_tfidf", "train_bm25",
            "train_reranker", "evaluate", "compare_baselines", "candidate_recall", "submit",
            "ensemble_evaluate", "ensemble_submit",
        ],
        default="train_bpr",
    )
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--reranker", choices=["lgbm", "catboost"], default="lgbm")
    parser.add_argument("--content", choices=["tfidf", "bm25"], default="tfidf")
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--cooc-tau", type=float, default=config.COOC_TAU)
    parser.add_argument("--lgbm-weight", type=float, default=0.5)
    args = parser.parse_args()

    print("Loading data")
    data = load_all()
    print(f"Mode {args.mode}")

    if args.mode == "train_bpr":
        run_train_bpr(data, full=args.full, force=args.force)
    elif args.mode == "train_tfidf":
        run_train_tfidf(data, force=args.force)
    elif args.mode == "train_bm25":
        run_train_bm25(data, force=args.force)
    elif args.mode == "train_reranker":
        run_train_reranker(
            data, reranker_name=args.reranker, content=args.content,
            force=args.force, cooc_tau=args.cooc_tau,
        )
    elif args.mode == "evaluate":
        run_evaluate(
            data, reranker_name=args.reranker, content=args.content,
            force=args.force, cooc_tau=args.cooc_tau,
        )
    elif args.mode == "compare_baselines":
        run_compare_baselines(data, args.reranker, content=args.content)
    elif args.mode == "candidate_recall":
        run_candidate_recall(data, content=args.content, k=args.candidate_k, cooc_tau=args.cooc_tau)
    elif args.mode == "submit":
        run_submit(
            data, reranker_name=args.reranker, content=args.content,
            force=args.force, cooc_tau=args.cooc_tau,
        )
    elif args.mode == "ensemble_evaluate":
        run_ensemble_evaluate(data, content=args.content, lgbm_weight=args.lgbm_weight)
    elif args.mode == "ensemble_submit":
        run_ensemble_submit(data, content=args.content, lgbm_weight=args.lgbm_weight)


if __name__ == "__main__":
    main()
