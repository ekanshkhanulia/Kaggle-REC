"""
main.py

train BPR and TF-IDF 

val test submit and reranking go in this

Run from project root:
  python main.py --mode train_bpr
  python main.py --mode train_tfidf
  python main.py --mode train_bpr --full
  
"""

from __future__ import annotations

import argparse

import pandas as pd
import config
from src.features import load_all
from src.retrieval.collaborative.bpr_matrix import BPRRetriever
from src.retrieval.content_based.tfidf_content import TfidfRetriever
from src.reranking.data_prep import *
from src.reranking.lgbm import LightGBMReranker
from src.reranking.catboost_ranker import CatBoostReranker
from src.metrics import recall_at_k
from src.reranking.cooccurrence import build_item_item_similarity

from src.retrieval.content_based.bm25 import BM25Retriever

def clear_bpr_artifacts(model_path, checkpoint_path) -> None:
    # delete old model/checkpoint/loss so retrain starts clean
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
    # default 80% split for val, --full uses entire train.csv
    bpr = BPRRetriever()
    if full:
        model_path = config.BPR_FULL_MODEL_PATH
        checkpoint_path = config.BPR_FULL_CHECKPOINT_PATH
        train_df = data["train"]
        print("Mode: train_bpr --full (final Kaggle model)")
        print(f"Training rows: {len(train_df)}")
    else:
        model_path = config.BPR_MODEL_PATH
        checkpoint_path = config.BPR_CHECKPOINT_PATH
        train_df = data["train_split"]
        print("Mode: train_bpr (dev model, 80% split)")
        print(f"Training rows: {len(train_df)}")

    if force:
        print("Force retrain, replacing old BPR artifacts")
        clear_bpr_artifacts(model_path, checkpoint_path)

    bpr.fit(
        train_df,
        resume=not force,
        model_path=model_path,
        checkpoint_path=checkpoint_path,
    )
    print(f"Done. Saved to {model_path}")


def load_content_retriever(item_text, content: str = "tfidf"):
    # load tf-idf or bm25 from artifacts, same interface for data_prep
    if content == "bm25":
        retriever = BM25Retriever()
        retriever.fit(item_text, resume=True)
    else:
        retriever = TfidfRetriever()
        retriever.fit(item_text, resume=True)
    return retriever


def run_train_tfidf(data: dict, force: bool = False) -> None:
    # train tfidf on item text once
    print("Mode: train_tfidf")
    print(f"Items: {len(data['item_text'])}")
    if force and config.TFIDF_MODEL_PATH.exists():
        print("Force retrain, replacing old TF-IDF artifact")
        config.TFIDF_MODEL_PATH.unlink()
    tfidf = TfidfRetriever()
    tfidf.fit(data["item_text"], resume=not force)
    print(f"Done. TF-IDF saved to {config.TFIDF_MODEL_PATH}")

def run_train_reranker(data: dict, reranker_name: str,content: str = "tfidf", force: bool = False) -> None:
    """Train the reranker on val_users using dev BPR (80% train)."""
    print(f"Mode: train_reranker ({reranker_name})")
    
    train_cache_path, _ = reranker_cache_paths(content)
    train_df = None if force else load_df(train_cache_path)

    # try cache first
    # train_df = None if force else load_df(config.RERANKER_TRAIN_DF_PATH)

    if train_df is None:
        # cache miss -> load fitted retrievers (dev BPR for training) + build dataset
        bpr = BPRRetriever()
        bpr.fit(data["train_split"], resume=True,
                model_path=config.BPR_MODEL_PATH,
                checkpoint_path=config.BPR_CHECKPOINT_PATH)
        
        # tfidf = TfidfRetriever()
        # tfidf.fit(data["item_text"], resume=True)
        content_model = load_content_retriever(data["item_text"], content=content)

        # precompute feature lookups from train_split
        item_feats = build_item_features(data["meta"], data["train_split"])
        user_feats = build_user_features(data["train_split"], item_feats)

        print("Building item-item co-occurrence matrix...")
        item_item_sim, item_to_idx_cooc = build_item_item_similarity(data["train_split"])

        # build training dataset for the reranker
        print("Building reranker training data...")
        train_df = build_reranker_dataset(
            users=list(data["val_targets"].keys()),
            #bpr=bpr, tfidf=tfidf,
            bpr=bpr, tfidf=content_model,
            seen_items=data["val_seen_items"],
            item_features=item_feats,
            user_features=user_feats,
            item_item_sim=item_item_sim,
            item_to_idx_cooc=item_to_idx_cooc,
            labels=data["val_targets"],
            k_retrieval=config.K_RETRIEVAL,
        )
        # save_df(train_df, config.RERANKER_TRAIN_DF_PATH)
        save_df(train_df, train_cache_path)
    
    print(f"Training rows: {len(train_df)}  positives: {train_df['label'].sum()}")

    # Drop user-groups with no positive candidates (groups with no positives provide no useful comparison signal, tested )
    users_with_positives = (train_df.groupby("user_id")["label"].max().pipe(lambda s: s[s == 1].index).tolist())
    print(
        f"Filtering to user-groups with at least 1 positive: "
        f"{len(users_with_positives):,} / {train_df['user_id'].nunique():,} users kept"
    )
    train_df = train_df[train_df["user_id"].isin(users_with_positives)].reset_index(drop=True)
    print(f"Training rows after filter: {len(train_df):,}")

    # train the chosen reranker
    if reranker_name == "lgbm":
        reranker = LightGBMReranker()
        model_path = config.LGBM_MODEL_PATH
    elif reranker_name == "catboost":
        reranker = CatBoostReranker()
        model_path = config.CATBOOST_MODEL_PATH
    else:
        raise ValueError(f"Unknown reranker: {reranker_name}")

    if force and model_path.exists():
        model_path.unlink()

    reranker.fit(train_df)
    reranker.save(model_path)
    print(f"Reranker saved to {model_path}")

def get_inference_df(data: dict, content: str = "tfidf", force: bool = False):
    """
    Build (or load from cache) the reranker inference DataFrame.
    Uses full BPR (100% train), no labels.
    """
    _, infer_cache_path = reranker_cache_paths(content)
    if not force:
        # df = load_df(config.RERANKER_INFERENCE_DF_PATH)
        df = load_df(infer_cache_path)
        if df is not None:
            return df

    print(f"Loading retrievers (full BPR + {content})...")
    bpr = BPRRetriever()
    bpr.fit(data["train"], resume=True,
            model_path=config.BPR_FULL_MODEL_PATH,
            checkpoint_path=config.BPR_FULL_CHECKPOINT_PATH)
    # tfidf = TfidfRetriever()
    # tfidf.fit(data["item_text"], resume=True)
    content_model = load_content_retriever(data["item_text"], content=content)

    print("Building feature lookups on full train...")
    item_feats = build_item_features(data["meta"], data["train"])
    user_feats = build_user_features(data["train"], item_feats)

    print("Building item-item co-occurrence matrix...")
    item_item_sim, item_to_idx_cooc = build_item_item_similarity(data["train"])

    print(f"Building inference dataset for {len(data['test_users']):,} users...")
    df = build_reranker_dataset(
        users=data["test_users"],
        # bpr=bpr, tfidf=tfidf,
        bpr=bpr, tfidf=content_model,
        seen_items=data["seen_items"],     # full-train-based seen
        item_features=item_feats,
        user_features=user_feats,
        item_item_sim=item_item_sim,
        item_to_idx_cooc=item_to_idx_cooc,
        labels=None,                       # no labels at inference
        k_retrieval=config.K_RETRIEVAL,
    )
    # save_df(df, config.RERANKER_INFERENCE_DF_PATH)
    save_df(df, infer_cache_path)

    return df

def top10_per_user(df, scores) -> dict[int, list[int]]:
    """
    Pick top-10 candidates per user, sorted by score desc.
    """
    scored = df[["user_id", "item_id"]].copy()
    scored["score"] = scores
    scored = scored.sort_values(["user_id", "score"], ascending=[True, False])
    top10 = (scored.groupby("user_id", sort=False)["item_id"].apply(lambda s: s.head(10).tolist()).to_dict())

    return top10

def run_evaluate(data: dict, reranker_name: str, content: str = "tfidf", force: bool = False) -> None:
    """
    Local Recall@10 on the test set, using:
      - full BPR (trained on 100% train)
      - TF-IDF
      - trained reranker (lgbm or catboost)
    """

    print(f"Mode: evaluate ({reranker_name})")

    # df = get_inference_df(data, force=force)
    df = get_inference_df(data, content=content, force=force)
    print(f"Inference df: {len(df):,} rows, {df['user_id'].nunique():,} users")

    # load reranker and score
    if reranker_name == "lgbm":
        reranker = LightGBMReranker()
        reranker.load(config.LGBM_MODEL_PATH)
    elif reranker_name == "catboost":
        reranker = CatBoostReranker()
        reranker.load(config.CATBOOST_MODEL_PATH)
    else:
        raise ValueError(f"Unknown reranker: {reranker_name}")

    print("Scoring with reranker...")
    scores = reranker.predict(df)

    print("Picking top-10 per user...")
    top10 = top10_per_user(df, scores)

    recall = recall_at_k(top10, data["test_targets"], k=config.TOP_K)
    print(f"Reranker Recall@{config.TOP_K}: {recall:.4f}")

def run_compare_baselines(data: dict, reranker_name: str, content: str = "tfidf") -> None:
    """
    Compare on test_targets:
      - Popularity (most-interacted items)
      - TF-IDF alone
      - BPR alone (full)
      - Pipeline (full BPR + TF-IDF + reranker)
    """

    print("Mode: compare_baselines\n")

    # popularity
    popular = (data["train"][config.COL_ITEM_ID].value_counts().head(50).index.tolist())
    pop_recs = {}
    for u in data["test_users"]:
        seen = data["seen_items"].get(u, set())
        pop_recs[u] = [i for i in popular if i not in seen][:10]
    pop_recall = recall_at_k(pop_recs, data["test_targets"], k=10)

    # BPR alone (full)
    bpr = BPRRetriever()
    bpr.fit(data["train"], resume=True,
            model_path=config.BPR_FULL_MODEL_PATH,
            checkpoint_path=config.BPR_FULL_CHECKPOINT_PATH)
    bpr_recs = bpr.recommend_all(data["test_users"], data["seen_items"], k=10)
    bpr_recall = recall_at_k(bpr_recs, data["test_targets"], k=10)

    # TF-IDF alone
    # tfidf = TfidfRetriever()
    # tfidf.fit(data["item_text"], resume=True)
    # tfidf_recs = tfidf.recommend_all(data["test_users"], data["seen_items"], k=10)
    # tfidf_recall = recall_at_k(tfidf_recs, data["test_targets"], k=10)
    content_model = load_content_retriever(data["item_text"], content=content)
    content_recs = content_model.recommend_all(data["test_users"], data["seen_items"], k=10)
    content_recall = recall_at_k(content_recs, data["test_targets"], k=10)
    content_label = "BM25 alone" if content == "bm25" else "TF-IDF alone"

    # pipeline: read cached inference df + load reranker.
    # df = pd.read_parquet(config.RERANKER_INFERENCE_DF_PATH)
    _, infer_cache_path = reranker_cache_paths(content)
    df = pd.read_parquet(infer_cache_path)
    reranker = LightGBMReranker()
    if reranker_name == "lgbm":
        reranker.load(config.LGBM_MODEL_PATH)
    elif reranker_name == "catboost":
        reranker = CatBoostReranker()
        reranker.load(config.CATBOOST_MODEL_PATH)

    scores = reranker.predict(df)
    scored = df[["user_id", "item_id"]].assign(score=scores)
    scored = scored.sort_values(["user_id", "score"], ascending=[True, False])
    pipeline_recs = (scored.groupby("user_id", sort=False)["item_id"].apply(lambda s: s.head(10).tolist()).to_dict())
    pipeline_recall = recall_at_k(pipeline_recs, data["test_targets"], k=10)

    # print table
    print(f"{'Method':30s} {'Recall@10':>10s}")
    print("-" * 42)
    print(f"{'Popularity':30s} {pop_recall:>10.4f}")
    # print(f"{'TF-IDF alone':30s} {tfidf_recall:>10.4f}")
    print(f"{content_label:30s} {content_recall:>10.4f}")
    print(f"{'BPR alone':30s} {bpr_recall:>10.4f}")
    print(f"{f'Pipeline ({reranker_name})':30s} {pipeline_recall:>10.4f}")

def run_submit(data: dict, reranker_name: str, content: str = "tfidf", force: bool = False) -> None:
    """
    Generate submission.csv for Kaggle.
    """
    print(f"Mode: submit ({reranker_name})")

    # df = get_inference_df(data, force=force)
    df = get_inference_df(data, content=content, force=force)

    # load reranker
    if reranker_name == "lgbm":
        reranker = LightGBMReranker()
        reranker.load(config.LGBM_MODEL_PATH)
    elif reranker_name == "catboost":
        reranker = CatBoostReranker()
        reranker.load(config.CATBOOST_MODEL_PATH)
    else:
        raise ValueError(f"Unknown reranker: {reranker_name}")

    print("Scoring with reranker...")
    scores = reranker.predict(df)

    print("Picking top-10 per user...")
    top10 = top10_per_user(df, scores)

    # build submission df matching the sample format:
    #   ID, user_id, item_id="i1,i2,...,i10"
    rows = []
    for user_id in data["test_users"]:
        items = top10.get(user_id, [])
        if len(items) != 10:
            raise ValueError(
                f"User {user_id} has {len(items)} recommendations, expected 10. "
                f"Check candidate pool / reranker output."
            )
        if len(set(items)) != 10:
            raise ValueError(
                f"User {user_id} has duplicate items in top-10: {items}"
            )
        rows.append({
            "ID": user_id,
            "user_id": user_id,
            "item_id": ",".join(str(i) for i in items[:10]),
        })

    sub = pd.DataFrame(rows)
    sub.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission written to {config.SUBMISSION_PATH}  ({len(sub):,} users)")



def reranker_cache_paths(content: str):
    # separate parquet cache so bm25 does not reuse tf-idf tables
    if content == "bm25":
        train_path = config.ARTIFACTS_DIR / "reranker_train_df_bm25.parquet"
        infer_path = config.ARTIFACTS_DIR / "reranker_inference_df_bm25.parquet"
    else:
        train_path = config.RERANKER_TRAIN_DF_PATH
        infer_path = config.RERANKER_INFERENCE_DF_PATH
    return train_path, infer_path



def run_train_bm25(data: dict, force: bool = False) -> None:
    # train bm25 on item text once
    print("Mode: train_bm25")
    print(f"Items: {len(data['item_text'])}")
    if force and config.BM25_MODEL_PATH.exists():
        print("Force retrain, replacing old BM25 artifact")
        config.BM25_MODEL_PATH.unlink()
    bm25 = BM25Retriever()
    bm25.fit(data["item_text"], resume=not force)
    print(f"Done. BM25 saved to {config.BM25_MODEL_PATH}")

def ensemble_top10(data: dict, content: str, lgbm_weight: float, catboost_weight: float) -> dict[int, list[int]]:
    df = get_inference_df(data, content=content, force=False)
    print(f"Inference df: {len(df):,} rows, {df['user_id'].nunique():,} users")
 
    # load both rerankers
    print("Loading LightGBM...")
    lgbm = LightGBMReranker()
    lgbm.load(config.LGBM_MODEL_PATH)
 
    print("Loading CatBoost...")
    cat = CatBoostReranker()
    cat.load(config.CATBOOST_MODEL_PATH)
 
    print("Scoring with LightGBM...")
    lgbm_scores = lgbm.predict(df)
    print("Scoring with CatBoost...")
    cat_scores = cat.predict(df)
 
    # per-user min-max normalize each model's scores to [0, 1]
    # (otherwise model with larger raw score range dominates the weighted sum)
    scored = df[["user_id", "item_id"]].copy()
    scored["lgbm"] = lgbm_scores
    scored["catboost"] = cat_scores
 
    def _per_user_minmax(g: "pd.Series") -> "pd.Series":
        lo, hi = g.min(), g.max()
        if hi == lo:
            return g * 0.0      # all-equal scores → all zero (rare edge case)
        return (g - lo) / (hi - lo)
 
    print("Normalizing scores per user...")
    scored["lgbm_norm"] = scored.groupby("user_id")["lgbm"].transform(_per_user_minmax)
    scored["catboost_norm"] = scored.groupby("user_id")["catboost"].transform(_per_user_minmax)
 
    # weighted combination
    scored["combined"] = (lgbm_weight * scored["lgbm_norm"] + catboost_weight * scored["catboost_norm"])
 
    # take top-10 per user
    print("Picking top-10 per user...")
    scored = scored.sort_values(["user_id", "combined"], ascending=[True, False])
    top10 = (scored.groupby("user_id", sort=False)["item_id"].apply(lambda s: s.head(10).tolist()).to_dict())

    return top10

def run_ensemble_evaluate(data: dict, content: str = "tfidf", lgbm_weight: float = 0.5) -> None:
    """
    Evaluate ensemble of LightGBM + CatBoost on test_targets.
    Weights are [lgbm_weight, 1 - lgbm_weight].
    """
    catboost_weight = 1.0 - lgbm_weight
    print(
        f"Mode: ensemble_evaluate  "
        f"(LGBM weight={lgbm_weight}, CatBoost weight={catboost_weight})"
    )
 
    top10 = ensemble_top10(data, content, lgbm_weight, catboost_weight)
    recall = recall_at_k(top10, data["test_targets"], k=config.TOP_K)
    print(f"\nEnsemble Recall@{config.TOP_K}: {recall:.4f}")
 
  
def run_ensemble_submit(data: dict, content: str = "tfidf", lgbm_weight: float = 0.5) -> None:
    """Generate submission.csv from the ensemble."""
    catboost_weight = 1.0 - lgbm_weight
    print(
        f"Mode: ensemble_submit  "
        f"(LGBM weight={lgbm_weight}, CatBoost weight={catboost_weight})"
    )
 
    top10 = ensemble_top10(data, content, lgbm_weight, catboost_weight)
 
    rows = []
    for user_id in data["test_users"]:
        items = top10.get(user_id, [])
        if len(items) != 10:
            raise ValueError(
                f"User {user_id} has {len(items)} recommendations, expected 10."
            )
        if len(set(items)) != 10:
            raise ValueError(
                f"User {user_id} has duplicate items in top-10: {items}"
            )
        rows.append({
            "ID": user_id,
            "user_id": user_id,
            "item_id": ",".join(str(i) for i in items),
        })
 
    sub = pd.DataFrame(rows)
    sub.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission written to {config.SUBMISSION_PATH}  ({len(sub):,} users)")

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid BPR + TF-IDF ")
    parser.add_argument(
        "--mode",
        choices=["train_bpr", "train_tfidf", "train_bm25",
                  "train_reranker", "evaluate", "compare_baselines", "submit",
                  "ensemble_evaluate", "ensemble_submit"],
        default="train_bpr",
        help="train_bpr | train_tfidf | train_bm25 | train_reranker | evaluate | compare_baselines | submit",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="with train_bpr train on 100%% train.csv, default is 80%% split for val",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="with train_bpr or train_tfidf delete old artifacts and retrain from scratch",
    )
    parser.add_argument(
        "--reranker",
        choices=["lgbm", "catboost"],
        default="lgbm",
        help="lgbm | catboost",
    )
    parser.add_argument(
        "--content",
        choices=["tfidf", "bm25"],
        default="tfidf",
        help="content retriever for hybrid pipeline, tfidf or bm25",
    )
    parser.add_argument(
        "--lgbm-weight",
        type=float,
        default=0.5,
        help="weight for LightGBM in the ensemble; catboost gets (1 - this)",
    )
    args = parser.parse_args()

    print("Loading data")
    data = load_all()
    print("Data loaded, dispatching mode", flush=True)
    print(f"args.mode = {args.mode!r}", flush=True)

    if args.mode == "train_bpr":
        run_train_bpr(data, full=args.full, force=args.force)
    elif args.mode == "train_tfidf":
        run_train_tfidf(data, force=args.force)
    elif args.mode == "train_bm25":
        run_train_bm25(data, force=args.force)
    elif args.mode == "train_reranker":
        run_train_reranker(data, reranker_name=args.reranker, content=args.content, force=args.force)
    elif args.mode == "evaluate":
        run_evaluate(data, reranker_name=args.reranker, content=args.content, force=args.force)
    elif args.mode == "compare_baselines":
        run_compare_baselines(data, args.reranker, content=args.content)
    elif args.mode == "submit":
        run_submit(data, reranker_name=args.reranker, content=args.content, force=args.force)
    elif args.mode == "ensemble_evaluate":
        print("Entering run_ensemble_evaluate", flush=True)
        run_ensemble_evaluate(data, content=args.content, lgbm_weight=args.lgbm_weight)
    elif args.mode == "ensemble_submit":
        run_ensemble_submit(data, content=args.content, lgbm_weight=args.lgbm_weight)

if __name__ == "__main__":
    main()

# python main.py --mode train_bpr --force ; python main.py --mode train_bpr --full --force
# python main.py --mode train_reranker --reranker lgbm --force ; python main.py --mode evaluate --reranker lgbm --force
# python main.py --mode train_bm25
# python main.py --mode train_reranker --content bm25 --reranker lgbm --force
# python main.py --mode evaluate --content bm25 --reranker lgbm --force
# python main.py --mode submit --content bm25 --reranker lgbm --force
