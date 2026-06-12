"""
main_pipeline.py

loads data trains both merges lists
  val mode, score on holdout from train.csv
  test mode, score vs test.csv labels
  submit mode, write outputs/submission.csv

hybrid = BPR + TF-IDF + RRF

Run from project root:
  python main_pipeline.py --mode train_bpr
  python main_pipeline.py --mode train_tfidf
  python main_pipeline.py --mode val
  python main_pipeline.py --mode train_bpr --full
  python main_pipeline.py --mode test
  python main_pipeline.py --mode submit
"""

from __future__ import annotations

import argparse

import pandas as pd

import config
from src.features import load_all
from src.metrics import recall_at_k
from src.retrieval.collaborative.bpr_matrix import BPRRetriever
from src.retrieval.content_based.tfidf_content import TfidfRetriever




# Each retriever returns TOP 100 before we merge down to 10 
# BPR might rank item X high, TF-IDF item Y so fusing needs both in the pool
CANDIDATE_K = 100

# RRF  Higher = flatter rank weights
RRF_K = 60  #how we merge BPR list and TF-IDF list into one


def get_popular_items(train_df: pd.DataFrame, n: int = 100) -> list[int]:
    # most clicked items in train, pad when bpr or tfidf return too few
    return (
        train_df[config.COL_ITEM_ID]
        .value_counts()
        .index.astype(int)
        .tolist()[:n]
    )


def reciprocal_rank_fusion(
    ranked_lists: list[list[int]],
    seen: set[int],
    k: int = config.TOP_K,
) -> list[int]:
    # merge bpr list and tfidf list into one top-k list
    # rrf uses rank not raw score cause scales differ
    scores: dict[int, float] = {}

    for ranked in ranked_lists:
        for rank, item_id in enumerate(ranked, start=1):
            # skip stuff user already clicked
            if item_id in seen:
                continue
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (RRF_K + rank)

    # sort by fused score take top k
    merged = sorted(scores.keys(), key=lambda item_id: scores[item_id], reverse=True)
    return merged[:k]


def fill_to_k(
    recs: list[int],
    seen: set[int],
    popular: list[int],
    k: int = config.TOP_K,
) -> list[int]:
    # pad to exactly k items if fusion gave too few
    out = list(recs)
    blocked = seen | set(out)  # seen in train  plus already picked

    for item_id in popular:
        if len(out) >= k:
            break
        if item_id not in blocked:
            out.append(item_id)
            blocked.add(item_id)

    return out[:k]


def hybrid_recommend_all(
    bpr: BPRRetriever,
    tfidf: TfidfRetriever,
    user_ids: list[int],
    seen_items: dict[int, set[int]],
    popular: list[int],
    k: int = config.TOP_K,
    candidate_k: int = CANDIDATE_K,
) -> dict[int, list[int]]:
    # per user bpr + tfidf candidates, rrf, pad to k
    # val uses val_seen_items, submit uses seen_items from full train
    # step  each retriever returns candidate_k items per user
    bpr_recs = bpr.recommend_all(user_ids, seen_items, k=candidate_k)
    tfidf_recs = tfidf.recommend_all(user_ids, seen_items, k=candidate_k)

    fused: dict[int, list[int]] = {}

    for user_id in user_ids:
        seen = seen_items.get(user_id, set())

        # step  merge the two ranked lists for this user
        lists = [
            bpr_recs.get(user_id, []),
            tfidf_recs.get(user_id, []),
        ]
        merged = reciprocal_rank_fusion(lists, seen, k=k)

        # step pad to exactly TOP_K if fusion returned too few
        fused[user_id] = fill_to_k(merged, seen, popular, k=k)

    return fused


def load_bpr(model_path=None, train_mode: str = "train_bpr") -> BPRRetriever:
    # load bpr from disk dont train
    path = model_path or config.BPR_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"No BPR model at {path}. "
            f"Run: python main_pipeline.py --mode {train_mode}"
        )
    bpr = BPRRetriever()
    bpr.load(path)
    return bpr


def load_tfidf() -> TfidfRetriever:
    # load tfidf from disk dont train
    if not config.TFIDF_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No TF-IDF model at {config.TFIDF_MODEL_PATH}. "
            "Run: python main_pipeline.py --mode train_tfidf"
        )
    tfidf = TfidfRetriever()
    tfidf.load(config.TFIDF_MODEL_PATH)
    return tfidf


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


def run_val(data: dict) -> None:
    # load dev bpr + tfidf score on last 20% holdout
    print("Mode: val")
    val_users = list(data["val_targets"].keys())
    print(f"Val users: {len(val_users)}")

    popular = get_popular_items(data["train_split"])

    print("Loading BPR (dev model)...")
    bpr = load_bpr(config.BPR_MODEL_PATH, train_mode="train_bpr")
    print("Loading TF-IDF")
    tfidf = load_tfidf()

    print("Generating fused recommendations")
    fused = hybrid_recommend_all(
        bpr,
        tfidf,
        val_users,
        data["val_seen_items"],
        popular,
    )

    bpr_only = bpr.recommend_all(val_users, data["val_seen_items"], k=config.TOP_K)
    tfidf_only = tfidf.recommend_all(val_users, data["val_seen_items"], k=config.TOP_K)

    print(f"BPR Recall@{config.TOP_K}:     {recall_at_k(bpr_only, data['val_targets']):.4f}")
    print(f"TF-IDF Recall@{config.TOP_K}:  {recall_at_k(tfidf_only, data['val_targets']):.4f}")
    print(f"Hybrid Recall@{config.TOP_K}:  {recall_at_k(fused, data['val_targets']):.4f}")


def run_test(data: dict) -> None:
    # load full bpr + tfidf score vs test.csv labels
    print("Mode: test ")
    test_targets = data["test_targets"]
    eval_users = list(test_targets.keys())
    print(f"Users scored: {len(eval_users)} (of {len(data['test_users'])} submit users)")

    popular = get_popular_items(data["train"])

    print("Loading BPR (final model)")
    bpr = load_bpr(config.BPR_FULL_MODEL_PATH, train_mode="train_bpr --full")
    print("Loading TF-IDF...")
    tfidf = load_tfidf()

    print("Generating recommendations")
    fused = hybrid_recommend_all(bpr, tfidf, eval_users, data["seen_items"], popular)
    bpr_only = bpr.recommend_all(eval_users, data["seen_items"], k=config.TOP_K)
    tfidf_only = tfidf.recommend_all(eval_users, data["seen_items"], k=config.TOP_K)

    print(f"BPR Recall@{config.TOP_K} (test.csv):     {recall_at_k(bpr_only, test_targets):.4f}")
    print(f"TF-IDF Recall@{config.TOP_K} (test.csv):  {recall_at_k(tfidf_only, test_targets):.4f}")
    print(f"Hybrid Recall@{config.TOP_K} (test.csv):  {recall_at_k(fused, test_targets):.4f}")


def run_submit(data: dict) -> None:
    
    print("Mode: submit")

    popular = get_popular_items(data["train"])

    print("Loading BPR ")
    bpr = load_bpr(config.BPR_FULL_MODEL_PATH, train_mode="train_bpr --full")
    print("Loading TF-IDF")
    tfidf = load_tfidf()

    print("Generating fused recommendations for test users")
    fused = hybrid_recommend_all(
        bpr,
        tfidf,
        data["test_users"],
        data["seen_items"],  # full train clicks, dont recommend again
        popular,
    )

    #  write CSV matching sample_sub
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sample = pd.read_csv(config.SAMPLE_SUBMISSION_PATH)
    rows = []
    for _, row in sample.iterrows():
        user_id = int(row[config.COL_USER_ID])
        sub_id = row["ID"]  
        items = fused.get(user_id, [])
       
        item_str = ",".join(str(item_id) for item_id in items[: config.TOP_K])
        rows.append(
            {
                "ID": sub_id,
                config.COL_USER_ID: user_id,
                config.COL_ITEM_ID: item_str,
            }
        )

    submission = pd.DataFrame(rows)
    submission.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Wrote {config.SUBMISSION_PATH} ({len(submission)} users)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid BPR + TF-IDF ")
    parser.add_argument(
        "--mode",
        choices=["train_bpr", "train_tfidf", "val", "test", "submit"],
        default="train_bpr",
        help="train_bpr | train_tfidf | val | test | submit",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="with train_bpr train on 100% train.csv, default is 80% split for val",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="with train_bpr or train_tfidf delete old artifacts and retrain from scratch",
    )
    args = parser.parse_args()

    print("Loading data")
    data = load_all()

    if args.mode == "train_bpr":
        run_train_bpr(data, full=args.full, force=args.force)
    elif args.mode == "train_tfidf":
        run_train_tfidf(data, force=args.force)
    elif args.mode == "val":
        run_val(data)
    elif args.mode == "test":
        run_test(data)
    else:
        run_submit(data)


if __name__ == "__main__":
    main()
