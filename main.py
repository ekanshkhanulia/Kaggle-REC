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

import config
from src.features import load_all
from src.retrieval.collaborative.bpr_matrix import BPRRetriever
from src.retrieval.content_based.tfidf_content import TfidfRetriever


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid BPR + TF-IDF ")
    parser.add_argument(
        "--mode",
        choices=["train_bpr", "train_tfidf"],
        default="train_bpr",
        help="train_bpr | train_tfidf",
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
    else:
        run_train_tfidf(data, force=args.force)


if __name__ == "__main__":
    main()
