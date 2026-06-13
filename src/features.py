"""Data loading and preprocessing."""

from __future__ import annotations

import pandas as pd

import config


def load_train() -> pd.DataFrame:
    df = pd.read_csv(config.TRAIN_PATH)
    return df.sort_values(
        [config.COL_USER_ID, config.COL_TIMESTAMP],
        kind="mergesort",
    ).reset_index(drop=True)


def load_item_meta() -> pd.DataFrame:
    return pd.read_csv(config.ITEM_META_PATH)


def build_item_text(meta: pd.DataFrame) -> pd.Series:
    meta = meta.set_index(config.COL_ITEM_ID)

    def _row_to_text(row: pd.Series) -> str:
        parts = []
        for col in config.ITEM_TEXT_COLUMNS:
            val = row.get(col, "")
            if pd.isna(val):
                continue
            parts.append(str(val))
        return " ".join(parts)

    return meta.apply(_row_to_text, axis=1)


def get_user_histories(train: pd.DataFrame) -> dict[int, list[int]]:
    histories: dict[int, list[int]] = {}
    for user_id, group in train.groupby(config.COL_USER_ID, sort=False):
        histories[int(user_id)] = group[config.COL_ITEM_ID].astype(int).tolist()
    return histories


def get_user_timed_histories(train: pd.DataFrame) -> dict[int, list[tuple[int, int]]]:
    timed: dict[int, list[tuple[int, int]]] = {}
    for user_id, group in train.groupby(config.COL_USER_ID, sort=False):
        timed[int(user_id)] = list(
            zip(
                group[config.COL_ITEM_ID].astype(int).tolist(),
                group[config.COL_TIMESTAMP].astype(int).tolist(),
            )
        )
    return timed


def get_user_seen_items(histories: dict[int, list[int]]) -> dict[int, set[int]]:
    return {user_id: set(items) for user_id, items in histories.items()}


def get_test_users() -> list[int]:
    sub = pd.read_csv(config.SAMPLE_SUBMISSION_PATH)
    return sub[config.COL_USER_ID].astype(int).unique().tolist()


def get_test_targets(test_users: list[int]) -> dict[int, list[int]]:
    test_df = pd.read_csv(config.TEST_PATH)
    test_df = test_df[test_df[config.COL_USER_ID].isin(test_users)]
    targets: dict[int, list[int]] = {}
    for user_id, group in test_df.groupby(config.COL_USER_ID, sort=False):
        targets[int(user_id)] = group[config.COL_ITEM_ID].astype(int).tolist()
    return targets


def temporal_train_val_split(
    train: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[int, list[int]]]:
    train_rows: list[pd.DataFrame] = []
    val_targets: dict[int, list[int]] = {}

    for user_id, group in train.groupby(config.COL_USER_ID, sort=False):
        n = len(group)
        if n < config.MIN_USER_INTERACTIONS_FOR_VAL:
            train_rows.append(group)
            continue

        n_val = max(1, int(n * config.VAL_HOLDOUT_RATIO))
        train_rows.append(group.iloc[:-n_val])
        val_items = group.iloc[-n_val:][config.COL_ITEM_ID].astype(int).tolist()
        val_targets[int(user_id)] = val_items

    train_split = pd.concat(train_rows, ignore_index=True)
    return train_split, val_targets


def load_all() -> dict:
    train = load_train()
    meta = load_item_meta()
    item_text = build_item_text(meta)
    test_users = get_test_users()
    test_targets = get_test_targets(test_users)
    train_split, val_targets = temporal_train_val_split(train)

    histories = get_user_histories(train)
    seen_items = get_user_seen_items(histories)

    val_histories = get_user_histories(train_split)
    val_seen_items = get_user_seen_items(val_histories)
    timed_histories = get_user_timed_histories(train)
    val_timed_histories = get_user_timed_histories(train_split)

    return {
        "train": train,
        "train_split": train_split,
        "val_targets": val_targets,
        "meta": meta,
        "item_text": item_text,
        "histories": histories,
        "seen_items": seen_items,
        "val_seen_items": val_seen_items,
        "timed_histories": timed_histories,
        "val_timed_histories": val_timed_histories,
        "test_users": test_users,
        "test_targets": test_targets,
    }


if __name__ == "__main__":
    data = load_all()
    print(f"train rows       {len(data['train'])}")
    print(f"train_split rows {len(data['train_split'])}")
    print(f"val users        {len(data['val_targets'])}")
    print(f"test users       {len(data['test_users'])}")
    print(f"items with text  {len(data['item_text'])}")
