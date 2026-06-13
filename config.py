from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "Data"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
ITEM_META_PATH = DATA_DIR / "item_meta.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
BPR_CHECKPOINT_PATH = ARTIFACTS_DIR / "bpr_checkpoint.npz"
BPR_MODEL_PATH = ARTIFACTS_DIR / "bpr_model.npz"
BPR_FULL_CHECKPOINT_PATH = ARTIFACTS_DIR / "bpr_full_checkpoint.npz"
BPR_FULL_MODEL_PATH = ARTIFACTS_DIR / "bpr_full_model.npz"
TFIDF_MODEL_PATH = ARTIFACTS_DIR / "tfidf_model.joblib"
BM25_MODEL_PATH = ARTIFACTS_DIR / "bm25_model.joblib"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR = PROJECT_ROOT / "outputs"
SUBMISSION_PATH = OUTPUT_DIR / "submission.csv"

TOP_K = 10
RANDOM_SEED = 42
VAL_HOLDOUT_RATIO = 0.2
MIN_USER_INTERACTIONS_FOR_VAL = 5

COL_USER_ID = "user_id"
COL_ITEM_ID = "item_id"
COL_TIMESTAMP = "timestamp"

ITEM_TEXT_COLUMNS = [
    "title",
    "main_category",
    "features",
    "description",
    "categories",
]

BPR_FACTORS = 64
BPR_ITERATIONS = 150
BPR_LEARNING_RATE = 0.01
BPR_REGULARIZATION = 0.01

TFIDF_MAX_FEATURES = 20_000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95

K_RETRIEVAL = 500
COOC_TAU = 7 * 24 * 60 * 60 * 1000
COOC_BASELINE_UNION_RECALL_AT_500 = 0.1817

META_COLS = ["user_id", "item_id", "label"]
CAT_COLS = [
    "main_category",
    "user_dominant_category",
    "store",
    "user_dominant_store",
]
RERANKER_TRAIN_DF_PATH = ARTIFACTS_DIR / "reranker_train_df.parquet"
RERANKER_INFERENCE_DF_PATH = ARTIFACTS_DIR / "reranker_inference_df.parquet"
SUBMISSION_PATH = ARTIFACTS_DIR / "submission.csv"

LGBM_MODEL_PATH = ARTIFACTS_DIR / "lgbm_reranker.joblib"
LGBM_OBJECTIVE = "lambdarank"
LGBM_METRIC = "ndcg"
LGBM_N_ESTIMATORS = 500
LGBM_LR = 0.05
LGBM_NUM_LEAVES = 63
LGBM_MIN_CHILD_SAMPLES = 20

CATBOOST_MODEL_PATH = ARTIFACTS_DIR / "catboost_reranker.cbm"
CATBOOST_TRAIN_DIR = ARTIFACTS_DIR / "catboost_info"
CATBOOST_SNAPSHOT_FILE = "catboost_reranker_checkpoint.snapshot"
CATBOOST_SNAPSHOT_PATH = CATBOOST_TRAIN_DIR / CATBOOST_SNAPSHOT_FILE
CATBOOST_SNAPSHOT_INTERVAL_SEC = 60
CATBOOST_LOSS = "YetiRank"
CATBOOST_ITER = 500
CATBOOST_LR = 0.05
CATBOOST_DEPTH = 6
