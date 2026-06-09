#Central configuration

from pathlib import Path


#PROJECT ROOT ----


#root folder
PROJECT_ROOT=Path(__file__).resolve().parent


# Local data folder (gitignored, download  data form Kaggle  )
DATA_DIR = PROJECT_ROOT / "Data"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
ITEM_META_PATH = DATA_DIR / "item_meta.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"


# saved models / checkpoints
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
# dev bpr, trained on 80% split, used for --mode val
BPR_CHECKPOINT_PATH = ARTIFACTS_DIR / "bpr_checkpoint.npz"
BPR_MODEL_PATH = ARTIFACTS_DIR / "bpr_model.npz"
# final bpr, full train.csv, used for test + submit
BPR_FULL_CHECKPOINT_PATH = ARTIFACTS_DIR / "bpr_full_checkpoint.npz"
BPR_FULL_MODEL_PATH = ARTIFACTS_DIR / "bpr_full_model.npz"
TFIDF_MODEL_PATH = ARTIFACTS_DIR / "tfidf_model.joblib"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

#  submission CSV and other outputs 
OUTPUT_DIR = PROJECT_ROOT / "outputs"
SUBMISSION_PATH = OUTPUT_DIR / "submission.csv"


#EVALUATION & SUBMISSION-------

TOP_K=10 #Recall@10
RANDOM_SEED=42

# last chunk of each user history held out for val (by time)
VAL_HOLDOUT_RATIO=0.2

# users with too few clicks stay all in train no val row
MIN_USER_INTERACTIONS_FOR_VAL=5


#DATA COLUMNS -------------

COL_USER_ID = "user_id"
COL_ITEM_ID = "item_id"
COL_TIMESTAMP = "timestamp"


#Text fields from item_meta.csv
ITEM_TEXT_COLUMNS = [
    "title",
    "main_category",
    "features",
    "description",
    "categories",
]



# BPR matrix factorization (collaborative retrieval)

BPR_FACTORS = 64          # embedding size for users and items
BPR_ITERATIONS = 100      # training iterations
BPR_LEARNING_RATE = 0.01  # how fast model learns
BPR_REGULARIZATION = 0.01 # prevents overfitting


#TF-IDF content retrieval

TFIDF_MAX_FEATURES = 20_000   # vocabulary size
TFIDF_NGRAM_RANGE = (1, 2)    # unigrams and bigrams
TFIDF_MIN_DF = 2              # ignore very rare words
TFIDF_MAX_DF = 0.95           # ignore very common word