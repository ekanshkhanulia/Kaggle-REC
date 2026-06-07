#Central configuration

from pathlib import Path


#PROJECT ROOT -------------


#root folder
PROJECT_ROOT=Path(__file__).resolve().parent


# Local data folder (gitignored — download  data form Kaggle  )
DATA_DIR = PROJECT_ROOT / "Data"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
ITEM_META_PATH = DATA_DIR / "item_meta.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"


#  submission CSV and other outputs 
OUTPUT_DIR = PROJECT_ROOT / "outputs"
SUBMISSION_PATH = OUTPUT_DIR / "submission.csv"


#EVALUATION & SUBMISSION---------------

TOP_K=10 #Recall@10

# Last fraction of each user's history held out for validation (by timestamp)
VAL_HOLDOUT_RATIO=0.2

# Users with fewer interactions stay fully in train (no val rows)
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