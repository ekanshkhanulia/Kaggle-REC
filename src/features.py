"""
features.py — data loading and preprocessing. 
"""

from __future__ import annotations # lets us write dict[int, list[int]] without quotes
import pandas as pd
import config


#load data
def load_train() -> pd.DataFrame:
    """  we sort cause we  need click/interaaction ini time order 
         Val split holds out the last cliks of evey user """

    df=pd.read_csv(config.TRAIN_PATH)

    df=df.sort_values(
        [config.COL_USER_ID , config.COL_TIMESTAMP], #SORT BY USER THEN time
        kind="mergesort",
    ).reset_index(drop=True)


    return df


# LOAD ITEM METADATA

def load_item_meta() -> pd.DataFrame:
    """
    Load item_meta.csv product info (title, category, description, etc.)
    
    """
    return pd.read_csv(config.ITEM_META_PATH) #  return the item_meta.csv dataframe


#build item text (for tf_idf )

def build_item_text(meta:pd.DataFrame) -> pd.Series:
    """
    Combine text columns into one string per item.
    """

    #set item_id as index so we can look up text instantly
    meta=meta.set_index(config.COL_ITEM_ID)

    def _row_to_text(row:pd.Series) -> str:
        parts=[]
        for col in config.ITEM_TEXT_COLUMNS: #Loops over columns 
            val=row.get(col, "") #row will give you a data under that column block 
            if pd.isna(val): #skip empty/msiing
                continue
            parts.append(str(val))
        return " ".join(parts) # join all fields 

    return meta.apply(_row_to_text,axis=1) # apply to every row (every item)

def get_user_histories(train:pd.DataFrame) -> dict[int,list[int]]:
    """Build user id
    cause
    BPR needs what each user clicked """

    histories:dict[int,list[int]]={} 

    for user_id,group in train.groupby(config.COL_USER_ID , sort=False):
        # group = all rows for this user
        histories[int(user_id)]=group[config.COL_ITEM_ID].astype(int).tolist()
    return histories

def get_user_seen_items(histories:dict[int,list[int]]) -> dict[int,set[int]]: # key is userid,a dn vlaue is set of items
    """set is for fast lookup like what user have already interacted with """
    seen_items={}
    for user_id,items in histories.items():
        seen_items[user_id]=set(items)
    return seen_items


def get_test_users()-> list[int]:
    """
    Load the list of users we must output recommendations for"""

    
    sub = pd.read_csv(config.SAMPLE_SUBMISSION_PATH)  # load sample_submission.csv
    return sub[config.COL_USER_ID].astype(int).unique().tolist()


def temporal_train_val_split(train:pd.DataFrame,) -> tuple[pd.DataFrame,dict[int,list[int]]]:

    train_rows:list[pd.DataFrame]=[] #traing
    val_targets:dict[int,list[int]]={}

    for user_id,group in train.groupby(config.COL_USER_ID,sort=False):
        n=len(group) #how many clicks this user has made

        #if not enough click to split
        if n<config.MIN_USER_INTERACTIONS_FOR_VAL:
            train_rows.append(group)
            continue
        
        #How many clicks to hold out for validation
        n_val=max(1,int(n*config.VAL_HOLDOUT_RATIO))

        # everything except last n_val rows → training
        train_rows.append(group.iloc[:-n_val])

        #last n_val rows->val targets(what we tru to predict)
        val_items=group.iloc[-n_val:][config.COL_ITEM_ID].astype(int).tolist()
        val_targets[int(user_id)] = val_items


    # combine all training rows back into one dataframe
    train_split = pd.concat(train_rows, ignore_index=True)
    return train_split, val_targets


def load_all() -> dict:
    train = load_train()                              # load + sort train.csv
    meta = load_item_meta()                           # load item_meta.csv
    item_text = build_item_text(meta)                 # one text string per item
    histories = get_user_histories(train)             # user → ordered click list
    seen_items = get_user_seen_items(histories)       # user → set of clicked items
    test_users = get_test_users()                     # users to predict for
    train_split, val_targets = temporal_train_val_split(train)  # split train/val 

    return {
        "train":train,
        "train_split":train_split,
        "val_targets":val_targets,
        "meta":meta,
        "item_text":item_text,
        "histories": histories,
        "seen_items": seen_items,
        "test_users": test_users,
    }

if __name__ == "__main__":
    data = load_all()
    print(f"train rows:        {len(data['train'])}")
    print(f"train_split rows:  {len(data['train_split'])}")
    print(f"val users:         {len(data['val_targets'])}")
    print(f"test users:        {len(data['test_users'])}")
    print(f"items with text:   {len(data['item_text'])}")
