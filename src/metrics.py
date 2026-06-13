import config
def recall_at_k(recommendations:dict[int,list[int]],
val_targets:dict[int,list[int]],
k:int=config.TOP_K,)-> float: #list of recommendation, values of hidden items, k

    scores=[]
    for user_id,true_items in val_targets.items():
        #get user's recommendation
        pred_items=recommendations.get(user_id,[])

        #only look at top k
        pred_set=set(pred_items[:k]) #convert it to set for fast lookup

        ## how many true items appear in top-k 
        hits=len(set(true_items) & pred_set)

        if true_items:          # if list is not empty
            recall = hits / len(true_items)
        else:                   
            recall = 0.0   

        scores.append(recall)
        # return average recall across all users
    return sum(scores) / len(scores) if scores else 0.0


def candidate_recall(
    candidates_by_user: dict[int, set[int]],
    val_targets: dict[int, list[int]],
) -> float:
    """Fraction of true items that appear anywhere in the candidate pool (per user, then averaged)."""
    scores = []
    for user_id, true_items in val_targets.items():
        pool = candidates_by_user.get(user_id, set())
        if not true_items:
            scores.append(0.0)
            continue
        hits = len(set(true_items) & pool)
        scores.append(hits / len(true_items))
    return sum(scores) / len(scores) if scores else 0.0