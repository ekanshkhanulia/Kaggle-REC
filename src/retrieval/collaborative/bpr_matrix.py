# Bayesian Personalized Ranking (collaborative filtering)

#it learnd from implicit feedback (user-item interactions)


from __future__ import annotations

import numpy as np
import scipy.sparse as sp  #for sparse matrix
from implicit.bpr import BayesianPersonalizedRanking


class BPRRetriever:
    def __init__(self):
        #bpr model for implicit lib
        #factor is  embedding size (how many dimensions per user/item vector)

        self.model=BayesianPersonalizedRanking(
            factors=config.BPR_FACTORS,
            iterations=config.BPR_ITERATIONS,
            learning_rate=config.BPR_LEARNING_RATE,
            regularization=config.BPR_REGULARIZATION,
            random_state=config.RANDOM_SEED,
            
        )

        # mappings between original ids and matrix indices like 0,1 ,2 etc
        #bpr need continuos ids for matrix factorization not original ids
        self.user_map:dict[int,int]={}
        self.item_map:dict[int,int]={}
        self.item_inv: dict[int, int] = {}   # matrix col index → item_id (reverse)
        
        # sparse user-item matrix (built during fit)
        self.user_item_matrix = None

    def fit(self,train_df:pd.Dataframe) -> None:



