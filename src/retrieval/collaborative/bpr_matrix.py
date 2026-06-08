# Bayesian Personalized Ranking (collaborative filtering)

#it learnd from implicit feedback (user-item interactions)


from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp  #for sparse matrix
from implicit.bpr import BayesianPersonalizedRanking

import config


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

    def fit(self,train_df:pd.DataFrame) -> None:
        """
        Build user-item matrix and train BPR.
        
        - Map user_ids and item_ids to contiguous indices (0,1,2...)
        - Build sparse user-item matrix (1 where user clicked item)
        - Train BPR on item-user matrix (implicit expects item-user)
        """

        #step -build mapping: row id to matrix index
        unique_users=sorted(train_df[config.COL_USER_ID].unique())
        unique_items = sorted(train_df[config.COL_ITEM_ID].unique())

        self.user_map={}
        for i,uid in enumerate(unique_users):
            self.user_map[uid]=i
        
        self.item_map={}
        for i,iid in enumerate(unique_items):
            self.item_map[iid]=i
        
        self.item_inv={}
        for iid,i in self.item_map.items():
            self.item_inv[i]=iid


        # Step convert each row's user_id/item_id to  indices
        rows = train_df[config.COL_USER_ID].map(self.user_map).values  # user indices so exmaple 3664  → 0 
        cols = train_df[config.COL_ITEM_ID].map(self.item_map).values  # item indices
        data = np.ones(len(rows), dtype=np.float32)                    # 1 

        
        
        
        
        n_users = len(unique_users)
        n_items = len(unique_items)

        #build sparse user-item matrix
        self.user_item_matrix=sp.csr_matrix((data,(rows,cols)),
        shape=(n_users,n_items))


        #step
        item_user=self.user_item_matrix.T.tocsr()


        #train bpr

        self.model.fit(item_user)


    def recommend(self,
        user_id:int,
        seen_items:set[int],
        k:int=config.TOP_K,)->list[int]:

        """
        Generate top-K item recommendations for 1 user.
        
        - Convert user_id to matrix index
        - Ask BPR model for top-K items
        - Filter already-seen items
        - Convert matrix indices back to real item_ids
        """

        # step:
        #check if user exist in traing data
        if user_id not in self.user_map: 
            return []  # cold start

        user_idx=self.user_map[user_id]


        #step
        #bpr recommendation
        # model.recommend returns (item_indices, scores) for this user
        item_indices,scores=self.model.recommend(
            user_idx,
            self.user_item_matrix[user_idx],
            N=k + len(seen_items), #caue seen items can be in the ,matirx so we need extra
            filter_already_liked_items=True,  # skip items user already clicked
        )




        #step
        #convert matrix indices back to real item_ids

        # Step  convert indices back to item_ids, skip seen items
        recommendations = []
        for idx in item_indices:
            item_id = self.item_inv[idx]       # matrix index → real item_id
            if item_id in seen_items:          # double-check seen filter
                continue
            recommendations.append(item_id)
            if len(recommendations) == k:      # stop when we have items equal to k
                break
        return recommendations 


    def recommend_all(self,
        user_ids: list[int],
        seen_items:dict[int,set[int]],
        k:int=config.TOP_K,
        )-> dict[int,list[int]]:
        """
        Generate top-K recommendations for a list of users.
        This is what main_pipeline.py calls"""

        all_recommendations = {}
        for user_id in user_ids:
            # get seen items for this user
            user_seen = seen_items.get(user_id, set())
            # get top-K recommendations for this user
            recs = self.recommend(user_id, user_seen, k)
            all_recommendations[user_id] = recs
        return all_recommendations




















