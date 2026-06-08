# Bayesian Personalized Ranking (collaborative filtering)

#it learnd from implicit feedback (user-item interactions)


from __future__ import annotations

import numpy as np
import pandas as pd

import config


class BPRRetriever:
    """
    BPR 
     idea is it is pairwise
      For user u, positive item i, negative item j:
        we want score(u,i) > score(u,j)
    score(u, i) = dot(user_embedding[u], item_embedding[i])
    """


    def __init__(self):
        # mappings between original ids and matrix indices like 0,1 ,2 etc
        #bpr need continuos ids for matrix factorization not original ids
        self.user_map:dict[int,int]={}
        self.item_map:dict[int,int]={}
        self.item_inv: dict[int, int] = {}   # matrix col index → item_id (reverse)

        # learned embeddings (built during fit)
        self.user_factors = None
        self.item_factors = None
        self.user_pos_items = {}

    def fit(self,train_df:pd.DataFrame) -> None:
        """
        Map ids, build user positive sets, train BPR embeddings from scratch.
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


        n_users = len(unique_users)
        n_items = len(unique_items)

        self.user_pos_items = {}
        for u in range(n_users):
            self.user_pos_items[u] = set() #created an empty set for each user

        for _, row in train_df.iterrows():
            u_idx = self.user_map[row[config.COL_USER_ID]]  #map converts to index
            i_idx = self.item_map[row[config.COL_ITEM_ID]]
            self.user_pos_items[u_idx].add(i_idx) #this user's set of clicked items

        factors = config.BPR_FACTORS
        rng = np.random.default_rng(42)
        self.user_factors = rng.normal(0, 0.01, size=(n_users, factors)).astype(np.float32) # a list of numbers that describes each user # filling a matrix with small random numbers
        self.item_factors = rng.normal(0, 0.01, size=(n_items, factors)).astype(np.float32)

        #training loop
        lr = config.BPR_LEARNING_RATE
        reg = config.BPR_REGULARIZATION
        for _ in range(config.BPR_ITERATIONS):
            for u, pos_items in self.user_pos_items.items(): #  e.g user 0 clicked items 5, 12, 8
                for i in pos_items:
                    j = int(rng.integers(0, n_items)) # random negative item index
                    while j in pos_items: # make sure it's not a positive item
                        j = int(rng.integers(0, n_items))
                    self._bpr_step(u, i, j, lr, reg)

    def _bpr_step(self, u: int, i: int, j: int, lr: float, reg: float) -> None:
        user = self.user_factors[u]
        pos = self.item_factors[i]
        neg = self.item_factors[j]

        score_pos = np.dot(user, pos) #user's taste vector and the item's profile vector, Multiply each matching pair
        score_neg = np.dot(user, neg)
        x_uij = score_pos - score_neg 
        #Turns the score gap into a push strength between 0 and 1
        sigmoid_neg = 1.0 / (1.0 + np.exp(x_uij))


        #update the vectors

        self.user_factors[u] += lr * (sigmoid_neg * (pos - neg) - reg * user)
        self.item_factors[i] += lr * (sigmoid_neg * user - reg * pos)
        self.item_factors[j] += lr * (sigmoid_neg * (-user) - reg * neg)


    def recommend(self,
        user_id:int,
        seen_items:set[int],
        k:int=config.TOP_K,)->list[int]:

        """
        Generate top-K item recommendations for 1 user.
        
        - Score all items with dot(user_embedding, item_embedding)
        - Return top-K unseen items
        """

        # step:
        #check if user exist in traing data
        if user_id not in self.user_map: 
            return []  # cold start

        user_idx=self.user_map[user_id]


        scores = self.item_factors @ self.user_factors[user_idx]
        ranked_indices = np.argsort(-scores)

        recommendations = []
        for idx in ranked_indices:
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




















