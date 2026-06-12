# Bayesian Personalized Ranking (collaborative filtering)

#it learnd from implicit feedback (user-item interactions)


from __future__ import annotations

import json

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
        self._train_row_count = 0  # match saved model to same train df size

    def fit(
        self,
        train_df: pd.DataFrame,
        resume: bool = True,
        model_path=None,
        checkpoint_path=None,
    ) -> None:
        # map ids build pos sets train embeddings, paths default from config
        from pathlib import Path

        model_path = Path(model_path or config.BPR_MODEL_PATH)
        checkpoint_path = Path(checkpoint_path or config.BPR_CHECKPOINT_PATH)
        self._model_path = model_path
        self._checkpoint_path = checkpoint_path
        self._train_row_count = len(train_df)

        # save resume, already finished? load and skip
        if resume and model_path.exists():
            meta_path = model_path.with_suffix(".meta.json")
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if meta.get("train_row_count") == self._train_row_count:
                    print(f"Loading saved BPR model from {model_path}")
                    self.load(model_path)
                    return

        # save resume, pc died mid train? pick up from checkpoint
        start_iter = 0
        loss_history: list[dict] = []
        if resume and checkpoint_path.exists():
            meta_path = checkpoint_path.with_suffix(".meta.json")
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if meta.get("train_row_count") == self._train_row_count:
                    start_iter = meta.get("completed_iteration", 0)
                    loss_history = meta.get("loss_history", [])
                    print(f"Resuming BPR from iteration {start_iter}/{config.BPR_ITERATIONS}")
                    self._load_checkpoint(checkpoint_path)

        if start_iter == 0:
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
            rng = np.random.default_rng(config.RANDOM_SEED)
            self.user_factors = rng.normal(0, 0.01, size=(n_users, factors)).astype(np.float32) # a list of numbers that describes each user # filling a matrix with small random numbers
            self.item_factors = rng.normal(0, 0.01, size=(n_items, factors)).astype(np.float32)

        #training loop
        lr = config.BPR_LEARNING_RATE
        reg = config.BPR_REGULARIZATION
        n_items = len(self.item_map)
        rng = np.random.default_rng(config.RANDOM_SEED)
        for iteration in range(start_iter, config.BPR_ITERATIONS):
            total_loss = 0.0
            n_steps = 0
            for u, pos_items in self.user_pos_items.items(): #  e.g user 0 clicked items 5, 12, 8
                for i in pos_items:
                    j = int(rng.integers(0, n_items)) # random negative item index
                    while j in pos_items: # make sure it's not a positive item
                        j = int(rng.integers(0, n_items))
                    total_loss += self._bpr_loss(u, i, j)
                    n_steps += 1
                    self._bpr_step(u, i, j, lr, reg)

            mean_loss = total_loss / n_steps if n_steps else 0.0
            loss_history.append({"iteration": iteration + 1, "loss": float(mean_loss)})

            # save resume checkpoint every iter rerun fit() to continue
            self._save_checkpoint(iteration + 1, loss_history)
            print(
                f"BPR iteration {iteration + 1}/{config.BPR_ITERATIONS} "
                f"loss={mean_loss:.4f} checkpoint saved"
            )

        # save resume done, write final model delete checkpoint
        self.save(self._model_path)
        loss_path = self._model_path.with_suffix(".loss.json")
        loss_path.write_text(json.dumps(loss_history), encoding="utf-8")
        print(f"BPR loss history saved to {loss_path}")
        if self._checkpoint_path.exists():
            self._checkpoint_path.unlink()
            self._checkpoint_path.with_suffix(".meta.json").unlink(missing_ok=True)
        print(f"BPR saved to {self._model_path}")

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

    def _bpr_loss(self, u: int, i: int, j: int) -> float:
        # bpr loss per triplet, -ln(sigmoid(score_pos - score_neg))
        x_uij = np.dot(self.user_factors[u], self.item_factors[i]) - np.dot(
            self.user_factors[u], self.item_factors[j]
        )
        return float(np.log1p(np.exp(-x_uij)))

    # save resume helpers, bpr_checkpoint.npz and bpr_model.npz
    def _save_checkpoint(self, completed_iteration: int, loss_history: list[dict]) -> None:
        pos_items_json = {str(u): list(items) for u, items in self.user_pos_items.items()}
        np.savez_compressed(
            self._checkpoint_path,
            user_factors=self.user_factors,
            item_factors=self.item_factors,
            user_map_keys=np.array(list(self.user_map.keys()), dtype=np.int64),
            user_map_vals=np.array(list(self.user_map.values()), dtype=np.int64),
            item_map_keys=np.array(list(self.item_map.keys()), dtype=np.int64),
            item_map_vals=np.array(list(self.item_map.values()), dtype=np.int64),
            pos_items_json=json.dumps(pos_items_json),
        )
        meta = {
            "train_row_count": self._train_row_count,
            "completed_iteration": completed_iteration,
            "loss_history": loss_history,
        }
        self._checkpoint_path.with_suffix(".meta.json").write_text(json.dumps(meta), encoding="utf-8")

    def _load_checkpoint(self, path) -> None:
        data = np.load(path, allow_pickle=False)
        self.user_factors = data["user_factors"]
        self.item_factors = data["item_factors"]
        self.user_map = {int(k): int(v) for k, v in zip(data["user_map_keys"], data["user_map_vals"])}
        self.item_map = {int(k): int(v) for k, v in zip(data["item_map_keys"], data["item_map_vals"])}
        self.item_inv = {i: iid for iid, i in self.item_map.items()}
        pos_items_json = json.loads(str(data["pos_items_json"]))
        self.user_pos_items = {int(u): set(items) for u, items in pos_items_json.items()}

    def save(self, path) -> None:
        pos_items_json = {str(u): list(items) for u, items in self.user_pos_items.items()}
        np.savez_compressed(
            path,
            user_factors=self.user_factors,
            item_factors=self.item_factors,
            user_map_keys=np.array(list(self.user_map.keys()), dtype=np.int64),
            user_map_vals=np.array(list(self.user_map.values()), dtype=np.int64),
            item_map_keys=np.array(list(self.item_map.keys()), dtype=np.int64),
            item_map_vals=np.array(list(self.item_map.values()), dtype=np.int64),
            pos_items_json=json.dumps(pos_items_json),
        )
        path.with_suffix(".meta.json").write_text(
            json.dumps({"train_row_count": self._train_row_count}), encoding="utf-8"
        )

    def load(self, path) -> None:
        data = np.load(path, allow_pickle=False)
        self.user_factors = data["user_factors"]
        self.item_factors = data["item_factors"]
        self.user_map = {int(k): int(v) for k, v in zip(data["user_map_keys"], data["user_map_vals"])}
        self.item_map = {int(k): int(v) for k, v in zip(data["item_map_keys"], data["item_map_vals"])}
        self.item_inv = {i: iid for iid, i in self.item_map.items()}
        pos_items_json = json.loads(str(data["pos_items_json"]))
        self.user_pos_items = {int(u): set(items) for u, items in pos_items_json.items()}
        meta_path = path.with_suffix(".meta.json")
        if meta_path.exists():
            self._train_row_count = json.loads(meta_path.read_text(encoding="utf-8")).get("train_row_count", 0)

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
