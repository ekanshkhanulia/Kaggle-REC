from __future__ import annotations
import joblib
import numpy as np
import config
from rank_bm25 import BM25Okapi

# BM25 works like search.
# Each item in item_meta is one document (title + category + description etc).
# We split that text into words and build an index over all items.
# For a user we take the words from items they already clicked and treat that as a search query.
# BM25 scores every item by how well its words match that query, rare matching words score higher.
# We return the highest scoring unseen items as recommendations.


def tokenize(text:str)-> list[str]:
    #  split item text into words
    return text.lower().split()

class BM25Retriever:
    def __init__(self):
        # step: empty state, filled in fit()
        self.item_ids: list[int] = []
        self.item_index: dict[int, int] = {}       # item_id ,row in bm25 index
        self.tokenized_corpus: list[list[str]] = []  # one word list per item
        self.bm25: BM25Okapi | None = None


    def fit(self, item_text,resume:bool=True)->None:
        #  load saved model if exists
        if resume and config.BM25_MODEL_PATH.exists():
            print(f"Loading saved BM25 from {config.BM25_MODEL_PATH}")
            self.load(config.BM25_MODEL_PATH)
            return

        #sorted item id
        self.item_ids=sorted(item_text.index.astype(int).tolist())


        #item_id -> row index
        self.item_index = {}
        for i in range(len(self.item_ids)):
            self.item_index[self.item_ids[i]] = i

        # tokenize each item's meta text
        self.tokenized_corpus = []
        for item_id in self.item_ids:
            self.tokenized_corpus.append(tokenize(str(item_text[item_id])))


        #  build bm25 search index over all items
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        # save 
        self.save(config.BM25_MODEL_PATH)
        print(f"BM25 saved to {config.BM25_MODEL_PATH}")


    def save(self, path) -> None:
        # save tokenized corpus  plus mappings, rebuild bm25 on load
        joblib.dump(
            {
                "item_ids": self.item_ids,
                "item_index": self.item_index,
                "tokenized_corpus": self.tokenized_corpus,
            },
            path,
        )


    def load(self, path) -> None:
        #  load back and rebuild bm inde
        data = joblib.load(path)
        self.item_ids = data["item_ids"]
        self.item_index = data["item_index"]
        self.tokenized_corpus = data["tokenized_corpus"]
        self.bm25 = BM25Okapi(self.tokenized_corpus)



    def recommend(self, user_id, seen_items, k=config.TOP_K):
        # cold user with no history
        if len(seen_items) == 0 or self.bm25 is None:
            return []

        query = self.build_user_profile(seen_items)
        if query is None:
            return []

        # score every item against user query
        scores = self.bm25.get_scores(query)
        ranked_indices = np.argsort(-scores)

        #  skip seen items, take top k
        recommendations = []
        for idx in ranked_indices:
            item_id = self.item_ids[idx]
            if item_id in seen_items:
                continue
            recommendations.append(item_id)
            if len(recommendations) == k:
                break
        return recommendations

    def recommend_all(self, user_ids, seen_items, k=config.TOP_K):
        #  loop users same 
        all_recommendations = {}
        for user_id in user_ids:
            user_seen = seen_items.get(user_id, set())
            all_recommendations[user_id] = self.recommend(user_id, user_seen, k)
        return all_recommendations
   

    def build_user_profile(self, seen_items: set[int]):
        # all words from items user clicked = search query
        tokens: list[str] = []
        # loop every item id the user already clicked in train
        for item_id in seen_items:
            if item_id in self.item_index: # skip if this item is not in our bm25 index
                row_idx = self.item_index[item_id] #get row number
                tokens.extend(self.tokenized_corpus[row_idx])# add all words from that item into the query list
        if not tokens:
            return None
        return tokens
    
    def score_candidates(self, user_profile, item_ids: list[int]) -> np.ndarray:
        #  reranker needs bm25 score on specific candidate items
        if user_profile is None or self.bm25 is None:
            return np.full(len(item_ids), np.nan, dtype=np.float32) #no query or no bm25 index built yet return
        all_scores = self.bm25.get_scores(user_profile)# score every item in catalog against the user query
        scores = np.full(len(item_ids), np.nan, dtype=np.float32)
        for k, item_id in enumerate(item_ids):
            if item_id in self.item_index:#only fill score if item exists in bm25 index
                scores[k] = float(all_scores[self.item_index[item_id]])
        return scores# return one score per candidate,





