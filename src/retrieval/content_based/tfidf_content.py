#TF-IDF content-based retrieval

"""Each item has a text document (title + category + description).
  - TF-IDF converts each document into a vector of numbers.
  - User profile = average of vectors of items they clicked.
  - Recommendation = items whose vectors are most similar to user profile."""

from __future__ import annotations

import joblib
import numpy as np
import config


from sklearn.feature_extraction.text import TfidfVectorizer #Converts text documents into TF-IDF vectors
from sklearn.metrics.pairwise import cosine_similarity #Measures how similar two vectors are


class TfidfRetriever:
    def __init__(self):
        self.vectorizer=TfidfVectorizer(
            max_features=config.TFIDF_MAX_FEATURES,  # vocab size cap not lot of words
            ngram_range=config.TFIDF_NGRAM_RANGE,    # unigrams and bigrams
            min_df=config.TFIDF_MIN_DF,        # ignore very rare words
            max_df=config.TFIDF_MAX_DF,       # ignore very common words like "the", "and", "is", etc.


        )


        #item_id-> row index in tfidf matrix
        self.item_ids:list[int]=[]
        self.item_index: dict[int, int] = {}  # item_id to row index



        self.tfidf_matrix=None

    def fit(self,item_text, resume: bool = True):
        """Build TF-IDF vectors for all items."""

        # save resume, skip if already on disk
        if resume and config.TFIDF_MODEL_PATH.exists():
            print(f"Loading saved TF-IDF from {config.TFIDF_MODEL_PATH}")
            self.load(config.TFIDF_MODEL_PATH)
            return

        #step:list of all item ids sorted 
        self.item_ids=sorted(item_text.index.astype(int).tolist())

        #step item_id -> row index in tfidf matrix
        self.item_index={}
        for i in range(len(self.item_ids)): #item_ids = [42, 8912, 42864, 61600, etc]
            item_id = self.item_ids[i] #i = 0  is  item_id = 42
            self.item_index[item_id] = i

        #step3 collect text in same order as item_ids
        texts=[]
        for item_id in self.item_ids:
            texts.append(item_text[item_id])


        # step convert texts to TF-IDF 
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)

        # save resume write to artifacts so next run skips
        self.save(config.TFIDF_MODEL_PATH)
        print(f"TF-IDF saved to {config.TFIDF_MODEL_PATH}")

    def save(self, path) -> None:
        # save resume dump vectorizer + matrix
        joblib.dump(
            {
                "vectorizer": self.vectorizer,
                "item_ids": self.item_ids,
                "item_index": self.item_index,
                "tfidf_matrix": self.tfidf_matrix,
            },
            path,
        )

    def load(self, path) -> None:
        # save resume load back from artifacts
        data = joblib.load(path)
        self.vectorizer = data["vectorizer"]
        self.item_ids = data["item_ids"]
        self.item_index = data["item_index"]
        self.tfidf_matrix = data["tfidf_matrix"]

    def recommend(self,user_id,seen_items,k=config.TOP_K):
        if len(seen_items) == 0:
            return []

        #step  collect TF-IDF vectors of items user clicked
        vectors=[]
        for item_id in seen_items:
            if item_id in self.item_index:
                row_idx=self.item_index[item_id]
                vectors.append(self.tfidf_matrix[row_idx])


        if len(vectors) == 0:
            return []

        # step : average them into one user profile vector
        user_profile=vectors[0]
        for j in range(1, len(vectors)):
            user_profile = user_profile + vectors[j] #vectors[0] = [0.0, 0.6, 0.0, 0.0, 0.4] # steel beaker digital multimeter lab
        user_profile = user_profile / len(vectors)

        # step cosine similarity  profile vs every item
        scores = cosine_similarity(user_profile, self.tfidf_matrix).flatten()
        
        # step  rank items by score, skip seen, take top k
        ranked_indices = np.argsort(-scores)
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
        all_recommendations = {}
        for user_id in user_ids:
            user_seen = seen_items.get(user_id, set())
            recs = self.recommend(user_id, user_seen, k)
            all_recommendations[user_id] = recs
        return all_recommendations
