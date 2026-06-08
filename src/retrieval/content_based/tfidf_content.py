#TF-IDF content-based retrieval

"""Each item has a text document (title + category + description).
  - TF-IDF converts each document into a vector of numbers.
  - User profile = average of vectors of items they clicked.
  - Recommendation = items whose vectors are most similar to user profile."""

from __future__ import annotations

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

