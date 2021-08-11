import pandas as pd
import numpy as np

import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def createVectorizer():
    return TfidfVectorizer()

def createTFIDFModel(tokenized_corpus, data_column, vectorizer):
    tfidf_model = vectorizer.fit_transform(tokenized_corpus[data_column])
    
    return tfidf_model

def processQuery(query, processing_func, vectorizer):
    query = processing_func(query)
    query = vectorizer.transform([query])
    
    return query

def getTFIDFRecs(model_scores, top_n, corpus):
    results = np.argsort(model_scores)[-top_n:]
    results = np.flip(results)

    final_recs = corpus.loc[corpus.index[results]]

    return final_recs

if __name__ == "__main__":
    wiki_token = pd.read_pickle("/assets/wikicfp_corpus.pkl")

    tfidfvec = createVectorizer()

    tfidf_model = createTFIDFModel(wiki_token, "tokenized_soup", tfidfvec)
    
    pickle.dump(tfidf_model, open("/workspaces/vra_conf_rec_app/assets/models/tfidfModelv0.1.pkl", "wb"))