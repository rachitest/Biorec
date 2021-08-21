import numpy as np
import pandas as pd

import pickle 

from rank_bm25 import BM25Okapi as bm

def createBMObject(corpus, data_col):
    wiki_bm = corpus[data_col].copy().to_list()
    wiki_bm_token = [doc.split(" ") for doc in wiki_bm]

    bm25 = bm(wiki_bm_token)

    return bm25

def getBM25Ranks(processing_func, query, model):
    query = processing_func(query)
    token_query = query.split(" ")
    doc_scores = model.get_scores(token_query)

    return doc_scores

def getRecs(model_scores, top_n, corpus):
    results = np.argsort(model_scores)[-top_n:]
    results = np.flip(results)

    final_recs = corpus.loc[corpus.index[results]]

    return final_recs

if __name__ == "__main__":
    pickle.HIGHEST_PROTOCOL = 4

    wiki_token = pd.read_pickle("/workspaces/vra_conf_rec_app/assets/wikicfp_corpus.pkl")

    bm25_model = createBMObject(wiki_token, "tokenized_soup")

    pickle.dump(bm25_model, open("/workspaces/vra_conf_rec_app/assets/models/bm25Modelv0.1.pkl", "wb"))