import gensim.models
import pickle

import numpy as np
import pandas as pd

def createDoc2VecObject(corpus, data_col):
    wiki_gen = corpus[data_col].copy().to_list()
    wiki_gen_token = [doc.split(" ") for doc in wiki_gen]

    wiki_gensim = [gensim.models.doc2vec.TaggedDocument(d, [i]) for i, d in enumerate(wiki_gen_token)]

    return wiki_gensim

def createModel(corpus):
    model = gensim.models.doc2vec.Doc2Vec(vector_size = 50, min_count = 2, epochs = 40)
    model.build_vocab(corpus)
    model.train(corpus, total_examples = model.corpus_count, epochs = model.epochs)

    return model

def getDoc2VecScores(processing_func, query, model):
    query = processing_func(query)
    token_query = query.split(" ")

    vector = model.infer_vector(token_query)
    sims = model.dv.most_similar([vector], topn=len(model.dv))

    return sims

def getDoc2VecRecs(model_scores, top_n, corpus):
    results = pd.DataFrame(model_scores[:top_n], columns = ["idx", "cos_sim"])
    final_recs = corpus.loc[corpus.index[results["idx"]]]

    return final_recs

if __name__ == "__main__":
    pickle.HIGHEST_PROTOCOL = 4

    wiki_token = pd.read_pickle("/workspaces/vra_conf_rec_app/assets/wikicfp_corpus.pkl")

    d2v_corpus = createDoc2VecObject(wiki_token, "tokenized_soup")

    d2v_model = createModel(d2v_corpus)

    d2v_model.save("/workspaces/vra_conf_rec_app/assets/models/d2vModelv0.1.pkl")