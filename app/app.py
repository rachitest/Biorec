import nltk
import streamlit as st

from io import StringIO

from cleaners.data_cleaning import readFolderST, betterDates, uniqueConfsPerYear, setLemmatizer, multiprocessApply, processCorpus, preprocessSentences
from recommenders.bm25_rec import createBMObject, getBM25Ranks, getRecs
from recommenders.doc2vec_rec import createDoc2VecObject, createModel, getDoc2VecScores, getDoc2VecRecs

st.set_page_config("Conference Recommendations", None, layout = "wide")
st.title("Conference Recommendations")
st.sidebar.title("Recommender Options")

# define functions locally

# pre download nltk data and set stop words + verb codes
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find("averaged_perceptron_tagger")
except LookupError:
    nltk.download("averaged_perceptron_tagger")
try:
    nltk.data.find("wordnet")
except LookupError:
    nltk.download("wordnet")
try:
    nltk.data.find("stopwords")
except LookupError:
    nltk.download("stopwords")

lemmatizer = setLemmatizer()

# get corpus from user to be used for the recommendations
input_files = st.sidebar.file_uploader("Choose your corpus", accept_multiple_files = True)

if input_files:
    wikicfp = readFolderST(input_files)
    wikicfp = uniqueConfsPerYear(wikicfp)
    wikicfp = betterDates(wikicfp)
    st.write("The following is your raw corpus:")
    st.dataframe(wikicfp)
else:
    st.write("You have not created a corpus yet")
    st.stop()

# process corpus to get it into the list of lists format used by BM25 and Doc2Vec
if wikicfp is not None:
    wiki_token = processCorpus(wikicfp)
    wiki_token = multiprocessApply(preprocessSentences, wiki_token, "soup", "processed_soup")
    st.write("The following is a sample of your tokenized corpus:")
    st.dataframe(wiki_token.head(1000), 1080)

# define batch element for choosing which type of recommender to use
with st.sidebar.form(key = "form_1"):
    rec_type = st.radio("Choose a recommender", ("BM25", "Doc2Vec"))
    query_type = st.radio("Choose query format", ("File", "Textbox"))
    number_of_recs = st.number_input("How many reccomendations would you like", 5, 50, value = 10, step = 5) 
    submit_button = st.form_submit_button(label = "Submit")

# create recommendations based on recommender algorithm and input type
if rec_type == "BM25" and query_type == "Textbox":
    query = st.sidebar.text_area("Enter your query", '', help = "Please enter the query you want conference recommendations for")
    if not query: 
        st.write("Please enter the query you want conference recommendations for")
        st.stop()
    bm25_model = createBMObject(wiki_token, "processed_soup")
    query_scores = getBM25Ranks(preprocessSentences, query, bm25_model)
    recs = getRecs(query_scores, number_of_recs, wikicfp)
    st.write(f"Here are the top {number_of_recs} recommendations for your query 🎉:")
    st.table(recs[["Conference Title", "Conference Webpage"]])
    st.stop()
elif rec_type == "BM25" and query_type == "File":
    query = st.sidebar.file_uploader("Choose your query", type = ["txt"], help = "Please select the file you want conference recommendations for (must be a txt file)")
    if query:
        stringio = StringIO(query.getvalue().decode("utf-8"))
        query_string = stringio.read()
    else: 
        st.write("Please enter the query you want conference recommendations for")
        st.stop()
    bm25_model = createBMObject(wiki_token, "processed_soup")
    query_scores = getBM25Ranks(preprocessSentences, query_string, bm25_model)
    recs = getRecs(query_scores, number_of_recs, wikicfp)
    st.write(f"Here are the top {number_of_recs} recommendations for your query 🎉:")
    st.table(recs[["Conference Title", "Conference Webpage"]])
    st.stop()
elif rec_type == "Doc2Vec" and query_type == "Textbox":
    query = st.sidebar.text_area("Enter your query", '', help = "Please enter the query you want conference recommendations for")
    if not query: 
        st.write("Please enter the query you want conference recommendations for")
        st.stop()
    d2v_corpus = createDoc2VecObject(wiki_token, "processed_soup")
    d2v_model = createModel(d2v_corpus)
    query_scores = getDoc2VecScores(preprocessSentences, query, d2v_model)
    recs = getDoc2VecRecs(query_scores, number_of_recs, wikicfp)
    st.write(f"Here are the top {number_of_recs} recommendations for your query 🎉:")
    st.table(recs[["Conference Title", "Conference Webpage"]])
    st.stop()
elif rec_type == "Doc2Vec" and query_type == "File":
    query = st.sidebar.file_uploader("Choose your query", type = ["txt"], help = "Please select the file you want conference recommendations for (must be a txt file)")
    if query:
        stringio = StringIO(query.getvalue().decode("utf-8"))
        query_string = stringio.read()
    else: 
        st.write("Please enter the query you want conference recommendations for")
        st.stop()
    d2v_corpus = createDoc2VecObject(wiki_token, "processed_soup")
    d2v_model = createModel(d2v_corpus)
    query_scores = getDoc2VecScores(preprocessSentences, query_string, d2v_model)
    recs = getDoc2VecRecs(query_scores, number_of_recs, wikicfp)
    st.write(f"Here are the top {number_of_recs} recommendations for your query 🎉:")
    st.table(recs[["Conference Title", "Conference Webpage"]])
    st.stop()