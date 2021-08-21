# [VRA: Conference Recommendation App](https://share.streamlit.io/rachitest/vra_conference_rec_app/main/app/streamlit_app.py)

## Table of Contents
1. [Background](#background)
2. [Usage](#usage)

### Background 
The Conference Recommender App was developed to simplify the publishing lives of academics!

In the present day there is a glut of conferences to submit and no user-friendly way of parsing which conferences would be most relevant. The goal of this conference recommender is to accept abstracts or resumes *(eventually!)* as input and to recommend relevant conferences series that would be a good avenues for publication on topics similar to the input.

### Usage
The Conference Recommender App can provide recommendation using 3 different algorithms (click on the link to learn more about each algorithm):
- [Okapi BM25 (ATIRE)](https://dl.acm.org/doi/10.1145/2682862.2682863)
- [Doc2Vec](https://cs.stanford.edu/~quocle/paragraph_vector.pdf)
- [TF-IDF](https://www.emerald.com/insight/content/doi/10.1108/eb026526/full/html)

<details> 
    <summary> 
    Here's what the home page of the app looks like
    </summary>

<br/><br/>
![Home Page](/readme_assets/home_page.png)
</details>

You have two optional expanders:
<details> 
    <summary> "Raw Corpus Preview" (Click to find out!)
    </summary>

<br/><br/>
The raw corpus preview shows the raw, unedited data that the recommender uses for recommendations, is human readable
<br/><br/>
![Raw Corpus Preview](/readme_assets/raw_corpus.png)
</details> 

and 

<details> 
    <summary> 
        "Tokenized Corpus Preview" (Click to find out!)
    </summary> 

<br/><br/>
The raw corpus preview shows the tokenized "soup" of data that the recommendation algorithms utilized, the soup is not human readable
<br/><br/>
![Raw Corpus Preview](/readme_assets/tokenized_corpus.png)
</details>