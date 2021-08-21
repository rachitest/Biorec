# VRA: Conference Recommendation App
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/rachitest/vra_conference_rec_app/main/app/streamlit_app.py)
## Table of Contents

1. [Background](#background)
2. [Layout](#Layout)
    1. [Home Page](#homepage)
    2. [Side Bar](#sidebar)
3. [Usage](#usage)
    1. [Text Guide](#textguide)
    2. [GIF Guide](#gifguide)
4. [Notes](#notes)
5. [Acknowledgements](#acknowledgements)

### Background

The Conference Recommender App was developed to simplify the publishing lives of academics!

In the present day there is a glut of conferences to submit and no user-friendly way of parsing which conferences would be most relevant. The goal of this conference recommender is to accept abstracts or resumes *(eventually!)* as input and to recommend relevant conferences series that would be a good avenues for publication on topics similar to the input.

### Layout

Here's what the home page of the app looks like:
<br/><br/>
![Home Page](/readme_assets/home_page.png)

<h4 id="homepage">Home Page</h4>

You have two optional expanders:
<details> 
    <summary> "Raw Corpus Preview": Shows the raw corpus preview shows the raw, unedited data that the recommender uses for recommendations; is human readable.
    </summary>

<br/><br/>
![Raw Corpus Preview](/readme_assets/raw_corpus.png)
</details> 

<details> 
    <summary> 
        "Tokenized Corpus Preview": Shows the raw corpus preview shows the tokenized "soup" of data that the recommendation algorithms utilize; the soup is not human readable.
    </summary> 

<br/><br/>
![Raw Corpus Preview](/readme_assets/tokenized_corpus.png)
</details>

<h4 id="sidebar">Side Bar</h4>

The side bar contains input and output options for the Conference Recommender App.

The Conference Recommender App can provide recommendation using 3 different algorithms:

- [Okapi BM25 (ATIRE)](https://dl.acm.org/doi/10.1145/2682862.2682863)
- [Doc2Vec](https://cs.stanford.edu/~quocle/paragraph_vector.pdf)
- [TF-IDF](https://www.emerald.com/insight/content/doi/10.1108/eb026526/full/html)

The user also gets to choose the format in which the query is entered in to the recommender. Currently available choices are as a text file (.txt extension only) or as a plaintext query into a text box.

The user also gets to choose the number of recommendations to recieve up to a maximum of 50 recommendation. The user can type out the exact number of recommendations required, or use the plus/minus buttons to increment or decrement the counter by steps of 5.

### Usage

<h4 id="textguide">Text Guide</h4>

Use the recommender in 7 easy steps:

1. Select the recommendation algorithm &#8594; **The default recommendation algorithm is BM25**
2. Select the input method &#8594; **The default input format is text file**
3. Enter the number of desired recommendations &#8594; **The default number of recommendations is 10**

*Upon the completion of step 3 press the  "Submit" button located below the number of recommendations*

4. Enter the query in the query input option available &#8594; **The default input option is file**
    <details> 
        <summary> 
            If you selected file you should see the following file picker 
        </summary>
    
    ![file picker](/readme_assets/file_picker.png) 
    </details>

    <details> 
        <summary> 
            If you selected text box you should see the following text box (press Cntrl + Enter to submit text box query)
        </summary>
    
    ![file picker](/readme_assets/text_box.png) 
    </details>

5. View your recommendations
6. Choose if you want to download the recommendations (Download only available in the form of a csv file) &#8594; **The default is Yes**
7. Repeat steps 1-6 for as many permutations as desired

<h4 id="gifguide">GIF Guide</h4>

Here I demonstrate how to use the Conference Recommendation App with the Doc2Vec algorithm and save the resulting recommendations.

![gif demo](/readme_assets/doc2vec_demo_upscaled_cropped.gif)

### Notes

For reviewers:

1. The downloaded csv file will contain a column with the title "Rating"
    1. Please use that column to provide ratings for the recommendations proivded on a scale from 1 to 5
        1. Where 1 indicates that the recommendation was not pertinent to your query and 5 indicates that the recommendation was completely pertinent to your query
2. Please give ratings for each recommendation algorithm

### Acknowledgements
