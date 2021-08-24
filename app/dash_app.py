import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import pandas as pd
import plotly.express as px

import pickle

from pathlib import Path

from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
from cleaners.data_cleaning import preprocessSentences
from recommenders.bm25_rec import getBM25Ranks, getRecs
from recommenders.doc2vec_rec import getDoc2VecScores, getDoc2VecRecs
from recommenders.tfidf_rec import createVectorizer, createTFIDFModel, processQuery, getTFIDFRecs

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets = external_stylesheets)

app.layout = html.Div([
    dcc.Tabs(id='tabs-example', value='tab-1', children=[
        dcc.Tab(label='Tab one', value='tab-1'),
        dcc.Tab(label='Tab two', value='tab-2'),
    ]),
    html.Div(id='tabs-example-content')
])

@app.callback(Output('tabs-example-content', 'children'),
                Input('tabs-example', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H3('Tab content 1')
        ])
    elif tab == 'tab-2':
        return html.Div([
            html.H3('Tab content 2')
        ])


if __name__ == '__main__':
    app.run_server(debug=True)

if __name__ == '__main__':
    app.run_server(debug=True)