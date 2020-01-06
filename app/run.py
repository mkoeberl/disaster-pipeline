import json
import plotly
import pandas as pd
import numpy as np
import pickle

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objects import Bar, Histogram
from sqlalchemy import create_engine
from pathlib import Path

import joblib


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages_categories', engine)

import os
path = Path(os.getcwd()).parent/'models'/'classifier.pkl'
print(str(path))

# load model
model = joblib.load(str(path))
#with open(str(path),'rb') as file:
#    model = pickle.load(file)




# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    category_names = list(df.columns[4:])
    values = np.sort(np.unique(df[category_names                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ].to_numpy().reshape((-1))))
    category_occurrence = {}                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
    for value in values:
        category_occurrence[value] = df[category_names].applymap(lambda x: 1 if x==value else 0).sum()

    lengths = df['message'].apply(lambda x: len(x))



    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    name='0',
                    x=category_names,
                    y=category_occurrence[0]
                ),
                Bar(
                    name='1',
                    x=category_names,
                    y=category_occurrence[1]
                ),
                Bar(
                    name='2',
                    x=category_names,
                    y=category_occurrence[2]
                )
            ],
            'layout':{
                'title': 'Distribution per category',
                'yaxis': {
                    'title': 'Count'
                },
                'xaxis': {
                    'title': 'Category'
                },
                'barmode': 'group'
            }
        },
        {
            'data' : [
                Histogram(
                    x=lengths
                )
            ],
            'layout':{
                'title': 'Histogram of message length in characters',
                'xaxis': {
                    'title': 'Message length'
                },
                'yaxis': {
                    'title': 'Count'
                }
            }

        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()