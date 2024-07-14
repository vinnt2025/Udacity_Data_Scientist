import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    """Tokenize and clean text data.

    This function performs the following steps:
    1. Tokenizes the input text into individual words.
    2. Lemmatizes each token to its base form.
    3. Converts each token to lowercase and removes any surrounding whitespace.
    4. Collects and returns the cleaned tokens as a list.

    Parameters:
    text (str): The text to be tokenized and cleaned.

    Returns:
    list: A list of cleaned and lemmatized tokens."""
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterMessages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """Render the index page with visualizations of category distributions.
    Parameters:
        None
    Returns:
        flask.Response: The rendered HTML template containing the visualizations."""
    
    # Extract data needed for visuals
    X = df['message']
    Y = df.iloc[:,4:]
    
    # Data for bar plot
    category_pct = Y.mean().sort_values(ascending= False)
    category = category_pct.index.str.replace('_', ' ')
    
    # Data for message counts by genre
    message_counts_by_genre = df.groupby('genre').count()['message']
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
       {
            'data': [
                Bar(
                    x=category,
                    y=category_pct
                )
            ],
            'layout': {
                'title': {
                    'text': 'Categories Type Distribution',
                    'font': {'size': 18, 'family': 'Arial', 'color': 'black', 'weight': 'bold'}
                },
                'yaxis': {
                    'title': {
                        'text':"Percentage",
                        'font': {'size': 15, 'family': 'Arial', 'color': 'black', 'weight': 'bold'}
                    },
                    'tickformat': ',.0%',
                },
                'xaxis': {
                    'title': {
                        'text':"Category Type",
                        'font': {'size': 15, 'family': 'Arial', 'color': 'black', 'weight': 'bold'}
                    },
                    'tickangle': 45
                },
                'height': 800, 
                'width': 1100,
                'margin': {
                    'l': 150, 
                    'r': 100,
                    'b': 200, 
                    't': 100, 
                    'pad': 4 
                }
            }
        },
        {
            'data': [
                Bar(
                    x=message_counts_by_genre.index,
                    y=message_counts_by_genre.values
                )
            ],
            'layout': {
                'title': 'Distribution of Messages by Genre',
                'xaxis': {'title': 'Genre'},
                'yaxis': {'title': 'Number of Messages'}
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
    """Handle user input and display classification results.

    This function retrieves the user's input query from the web request, uses the trained model 
    to predict the classifications for the input, and prepares the results for rendering.
    Parameters:
        None
    Returns:
        flask.Response: The rendered HTML template displaying the query and classification results."""
    
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
    """Start the Flask web application.
    
    Parameters:
        None
    Returns:
        None"""
    
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()