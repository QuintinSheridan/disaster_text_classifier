import json
import plotly
import pandas as pd

from plotly.graph_objs import Heatmap

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


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
df = pd.read_sql_table('train_test_data', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
   
    labels_df = df.drop(['id', 'original', 'message','genre'], axis =1)
    labels = labels_df.columns
    label_counts = labels_df.sum().values
 
    corr = labels_df.corr()
    sns_colorscale = [[0.0, '#3f7f93'], #cmap = sns.diverging_palette(220, 10, as_cmap = True)
                    [0.071, '#5890a1'],
                    [0.143, '#72a1b0'],
                    [0.214, '#8cb3bf'],
                    [0.286, '#a7c5cf'],
                    [0.357, '#c0d6dd'],
                    [0.429, '#dae8ec'],
                    [0.5, '#f2f2f2'],
                    [0.571, '#f7d7d9'],
                    [0.643, '#f2bcc0'],
                    [0.714, '#eda3a9'],
                    [0.786, '#e8888f'],
                    [0.857, '#e36e76'],
                    [0.929, '#de535e'],
                    [1.0, '#d93a46']]

    #heat = go.Heatmap(
    #z=corr,
    #x=labels,
    #y=labels,
    #xgap=1, ygap=1,
    #colorscale=sns_colorscale,
    #colorbar_thickness=20,
    #colorbar_ticklen=3,
    #hovertext =hovertext,
    #hoverinfo='text'
    #)


    #title = 'Correlation Matrix'               

    #layout = go.Layout(
    #             title_text=title, title_x=0.5, 
    #             width=600, height=600,
    #             xaxis_showgrid=False,
    #             yaxis_showgrid=False,
    #             yaxis_autorange='reversed') 
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
                    x=labels,
                    y=label_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Heatmap(
                    z=corr.values,
                    x=labels,
                    y=labels
                    #xgap=1, ygap=1,
                    #colorscale=sns_colorscale,
                    #colorbar_thickness=20,
                    #colorbar_ticklen=3,
                    #hovertext =labels,
                    #hoverinfo='text'
                    )
            ],

            'layout': {
                'title': 'Category Correlation',
                'xaxis_showgrid': 'False',
                'yaxis_showgrid': 'F',   
                'yaxis_autorange':'reversed'
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
