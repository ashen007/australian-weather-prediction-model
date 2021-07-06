import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import dash_core_components as dcc
import dash_html_components as html

from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from feature_engine.encoding import OrdinalEncoder
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from dash.dependencies import Input, Output
from app import app

# data manipulations
data = pd.read_pickle('data/aus_weather_cln_without_encoding.pkl')
data['Date'] = pd.to_datetime(data['Date'])
data['RainToday'] = data['RainToday'].apply(lambda x: 1 if x == 'Yes' else 0)
data['RainTomorrow'] = data['RainTomorrow'].apply(lambda x: 1 if x == 'Yes' else 0)

# encode location feature
location_encoder = OrdinalEncoder(encoding_method='arbitrary',
                                  variables=['Location'])
location_encoder.fit(data)
data = location_encoder.transform(data)

# split data in to train and validation
input_feature = ['Location', 'Rainfall', 'Sunshine', 'WindGustSpeed', 'Humidity9am', 'Humidity3pm',
                 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'RainToday']
train_x, test_x, train_y, test_y = train_test_split(data[input_feature], data['RainTomorrow'],
                                                    test_size=0.2, random_state=64)

rainfall = 'Rainfall'
model_input_features = ['Sunshine', 'WindGustSpeed', 'Humidity9am', 'Humidity3pm',
                        'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'RainToday', 'RainTomorrow']


@app.callback(Output('feature-matrix', 'figure'),
              Input('sample-size', 'value'))
def update_sample_space(sample_size):
    temp = data.sample(n=sample_size, random_state=64)
    feature_relation = go.Figure(go.Parcoords(line=dict(color=temp['Rainfall'],
                                                        colorscale='ice'
                                                        ),
                                              dimensions=[dict(label=feature,
                                                               values=temp[feature],
                                                               range=[temp[feature].min(), temp[feature].max()]) for
                                                          feature in model_input_features]))
    feature_relation.update_layout(height=700,
                                   title=dict(text='Features that chosen by Pearson correlation coefficient algorithm',
                                              xanchor='center',
                                              yanchor='top',
                                              x=0.5,
                                              y=0.97,
                                              font=dict(size=18)
                                              ),
                                   margin=dict(t=100)
                                   )

    return feature_relation


@app.callback(Output('roc', 'figure'),
              Output('pred-prob', 'figure'),
              Input('model', 'value'))
def update_model(model):
    roc_figure = go.Figure()
    hist_figure = go.Figure()

    if model == 'knn':
        with open('models/rain_knn_model.pkl', 'rb') as file:
            model = pd.read_pickle(file)
    if model == 'log':
        with open('models/rain_logistic_model.pkl', 'rb') as file:
            model = pd.read_pickle(file)
    if model == 'tree':
        with open('models/rain_tree_model.pkl', 'rb') as file:
            model = pd.read_pickle(file)

    y_score = model.predict_proba(test_x)
    fpr, tpr, thresholds = roc_curve(y, y_score)

    return roc_figure, hist_figure


layout = html.Div([
    html.Section([
        dcc.Graph(id='feature-matrix'),
        html.Div([html.Label('sample size',
                             style={'margin-bottom': '12px'}),
                  dcc.Slider(id='sample-size',
                             min=2000,
                             max=16000,
                             value=2000,
                             marks={str(i): str(i) for i in [2000, 4000, 8000, 16000]})
                  ],
                 style={'padding': '50px 25px 25px'}),
    ],
        style={'box-sizing': 'border-box',
               'margin': '25px',
               'box-shadow': '0px 1px 3px rgb(0 0 0 / 12%),'
                             '0px 1px 2px rgb(0 0 0 / 24%)'
               }
    ),
    html.Section([
        html.Div([
            dcc.Graph(id='roc'),
            dcc.Graph(id='pred-prob')
        ]),
        html.Div([
            dcc.Dropdown(id='model',
                         options=[
                             {'label': 'k Nearest Neighbor', 'value': 'knn'},
                             {'label': 'Logistic Classification', 'value': 'log'},
                             {'label': 'CART Tree', 'value': 'tree'}
                         ],
                         value='log')
        ])
    ])
])
