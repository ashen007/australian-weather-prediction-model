import numpy as np
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
from dash.dependencies import Input, Output, State
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
              Output('fpr-tpr', 'figure'),
              Output('conf-matrix', 'figure'),
              Output('hist-fig', 'figure'),
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

    y_score = model.predict_proba(test_x)[:, 1]
    predictions = model.predict(test_x)
    fpr, tpr, thresholds = roc_curve(test_y, y_score)

    comp_df = pd.DataFrame(np.array(test_y), columns=['true label'])
    comp_df['predicted_label'] = list(predictions)
    true_labels = comp_df[comp_df['true label'] == 1]
    false_labels = comp_df[comp_df['true label'] == 0]
    tp = np.sum(true_labels['true label'] == true_labels['predicted_label'])
    tn = true_labels.shape[0] - tp
    fp = np.sum(false_labels['true label'] == false_labels['predicted_label'])
    fn = false_labels.shape[0] - fp

    # roc and auc

    roc_figure = go.Figure(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        marker=dict(color='#0C7BB3'),
        fill='tozeroy',
    ))
    roc_figure.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    roc_figure.update_layout(title=f'ROC Curve, AUC={auc(fpr, tpr):.4f}',
                             height=700,
                             plot_bgcolor='#fff',
                             xaxis=dict(title='False Positive Rate',
                                        gridcolor='#F5F5F5'),
                             yaxis=dict(title='True Positive Rate',
                                        zeroline=False,
                                        gridcolor='#F5F5F5')
                             )

    # confusion matrix
    confusion_matrix = px.pie(values=[tp, tn, fp, fn],
                              names=['True positive', 'True negative', 'False positive', 'False negative'],
                              color_discrete_sequence=px.colors.sequential.ice)

    confusion_matrix.update_layout(height=600,
                                   title=dict(text='Confusion matrix',
                                              xanchor='center',
                                              yanchor='top',
                                              x=0.5,
                                              y=0.93,
                                              font=dict(size=18)
                                              ))

    # compare true labels and model predictions

    hist_figure = go.Figure()
    hist_figure.add_trace(go.Histogram(x=test_y,
                                       marker=dict(color='#193441'),
                                       name='True label',
                                       ))
    hist_figure.add_trace(go.Histogram(x=predictions,
                                       marker=dict(color='#56B9EA'),
                                       name='Predicted label',
                                       ))
    hist_figure.update_layout(height=600,
                              title=dict(text='True labels and predicted label counts',
                                         xanchor='center',
                                         yanchor='top',
                                         x=0.5,
                                         y=0.93,
                                         font=dict(size=18)
                                         ),
                              xaxis=dict(title='Class',
                                         gridcolor='#F5F5F5'),
                              yaxis=dict(title='Count',
                                         zeroline=False,
                                         gridcolor='#F5F5F5'),
                              plot_bgcolor='#fff',
                              barmode='stack',
                              bargap=0.1
                              )

    fig_hist = px.histogram(
        x=y_score,
        color=test_y,
        color_discrete_sequence=px.colors.sequential.Blues_r,
        nbins=50,
        labels=dict(color='True Labels', x='Score')
    )

    fig_hist.update_layout(height=600,
                           title=dict(text='Prediction probabilities',
                                      xanchor='center',
                                      yanchor='top',
                                      x=0.5,
                                      y=0.93,
                                      font=dict(size=18)
                                      ),
                           plot_bgcolor='#fff',
                           xaxis=dict(gridcolor='#F5F5F5'),
                           yaxis=dict(zeroline=False,
                                      gridcolor='#F5F5F5'))

    # compare fpr and tpr for every threshold
    df = pd.DataFrame({
        'False Positive Rate': fpr,
        'True Positive Rate': tpr
    }, index=thresholds)
    df.index.name = "Thresholds"
    df.columns.name = "Rate"

    fig_thresh = px.line(
        df, title='TPR and FPR at every threshold',
        height=700
    )

    fig_thresh.update_layout(plot_bgcolor='#fff',
                             xaxis=dict(gridcolor='#F5F5F5'),
                             yaxis=dict(zeroline=False,
                                        gridcolor='#F5F5F5'))

    return roc_figure, hist_figure, fig_thresh, confusion_matrix, fig_hist


@app.callback(Output('model-output', 'figure'),
              Output('predicted-label', 'children'),
              Input('submit-button', 'n_clicks'),
              Input('model', 'value'),
              State('location', 'value'),
              State('rainfall', 'value'),
              State('sunshine', 'value'),
              State('windGustSpeed', 'value'),
              State('humidity9am', 'value'),
              State('humidity3pm', 'value'),
              State('pressure9am', 'value'),
              State('pressure3pm', 'value'),
              State('cloud9am', 'value'),
              State('cloud3pm', 'value'),
              State('rain-today', 'value'))
def make_prediction(n_clicks, model, location, rainfall, sunshine, windGustSpeed,
                    humidity9am, humidity3pm, pressure9am, pressure3pm,
                    cloud9am, cloud3pm, raintoday):
    if model == 'knn':
        with open('models/rain_knn_model.pkl', 'rb') as file:
            model = pd.read_pickle(file)
    if model == 'log':
        with open('models/rain_logistic_model.pkl', 'rb') as file:
            model = pd.read_pickle(file)
    if model == 'tree':
        with open('models/rain_tree_model.pkl', 'rb') as file:
            model = pd.read_pickle(file)

    inputs = np.asarray([location, rainfall, sunshine, windGustSpeed,
                         humidity9am, humidity3pm, pressure9am, pressure3pm,
                         cloud9am, cloud3pm, raintoday]).reshape(1, -1)
    prob = model.predict_proba(inputs)
    pred = model.predict(inputs)

    if pred == 1:
        label = f'Will it rain: Yes ({np.round(prob[0][1] * 100, 2)}%)'
    elif pred == 0:
        label = f'Will it rain: No ({np.round(prob[0][0] * 100, 2)}%)'
    else:
        label = 'Error'

    model_output = go.Figure(go.Bar(
        x=['NO', 'YES'],
        y=[prob[0][0], prob[0][1]],
        marker=dict(color=['#193441', '#56B9EA'])
    ))

    model_output.update_layout(title='Prediction probability of each label',
                               height=600,
                               plot_bgcolor='#fff',
                               xaxis=dict(title='Labels',
                                          gridcolor='#F5F5F5'),
                               yaxis=dict(title='Probability',
                                          gridcolor='#F5F5F5')
                               )

    return model_output, label


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
            html.Div([dcc.Graph(id='pred-prob',
                                style={'width': '32%',
                                       'box-shadow': '0px 1px 3px rgb(0 0 0 / 12%),'
                                                     '0px 1px 2px rgb(0 0 0 / 24%)'
                                       }),
                      dcc.Graph(id='hist-fig',
                                style={'width': '32%',
                                       'box-shadow': '0px 1px 3px rgb(0 0 0 / 12%),'
                                                     '0px 1px 2px rgb(0 0 0 / 24%)'
                                       }),
                      dcc.Graph(id='conf-matrix',
                                style={'width': '32%',
                                       'box-shadow': '0px 1px 3px rgb(0 0 0 / 12%),'
                                                     '0px 1px 2px rgb(0 0 0 / 24%)'
                                       })],
                     style={'display': 'flex',
                            'flex-direction': 'row',
                            'flex-wrap': 'nowrap',
                            'justify-content': 'space-evenly',
                            'box-sizing': 'border-box',
                            'margin': '0 5px 25px'
                            }),
            html.Div([
                html.Div([
                    dcc.Dropdown(id='model',
                                 options=[
                                     {'label': 'k Nearest Neighbor', 'value': 'knn'},
                                     {'label': 'Logistic Classification', 'value': 'log'},
                                     {'label': 'CART Tree', 'value': 'tree'}
                                 ],
                                 value='log')
                ],
                    style={'width': '30%',
                           'margin': '0 0 0 20px'}
                )
            ],
                style={'margin': '0 25px',
                       'padding': '25px 0',
                       'background-color': '#fff',
                       'box-shadow': '0px 1px 3px rgb(0 0 0 / 12%),'
                                     '0px 1px 2px rgb(0 0 0 / 24%)'
                       }
            ),
            html.Div(
                html.Div([dcc.Graph(id='roc',
                                    style={'width': '50%'}),
                          dcc.Graph(id='fpr-tpr',
                                    style={'width': '50%'})],
                         style={'display': 'flex',
                                'justify-content': 'center',
                                'box-sizing': 'border-box',
                                'margin': '0 25px',
                                'box-shadow': '0px 1px 3px rgb(0 0 0 / 12%),'
                                              '0px 1px 2px rgb(0 0 0 / 24%)'
                                }
                         )
            )
        ])
    ]),
    html.Section([
        html.Div([
            html.Div([html.Label(children='Location',
                                 style={'margin-bottom': '12px'}),
                      dcc.Dropdown(id='location',
                                   options=[
                                       {'label': 'Albury', 'value': 0},
                                       {'label': 'BadgerysCreek', 'value': 1},
                                       {'label': 'Cobar', 'value': 2},
                                       {'label': 'CoffsHarbour', 'value': 3},
                                       {'label': 'Moree', 'value': 4},
                                       {'label': 'Newcastle', 'value': 5},
                                       {'label': 'NorahHead', 'value': 6},
                                       {'label': 'NorfolkIsland', 'value': 7},
                                       {'label': 'Penrith', 'value': 8},
                                       {'label': 'Richmond', 'value': 9},
                                       {'label': 'Sydney', 'value': 10},
                                       {'label': 'SydneyAirport', 'value': 11},
                                       {'label': 'WaggaWagga', 'value': 12},
                                       {'label': 'Williamtown', 'value': 13},
                                       {'label': 'Wollongong', 'value': 14},
                                       {'label': 'Canberra', 'value': 15},
                                       {'label': 'Tuggeranong', 'value': 16},
                                       {'label': 'MountGinini', 'value': 17},
                                       {'label': 'Ballarat', 'value': 18},
                                       {'label': 'Bendigo', 'value': 19},
                                       {'label': 'Sale', 'value': 20},
                                       {'label': 'MelbourneAirport', 'value': 21},
                                       {'label': 'Melbourne', 'value': 22},
                                       {'label': 'Mildura', 'value': 23},
                                       {'label': 'Nhil', 'value': 24},
                                       {'label': 'Portland', 'value': 25},
                                       {'label': 'Watsonia', 'value': 26},
                                       {'label': 'Dartmoor', 'value': 27},
                                       {'label': 'Brisbane', 'value': 28},
                                       {'label': 'Cairns', 'value': 29},
                                       {'label': 'GoldCoast', 'value': 30},
                                       {'label': 'Townsville', 'value': 31},
                                       {'label': 'Adelaide', 'value': 32},
                                       {'label': 'MountGambier', 'value': 33},
                                       {'label': 'Nuriootpa', 'value': 34},
                                       {'label': 'Woomera', 'value': 35},
                                       {'label': 'Albany', 'value': 36},
                                       {'label': 'Witchcliffe', 'value': 37},
                                       {'label': 'PearceRAAF', 'value': 38},
                                       {'label': 'PerthAirport', 'value': 39},
                                       {'label': 'Perth', 'value': 40},
                                       {'label': 'SalmonGums', 'value': 41},
                                       {'label': 'Walpole', 'value': 42},
                                       {'label': 'Hobart', 'value': 43},
                                       {'label': 'Launceston', 'value': 44},
                                       {'label': 'AliceSprings', 'value': 45},
                                       {'label': 'Darwin', 'value': 46},
                                       {'label': 'Katherine', 'value': 47},
                                       {'label': 'Uluru', 'value': 48}
                                   ],
                                   value=20)],
                     style={'width': '20%'}),
            html.Div([html.Div([html.Label(children='Rainfall',
                                           style={'margin-bottom': '12px'}),
                                dcc.Input(id='rainfall', type='number', step=0.001, value=6.2)],
                               style={'margin': '0 35px 0 0'}),
                      html.Div([html.Label(children='Sunshine',
                                           style={'margin-bottom': '12px'}),
                                dcc.Input(id='sunshine', type='number', step=0.001, value=10.5)],
                               style={'margin': '0 35px 0 0'}),
                      html.Div([html.Label(children='Wind gust speed',
                                           style={'margin-bottom': '12px'}),
                                dcc.Input(id='windGustSpeed', type='number', step=0.001, value=98.0)],
                               style={'margin': '0 35px 0 0'})],
                     style={'display': 'flex',
                            'margin': '20px 0'}),
            html.Div([html.Div([html.Label('Humidity at 9am',
                                           style={'margin-bottom': '12px'}),
                                dcc.Input(id='humidity9am', type='number', step=0.001, value=48.0)],
                               style={'margin': '0 35px 0 0'}),
                      html.Div([html.Label('Humidity at 3pm',
                                           style={'margin-bottom': '12px'}),
                                dcc.Input(id='humidity3pm', type='number', step=0.001, value=32.0)],
                               style={'margin': '0 35px 0 0'})],
                     style={'display': 'flex',
                            'margin': '20px 0'}
                     ),
            html.Div([html.Div([html.Label('Pressure at 9am',
                                           style={'margin-bottom': '12px'}),
                                dcc.Input(id='pressure9am', type='number', step=0.001, value=1002.900)],
                               style={'margin': '0 35px 0 0'}),
                      html.Div([html.Label('Pressure at 3pm',
                                           style={'margin-bottom': '12px'}),
                                dcc.Input(id='pressure3pm', type='number', step=0.001, value=999.300)],
                               style={'margin': '0 35px 0 0'})],
                     style={'display': 'flex',
                            'margin': '20px 0'}
                     ),
            html.Div([html.Div([html.Label('Cloud density at 9am (1-10)',
                                           style={'margin-bottom': '12px'}),
                                dcc.Input(id='cloud9am', type='number', step=0.001, value=2.0)],
                               style={'margin': '0 35px 0 0'}),
                      html.Div([html.Label('Cloud density at 3pm (1-10)',
                                           style={'margin-bottom': '12px'}),
                                dcc.Input(id='cloud3pm', type='number', step=0.001, value=3.0)],
                               style={'margin': '0 35px 0 0'})],
                     style={'display': 'flex',
                            'margin': '20px 0'}
                     ),
            html.Div([html.Label('Rain today',
                                 style={'margin-bottom': '12px'}),
                      dcc.RadioItems(id='rain-today',
                                     options=[
                                         {'label': 'Yes', 'value': 1},
                                         {'label': 'No', 'value': 0}
                                     ],
                                     value=1)],
                     style={'margin-bottom': '20px'}),
            html.Button(id='submit-button',
                        n_clicks=0,
                        children='Get prediction',
                        style={'margin': '20px 0px'}),

        ],
            style={'box-sizing': 'border-box',
                   'padding': '35px',
                   'width': '50%'}
        ),
        html.Div([
            html.Div(children=[dcc.Graph(id='model-output')]),
            html.Div(id='predicted-label',
                     style={'font-size': '32px',
                            'font-weight': 'bolder',
                            'letter-spacing': '1px'})
        ],
            style={'box-sizing': 'border-box',
                   'padding': '35px',
                   'width': '50%'}
        )
    ],
        style={'display': 'flex',
               'margin': '25px',
               'background-color': '#fff',
               'box-shadow': '0px 1px 3px rgb(0 0 0 / 12%),'
                             '0px 1px 2px rgb(0 0 0 / 24%)'
               }
    )
])
