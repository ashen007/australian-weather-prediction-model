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
                                              y=0.95,
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
                                         y=0.95,
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
                                      y=0.95,
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
                style={'width': '97%',
                       'margin': '0 auto',
                       'padding': '25px 0',
                       'background-color': '#fff'}
            ),
            html.Div([dcc.Graph(id='roc',
                                style={'width': '48.5%'}),
                      dcc.Graph(id='fpr-tpr',
                                style={'width': '48.5%'})],
                     style={'display': 'flex',
                            'justify-content': 'center',
                            'box-sizing': 'border-box',
                            'margin': '0 0',
                            'box-shadow': '0px 1px 3px rgb(0 0 0 / 12%),'
                                          '0px 1px 2px rgb(0 0 0 / 24%)'
                            })
        ])
    ])
])
