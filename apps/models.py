import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import dash_core_components as dcc
import dash_html_components as html

from dash.dependencies import Input, Output
from app import app

# feature relationships
data = pd.read_pickle('data/aus_weather_cln_without_encoding.pkl')
data['Date'] = pd.to_datetime(data['Date'])
data['RainToday'] = data['RainToday'].apply(lambda x: 1 if x == 'Yes' else 0)
data['RainTomorrow'] = data['RainTomorrow'].apply(lambda x: 1 if x == 'Yes' else 0)

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


layout = html.Div([html.Section([
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
)
])
