import dash
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

data = pd.read_pickle('data/aus_weather_cln_without_encoding.pkl')
data['Date'] = pd.to_datetime(data['Date'])
model_input_features = ['Location', 'Rainfall', 'Sunshine', 'WindGustSpeed', 'Humidity9am', 'Humidity3pm',
                        'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'RainToday']

app = dash.Dash(__name__,
                external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'],
                assets_folder='assets/')


# aggregations and manipulation
def seasons(month):
    """
    decide the season that date belongs to
    :param month:
    :return:
    """
    if 3 <= month <= 5:
        return 'spring'
    if 6 <= month <= 8:
        return 'summer'
    if 9 <= month <= 11:
        return 'autumn'

    return 'winter'


data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Season'] = data['Date'].dt.month.apply(seasons)

years = data['Year'].unique()
season = ['spring', 'summer', 'autumn', 'winter']


# graphs and call-backs
@app.callback(Output('time-series', 'figure'),
              Input('location', 'value'),
              Input('year', 'value'))
def update_timeseries(location, year):
    temp = data[(data['Location'] == location) & (data['Year'] == year)]

    figure = go.Figure(go.Scatter(x=temp['Date'],
                                  y=temp['Rainfall'],
                                  mode='lines',
                                  marker=dict(color='#053BA6')))
    figure.update_layout(title=dict(text='How much rainfall does get',
                                    xanchor='center',
                                    yanchor='top',
                                    x=0.5,
                                    y=0.9,
                                    font=dict(size=18)
                                    ),
                         xaxis=dict(title='Date',
                                    gridcolor='#F5F5F5'),
                         yaxis=dict(title='Rainfall (mm)',
                                    gridcolor='#F5F5F5'),
                         plot_bgcolor='#fff',
                         height=600)
    figure.update_xaxes(rangeslider_visible=True)
    return figure


@app.callback(Output('pressure9am', 'figure'),
              Input('location', 'value'),
              Input('year', 'value'))
def update_pressure9am(location, year):
    pres9am_vs_rain = go.Figure()
    temp = data[(data['Location'] == location) & (data['Year'] == year)]

    for s in temp['Season'].unique():
        pres9am_vs_rain.add_trace(go.Scatter3d(x=temp[temp['Season'] == s]['Humidity9am'],
                                               y=temp[temp['Season'] == s]['Pressure9am'],
                                               z=temp[temp['Season'] == s]['Rainfall'],
                                               mode='markers',
                                               opacity=0.9,

                                               marker=dict(size=10,
                                                           line=dict(width=0.5,
                                                                     color='#fff')),
                                               name=s)
                                  )

    pres9am_vs_rain.update_layout(title=dict(text='Pressure and Humidity at morning',
                                             xanchor='center',
                                             yanchor='top',
                                             x=0.5,
                                             y=0.9,
                                             font=dict(size=18)),
                                  scene=dict(xaxis=dict(title='Humidity at 9am',
                                                        backgroundcolor="#fff",
                                                        gridcolor="#F5F5F5",
                                                        ),
                                             yaxis=dict(title='Pressure at 9am',
                                                        backgroundcolor="#fff",
                                                        gridcolor="#F5F5F5",
                                                        ),
                                             zaxis=dict(title='Rainfall (mm)',
                                                        backgroundcolor="#fff",
                                                        gridcolor="#F5F5F5",
                                                        )),
                                  height=650,
                                  margin=dict(b=0, l=0, r=0),
                                  legend=dict(orientation='h'),
                                  plot_bgcolor='#fff'
                                  )
    return pres9am_vs_rain


@app.callback(Output('pressure3pm', 'figure'),
              Input('location', 'value'),
              Input('year', 'value'))
def update_pressure3pm(location, year):
    pres3pm_vs_rain = go.Figure()
    temp = data[(data['Location'] == location) & (data['Year'] == year)]

    for s in temp['Season'].unique():
        pres3pm_vs_rain.add_trace(go.Scatter3d(x=temp[temp['Season'] == s]['Humidity3pm'],
                                               y=temp[temp['Season'] == s]['Pressure3pm'],
                                               z=temp[temp['Season'] == s]['Rainfall'],
                                               mode='markers',
                                               opacity=0.9,
                                               marker=dict(size=10,
                                                           line=dict(width=0.5,
                                                                     color='#fff')),
                                               name=s)
                                  )

    pres3pm_vs_rain.update_layout(title=dict(text='Pressure and Humidity at evening',
                                             xanchor='center',
                                             yanchor='top',
                                             x=0.5,
                                             y=0.9,
                                             font=dict(size=18)),
                                  scene=dict(xaxis=dict(title='Humidity at 3pm',
                                                        backgroundcolor="#fff",
                                                        gridcolor="#F5F5F5",
                                                        ),
                                             yaxis=dict(title='Pressure at 3pm',
                                                        backgroundcolor="#fff",
                                                        gridcolor="#F5F5F5",
                                                        ),
                                             zaxis=dict(title='Rainfall (mm)',
                                                        backgroundcolor="#fff",
                                                        gridcolor="#F5F5F5",
                                                        )),
                                  height=650,
                                  margin=dict(b=0, l=0, r=0),
                                  legend=dict(orientation='h'),
                                  )
    return pres3pm_vs_rain


# web page structure


app.layout = html.Div(id='main',
                      children=[
                          html.Header(id='header-top',
                                      children=[
                                          html.Nav([
                                              html.Div(id='main-title',
                                                       children='AUSTRALIA TOMORROW WEATHER PREDICTION MODEL',
                                                       style={'padding': '16px 0 16px 32px',
                                                              'font-size': '24px',
                                                              'font-family': 'sans-serif',
                                                              'font-weight': '600'})
                                          ])
                                      ], style={'background-color': '#fff',
                                                'box-shadow': '0px -8px 20px 0px'
                                                }),
                          html.Section([
                              html.Div([dcc.Dropdown(id='location',
                                                     options=[{'label': 'Albury', 'value': 'Albury'},
                                                              {'label': 'BadgerysCreek', 'value': 'BadgerysCreek'},
                                                              {'label': 'Cobar', 'value': 'Cobar'},
                                                              {'label': 'CoffsHarbour', 'value': 'CoffsHarbour'},
                                                              {'label': 'Moree', 'value': 'Moree'},
                                                              {'label': 'Newcastle', 'value': 'Newcastle'},
                                                              {'label': 'NorahHead', 'value': 'NorahHead'},
                                                              {'label': 'NorfolkIsland', 'value': 'NorfolkIsland'},
                                                              {'label': 'Penrith', 'value': 'Penrith'},
                                                              {'label': 'Richmond', 'value': 'Richmond'},
                                                              {'label': 'Sydney', 'value': 'Sydney'},
                                                              {'label': 'SydneyAirport', 'value': 'SydneyAirport'},
                                                              {'label': 'WaggaWagga', 'value': 'WaggaWagga'},
                                                              {'label': 'Williamtown', 'value': 'Williamtown'},
                                                              {'label': 'Wollongong', 'value': 'Wollongong'},
                                                              {'label': 'Canberra', 'value': 'Canberra'},
                                                              {'label': 'Tuggeranong', 'value': 'Tuggeranong'},
                                                              {'label': 'MountGinini', 'value': 'MountGinini'},
                                                              {'label': 'Ballarat', 'value': 'Ballarat'},
                                                              {'label': 'Bendigo', 'value': 'Bendigo'},
                                                              {'label': 'Sale', 'value': 'Sale'},
                                                              {'label': 'MelbourneAirport',
                                                               'value': 'MelbourneAirport'},
                                                              {'label': 'Melbourne', 'value': 'Melbourne'},
                                                              {'label': 'Mildura', 'value': 'Mildura'},
                                                              {'label': 'Nhil', 'value': 'Nhil'},
                                                              {'label': 'Portland', 'value': 'Portland'},
                                                              {'label': 'Watsonia', 'value': 'Watsonia'},
                                                              {'label': 'Dartmoor', 'value': 'Dartmoor'},
                                                              {'label': 'Brisbane', 'value': 'Brisbane'},
                                                              {'label': 'Cairns', 'value': 'Cairns'},
                                                              {'label': 'GoldCoast', 'value': 'GoldCoast'},
                                                              {'label': 'Townsville', 'value': 'Townsville'},
                                                              {'label': 'Adelaide', 'value': 'Adelaide'},
                                                              {'label': 'MountGambier', 'value': 'MountGambier'},
                                                              {'label': 'Nuriootpa', 'value': 'Nuriootpa'},
                                                              {'label': 'Woomera', 'value': 'Woomera'},
                                                              {'label': 'Albany', 'value': 'Albany'},
                                                              {'label': 'Witchcliffe', 'value': 'Witchcliffe'},
                                                              {'label': 'PearceRAAF', 'value': 'PearceRAAF'},
                                                              {'label': 'PerthAirport', 'value': 'PerthAirport'},
                                                              {'label': 'Perth', 'value': 'Perth'},
                                                              {'label': 'SalmonGums', 'value': 'SalmonGums'},
                                                              {'label': 'Walpole', 'value': 'Walpole'},
                                                              {'label': 'Hobart', 'value': 'Hobart'},
                                                              {'label': 'Launceston', 'value': 'Launceston'},
                                                              {'label': 'AliceSprings', 'value': 'AliceSprings'},
                                                              {'label': 'Darwin', 'value': 'Darwin'},
                                                              {'label': 'Katherine', 'value': 'Katherine'},
                                                              {'label': 'Uluru', 'value': 'Uluru'}],
                                                     value='Albury',
                                                     style={'width': '48%'}),
                                        html.Div(dcc.Slider(id='year',
                                                            min=np.min(years),
                                                            max=np.max(years),
                                                            marks={str(i): str(i) for i in years},
                                                            value=years.max()),
                                                 style={'width': '80%',
                                                        'margin': '0 auto'})],
                                       style={'width': '90%',
                                              'margin': '0 auto',
                                              'padding': '35px 25px',
                                              'display': 'flex'}),
                              dcc.Graph(id='time-series')
                          ],
                              style={'margin-top': '50px',
                                     'margin-bottom': '25px'}),
                          html.Section(id='compare',
                                       children=[
                                           html.Div(
                                               [
                                                   html.Div(
                                                       dcc.Graph(id='pressure9am'),
                                                       # style={'width': '48%'}
                                                   ),
                                                   html.Div(
                                                       dcc.Graph(id='pressure3pm'),
                                                       # style={'width': '48%'}
                                                   )],
                                               style={'width': '95%',
                                                      'margin': '0 auto',
                                                      'display': 'flex',
                                                      'flex-direction': 'row',
                                                      'justify-content': 'space-around'}
                                           )
                                       ],
                                       style={'margin': '25px 0'})
                      ],
                      )

app.run_server(debug=False)
