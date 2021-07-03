import dash
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
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month



years = data['Year'].unique()


# graphs and call-backs
@app.callback(Output('time-series', 'figure'),
              Input('location', 'value'),
              Input('year', 'value'))
def update_graph(location, year):
    temp = data[(data['Location'] == location) & (data['Year'] == year)]

    figure = go.Figure(go.Scatter(x=temp['Date'],
                                  y=temp['Rainfall'],
                                  mode='lines',
                                  marker=dict(color='#053BA6')))
    figure.update_layout(xaxis=dict(title='Date',
                                    gridcolor='#F5F5F5'),
                         yaxis=dict(title='Rainfall (mm)',
                                    gridcolor='#F5F5F5'),
                         plot_bgcolor='#fff',
                         height=600)
    return figure


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
                                        dcc.Dropdown(id='year',
                                                     options=[{'label': str(i), 'value': i} for i in years],
                                                     value=years.max(),
                                                     style={'width': '48%'})],
                                       style={'width': '90%', 'margin': '0 auto', 'display': 'flex'}),
                              dcc.Graph(id='time-series')
                          ],
                              style={'margin-top': '50px'})
                      ],
                      )

app.run_server(debug=False)
