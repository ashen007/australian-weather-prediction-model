import dash
import pandas as pd
import plotly.graph_objs as go
import dash_core_components as dcc
import dash_html_components as html

app = dash.Dash(__name__,
                assets_folder='assets/')

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
                          html.Section(id='time-series',
                                       children=[])
                      ])

app.run_server(debug=False)
