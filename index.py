import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app
from app import server
from apps import analysis_app, models

app.layout = html.Div([
    html.Div([
        html.Header(id='header-top',
                    children=[
                        html.Nav([
                            html.Div(id='main-title',
                                     children='AUSTRALIA TOMORROW WEATHER PREDICTION MODEL',
                                     style={'padding': '16px 0 16px 32px',
                                            'font-size': '24px',
                                            'font-family': 'sans-serif',
                                            'font-weight': '600',
                                            'width': '80%'
                                            }),
                            dcc.Link('Analysis Report', href='/apps/analysis_app',
                                     style={'width': '10%',
                                            'color': '#053BA6',
                                            'line-height': '2.2',
                                            'text-align': 'end',
                                            'box-sizing': 'border-box',
                                            'padding': '16px 8px 16px 0'
                                            }),
                            dcc.Link('Model', href='/',
                                     style={'width': '10%',
                                            'color': '#053BA6',
                                            'line-height': '2.2',
                                            'text-align': 'start',
                                            'box-sizing': 'border-box',
                                            'padding': '16px 0 16px 8px'
                                            })
                        ], style={'display': 'flex'}),
                    ], style={'background-color': '#fff',
                              'box-shadow': '0px -8px 20px 0px'
                              }
                    ),
    ], className='links'),
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content', children=[])
])


@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/apps/analysis_app':
        return analysis_app.layout
    if pathname == '/':
        return models.layout


if __name__ == '__main__':
    app.run_server(debug=False)
