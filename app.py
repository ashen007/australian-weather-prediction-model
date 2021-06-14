import plotly.graph_objs as go
import dash
import dash_core_components as dcc
import dash_html_components as html

app = dash.Dash()

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
                                      ])
                      ])

app.run_server(debug=False)
