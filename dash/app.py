# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
# df = pd.DataFrame({
#     "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
#     "Amount": [4, 1, 2, 2, 4, 5],
#     "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
# })

df = pd.read_csv('../CleanData/CompleteMerged.csv')


average_crime_per_year = df.groupby('year').mean().reset_index()
fig = px.line(average_crime_per_year, x="year", y="violent_crime")

app.layout = html.Div(children=[
    html.H1(children='On Defunding Police'),

    html.Div(children='''
        Violent Crime over time.
    '''),

    dcc.Graph(
        id='example-graph',
        figure=fig
    ),
    html.H6("Change the value to display another variable!"),
    html.Div(["Input: ",
              dcc.Input(id='my-input', value='initial value', type='text')]),
    dcc.Graph(id='graph-with-input')
])

@app.callback(
    Output('graph-with-input', 'figure'),
    Input('my-input', 'value'))
def update_figure(input_value):
    if input_value in list(df.columns):
        fig = px.line(average_crime_per_year, x="year", y=input_value)
    else:
        fig = px.line(average_crime_per_year, x="year", y="violent_crime")

    fig.update_layout(transition_duration=500)

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)