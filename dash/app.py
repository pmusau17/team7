# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import geopandas as gpd
import json

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


# Load the data
df = pd.read_csv('../CleanData/CompleteMerged.csv')

# Load the GeoJson Data
relevant_areas = gpd.read_file('../CleanData/relevant_areas.json')

# create options for drop down
labels = []
for i in list(df.columns):
    item = {'label': '{}'.format(i), 'value': '{}'.format(i)}
    labels.append(item)


average_crime_per_year = df.groupby('year').mean().reset_index()
fig = px.line(average_crime_per_year, x="year", y="violent_crime")
fig2 = px.scatter(df, x="police", y="violent_crime",color='Group',hover_data=['city_merge_name','violent_crime','police','population'])


fig3 = px.choropleth(df[df.year==2017], geojson=json.loads(relevant_areas.to_json()), color="Group",
                    locations="city_merge_name", featureidkey="properties.city_name",
                    projection="albers usa"
                   )
fig3.update_layout(
        title = 'Metropolitan Areas by Cluster Label'
)


app.layout = html.Div(children=[
    html.H1(children='On Defunding Police'),

    html.Div(children='''
        Violent Crime over time.
    '''),
    dcc.Graph(
        id='example-graph',
        figure=fig
    ),

    # drop down menu
    dcc.Dropdown(
        id='demo-dropdown',
        options=labels,
        value='violent_crime'
    ),

    dcc.Graph(
        id='scatter',
        figure=fig2
    ),
    html.H6("Change the value to display another variable!"),
    html.Div(["Input: ",
              dcc.Input(id='my-input', value='initial value', type='text')]),
    dcc.Graph(id='graph-with-input'),

    dcc.Graph(
        id='map',
        figure=fig3
    ),
])

# the decorator defines the callback
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


@app.callback(
    Output('scatter', 'figure'),
    Input('demo-dropdown', 'value'))
def update_scatter(input_value):
    fig= px.scatter(df, x="police", y=input_value,color='Group',hover_data=['city_merge_name','violent_crime','police','population'])

    fig.update_layout(transition_duration=500)

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)