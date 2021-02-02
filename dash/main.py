# Basics Requirements
import pathlib
import os
import dash
from dash.dependencies import Input, Output, State, ClientsideFunction
from dash.exceptions import PreventUpdate
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import plotly.express as px
import geopandas as gpd
import json


# Dash Bootstrap Components
import dash_bootstrap_components as dbc

# Data
import math
import numpy as np
import datetime as dt
import pandas as pd
import json

# Recall app
from app_handle import app

# LOAD THE DIFFERENT FILES
from lib import sidebar, stats, title

app.layout = html.Div(
    [title.title,stats.stats, sidebar.sidebar,],
    className="ds4a-app",  # You can also add your own css files by locating them into the assets folder
)

color_discrete_map = {"1":'#636EFA',"2":'#EF553B',"3":'#00CC96',"4":'#AB63FA',"5":'#FFA15A',"6":'#19D3F3'}
#[, , , , , , '#FF6692', '#B6E880', '#FF97FF', '#FECB52']

# Load the data
df = pd.read_csv('data/CleanDataScatter.csv')


@app.callback(
    Output('map', 'figure'),
    Output('violent-crime-bar','figure'),
    Input('year_dropdown', 'value'),
    Input('city_dropdown', 'value'),
    Input('ds4a-checklist', 'value')
)
def update_map(year_value,city_value,ds4a_value):
    df['Group'] = df['Group'].apply(str).apply(str.strip)
    # Select the appropriate groups
    df_group=df[df.Group.isin(list(ds4a_value))]
    # get the correct year:
    if(year_value!='All Years'):
        df_group= df_group[df_group.Year==int(year_value)]
    # get the correct city
    if not isinstance(city_value, list):
        city_value = [city_value]
    if ('All Cities' not in city_value):
        df_group = df_group[df_group.City.isin(city_value)]

    fig3 = px.scatter_geo(df_group.sort_values(by='Group'),
                    lat="Latitude",
                    lon="Longitude",
                    hover_name="City Name",
                    projection="albers usa",
                    size="Violent Crime",
                    color_discrete_map=color_discrete_map,
                    hover_data={'City':True,'Violent Crime':True,'Property Crime':True,'Police Spending':True,'Total Revenue':True,"Longitude":False,"Latitude":False,"Group":False},
                    color="Group")


    fig3.update_layout(
        title = 'Metropolitan Areas by Cluster Label'
    )


    by_group = df_group.groupby(['Year','Group']).mean().reset_index()
    by_group['Group'] = by_group['Group'].astype(str) 
    by_group['Year'] = by_group['Year'].astype(int)
    fig = px.bar(by_group, x="Year", y="Violent Crime",color='Group',color_discrete_map=color_discrete_map,)
    fig.update_layout(barmode='group',title='Violent Crime Over Time (Rate per 100,000)')

    return fig3,fig


@app.callback(
    Output('scatter', 'figure'),
    Input('demo-dropdown', 'value'))
def update_scatter(input_value):
    if(input_value not in list(df.columns)):
        input_value='Violent Crime'
    df['Group'] = df['Group'].apply(str)
    fig= px.scatter(df.sort_values(by='Group'), x="Police Spending", y=input_value,color='Group',hover_data=['City','Violent Crime','Police Spending','Population'])

    fig.update_layout(transition_duration=500)

    return fig

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port="8050", debug=True)