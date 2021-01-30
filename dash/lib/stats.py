import dash
from dash.dependencies import Input, Output, State, ClientsideFunction
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px


from datetime import datetime as dt
import json
import numpy as np
import pandas as pd
import os
import geopandas as gpd
# Recall app
from app import app

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

# create the df for visualizing violent crime per group
by_group = df.groupby(['year','Group']).mean().reset_index()
by_group['Group'] = by_group['Group'].astype(str) 
by_group['violent_crime'] = by_group['violent_crime'] * 10**5
fig = px.bar(by_group, x="year", y="violent_crime",color='Group')
fig.update_layout(barmode='group',title='Violent Crime Over Time (Rate per 100,000)')




fig2 = px.scatter(df, x="police", y="violent_crime",color='Group',hover_data=['city_merge_name','violent_crime','police','population'])


fig3 = px.choropleth(df[df.year==2017], geojson=json.loads(relevant_areas.to_json()), color="Group",
                    locations="city_merge_name", featureidkey="properties.city_name",
                    projection="albers usa"
                   )
fig3.update_layout(
        title = 'Metropolitan Areas by Cluster Label'
)

stats = html.Div(
    [        
        html.P('2020 was a pivotal year around the world within the context of the conversation around racial equity, justice, and equality. Specifically, in the United States, thousands of protestors and activists have taken to the streets to call for police reform and support for policies that effectively address the underlying factors that contribute to crime, poverty, and homelessness. One of the most poignant demands of these individuals has been a call for a complete reimagining of law enforcement in the United States. This project centers on one facet of this demand which is the reallocation of police budgets or more broadly “defunding the police.” What does this call entail and is there merit to considering its implementation?'),
        
        dcc.Graph(
        id='violent-crime-bar',
        figure=fig
        ),

        html.P('Specifically, our goal is to discover if a reallocation of city budgets will result in an improved quality of life for city residents. However before proceeding, what do we mean by an “improved quality of life?”'),

        dcc.Graph(
        id='map',
        figure=fig3
        ),

        # drop down menu
        dcc.Dropdown(
        id='demo-dropdown',
        options=labels,
        value='violent_crime',
        style=dict(
            width='40%',
        )
        ),

        dcc.Graph(
        id='scatter',
        figure=fig2
        )
    ],
    className="ds4a-body",
)
