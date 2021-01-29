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
from lib import sidebar, stats

app.layout = html.Div(
    [stats.stats, sidebar.sidebar,],
    className="ds4a-app",  # You can also add your own css files by locating them into the assets folder
)



# Load the data
df = pd.read_csv('../CleanData/CompleteMerged.csv')
# Load the GeoJson Data
relevant_areas = gpd.read_file('../CleanData/relevant_areas.json')



@app.callback(
    Output('scatter', 'figure'),
    Input('demo-dropdown', 'value'))
def update_scatter(input_value):
    fig= px.scatter(df, x="police", y=input_value,color='Group',hover_data=['city_merge_name','violent_crime','police','population'])

    fig.update_layout(transition_duration=500)

    return fig

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port="8050", debug=True)