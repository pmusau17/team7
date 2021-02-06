# Basics Requirements
import pathlib
import dash
from dash.dependencies import Input, Output, State, ClientsideFunction
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd


# Dash Bootstrap Components
import dash_bootstrap_components as dbc

# Data
import json
from datetime import datetime as dt

# Recall app
from app import app


####################################################################################
# Add the DS4A_Img
####################################################################################

DS4A_Img = html.Div(
    children=[html.Img(src=app.get_asset_url("ds4a2.png"), id="ds4a-image",style={'width':'100%'})],
)

#############################################################################
# State Dropdown Card
#############################################################################

# Load the data
df = pd.read_csv('data/CleanData.csv')

## Create Options
items = ['All Years']+[str(i) for i in range(2010,2018)]
options = []
for i in items: 
    op = {'label': i, 'value': i}
    options.append(op)

cities = ['All Cities'] + sorted(df['City'].unique().tolist())
city_options = []
for i in cities: 
    op = {'label': i, 'value': i}
    city_options.append(op)

## Year Drop down
drop = dcc.Dropdown(
    options=options,
    value='All Years',
    multi=False,
    id='year_dropdown'
)  


## Button to go back to all defaults
reset = html.Button('Reset', id='button')


### City Drop Down
city_drop = dcc.Dropdown(
    options=city_options,
    multi=True,
    value="All Cities", 
    id = 'city_dropdown'
)  


### Checklist for Cluster Groups

checklist = dcc.Checklist(
    options=[
        {'label': 'Group 1', 'value': '1'},
        {'label': 'Group 2', 'value': '2'},
        {'label': 'Group 3', 'value': '3'},
        {'label': 'Group 4', 'value': '4'},
        {'label': 'Group 5', 'value': '5'},
        {'label': 'Group 6', 'value': '6'}
    ],
    value=['1','2','3','4','5','6'],
    labelStyle={'display': 'block'}, id ="ds4a-checklist",
)  

##############################################################################
# Date Picker Card
##############################################################################


#############################################################################
# Sidebar Layout
#############################################################################
sidebar = html.Div(
    [
        DS4A_Img,  # Add the DS4A_Img located in the assets folder
        html.Hr(),  # Add an horizontal line
        ####################################################
        # Place the rest of Layout here
        ####################################################
        html.H5('Year',className='year_label'),
        drop,
        html.H5('City',className='year_label'),
        city_drop,
        html.H5('Cluster Group',className='year_label'),
        checklist,
        #reset,
    ],
    className="ds4a-sidebar",
)
