# Basics Requirements
import pathlib
import dash
from dash.dependencies import Input, Output, State, ClientsideFunction
import dash_core_components as dcc
import dash_html_components as html


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

## Create Options
items = ['All Years']+[str(i) for i in range(2010,2018)]
options = []
for i in items: 
    op = {'label': i, 'value': i}
    options.append(op)

## Year Drop down
drop = dcc.Dropdown(
    options=options,
    value=['All Years'],
    multi=False
)  


##

reset = html.Button('Reset', id='button')



### Checklist for Cluster Groups

checklist = dcc.Checklist(
    options=[
        {'label': 'Group 1', 'value': 'Group 1'},
        {'label': 'Group 2', 'value': 'Group 2'},
        {'label': 'Group 3', 'value': 'Group 3'},
        {'label': 'Group 4', 'value': 'Group 4'},
        {'label': 'Group 5', 'value': 'Group 5'}
    ],
    value=['Group 1', 'Group 2','Group 3','Group 4','Group 5'],
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
        drop,
        checklist,
        reset,
    ],
    className="ds4a-sidebar",
)
