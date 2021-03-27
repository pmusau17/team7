import dash
import dash_table
from dash.dependencies import Input, Output, State, ClientsideFunction
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import statsmodels.formula.api as sm
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime as dt
import json
import numpy as np
import pandas as pd
import os
import geopandas as gpd
# Recall app
from app_handle import app

from .definitions import regress_vars


def reapply_names(row):
    if row in list(regress_vars.values()):
        return regress_names[row]
    else:
        return row

# Load the data
df = pd.read_csv('data/CleanDataScatter.csv')
regress_names = {value:key for key, value in regress_vars.items()}
color_discrete_map = {"1":'#636EFA',"2":'#EF553B',"3":'#00CC96',"4":'#AB63FA',"5":'#FFA15A',"6":'#19D3F3'}

regress = df[df['City']!='Washington, DC']
regress['Violent Crime'] = regress['Violent Crime'] * (10**-5)
regress['Property Crime'] = regress['Property Crime'] * (10**-5)

regress= regress.rename(columns=regress_vars)
noDC = regress
noDC['Fits']= 4.662918e-03+ noDC['rev_total']*1.699413e-07
+ noDC['public_welfare']*-2.784823e-06 + noDC['welfare_cash']*3.215736e-06 
+noDC['correction']*-3.820413e-06 + noDC['housing_commdevt']*-1.767486e-06 
+ noDC['youth_poverty']*7.044471e-05 
+noDC['estimate_employed']*-1.596820e-02 +noDC["income_deficit"]*-2.441968e-07
+ noDC["total_estimate_age_under_18_years"]*-3.196742e-02+ noDC["total_estimate_sex_male"]*4.768845e-02
+ noDC["total_estimate_age_65_years_and_over"]*-1.251370e+03+noDC["percent_race_one_race_asian"]*-6.814841e-05
+ noDC["percent_black"]*3.520207e-05+noDC["percent_hawaiian"]*1.294790e-03+ noDC["percent_race_one_race_white"] * -4.546245e-05+noDC["percent_hispanic_or_latino"] * 1.482309e-05

noDC = noDC.rename(columns=lambda x: reapply_names(x))
fig_reg = px.scatter(noDC, y='Violent Crime', x='Fits', trendline="ols")
noDC['Group'] = noDC['Group'].apply(str)
fig_reg_group = px.scatter(noDC, y='Violent Crime', x='Fits', trendline="ols",color = "Group",color_discrete_map=color_discrete_map)

fig_reg.update_layout(
    paper_bgcolor='#F8F9F9'
)


fig_reg_group.update_layout(
    paper_bgcolor='#F8F9F9'
)