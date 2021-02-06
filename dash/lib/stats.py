import dash
import dash_table
from dash.dependencies import Input, Output, State, ClientsideFunction
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from datetime import datetime as dt
import json
import numpy as np
import pandas as pd
import os
import geopandas as gpd
# Recall app
from app_handle import app

# useful definitions that I've conveniently moved elsewhere
from .definitions import relevant_variables,pc_n,pc_p,vc_n,vc_p,races
from .text import *

# Load the data
df = pd.read_csv('data/CleanDataScatter.csv')
color_discrete_map = {"1":'#636EFA',"2":'#EF553B',"3":'#00CC96',"4":'#AB63FA',"5":'#FFA15A',"6":'#19D3F3'}
# Create Options For Drop Down Menu




labels = []
for i in list(relevant_variables):
    item = {'label': '{}'.format(i), 'value': '{}'.format(i)}
    labels.append(item)

relevant_variables2 = ['Violent Crime', 'Property Crime']
labels2 = []
for i in list(relevant_variables2):
    item = {'label': '{}'.format(i), 'value': '{}'.format(i)}
    labels2.append(item)


average_crime_per_year = df.groupby('Year').mean().reset_index()

# create the df for visualizing violent crime per group
by_group = df.groupby(['Year','Group']).mean().reset_index()
by_group['Group'] = by_group['Group'].astype(str) 


# bar plot of violent crime
fig = px.bar(by_group, x="Year", y="Violent Crime",color='Group')
fig.update_layout(barmode='group',title='',plot_bgcolor='#F8F9F9')

# Line plot of police and violent_crime

fig4 = px.line(average_crime_per_year, x="Year", y=["Violent Crime","Police Spending"])


subfig2 = make_subplots(specs=[[{"secondary_y": True}]])
fig5 = px.line(average_crime_per_year, x="Year", y=["Violent Crime"],labels={"Violent Crime": "Violent Crime"})
fig5.update_traces(line_color='rgb(204, 102, 119)',name="Violent Crime")
fig6 = px.line(average_crime_per_year, x="Year", y=["Police Spending"],labels={"Police Spending": "Police Spending"})
fig6.update_traces(line_color='rgb(95, 70, 144)',name="Police Spending")
fig6.update_traces(yaxis="y2")
fig6.update_layout(showlegend=True)
subfig2.add_traces(fig5.data + fig6.data)
subfig2.layout.xaxis.title="Year"
subfig2.layout.yaxis.title="Violent Crime (Rate per 100,00)"
subfig2.layout.yaxis2.title="Police Spending"
subfig2.layout.paper_bgcolor='#F8F9F9'


subfig = make_subplots(specs=[[{"secondary_y": True}]])
fig5 = px.line(average_crime_per_year, x="Year", y=["Violent Crime"],labels={"Violent Crime": "Violent Crime"})
fig5.update_traces(line_color='rgb(204, 102, 119)',name="Violent Crime")
fig6 = px.line(average_crime_per_year, x="Year", y=["Percent Labor Force (Age 16 and Over)"],labels={"Percent Labor Force (Age 16 and Over)": "Percent Labor Force (Age 16 and Over)"})
fig6.update_traces(line_color='rgb(95, 70, 144)',name="Labor Force")
fig6.update_traces(yaxis="y2")
fig6.update_layout(showlegend=True)
subfig.add_traces(fig5.data + fig6.data)
subfig.layout.xaxis.title="Year"
subfig.layout.yaxis.title="Violent Crime (Rate per 100,00)"
subfig.layout.yaxis2.title="Percent Labor Force (Age 16 and Over)"



subfig3 = make_subplots(specs=[[{"secondary_y": True}]])
fig5 = px.line(average_crime_per_year, x="Year", y=["Violent Crime"],labels={"Violent Crime": "Violent Crime"})
fig5.update_traces(line_color='rgb(204, 102, 119)',name="Violent Crime")
fig6 = px.line(average_crime_per_year, x="Year", y=["Housing and Community Development"],labels={"Housing and Community Development": "Housing and Community Development"})
fig6.update_traces(line_color='rgb(95, 70, 144)',name="Housing")
fig6.update_traces(yaxis="y2")
fig6.update_layout(showlegend=True)
subfig3.add_traces(fig5.data + fig6.data)
subfig3.layout.xaxis.title="Year"
subfig3.layout.yaxis.title="Violent Crime (Rate per 100,00)"
subfig3.layout.yaxis2.title="Housing and Community Development"
subfig3.layout.paper_bgcolor='#F8F9F9'



# Correlation  Summarized Plot
corrs = df[vc_p+vc_n+['Violent Crime','Property Crime','Group']].groupby("Group").median().corr()
disp = corrs[['Property Crime', 'Violent Crime']].iloc[0:-2].sort_values(by='Property Crime',ascending=False)
fig7 = px.imshow(disp)
fig7.update_layout(
    paper_bgcolor='#F8F9F9',
    height=700
)

# Correlation  Summarized Plot
corrs = df[pc_p+pc_n+['Violent Crime','Property Crime','Group']].groupby("Group").median().corr()
disp = corrs[['Property Crime', 'Violent Crime']].iloc[0:-2].sort_values(by='Property Crime',ascending=False)
fig8 = px.imshow(disp)
fig8.update_layout(
    autosize=True,
    height=1200,
    paper_bgcolor='#F8F9F9'
)



race_analysis = []
for i in range(len(races)):
    race_melt = df[vc_p+vc_n+['Group']+races[i:i+1]].corr()
    race_analysis.append(race_melt[races[i:i+1]].iloc[:-2])


race_analysis = pd.concat(race_analysis,axis=1)
fig9 = px.imshow(race_analysis)
fig9.update_layout(
    autosize=True,
    height=600,
    paper_bgcolor='#F8F9F9'
)

# Table 
tab = round(df.groupby('Group').median()[['Violent Crime','Property Crime',
                                          'Percent of Individuals Below Poverty (Ages 18-64)','Police Spending']].reset_index(),2)




#fig.update_layout(barmode='group',title='')
df['Group'] = df['Group'].apply(str) 
fig2 = px.scatter(df.sort_values(by='Group'), x="Police Spending", y="Violent Crime",color='Group',hover_data=['City','Violent Crime','Police Spending','Population'],color_discrete_map=color_discrete_map,)

df['Group'] = df['Group'].apply(str) 
fig3 = px.scatter_geo(df,
                    lat="Latitude",
                    lon="Longitude",
                    hover_name="City",
                    projection="albers usa",
                    color="Group",
                    size="Violent Crime",
                    color_discrete_map=color_discrete_map,
                    hover_data=['City','Violent Crime','Property Crime','Police Spending','Total Revenue'])

fig3.update_layout(
        title = 'Metropolitan Areas by Cluster Label'
)

stats = html.Div(
    [        
        html.H3("Background and Motivation"),
        html.P(p1),
        
        html.P(p3),
        html.H3("Budget Allocations and Crime"),
        html.P(p4),
        dcc.Graph(
        id='violent-crime-bar',
        figure=fig
        ),

        

        dcc.Graph(
        id='map',
        figure=fig3
        ),

        # drop down menu scatter plot1
        dcc.Dropdown(
        id='demo-dropdown',
        options=labels,
        value='Police Spending',
        ),

        # drop down menu scatter plot1
        dcc.Dropdown(
        id='demo-dropdown2',
        options=labels2,
        value='Violent Crime',
        ),

        dcc.Graph(
        id='scatter',
        figure=fig2
        ),


        html.H3("What has the largest impact on reducing Crime?"),
        html.P(p5),
        dcc.Graph(
        id='violent_crime-spending',
        figure=subfig2
        ),

        html.P(p6),
        dcc.Graph(
        id='violent-crime-labor',
        figure=subfig3
        ),


        html.H3("Demographic Data"),
        html.P(p8),
        dcc.Graph(
        id='correlation_violent_crime',
        figure=fig7
        ),


        # html.H3("Relevant Correlations"),
        # html.P(p5),
        # dcc.Graph(
        # id='correlation_property_crime',
        # figure=fig8
        # ),

        html.H3("Race and Crime"),
        html.P(p7),
        dcc.Graph(
        id='correlation_races_crime',
        figure=fig9
        ),


        html.H3("Statistical Analysis"),
        html.P(p10),
        
        dcc.Markdown('''
            1. Poverty rates are significantly different across all levels, with High crime having the highest poverty levels.
            2. Police spending is significantly different between high and average crime levels, but not different btw low and average.
            3. Income deficit is different amongst all levels, with high crime areas having the highest income deficit
        '''),

        dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in tab.columns],
        data=tab.to_dict('records'),
        ),


        html.H3("Final Notes"),
        html.P(p9),

    
    ],
    className="ds4a-body",
)
