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
from plotly.subplots import make_subplots
from lib.definitions import relevant_variables,pc_n,pc_p,vc_n,vc_p,races,regress_vars
from lib.regression import reapply_names

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
app.config.suppress_callback_exceptions = True

# LOAD THE DIFFERENT FILES
from lib import sidebar, stats, title


server = app.server



app.layout = html.Div(
    [title.title,stats.stats, sidebar.sidebar,],
    className="ds4a-app",  # You can also add your own css files by locating them into the assets folder
)

color_discrete_map = {"1":'#636EFA',"2":'#EF553B',"3":'#00CC96',"4":'#AB63FA',"5":'#FFA15A',"6":'#19D3F3'}

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
    if(len(ds4a_value)==0 or len(city_value)==0):
        fig1 = go.Figure().add_annotation(x=2, y=2,text="No Data to Display",font=dict(family="sans serif",size=25,color="crimson"),showarrow=False,yshift=10).update_layout(paper_bgcolor='#F8F9F9')
        fig2 = go.Figure().add_annotation(x=2, y=2,text="No Data to Display",font=dict(family="sans serif",size=25,color="crimson"),showarrow=False,yshift=10).update_layout(paper_bgcolor='#F8F9F9')
        return fig1,fig2

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
                    hover_name="City",
                    projection="albers usa",
                    size="Violent Crime",
                    color_discrete_map=color_discrete_map,
                    hover_data={'City':True,'Violent Crime':True,'Property Crime':True,'Police Spending':True,'Total Revenue':True,"Longitude":False,"Latitude":False,"Group":False},
                    color="Group")


    fig3.update_layout(
        title = '',paper_bgcolor='#F8F9F9',plot_bgcolor='#F8F9F9',geo=dict(bgcolor= '#F8F9F9',landcolor='rgb(136,136,136)')
    )


    by_group = df_group.groupby(['Year','Group']).mean().reset_index()
    by_group['Group'] = by_group['Group'].astype(str) 
    by_group['Year'] = by_group['Year'].astype(int)
    fig = px.bar(by_group, x="Year", y="Violent Crime",color='Group',color_discrete_map=color_discrete_map,)
    fig.update_layout(barmode='group',title='',paper_bgcolor='#F8F9F9',yaxis_title="Violent Crime (Rate per 100,000)")

    return fig3,fig


@app.callback(
    Output('scatter', 'figure'),
    Input('demo-dropdown', 'value'),
    Input('demo-dropdown2', 'value'))
def update_scatter(input_value,input_value2):
    if(input_value not in list(df.columns)):
        input_value='Police Spending'
    df['Group'] = df['Group'].apply(str)
    fig= px.scatter(df.sort_values(by='Group'), x=input_value, y=input_value2,color='Group',hover_data=['City','Violent Crime','Police Spending','Population'])

    fig.update_layout(transition_duration=500,paper_bgcolor='#F8F9F9')

    return fig



@app.callback(
    Output('violent_crime-spending', 'figure'),
    Output('violent-crime-labor','figure'),
    Output('correlation_races_crime','figure'),
    Input('year_dropdown', 'value'),
    Input('city_dropdown', 'value'),
    Input('ds4a-checklist', 'value')
)

def update_analysis_plots(year_value,city_value,ds4a_value):
    if(len(ds4a_value)==0 or len(city_value)==0):
        fig1 = go.Figure().add_annotation(x=2, y=2,text="No Data to Display",font=dict(family="sans serif",size=25,color="crimson"),showarrow=False,yshift=10).update_layout(paper_bgcolor='#F8F9F9')
        fig2 = go.Figure().add_annotation(x=2, y=2,text="No Data to Display",font=dict(family="sans serif",size=25,color="crimson"),showarrow=False,yshift=10).update_layout(paper_bgcolor='#F8F9F9')
        fig3 = go.Figure().add_annotation(x=2, y=2,text="No Data to Display",font=dict(family="sans serif",size=25,color="crimson"),showarrow=False,yshift=10).update_layout(paper_bgcolor='#F8F9F9')
        return fig1,fig2,fig3

    df['Group'] = df['Group'].apply(str).apply(str.strip)
    df_group=df[df.Group.isin(list(ds4a_value))]
    # get the correct year:
    if(year_value!='All Years'):
        df_group= df_group[df_group.Year==int(year_value)]
    # get the correct city
    if not isinstance(city_value, list):
        city_value = [city_value]
    if ('All Cities' not in city_value):
        df_group = df_group[df_group.City.isin(city_value)]
    
    average_crime_per_year = df_group.groupby('Year').mean().reset_index()
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
    cols = vc_p+vc_n+['Violent Crime','Property Crime','Group']
    passes= True
    for i in cols:
        if not (i in list(df_group.columns)):
            passes=False
            print("Fail")
            break


    if(passes):
        race_analysis = []
        for i in range(len(races)):
            race_melt = df_group[vc_p+vc_n+['Group']+races[i:i+1]].corr()
            race_analysis.append(race_melt[races[i:i+1]].iloc[:-2])


        race_analysis = pd.concat(race_analysis,axis=1)
        fig9 = px.imshow(race_analysis)
        fig9.update_layout(
            autosize=True,
            height=600,
            paper_bgcolor='#F8F9F9'
        )
    else:
        fig9 = go.Figure().add_annotation(x=2, y=2,text="No Data to Display",font=dict(family="sans serif",size=25,color="crimson"),showarrow=False,yshift=10).update_layout(paper_bgcolor='#F8F9F9')

    return subfig2,subfig3,fig9


@app.callback(
    Output('regression_fit_group', 'figure'),
    Input('ds4a-checklist', 'value')
)
def update_regression_plot(ds4a_value):
    if(len(ds4a_value)==0):
        fig1 = go.Figure().add_annotation(x=2, y=2,text="No Data to Display",font=dict(family="sans serif",size=25,color="crimson"),showarrow=False,yshift=10).update_layout(paper_bgcolor='#F8F9F9')
        return fig1

    df['Group'] = df['Group'].apply(str).apply(str.strip)
    df_group=df[df.Group.isin(list(ds4a_value))]
    regress = df_group[df_group['City']!='Washington, DC']
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
    + noDC["percent_black"]*3.520207e-05+noDC["percent_hawaiian"]*1.294790e-03+ noDC["percent_race_one_race_white"] * -4.546245e-05
    +noDC["percent_hispanic_or_latino"] * 1.482309e-05

    noDC = noDC.rename(columns=lambda x: reapply_names(x))
    noDC['Group'] = noDC['Group'].apply(str)
    noDC['Violent Crime'] = noDC['Violent Crime']*(10**5)
    noDC['Violent Crime Prediction (Rate per 100,000)'] = noDC['Fits']*(10**5)

    fig_reg_group = px.scatter(noDC, y='Violent Crime', x='Violent Crime Prediction (Rate per 100,000)', trendline="ols",color = "Group",color_discrete_map=color_discrete_map)
    fig_reg_group.update_layout(
        paper_bgcolor='#F8F9F9'
    )
    

    return fig_reg_group
    

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run_server(host='0.0.0.0', port=port,debug=True,dev_tools_ui=False)