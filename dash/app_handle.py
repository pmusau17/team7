#######################################################
# Main APP definition.
#
# Dash Bootstrap Components used for main theme and better
# organization.
#######################################################

import dash
import dash_bootstrap_components as dbc


external_stylesheets = ['https://stackpath.bootstrapcdn.com/bootswatch/4.5.2/cosmo/bootstrap.min.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
#We need this for function callbacks not present in the app.layout
app.config.suppress_callback_exceptions = True
server = app.server