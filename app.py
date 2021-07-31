import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(
    __name__, external_stylesheets=external_stylesheets,meta_tags=[{"name": "viewport", "content": "width=device-width"}],
)
app.title = "Sales Prediction"
server = app.server