import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from app import app
from apps.dashboard import prediction_layout

# Describe the layout/ UI of the app
app.layout = html.Div(
    [dcc.Location(id="url", refresh=False),
     html.Div([
         dcc.Link('Sales Prediction', href='/apps/prediction_layout'),
         ], className="row"),
     html.Div(id="page-content")]
)

# Update page
@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def display_page(pathname):
    if pathname == '/apps/Sales_Prediction':
        return prediction_layout.layout

    else:
        return prediction_layout.layout


if __name__ == "__main__":
    app.run_server(debug=False)