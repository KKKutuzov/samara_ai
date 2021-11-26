import dash
import pandas as pd
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import datetime
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.io as poi


''' READ DATA '''


external_stylesheets = [
    {
        "href": "https://fonts.googleapis.com/css2?"
                "family=Lato:wght@400;700&display=swap",
        "rel": "stylesheet",
    },
]

app = dash.Dash(__name__,
                external_stylesheets=external_stylesheets)
app.title = "Tiger Analytics"


''' LAYOUT '''

app.layout = html.Div([
    html.Div(
            children=[
                html.P(children="🐯", className="header-emoji"),
                html.H1(
                    children="Tiger Analytics", className="header-title"
                ),
                html.P(
                    children="Analyze the behavior of tiger"
                    " and the number of tiger sold in the Russia"
                    " between 2015 and 2021",
                    className="header-description",
                ),
            ],
            className="header",
            style={'margin-bottom': 40}
        ),
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '0px',
            'margin-bottom': '40px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    dbc.Row([
        dbc.Col(
            html.Div(id='output-image-upload'),
            width={"size": 6, "offset": 3}),
    ],
        style={'margin-bottom': 40}
    )
    ],
    style={'margin-left': '60px',
           'margin-right': '60px'})

''' CALLBACKS '''


def parse_contents(contents):
    return html.Div([
        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src=contents,
                 style={
                     'width': 200,
                     'height': 100
                 }),
    ])


@app.callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents')
              )
def update_output(list_of_contents):
    if list_of_contents is not None:
        children = [
            parse_contents(c) for c in
            zip(list_of_contents)]
        return children


if __name__ == '__main__':
    app.run_server(debug=True)

