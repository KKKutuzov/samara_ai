import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import datetime
import random
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.io as poi


codes  = {'1' : 'тигр', '2' : 'леопард', '3' : 'хз'}


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
        html.H1('На фотографии изображены'),
        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src=contents,
                 style={
                     'height': 400,
                     'display' : 'block', 
                     'margin-left' : 'auto', 
                     'margin-right' : 'auto'
                 }),
    ])

def predict(img):
    return random.randint(0, 3)

@app.callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'),
              )
def update_output(list_of_contents, list_of_names):
    if list_of_contents is not None:
        tigers = []
        leopards = []
        another = []
        for elm in zip(list_of_contents,list_of_names):
            flag = predict(elm)
            if flag == 1:
                tigers.append(elm)
                continue
            if flag == 2:
                leopards.append(elm)
                continue
            another.append(elm)
        children = []
        if len(tigers) != 0:
            if len(tigers) == 1:
                data = [html.H1('На фотографии изображен тигр: ')]
            else:
                data = [html.H1('На фотографиях изображены тигры:  ')]
            for elm in tigers:
                data.append(html.Img(src=elm[0],
                        style={
                            'height': 400,
                            'display' : 'block', 
                            'margin-left' : 'auto', 
                            'margin-right' : 'auto'
                        }))
                data.append(html.H2(elm[1],
                    style = {'text-align': 'center'}
                ))
            children.append(html.Div(data))

        if len(leopards) != 0:
            if len(leopards) == 1:
                data = [html.H1('На фотографии изображен леопард: ')]
            else:
                data = [html.H1('На фотографиях изображены леопарды: ')]
            for elm in leopards:
                data.append(html.Img(src=elm[0],
                        style={
                            'height': 400,
                            'display' : 'block', 
                            'margin-left' : 'auto', 
                            'margin-right' : 'auto'
                        }))
                data.append(html.H2(elm[1],
                    style = {'text-align': 'center'}
                ))
            children.append(html.Div(data))

        if len(another) != 0:
            if len(another) == 1:
                data = [html.H1('На фотографии изображен не тигр и не леопард: ')]
            else:
                data = [html.H1('На фотографиях изображены не тигры и не леопарды: ')]
            for elm in another:
                data.append(html.Img(src=elm[0],
                        style={
                            'height': 400,
                            'display' : 'block', 
                            'margin-left' : 'auto', 
                            'margin-right' : 'auto'
                        }))
                data.append(html.H2(elm[1],
                    style = {'text-align': 'center'}
                ))
            children.append(html.Div(data))
        return children


if __name__ == '__main__':
    app.run_server(debug=True)

