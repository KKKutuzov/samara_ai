import datetime

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

app.layout = html.Div([
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
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-image-upload'),
])

def parse_contents(contents):
    return html.Div([
        html.H5('Входное изображение:'),
        html.Img(src=contents,height=400,style={'display' : 'block', 'margin-left' : 'auto', 'margin-right' : 'auto'}),
    ])

@app.callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents'))
def update_output(list_of_contents):
    if list_of_contents is not None:
        children = [
            parse_contents(c) for c in
            list_of_contents]
        return children

if __name__ == '__main__':
    app.run_server(debug=True)