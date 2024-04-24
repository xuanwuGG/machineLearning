import dash
import pandas as pd
from dash import dash_table
from dash import  dcc
from dash import html
from dash.dependencies import Input, Output
import main 
import io
import base64

app = dash.Dash(__name__)

df = pd.read_csv("C:/Users/10937/Documents/数据集/archive/JokeText.csv")
df['Score'] = 0
df = df[['JokeId', 'Score', 'JokeText']]
global similarity_matrix


app.layout = html.Div([
    html.H1('Machine Learning Work Design', style={'text-align': 'center'}),
    dcc.Upload(
        id='upload-data',
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
    ),
    dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i, 'editable': (i == 'Score')} for i in df.columns],
        data=df.to_dict('records'),
        style_cell={'text-align': 'left'},
        page_size=5,
    ),

    dash_table.DataTable(
    id='table2',
    columns=[{"name": "", "id": ""}],
    data=[{}]
    ),

    html.Div([
        dcc.Input(id='item-id', type='text', placeholder='enter item id'),
        dcc.Input(id='num-recommendations', type='text', placeholder='enter recommended num'),
        html.Button('submit', id='submit-button', n_clicks=0),
    ], style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center', 'height': '50vh'}),
    html.Div(id='output-container', style={'text-align': 'center'})
])

def parse_contents(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    return io.StringIO(decoded.decode('utf-8'))
#限制输入区间
@app.callback(
    Output('table', 'data'),
    [Input('table', 'data_timestamp')],
    [dash.dependencies.State('table', 'data')]
)
def update_data(timestamp, rows):
    for row in rows:
        row['Score'] = min(max(float(row['Score']), -10), 10)
    return rows

#获取上传文件
@app.callback(
    Output('output-container', 'children'),
    [Input('upload-data', 'contents')]
)
def update_matrix(contents):
    global similarity_matrix
    if contents is not None:
        path=parse_contents(contents)
        similarity_matrix = main.matrixModify(path)        
        
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0',port=8051)
    