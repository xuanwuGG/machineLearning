import dash
from dash import dcc
from dash import html
import time
from main import NMF_get_predictions, SVD_get_predictions
from surprise import accuracy

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('Machine Learning Working Design', style={'textAlign': 'center'}),
    dcc.Dropdown(
        id='dropdown',
        options=[
            {'label': 'SVD', 'value': 'SVD'},
            {'label': 'NMF', 'value': 'NMF'}
        ],
        value=None,
        style={'width': '50%', 'margin': 'auto'}
    ),
       html.Div(
        dcc.Loading(
            id="loading",
            type="circle",
            fullscreen=True,
            children=html.Div(id='output-container')
        ),
        style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center', 'height': '500px'}  # 修改这里
    )
])

@app.callback(
    dash.dependencies.Output('output-container', 'children'),
    [dash.dependencies.Input('dropdown', 'value')])
def update_output(value):
    if value is None:
        return html.Div()
    else:
        if value == 'SVD':
            predictions=SVD_get_predictions()
        elif value == 'NMF':
            predictions = NMF_get_predictions()
            rmse = accuracy.rmse(predictions)  # Assuming you have a function to calculate RMSE
            return html.Div([
                html.H3('You have selected "{}"'.format(value), style={'textAlign': 'center'}),
                html.H4('RMSE: {}'.format(rmse), style={'textAlign': 'center'})
            ])
        return html.Div([
            html.H3('You have selected "{}"'.format(value), style={'textAlign': 'center'})
        ])

if __name__ == '__main__':
    app.run_server(debug=True)