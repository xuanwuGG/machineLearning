import dash
from dash import dcc
from dash import html
import time
from main import NMF_get_predictions, SVD_get_predictions
from surprise import accuracy
import plotly.graph_objs as go

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
    html.Div([
        dcc.Input(id='start', type='number', placeholder='Start index'),
        dcc.Input(id='end', type='number', placeholder='End index')
    ], style={'display': 'flex', 'justify-content': 'center', 'margin-top': '120px'}), 
    html.Div(id='output-container')
])

@app.callback(
    dash.dependencies.Output('output-container', 'children'),
    [dash.dependencies.Input('dropdown', 'value'),
     dash.dependencies.Input('start', 'value'),
     dash.dependencies.Input('end', 'value')])
def update_output(value, start, end):
    if value is None or start is None or end is None:
        return html.Div()
    else:
        if value == 'SVD':
            predictions = SVD_get_predictions()
        elif value == 'NMF':
            predictions = NMF_get_predictions()
        # 过滤出编号在指定区间的预测结果
        filtered_predictions = predictions[start:end]
        # 提取真实分和预测分
        actual_scores = [pred.r_ui for pred in filtered_predictions]
        predicted_scores = [pred.est for pred in filtered_predictions]
        # 创建一个图表
        figure = go.Figure(
            data=[
                go.Scatter(y=actual_scores, name='Actual'),
                go.Scatter(y=predicted_scores, name='Predicted')
            ],
            layout=go.Layout(title="Scores")
        )
        # 创建一个直方图
        histogram = go.Figure(
            data=[
                go.Histogram(x=predicted_scores, name='Predicted')
            ],
            layout=go.Layout(title="Histogram of Predicted Scores")
        )
        # 返回一个dcc.Graph组件和一个直方图
        return html.Div([
            dcc.Graph(figure=figure),
            dcc.Graph(figure=histogram)
        ])

if __name__ == '__main__':
    app.run_server(debug=True)