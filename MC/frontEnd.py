import dash
from dash import dcc
from dash import html
import time
from main import NMF_get_predictions, SVD_get_predictions
from surprise import accuracy
import plotly.graph_objs as go
import pandas as pd
import base64
import io

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('Machine Learning Working Design', style={'textAlign': 'center'}),
    html.Label('adjust epochs:'),
    dcc.Slider(
        id='slider',
        min=0,
        max=20,
        step=1,
    ),
    html.Div(id='slider-output-container'),
    dcc.Dropdown(
        id='dropdown',
        options=[
            {'label': 'SVD', 'value': 'SVD'},
            {'label': 'NMF', 'value': 'NMF'}
        ],
        style={'width': '50%', 'margin': 'auto'}
    ),
    html.Div([
        dcc.Input(id='start', type='number', placeholder='Start index'),
        dcc.Input(id='end', type='number', placeholder='End index'),
        html.Button('Submit', id='submit_button', n_clicks=0)
    ], style={'display': 'flex', 'justify-content': 'center', 'margin-top': '80px'}), 
    dcc.Markdown('''
                 ## 须知  
                    1. 网站源程序**未进行任何健壮性相关检测**，请按照步骤操作  
                    2. 请先选择算法，再输入编号范围  
                    3. 网站加载时，**不建议**再进行任何操作  
                    4. 数据集应放置在与源程序同一目录下
                 '''),
    html.Div(id='output-container')
])


@app.callback(
    dash.dependencies.Output('output-container', 'children'),
    [dash.dependencies.Input('submit_button', 'n_clicks'),
     dash.dependencies.Input('slider', 'value')],
    [dash.dependencies.State('dropdown', 'value'),
     dash.dependencies.State('start', 'value'),
     dash.dependencies.State('end', 'value')])
def update_output(n_clicks,slider_value,value, start, end):
   if n_clicks > 0:
        if value is None or start is None or end is None:
            return html.Div()
        else:
            if value == 'SVD':
                predictions = SVD_get_predictions(n_epochs=int(slider_value))
            elif value == 'NMF':
                predictions = NMF_get_predictions(n_epochs=int(slider_value))
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