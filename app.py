import dash
from dash.dependencies import Input, Output, State
from dash import dcc
from dash import html
from random import random
import plotly
from loguru import logger
import sys
import pandas as pd
import numpy as np
import plot_nn
import dash_cytoscape as cyto
import networkx as nx
import plotly.graph_objects as go
from networks import NetworkDOS
import torch

num = 10  # number of iterations


# print(networkgraph)
# For scripts
logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time}</green> | <level>{level} | {message}</level>",
    level="TRACE",
)


app = dash.Dash(__name__)
app.layout = html.Div(
    html.Div(
        [
            dcc.Graph(id="live-update-graph-scatter", animate=True),
            html.Div(id="step"),
            # cyto.Cytoscape(elements=networkgraph),
            dcc.Interval(
                id="interval-component",
                interval=1 * 1000,
            ),
        ]
    )
)


@app.callback(
    [
        Output("live-update-graph-scatter", "figure"),
        Output("step", "children"),
    ],
    Input("interval-component", "n_intervals"),
)
def update_graph_scatter(x):
    if plot_nn.epoch == 29:
        plot_nn.epoch = 0
        plot_nn.step += 1
    logger.trace("Play action.")
    model = NetworkDOS(nb_stocks=1, hidden_size=4)
    fpath = f"/Users/andreferdinand/Desktop/Coding2/output/neural_networks/phase_{plot_nn.step}/model_epoch_{plot_nn.epoch}.pt"
    checkpoint = torch.load(fpath)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.debug("happy")
    stock_values = np.reshape([1., 2., 3., 4., 5., 6.], newshape=(6, 1))
    input = torch.from_numpy(stock_values).double()
    logger.debug(f"input{input}")
    #input = X_inputs = torch.from_numpy(stock_values).double()
    model.train(False)
    out_data = model(input)

    reward_figure = go.Figure(
        go.Scatter(
            x=np.arange(len(6)),
            y=out_data,
        )
    )
   # print(np.concatenate(play_trained_model.old_rewards).ravel())
    return reward_figure, plot_nn.step
    # return {'data': traces}


if __name__ == "__main__":
    app.run_server(debug=True, port=8091)
