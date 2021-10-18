import dash
from dash.dependencies import Input, Output, State
from dash import dcc
from dash import html
from random import random
from loguru import logger
import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
sys.path.insert(0, os.path.dirname("."))
import torch
import configs.app_config as app_config 

# application, which compares the convergence of two algorithm

# logger for debugging the code
logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time}</green> | <level>{level} | {message}</level>",
    level="TRACE",
)

# initializing of the web application
app = dash.Dash(__name__)
app.layout = html.Div(
    html.Div(
        [
            dcc.Graph(id="live-update-graph-scatter", animate=True),
            html.Div(id="step"),
            dcc.Interval(
                id="interval-component",
                interval=1 * 1000,
            ),
        ]
    )
)

# callback for trainings animation


@app.callback(
    [
        Output("live-update-graph-scatter", "figure"),
        Output("step", "children"),
    ],
    Input("interval-component", "n_intervals"),
)
def update_graph_scatter(x):
   
    logger.trace("Play action.")
    # if the maximum number of epochs is reached, switch to the next step of the algorithm
    model = app_config.network_weight
    # import the first model
    fpath = f"/Users/andreferdinand/Desktop/Coding2/output/weights_log/model_epoch_{app_config.list_of_numbers_merged[app_config.epoch_weight]}.pt"
    checkpoint = torch.load(fpath)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.train(False)
    model.double()
    # Creation of a vector to plot the function in the web application
    stock_values = np.linspace(1, 6, num=50)
    stock_values = np.reshape(stock_values, (50, 1))
    input = torch.from_numpy(stock_values)
    input = input.double()
    # if log model is used, the exponential function is needed
    if app_config.LOG_MODEL_TWO_OUT_WEIGHT:
        out_data = np.exp(model(input)[:, 0].detach().numpy().flatten())
    else:
        out_data = model(input).detach().numpy().flatten()
    logger.debug(out_data)
    # get loss of the algorithm on the trainingssample
    loss = checkpoint['loss']
        
    # plot the decision function of the algorithm
    reward_figure1 = go.Figure(
        go.Scatter(
            x=list(np.linspace(1, 6, 50)),
            y=list(out_data),
        )
    )
    reward_figure1.update_layout(yaxis_range=[0, 1])
    # plot information about the game
    text1 = f" learning epoch: {app_config.list_of_numbers_merged[app_config.epoch_weight]} \n loss: {loss} \n"

    app_config.epoch_weight += 1
    return reward_figure1, text1


if __name__ == "__main__":
    app.run_server(debug=True, port=8092)
