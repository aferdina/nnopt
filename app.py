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
#import dash_cytoscape as cyto
#import networkx as nx
import plotly.graph_objects as go
from networks import NetworkDOS
from networks import NetworkeasyDOS
import torch
import corr_stopp

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
            dcc.Graph(id="live-update-graph-scatter2", animate=True),
            html.Div(id="step2"),
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
        Output("live-update-graph-scatter2", "figure"),
        Output("step2", "children"),
    ],
    Input("interval-component", "n_intervals"),
)
def update_graph_scatter(x):
    if plot_nn.epoch == 29:
        plot_nn.epoch = 0
        plot_nn.step += 1
    logger.trace("Play action.")
    model = NetworkDOS(nb_stocks=1, hidden_size=4)
    #generate non copied case
    fpath = f"../output/neural_networks3/phase_{plot_nn.step}/model_epoch_{plot_nn.epoch}.pt"
    checkpoint = torch.load(fpath)
    model.load_state_dict(checkpoint['model_state_dict'])
    stock_values_test = np.reshape([1,2,3,4,5,6],(6,1))
    input_test = torch.from_numpy(stock_values_test).double()
    stock_values = np.linspace(1,6,num=50)
    stock_values = np.reshape(stock_values,(50,1))
    input = torch.from_numpy(stock_values)
    input = input.double()
    model.train(False)
    model.double()
    out_data = model(input)
    #logger.debug(f"list of q values{np.round(model(input_test).detach().numpy().flatten().tolist())}")
    loss = checkpoint['loss']
    #logger.debug(f"true stopping times {corr_stopp.stoppingtimes[plot_nn.step+1,:].flatten()}")
    if plot_nn.epoch ==28:   
        logger.debug(f"real qs{np.round(corr_stopp.stoppingtimes[plot_nn.step,:].flatten())}")
        logger.debug(f"approx. qs{np.round(model(input_test).detach().numpy().flatten())}")
        corr_stopp.mistakes += np.sum(np.abs(corr_stopp.stoppingtimes[plot_nn.step,:].flatten()-np.round(model(input_test).detach().numpy().flatten())))

    
    reward_figure1 = go.Figure(
        go.Scatter(
            x=list(np.linspace(1,6,50)),
            y=list(out_data.detach().numpy().flatten()),
        )
    )
    
    reward_figure1.update_layout(yaxis_range=[0,1])
    text1 = f"step in the game: {plot_nn.step} \n learning epoch: {plot_nn.epoch} \n loss: {loss} \n number of mistakes {corr_stopp.mistakes}"

    #generate copied model 
    fpath2 = f"../output/neural_networkscopy3/phase_{plot_nn.step}/model_epoch_{plot_nn.epoch}.pt"
    checkpoint2 = torch.load(fpath2)
    model.load_state_dict(checkpoint2['model_state_dict'])
    model.train(False)
    model.double()
    out_data2 = model(input)
    loss2 = checkpoint2['loss']
    #logger.debug(f"list of q values copied {np.round(mod   el(input_test).detach().numpy().flatten().tolist())}")
    if plot_nn.epoch ==28:
        logger.debug(f"real qs{np.round(corr_stopp.stoppingtimes[plot_nn.step,:].flatten())}")
        logger.debug(f"approx. qs copy{np.round(model(input_test).detach().numpy().flatten())}")
        corr_stopp.mistakes2 += np.sum(np.abs(corr_stopp.stoppingtimes[plot_nn.step,:]-np.round(model(input_test).detach().numpy().flatten())))

    
    reward_figure2 = go.Figure(
        go.Scatter(
            x=list(np.linspace(1,6,50)),
            y=list(out_data2.detach().numpy().flatten()),
        )
    )

    reward_figure2.update_layout(yaxis_range=[0,1])
    text2 = f"step in the game: {plot_nn.step} \n learning epoch: {plot_nn.epoch} \n loss: {loss2} \n number of mistakes {corr_stopp.mistakes2}"
    plot_nn.epoch +=1
    return reward_figure1, text1, reward_figure2, text2
    # return {'data': traces}


if __name__ == "__main__":
    app.run_server(debug=True, port=8091)
 