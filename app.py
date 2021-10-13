import dash
from dash.dependencies import Input, Output, State
from dash import dcc
from dash import html
from random import random
from dash import dash_table
import plotly
from loguru import logger
import sys
import pandas as pd
import numpy as np
from torch._C import Graph
from networks import NetworklogDOS
import plot_nn
#import dash_cytoscape as cyto
#import networkx as nx
import plotly.graph_objects as go

import torch
import corr_stopp
import app_config

emp_qvalues1 = pd.read_csv(
    f"../output/{app_config.PATH_ONE}/emp_qvalues.csv")
emp_qvalues1.index = np.arange(1, len(emp_qvalues1) + 1)
emp_qvalues1.reset_index(inplace=True)

emp_steps_qvalues1 = pd.read_csv(
    f"../output/{app_config.PATH_ONE}/emp_step_qvalues.csv")
emp_steps_qvalues1.index = np.arange(1, len(emp_steps_qvalues1) + 1)
emp_steps_qvalues1.reset_index(inplace=True)

emp_qvalues2 = pd.read_csv(
    f"../output/{app_config.PATH_TWO}/emp_qvalues.csv")
emp_qvalues2.index = np.arange(1, len(emp_qvalues2) + 1)
emp_qvalues2.reset_index(inplace=True)

emp_steps_qvalues2 = pd.read_csv(
    f"../output/{app_config.PATH_TWO}/emp_step_qvalues.csv")
emp_steps_qvalues2.index = np.arange(1, len(emp_steps_qvalues2) + 1)
emp_steps_qvalues2.reset_index(inplace=True)

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
            html.Div(id="txtfeld"), 
            dash_table.DataTable(id='table1', columns=[
                                 {"name": i, "id": i} for i in emp_steps_qvalues1.columns], data=emp_steps_qvalues1.to_dict('records'), page_size=5, style_data_conditional=[
                {
                    'if': {
                        'filter_query': '{{index}} = {}'.format(1),
                    },
                    'backgroundColor': '#FF4136',
                    'color': 'black'
                }, ]),
            dash_table.DataTable(id='table2', columns=[
                                 {"name": i, "id": i} for i in emp_qvalues1.columns], data=emp_qvalues1.to_dict('records'), page_size=5, style_data_conditional=[
                {
                    'if': {
                        'filter_query': '{{index}} = {}'.format(1),
                    },
                    'backgroundColor': '#FF4136',
                    'color': 'black'
                }, ]),
            dcc.Graph(id="live-update-graph-scatter2", animate=True),
            html.Div(id="step2"),
            dash_table.DataTable(id='table3', columns=[
                                 {"name": i, "id": i} for i in emp_steps_qvalues2.columns], data=emp_steps_qvalues2.to_dict('records'), page_size=5, style_data_conditional=[
                {
                    'if': {
                        'filter_query': '{{index}} = {}'.format(1),
                    },
                    'backgroundColor': '#FF4136',
                    'color': 'black'
                }, ]),
            dash_table.DataTable(id='table4', columns=[
                                 {"name": i, "id": i} for i in emp_qvalues2.columns], data=emp_qvalues1.to_dict('records'), page_size=5, style_data_conditional=[
                {
                    'if': {
                        'filter_query': '{{index}} = {}'.format(1),
                    },
                    'backgroundColor': '#FF4136',
                    'color': 'black'
                }, ]),
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
        Output("table1","style_data_conditional"),
        Output("table2","style_data_conditional"),
        Output("table3","style_data_conditional"),
        Output("table4","style_data_conditional"),
    ],
    Input("interval-component", "n_intervals"),
)
def update_graph_scatter(x):
    if plot_nn.epoch == app_config.NB_EPOCHS:
        plot_nn.epoch = 0
        plot_nn.step += 1
    logger.trace("Play action.")
    model = app_config.network
    # generate non copied case
    fpath = f"../output/{app_config.PATH_ONE}/phase_{plot_nn.step}/model_epoch_{plot_nn.epoch}.pt"
    checkpoint = torch.load(fpath)
    model.load_state_dict(checkpoint['model_state_dict'])
    stock_values_test = np.reshape([1, 2, 3, 4, 5, 6], (6, 1))
    input_test = torch.from_numpy(stock_values_test).double()
    stock_values = np.linspace(1, 6, num=50)
    stock_values = np.reshape(stock_values, (50, 1))
    input = torch.from_numpy(stock_values)
    input = input.double()
    model.train(False)
    model.double()
    if app_config.LOG_MODEL_TWO_OUT:
        out_data = np.exp(model(input)[:, 0].detach().numpy().flatten())
        out_test_data = np.round(
            np.exp(model(input_test)[:, 0].detach().numpy().flatten()))
    else:
        out_data = model(input).detach().numpy().flatten()
        out_test_data = np.round(model(input_test).detach().numpy().flatten())
    logger.debug(out_data)
    logger.debug(out_test_data)
    loss = checkpoint['loss']
    if plot_nn.epoch == app_config.NB_EPOCHS-1:
        logger.debug(
            f"real qs{np.round(corr_stopp.stoppingtimes[plot_nn.step,:].flatten())}")
        logger.debug(
            f"approx. qs{np.round(model(input_test).detach().numpy().flatten())}")
        corr_stopp.mistakes += np.sum(np.abs(np.round(corr_stopp.stoppingtimes[plot_nn.step, :].flatten(
        ))-out_test_data))

    reward_figure1 = go.Figure(
        go.Scatter(
            x=list(np.linspace(1, 6, 50)),
            y=list(out_data),
        )
    )

    reward_figure1.update_layout(yaxis_range=[0, 1])
    text1 = f"step in the game: {plot_nn.step} \n learning epoch: {plot_nn.epoch} \n loss: {loss} \n number of mistakes {corr_stopp.mistakes}"

    # generate copied model
    fpath2 = f"../output/{app_config.PATH_TWO}/phase_{plot_nn.step}/model_epoch_{plot_nn.epoch}.pt"
    checkpoint2 = torch.load(fpath2)
    model.load_state_dict(checkpoint2['model_state_dict'])
    model.train(False)
    model.double()
    if app_config.LOG_MODEL_TWO_OUT:
        out_data2 = np.exp(model(input)[:, 0].detach().numpy().flatten())
        out_test_data2 = np.round(
            np.exp(model(input_test)[:, 0].detach().numpy().flatten()))
    else:
        out_data2 = model(input).detach().numpy().flatten()
        out_test_data2 = np.round(model(input_test).detach().numpy().flatten())
    logger.debug(out_data)
    logger.debug(out_test_data)
    loss2 = checkpoint2['loss']
    if plot_nn.epoch == app_config.NB_EPOCHS-1:
        logger.debug(
            f"real qs{np.round(corr_stopp.stoppingtimes[plot_nn.step,:].flatten())}")
        logger.debug(
            f"approx. qs copy{np.round(model(input_test).detach().numpy().flatten())}")
        corr_stopp.mistakes2 += np.sum(np.abs(np.round(corr_stopp.stoppingtimes[plot_nn.step, :].flatten(
        ))-out_test_data2))

    reward_figure2 = go.Figure(
        go.Scatter(
            x=list(np.linspace(1, 6, 50)),
            y=list(out_data2),
        )
    )

    reward_figure2.update_layout(yaxis_range=[0, 1])
    text2 = f"step in the game: {plot_nn.step} \n learning epoch: {plot_nn.epoch} \n loss: {loss2} \n number of mistakes {corr_stopp.mistakes2}"
    plot_nn.epoch += 1
    style_data_conditional=[
                {
                    'if': {
                        'filter_query': '{{index}} = {}'.format(plot_nn.step),
                    },
                    'backgroundColor': '#FF4136',
                    'color': 'black'
                }, ]
    return reward_figure1, text1, reward_figure2, text2, style_data_conditional, style_data_conditional, style_data_conditional, style_data_conditional
    # return {'data': traces}


if __name__ == "__main__":
    app.run_server(debug=True, port=8092)
