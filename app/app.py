import dash
from dash.dependencies import Input, Output, State
from dash import dcc
from dash import html
from random import random
from dash import dash_table
from loguru import logger
import sys
import pandas as pd
import numpy as np
from torch._C import Graph
import plotly.graph_objects as go
import os
import torch
sys.path.insert(0, os.path.dirname("."))
import configs.corr_stopp as corr_stopp
import configs.app_config as app_config

# application, which compares the convergence of two algorithm

# create datatable of the empirical q values
# emp_qvalues are q values from the whole dataset, emp_step_qvalues are the empirical q-values given the decisions of the algorithm
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
            html.Div([html.P('Empirical Q-Values, depending on policy'), ]),
            dash_table.DataTable(id='table1', columns=[
                                 {"name": i, "id": i} for i in emp_steps_qvalues1.columns], data=emp_steps_qvalues1.to_dict('records'), page_size=5, style_data_conditional=[
                {
                    'if': {
                        'filter_query': '{{index}} = {}'.format(1),
                    },
                    'backgroundColor': '#FF4136',
                    'color': 'black'
                }, ]),
            html.Div([html.P('Empirical Q-Values of the trainingsset'), ]),
            dash_table.DataTable(id='table2', columns=[
                                 {"name": i, "id": i} for i in emp_qvalues1.columns], data=emp_qvalues1.to_dict('records'), page_size=5, style_data_conditional=[
                {
                    'if': {
                        'filter_query': '{{index}} = {}'.format(1),
                    },
                    'backgroundColor': '#FF4136',
                    'color': 'black'
                }, ]),
            # same structure for the second model
            dcc.Graph(id="live-update-graph-scatter2", animate=True),
            html.Div(id="step2"),
            html.Div([html.P('Empirical Q-Values, depending on policy'), ]),
            dash_table.DataTable(id='table3', columns=[
                                 {"name": i, "id": i} for i in emp_steps_qvalues2.columns], data=emp_steps_qvalues2.to_dict('records'), page_size=5, style_data_conditional=[
                {
                    'if': {
                        'filter_query': '{{index}} = {}'.format(1),
                    },
                    'backgroundColor': '#FF4136',
                    'color': 'black'
                }, ]),
            html.Div([html.P('Empirical Q-Values of the trainingsset'), ]),
            dash_table.DataTable(id='table4', columns=[
                                 {"name": i, "id": i} for i in emp_qvalues2.columns], data=emp_qvalues2.to_dict('records'), page_size=5, style_data_conditional=[
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

# callback for trainings animation


@app.callback(
    [
        Output("live-update-graph-scatter", "figure"),
        Output("step", "children"),
        Output("live-update-graph-scatter2", "figure"),
        Output("step2", "children"),
        Output("table1", "style_data_conditional"),
        Output("table2", "style_data_conditional"),
        Output("table3", "style_data_conditional"),
        Output("table4", "style_data_conditional"),
    ],
    Input("interval-component", "n_intervals"),
)
def update_graph_scatter(x):
    # if the maximal number of epochs
    if app_config.epoch == app_config.NB_EPOCHS:
        app_config.epoch = app_config.NB_EPOCHS_START
        app_config.step += 1
    logger.trace("Play action.")
    # if the maximum number of epochs is reached, switch to the next step of the algorithm
    model = app_config.network
    # import the first model
    fpath = f"../output/{app_config.PATH_ONE}/phase_{app_config.step}/model_epoch_{app_config.epoch}.pt"
    checkpoint = torch.load(fpath)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.train(False)
    model.double()
    # Creation of a vector to evaluate the goodness of the algorithm
    stock_values_test = np.reshape([1, 2, 3, 4, 5, 6], (6, 1))
    input_test = torch.from_numpy(stock_values_test).double()
    # Creation of a vector to plot the function in the web application
    stock_values = np.linspace(1, 6, num=50)
    stock_values = np.reshape(stock_values, (50, 1))
    input = torch.from_numpy(stock_values)
    input = input.double()
    # if log model is used, the exponential function is needed
    if app_config.LOG_MODEL_TWO_OUT:
        out_data = np.exp(model(input)[:, 0].detach().numpy().flatten())
        out_test_data = np.round(
            np.exp(model(input_test)[:, 0].detach().numpy().flatten()))
    else:
        out_data = model(input).detach().numpy().flatten()
        out_test_data = np.round(model(input_test).detach().numpy().flatten())
    logger.debug(out_data)
    logger.debug(out_test_data)
    # get loss of the algorithm on the trainingssample
    loss = checkpoint['loss']
    # evaluate the algorithm with respect to the theoretical stopping times
    if app_config.epoch == app_config.NB_EPOCHS-1:
        logger.debug(
            f"real qs{np.round(corr_stopp.stoppingtimes[app_config.step,:].flatten())}")
        logger.debug(
            f"approx. qs{out_test_data}")
        corr_stopp.mistakes += np.sum(np.abs(np.round(corr_stopp.stoppingtimes[app_config.step, :].flatten(
        ))-out_test_data))
    # plot the decision function of the algorithm
    reward_figure1 = go.Figure(
        go.Scatter(
            x=list(np.linspace(1, 6, 50)),
            y=list(out_data),
        )
    )
    reward_figure1.update_layout(yaxis_range=[0, 1])
    # plot information about the game
    text1 = f"step in the game: {app_config.step} \n learning epoch: {app_config.epoch} \n loss: {loss} \n number of mistakes {corr_stopp.mistakes}"
    # generate second model identical to the first one
    fpath2 = f"../output/{app_config.PATH_TWO}/phase_{app_config.step}/model_epoch_{app_config.epoch}.pt"
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
    logger.debug(out_data2)
    logger.debug(out_test_data2)
    loss2 = checkpoint2['loss']
    if app_config.epoch == app_config.NB_EPOCHS-1:
        logger.debug(
            f"real qs{np.round(corr_stopp.stoppingtimes[app_config.step,:].flatten())}")
        logger.debug(
            f"approx. qs copy{out_test_data2}")
        corr_stopp.mistakes2 += np.sum(np.abs(np.round(corr_stopp.stoppingtimes[app_config.step, :].flatten(
        ))-out_test_data2))
    reward_figure2 = go.Figure(
        go.Scatter(
            x=list(np.linspace(1, 6, 50)),
            y=list(out_data2),
        )
    )
    reward_figure2.update_layout(yaxis_range=[0, 1])
    text2 = f"step in the game: {app_config.step} \n learning epoch: {app_config.epoch} \n loss: {loss2} \n number of mistakes {corr_stopp.mistakes2}"
    app_config.epoch += 1
    style_data_conditional = [
        {
            'if': {
                'filter_query': '{{index}} = {}'.format(app_config.step),
            },
            'backgroundColor': '#FF4136',
            'color': 'black'
        }, ]
    return reward_figure1, text1, reward_figure2, text2, style_data_conditional, style_data_conditional, style_data_conditional, style_data_conditional


if __name__ == "__main__":
    app.run_server(debug=True, port=8092)
