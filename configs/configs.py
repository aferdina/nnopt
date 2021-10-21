# ATTENTION: only one parameter set should be selected, multiple configuration may leed to a problem

from dataclasses import dataclass
import typing
from typing import Iterable
import numpy as np

FigureType = typing.NewType('FigureType', str)
TablePrice = FigureType("TablePrice")
TableDuration = FigureType("TableDuration")
PricePerNbPaths = FigureType("PricePerNbPaths")


@dataclass
class _DefaultConfig:
    #algorithm using for function approximation
    algos: Iterable[str] = ("logsoftDOS",) 
    # number of dates for exercising the option
    nb_dates: Iterable[int] = (20,)
    # used stock model for creating the path samples
    stock_models: Iterable[str] = ('dice_model',)
    # strike price 
    strikes: Iterable[int] = (0,)
    #number of paths generated from 'stock_model'; 50% for training, 50% for evaluating 
    nb_paths: Iterable[int] = (600,)
    # number of algorithm iterations 
    nb_runs: int = 1
    # number of stocks, used in the algorithm 
    nb_stocks: Iterable[int] = (1,)
    #payoff function, for evaluating the option
    payoffs: Iterable[str] = ('MaxCall',)
    # number of nodes in the hidden layer
    hidden_size: Iterable[int] = (4,)
    hidden_size2: Iterable[int] = (4,)
    #number of epochs for training the neural network
    nb_epochs: Iterable[int] = (10,)
    # discount factor of the game 
    gamma: Iterable[float] = (1.,)
    # parameter which defines how many times the same learning rate should be used
    step_size: Iterable[int] = (50,)
    # tetermination criteria 
    eps: Iterable[float] = (0.001, )
    # start learning rate of the algorithm 
    lr: Iterable[float] = (0.001,)
    # bool, whether weights of last period should be used for initializing the neural network in the next step
    copy: Iterable[bool] = (True,)
    # bool, whether only paths in the money should be used for training 
    train_ITM_only: Iterable[bool] = (True,)
    # use whole path in non markovian setup
    use_path: Iterable[bool] = (False,)
    # location, where game information should be stored
    storage_loc: Iterable[str] = ("neural_networks4_copy_const",)
    # bool, whether neural net is initialized constant at 0.5
    start_const: Iterable[bool] = (True,)
    representations: Iterable[str] = ('TablePriceDuration',)
    # When adding a filter here, also add to filtering.py.


'''
Comparison prices and computation time
'''


@dataclass
class _DimensionTable(_DefaultConfig):
    nb_stocks: Iterable[int] = (1,)


table_spots_Dim_BS_MaxCall = _DimensionTable()
