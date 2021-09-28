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
  algos: Iterable[str] = ("DOS",)
  nb_dates: Iterable[int] = (20,)
  stock_models: Iterable[str] = ('dice_model',)
  strikes: Iterable[int] = (0,)
  nb_paths: Iterable[int] = (10,20,)
  nb_runs: int = 30
  nb_stocks: Iterable[int] = (1,)
  payoffs: Iterable[str] = ('Identity',)
  hidden_size: Iterable[int] = (4,)
  hidden_size2: Iterable[int] = (4,)
  nb_epochs: Iterable[int] = (30,)
  gamma: Iterable[float] = (1.,)
  step_size: Iterable[int] = (50,)
  eps: Iterable[float] = (0.001, )
  lr: Iterable[float] = (0.01,)
  train_ITM_only: Iterable[bool] = (True,)
  use_path: Iterable[bool] = (False,)
  representations: Iterable[str] = ('TablePriceDuration',)
  # When adding a filter here, also add to filtering.py.


'''
Comparison prices and computation time
'''

@dataclass
class _DimensionTable(_DefaultConfig):
  nb_stocks: Iterable[int] = (1,5)


table_spots_Dim_BS_MaxCall = _DimensionTable()