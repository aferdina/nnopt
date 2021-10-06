""" Computes the American option price using the deep optimal stopping (DOS).

It is the implementation of the deep optimal stopping (DOS) introduced in
(deep optimal stopping, Becker, Cheridito and Jentzen, 2020).
TODO: rewrite such that new initializazion is needed
"""
import traceback
from matplotlib.pyplot import step
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as tdata
from torch.utils.data import distributed
import os
from loguru import logger
import corr_stopp

import backward_induction
import networks
import traceback

def init_weights(m):
  if isinstance(m, torch.nn.Linear):
    torch.manual_seed(42)
    # torch.nn.init.zeros_(m.weight)
    torch.nn.init.xavier_uniform_(m.weight)
    m.bias.data.fill_(0.01)



class DeepOptimalStopping(backward_induction.AmericanOptionPricer):
  """Computes the American option price using the deep optimal stopping (DOS)
  """

  def __init__(self, model, payoff, nb_epochs=20, nb_batches=None,
               hidden_size=10, use_path=False, eps=0.001, copy=True):
    del nb_batches
    super().__init__(model, payoff ,use_path=use_path)
    self.hidden_size = hidden_size
    self.eps = eps
    self.nb_epochs = nb_epochs
    self.copy = copy
    if self.use_path:
      self.state_size = model.nb_stocks * (model.nb_dates+1)
    else:
      self.state_size = model.nb_stocks
    if copy:
      self.neural_stopping = OptimalStoppingOptimization(
        self.state_size, self.model.nb_paths, hidden_size=self.hidden_size,
        nb_iters=self.nb_epochs,eps=0.001)  

  def stop(self, step, stock_values, immediate_exercise_values,
           discounted_next_values, copy=True, h=None, new_init =False):
    """ see base class """
  
    if not copy:
          self.neural_stopping = OptimalStoppingOptimization(
            self.state_size, self.model.nb_paths, hidden_size=self.hidden_size,
            nb_iters=self.nb_epochs,eps = 0.001)    
    #logger.debug(f"step given by {step}")
    if self.use_path:
      # shape [paths, stocks, dates up to now]
      stock_values = np.flip(stock_values, axis=2)
      # add zeros to get shape [paths, stocks, dates+1]
      stock_values = np.concatenate(
        [stock_values, np.zeros(
          (stock_values.shape[0], stock_values.shape[1],
           self.model.nb_dates + 1 - stock_values.shape[2]))], axis=-1)
      stock_values = stock_values.reshape((stock_values.shape[0], -1))
    
    self.neural_stopping.train_network(step,
      stock_values[:self.split],
      immediate_exercise_values.reshape(-1, 1)[:self.split],
      discounted_next_values[:self.split])
    inputs = stock_values
    stopping_rule = self.neural_stopping.evaluate_network(inputs)
    S = np.reshape([1,2,3,4,5,6],(6,1))
    try: 
      logger.debug(np.round(self.neural_stopping.evaluate_network(S)))
    except Exception:
      print(traceback.format_exc())
      
    return stopping_rule




class OptimalStoppingOptimization(object):
  """Train/evaluation of the neural network used for the stopping decision"""

  def __init__(self, nb_stocks, nb_paths, hidden_size=10, nb_iters=20,
               batch_size=2000, eps = 0.001):
    self.eps = eps
    self.nb_stocks = nb_stocks
    self.nb_paths = nb_paths
    self.nb_iters = nb_iters
    self.batch_size = batch_size
    self.network = networks.NetworkDOS(
      self.nb_stocks, hidden_size=hidden_size).double()
    self.network.apply(init_weights)  

  def _Loss(self, X):
    return -torch.mean(X)

  def train_network(self, step, stock_values, immediate_exercise_value,
                    discounted_next_values):
    optimizer = optim.Adam(self.network.parameters())
    discounted_next_values = torch.from_numpy(discounted_next_values).double()
    immediate_exercise_value = torch.from_numpy(immediate_exercise_value).double()
    X_inputs = torch.from_numpy(stock_values).double()
    #logger.debug("debug train_network")
    self.network.train(True)
    ones = torch.ones(len(discounted_next_values))
    for i in range(self.nb_iters):
      for batch in tdata.BatchSampler(
              tdata.RandomSampler(range(len(X_inputs)), replacement=False),
              batch_size=self.batch_size, drop_last=False):

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
          outputs = self.network(X_inputs[batch])
          values = (immediate_exercise_value[batch] * outputs +
                    discounted_next_values[batch] * (ones[batch] - outputs))
          loss = self._Loss(values)
          loss.backward()
          optimizer.step()
      fpath = os.path.join(os.path.dirname(__file__),f"../output/neural_networks3/phase_{step}")
      os.makedirs(fpath, exist_ok=True)
      tmp_path = fpath + f"/model_epoch_{i}.pt"
      #logger.debug(f"loss: {loss}")
      torch.save({
                  'epoch': i,
                  'model_state_dict': self.network.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'loss': loss,
                  }, tmp_path)
      #logger.debug("Hello")
      
    

  def log_train_network(self, stock_values, immediate_exercise_value,
                    discounted_next_values):
    optimizer = optim.Adam(self.network.parameters())
    discounted_next_values = torch.from_numpy(discounted_next_values).double()
    immediate_exercise_value = torch.from_numpy(immediate_exercise_value).double()
    X_inputs = torch.from_numpy(stock_values).double()

    self.network.train(True)
    ones = torch.ones(len(discounted_next_values))
    for _ in range(self.nb_iters):
      for batch in tdata.BatchSampler(
              tdata.RandomSampler(range(len(X_inputs)), replacement=False),
              batch_size=self.batch_size, drop_last=False):

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
          outputs = self.network(X_inputs[batch]).reshape(-1)
          values = (immediate_exercise_value[batch].reshape(-1) * torch.log(outputs+torch.finfo(torch.float32).eps) +
                    discounted_next_values[batch] * torch.log((ones[batch] - outputs)+torch.finfo(torch.float32).eps))
          loss = self._Loss(values)
          loss.backward()
          optimizer.step()

  def train_network_abbruch(self, stock_values, immediate_exercise_value, discounted_next_values):
        eps = torch.tensor(self.eps,dtype=torch.float64)
        optimizer = optim.Adam(self.network.parameters())
        discounted_next_values = torch.from_numpy(discounted_next_values).double()
        immediate_exercise_value = torch.from_numpy(immediate_exercise_value).double()
        inputs = stock_values
        X_inputs = torch.from_numpy(inputs).double()
        self.network.train(True)
        ones = torch.ones(len(discounted_next_values))
        T=True
        while T:
              for batch in tdata.BatchSampler(tdata.RandomSampler(range(len(X_inputs)),replacement=False),batch_size=self.batch_size,drop_last=False):
                    OSV_old = torch.tensor(-1000.,dtype=torch.float64)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(True):
                      outputs = self.network(X_inputs[batch]).reshape(-1)
                      values = (immediate_exercise_value[batch].reshape(-1) * outputs + discounted_next_values[batch] * (ones[batch] - outputs))
                      loss = self._Loss(values)
                      loss.backward()
                      optimizer.step()
                      if torch.abs(loss-OSV_old)<eps:
                            T = False
                            break
                      else:
                            OSV_old = loss

  def log_train_network_abbruch(self, stock_values, immediate_exercise_value, discounted_next_values):
        eps = torch.tensor(self.eps,dtype=torch.float64)
        optimizer = optim.Adam(self.network.parameters())
        discounted_next_values = torch.from_numpy(discounted_next_values).double()
        immediate_exercise_value = torch.from_numpy(immediate_exercise_value).double()
        inputs = stock_values
        X_inputs = torch.from_numpy(inputs).double()
        self.network.train(True)
        ones = torch.ones(len(discounted_next_values))
        T=True
        while T:
              for batch in tdata.BatchSampler(tdata.RandomSampler(range(len(X_inputs)),replacement=False),batch_size=self.batch_size,drop_last=False):
                    OSV_old = torch.tensor(-1000.,dtype=torch.float64)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(True):
                      outputs = self.network(X_inputs[batch]).reshape(-1)
                      values = (immediate_exercise_value[batch].reshape(-1) * outputs + discounted_next_values[batch] * (ones[batch] - outputs))
                      loss = self._Loss(values)
                      loss.backward()
                      optimizer.step()
                      if torch.abs(loss-OSV_old)<eps:
                            T = False
                            break
                      else:
                            OSV_old = loss

  def evaluate_network(self, X_inputs):
    self.network.train(False)
    X_inputs = torch.from_numpy(X_inputs).double()
    outputs = self.network(X_inputs)
    return outputs.view(len(X_inputs)).detach().numpy()
