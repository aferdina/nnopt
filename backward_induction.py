"""Base class that optimal stopping value using backward recusrion.

All algorithms that are using a backward recusrion such as
LSM (Least squares Monte Carlo),
NLSM (Neural Least squares Monte Carlo),
RLSM (Randomized Least squares Monte Carlo)
and DOS (Deep Optimal Stopping) are inherited from this class.
TODO: rewrite function in cython 
"""

import numpy as np
import time
from loguru import logger
from numpy.lib.function_base import copy
import sys
import os

class AmericanOptionPricer:
  """Computes the price of an American Option using backward recusrion.
  """
  def __init__(self, model, payoff,use_rnn=False, train_ITM_only=True,
               use_path=False, copy=True, values=[1,2,3,4,5,6], storage_loc = "/neural_networks_4"):

    #class model: The stochastic process model of the stock (e.g. Black Scholes).
    self.model = model

    #class payoff: The payoff function of the option (e.g. Max call).
    self.payoff = payoff

    #bool: randomized neural network is replaced by a randomized recurrent NN.
    self.use_rnn = use_rnn

    #bool: x_k is replaced by the entire path (x_0, .., x_k) as input of the NN.
    self.use_path = use_path

    #bool: only the paths that are In The Money (ITM) are used for the training.
    self.train_ITM_only = train_ITM_only

    #bool: copy weights or not
    self.copy = copy

    #list: possible values in dice game 
    self.values = values

    #str: place where information are stored
    self.storage_loc = storage_loc


  def calculate_continuation_value(self):
    """Computes the continuation value of an american option at a given date.

    All algorithms that inherited from this class (AmericanOptionPricer) where
    the continuation value is approximated by basis functions (LSM),
    neural networks (NLSM), randomized neural networks (RLSM), or
    recurrent randomized neural networks (RRLSM) only differ by a this function.

    The number of paths determines the size of the arrays.

    Args:
      values (np array): the option price of the next date (t+1).
      immediate_exercise_value (np array): the payoff evaluated with the current
       stock price (date t).
      stock_paths_at_timestep (np array): The stock price at the current date t.

    Returns:
      np array: the option price at current date t if we continue until next
       date t+1.
    """
    raise NotImplementedError

  def stop(self, stock_values, immediate_exercise_values,
           discounted_next_values, h=None):
    """Returns a vector of {0, 1}s (one per path) for a given data, where:
        1 means stop, and
        0 means continue.

    The optimal stopping algorithm (DOS) where the optimal stopping is
    approximated by a neural network has a different function "stop".
    """
    stopping_rule = np.zeros(len(stock_values))
    if self.use_rnn:
      continuation_values = self.calculate_continuation_value(
          discounted_next_values,
          immediate_exercise_values, h)
    else:
      continuation_values = self.calculate_continuation_value(
        discounted_next_values,
        immediate_exercise_values, stock_values)
    if self.train_ITM_only:
      which = (immediate_exercise_values > continuation_values) & \
              (immediate_exercise_values > np.finfo(float).eps)
    else:
      which = immediate_exercise_values > continuation_values
    stopping_rule[which] = 1
    return stopping_rule

  def price(self):
    """It computes the price of an American Option using a backward recusrion.
    """
    logger.debug("start pricing")
    t1 = time.time() 
    stock_paths = self.model.generate_paths()
    print("time path gen: {}".format(time.time()-t1), end=" ")
    self.split = int(len(stock_paths)/2)
    emp_qvalues = self.model.get_emp_qvalues(stock_paths[self.split:,:,:])
    emp_stopping = self.model.get_emp_stopping_rule(stock_paths[self.split:,:,:])
    logger.debug("stocks are generated")
    fpath = f'../output/{self.storage_loc}/'
    os.makedirs(fpath, exist_ok=True)
    tmp_fpath_emp_q = fpath + 'emp_qvalues.csv'
    tmp_fpath_emp_stopp = fpath + "emp_stopp.csv"
    np.savetxt(tmp_fpath_emp_q, emp_qvalues, fmt='%1.4f',delimiter=",")
    np.savetxt(tmp_fpath_emp_stopp, emp_stopping, fmt='%1.4f',delimiter=",")
    step = 1
    if self.use_rnn:
      hs = self.compute_hs(stock_paths)
    disc_factor = 1
    immediate_exercise_value = self.payoff.eval(stock_paths[:, :, -1])
    values = immediate_exercise_value
    emp_step_qvalues = np.asmatrix(self.values)
    for date in range(stock_paths.shape[2] - 2, 0, -1):
      immediate_exercise_value = self.payoff.eval(stock_paths[:, :, date])

      #empirical Q values for trainingsample
      logger.debug(f"empirical Q vaules:")
      liste = []
      for i in self.values:
        which2 = (immediate_exercise_value[self.split:] == i)
        helpi = values[self.split:]
        liste.append(np.mean(helpi[which2]))
      emp_step_qvalues = np.concatenate((emp_step_qvalues,np.array(liste).reshape((1,len(self.values)))),axis=0)
      logger.debug(emp_step_qvalues)


      if self.use_rnn:
        h = hs[date]
      else:
        h = None
      if self.use_path:
        stopping_rule = self.stop(step, 
          stock_paths[:, :, :date+1], immediate_exercise_value,
          values * disc_factor, copy=self.copy, h=h)
      else:
        stopping_rule = self.stop(step, 
          stock_paths[:, :, date], immediate_exercise_value,
          values * disc_factor, copy=self.copy, h=h)
      which = (stopping_rule > 0.5)
      values[which] = immediate_exercise_value[which]
      values[~which] *= disc_factor
      
      step +=1
    payoff_0 = self.payoff.eval(stock_paths[:, :, 0])[0]
    tmp_fpath_emp_step_q = fpath + "emp_step_qvalues.csv"
    np.savetxt(tmp_fpath_emp_step_q, emp_step_qvalues, fmt='%1.4f',delimiter=",")
    # IMPORTANT change to values[self.split:] to get test error
    return max(payoff_0, np.mean(values[:self.split]) * disc_factor)