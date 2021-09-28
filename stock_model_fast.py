""" Underlying model of the stochastic processes that are used:
- Black Scholes
- Heston
- Fractional Brownian motion
"""
"""
******************************************************************
TO DO: implement faster way to construct paths of brownian motion DONE
       implement faster way to construct paths of fractional brownian motion 
       implement code faster with cython 
******************************************************************
"""
import math
import numpy as np
import matplotlib.pyplot as plt

NB_JOBS_PATH_GEN = 1

#generating dicing model to evaluate action functions and so on

class Model_dice:
  def __init__(self, values, prob):
      self.values = values
      self.prob = prob

  def generate_paths(self,nb_paths=1, length=1):
    """
    Parameters
    --------------
    np_paths (int): number of generating paths

    Return
    --------------
    numpy array: shape(nb_paths,length of trajectory) 
      """

    return np.random.choice(a=self.values,size = (nb_paths,length),p=self.prob)


