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
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

NB_JOBS_PATH_GEN = 1

#generating dicing model to evaluate action functions and so on

class Model_dice:
  def __init__(self, values, prob, nb_paths, nb_stocks, nb_dates):
      self.values = np.array(values)
      self.prob = np.array(prob)
      self.nb_paths = nb_paths
      self.nb_stocks = nb_stocks
      self.nb_dates = nb_dates

  def generate_paths(self):
    """
    Parameters
    --------------
    np_paths (int, optional): number of generating paths
    nb_stocks (int, optional): number of dices in the game
    nb_dates (int, optional): number of stopping dates in the game

    Return
    --------------
    numpy array: shape(nb_paths,length of trajectory) 
      """
    logger.debug(f"in function")
    return np.random.choice(a=self.values,size = (self.nb_paths, self.nb_stocks, self.nb_dates),p=self.prob)

  def plot_paths(self):
    """plot paths of model 

    Args:
        nb_paths (int, optional): number of generating paths. Defaults to 1.
        length (int, optional): length of trajectory. Defaults to 1.
    """

    data = np.random.choice(a=self.values,size = (self.nb_paths,self.nb_dates),p=self.prob)
    color = ["b","g","r","c","m","y","k","w"]
    x = np.arange(0,self.nb_dates)
    for col in range(self.nb_paths):
          plt.plot(x, data[:, col], color[col % 8], label='path_'+str(col))
    plt.legend()
    plt.savefig("picture.png")
    plt.close()



