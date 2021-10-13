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
# %%
from os import replace
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from numpy.core.defchararray import array
from numpy.core.fromnumeric import reshape, shape
import payoff
import sys
NB_JOBS_PATH_GEN = 1


_PAYOFFS = {
    "MaxPut": payoff.MaxPut,
    "MaxCall": payoff.MaxCall,
    "GeometricPut": payoff.GeometricPut,
    "BasketCall": payoff.BasketCall,
    "Identity": payoff.Identity,
    "Max": payoff.Max,
    "Mean": payoff.Mean,
}


# generating dicing model to evaluate action functions and so on

class Model_dice:
    def __init__(self, values, prob, nb_paths, nb_stocks, nb_dates, payoff):
        self.values = np.array(values)
        self.prob = np.array(prob)
        self.nb_paths = nb_paths
        self.nb_stocks = nb_stocks
        self.nb_dates = nb_dates
        self.payoff = _PAYOFFS[payoff](strike=0)

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
        return np.random.choice(a=self.values, size=(self.nb_paths, self.nb_stocks, self.nb_dates), p=self.prob)

    def plot_paths(self):
        """plot paths of model 

        Args:
            nb_paths (int, optional): number of generating paths. Defaults to 1.
            length (int, optional): length of trajectory. Defaults to 1.
        """

        data = np.random.choice(a=self.values, size=(
            self.nb_paths, self.nb_dates), p=self.prob,replace=True)
        color = ["b", "g", "r", "c", "m", "y", "k", "w"]
        x = np.arange(0, self.nb_dates)
        for col in range(self.nb_paths):
            plt.plot(x, data[:, col], color[col % 8], label='path_'+str(col))
        plt.legend()
        plt.savefig("picture.png")
        plt.close()

    def get_emp_qvalues(self, sample):
          S = self.payoff.eval(np.array(self.values).reshape(6,1))
          result = np.asmatrix(S)
          for i in range(self.nb_dates-1,0, -1):
              logger.debug(f"i is {i}")
              liste = []
              for s in S:
                    # TODO: backwardind.
                    qvalue = max(np.mean(sample[sample[:, 0, i-1]==s,0,i]),s)
                    liste.append(qvalue)
                    #sample[sample[:, 0, i-1]==s,0,i-1] = np.maximum(np.zeros_like(sample[sample[:, 0, i-1]==s,0,i]) * qvalue,sample[sample[:, 0, i-1]==s,0,i-1]) 
                    # der Teil sample[sample[:, 0, i-1]==s,0,i-1] ist immer s, also ist das das selbe maximum das du oben schon gebildet hast
                    sample[sample[:, 0, i-1]==s,0,i-1] = np.ones_like(sample[sample[:, 0, i-1]==s,0,i-1]) * qvalue
              result = np.concatenate((result, np.array(liste).reshape((1,len(self.values)))),axis=0)
              logger.debug(f"result is {result}")
          return result

    def get_emp_stopping_rule(self, sample):
          qvalues = self.get_emp_qvalues(sample)
          result = (qvalues <= np.array([self.values]*(self.nb_dates),dtype=np.float32))
          result = result.astype("float32")
          return result
# %%
if __name__ == "__main__":
    values = [1, 2, 3, 4, 5, 6]
    prob = [0.1, 0.1, 0.1, 0.4, 0.2, 0.1]
    nb_stocks = 1
    nb_paths = 200
    nb_dates = 3
    modelo = Model_dice(values=values, prob=prob,
                        nb_paths=nb_paths, nb_dates=nb_dates,nb_stocks=nb_stocks,payoff="Identity")
    paths = modelo.generate_paths()
    modelo.get_emp_qvalues(paths)

# %%
