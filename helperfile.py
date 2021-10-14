### this file is created to test some funcionality of the code


from stock_model_fast import Model_dice
import numpy as np
from payoff import Identity, MaxCall, Mean
from DOS import OptimalStoppingOptimization
#specify values

""" values = [1,2,3,4,5,6]
prob = [0.2,0.1,0.1,0.3,0.2,0.1]
data_model = Model_dice(values=values,prob=prob,nb_dates=10,nb_paths=10,nb_stocks=1)
data = data_model.generate_paths()
pay = Identity(strike=None)
pay2 = Mean(strike=None)
print(data.shape)
print(pay(data))
print(pay2(data))
 """

def f(x):
    x = x +2 
    return x

if __name__ == "__main__":
    A = 4
    b = f(A)
   
    print(A)