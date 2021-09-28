from stock_model_fast import Model_dice
import numpy as np
from payoff import Identity, MaxCall, Mean
#specify values

values = np.array([1,2,3,4,5,6],dtype=np.int)
prob = np.array([0.2,0.1,0.1,0.3,0.2,0.1])
data_model = Model_dice(values=values,prob=prob,nb_dates=10,nb_paths=5,nb_stocks=1)
data = data_model.generate_paths()
pay = Identity(strike=None)
pay2 = Mean(strike=None)
print(data)
print(pay(data))
print(pay2(data))