from stock_model_fast import Model_dice
import numpy as np

#specify values

values = np.array([1,2,3,4,5,6],dtype=np.int)
prob = np.array([0.2,0.1,0.1,0.3,0.2,0.1])
data_model = Model_dice(values=values,prob=prob)

print(data_model.generate_paths(10,10))