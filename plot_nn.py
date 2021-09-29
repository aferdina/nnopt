import torch
import numpy as np
from networks import NetworkDOS
from loguru import logger

step = 1
epoch = 0
def play():
    
    model = NetworkDOS(nb_stocks=1, hidden_size=4)
    fpath = f"/Users/andreferdinand/Desktop/Coding2/output/neural_networks/phase_{step}/model_epoch_{epoch}.pt"
    checkpoint = torch.load(fpath)
    model.load_state_dict(checkpoint['model_state_dict'])
    stock_values = np.reshape(np.array([1,2,3,4,5,6]), newshape=(6, 1))
    logger.debug(f"input {stock_values}")
    input = torch.from_numpy(stock_values).double()
    input = input.type(torch.double)
    #logger.debug(input.type())
    #input = X_inputs = torch.from_numpy(stock_values).double()
    model.train(False)
    out_data = model(input)

if __name__=="__main__":
    play()