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
    stock_values = np.reshape([1.,2.,3.,4.,5.,6.],(6,1))
    input = torch.from_numpy(stock_values)
    input = input.double()
    model.train(False)
    model.double()
    out_data = model(input)
    print(out_data)

if __name__=="__main__":
    play() 