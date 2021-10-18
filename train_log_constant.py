import traceback
from matplotlib.pyplot import step
import numpy as np
from numpy.core.fromnumeric import trace
import torch
import torch.optim as optim
import torch.utils.data as tdata
from torch.utils.data import distributed
import os
from loguru import logger
import networks
import traceback
import sys
import numpy as np
import time 
from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here


#approximate specific function

logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time}</green> | <level>{level} | {message}</level>",
    level="TRACE",
)

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.manual_seed(42)
        # torch.nn.init.zeros_(m.weight)
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class TrainLOG_CONSTANT(object):
    """Train/evaluation of the neural network used for the stopping decision"""

    def __init__(self, nb_stocks, hidden_size=4, const_value=5 , support= np.linspace(1,6,endpoint=True,num = 1000), 
                 batch_size=2000, eps=0.001, storage_loc="weights_log"):
        self.support = support.reshape((1000,1))
        self.const_value = const_value
        self.eps = eps
        self.nb_stocks = nb_stocks
        self.storage_loc = storage_loc
        self.batch_size = batch_size
        self.network = networks.NetworksoftlogDOS(
            self.nb_stocks, hidden_size=hidden_size).double()
        self.network.apply(init_weights)
        self.writer = SummaryWriter(f'runs/wight_const_{int(time.time())}')

    def _Loss(self, X):
        return -torch.mean(X)*2

    def get_weigthts(self):
        #set up vectors
        logger.debug(f"size im is {self.support.shape}")
        immediate_exercise_value = np.ones_like(self.support)*self.const_value
        discounted_next_values = np.ones_like(self.support)*self.const_value
        test_values = torch.from_numpy(np.ones_like(self.support)* 0.5).double()
        #initialize optimizer for the game 
        optimizer = optim.Adam(self.network.parameters())
        #create matrix to use pointwise multiplication to calculate the loss 
        mat = np.concatenate((immediate_exercise_value.reshape(immediate_exercise_value.shape[0],1),discounted_next_values.reshape(len(discounted_next_values),1)),axis=1)
        logger.debug(f"matrix is {mat}")
        mat = torch.from_numpy(mat).double()
        #cast input of neural network
        X_inputs = torch.from_numpy(self.support).double()
        self.network.train(True)
        step = 0
        self.writer.add_graph(self.network,X_inputs)
        fpath = os.path.join(os.path.dirname(
                    __file__), f"../output/{self.storage_loc}")
        os.makedirs(fpath, exist_ok=True)
        logger.debug("running")
        while True:
            for batch in tdata.BatchSampler(
                    tdata.RandomSampler(
                        range(len(X_inputs)), replacement=False),
                    batch_size=self.batch_size, drop_last=False):
                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    outputs = self.network(X_inputs[batch])
                    values = torch.mul(mat[batch,:],outputs)
                    loss = self._Loss(values)
                    loss.backward()
                    optimizer.step()
                self.network.train(False)
                lossy = np.mean(torch.abs(torch.exp(self.network(X_inputs))[:,0]-test_values).detach().numpy())
                logger.debug(f"loss is given by {lossy}")
                self.network.train(True)
                logger.debug(f"{lossy<self.eps}")
                step +=1
                if step % 500 ==0:
                    self.writer.add_scalar("Loss/train", loss, step)
                    self.writer.add_histogram("layer1.bias",self.network.layer1.bias,step)
                    self.writer.add_histogram("layer1.weight",self.network.layer1.weight,step)
                    self.writer.add_histogram("layer1.weight.grad",self.network.layer1.weight.grad,step)
                    self.writer.add_histogram("layer3.bias",self.network.layer3.bias,step)
                    self.writer.add_histogram("layer3.weight",self.network.layer3.weight,step)
                    self.writer.add_histogram("layer3.weight.grad",self.network.layer3.weight.grad,step)

                if step % 50 ==0:
                    tmp_path = fpath + f"/model_epoch_{step}.pt"
                    torch.save({
                        'model_state_dict': self.network.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                    }, tmp_path)
                if lossy < self.eps:
                    tmp_path = fpath + f"/model_epoch_final.pt"
                    torch.save({
                        'model_state_dict': self.network.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                    }, tmp_path)
                    self.writer.flush()
                    self.writer.close()
                    return
                
if __name__ == "__main__":
    training = TrainLOG_CONSTANT(nb_stocks=1)
    training.get_weigthts()