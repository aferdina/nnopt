""" Implementing of the Deep Optimal Stopping Algorithm 
"""
from matplotlib.pyplot import step
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as tdata
import os
from loguru import logger
import traceback
import sys

sys.path.insert(0, os.path.dirname("."))
import utils.networks as networks
import backward_induction
# delcare init function to reset the weights of the neural network after a step of the backward recursion
import functools
import time
from torch.utils.tensorboard import SummaryWriter

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.manual_seed(42)
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class DeepOptimalStopping(backward_induction.AmericanOptionPricer):
    """Computes the American option price using the deep optimal stopping (DOS)
    """

    def __init__(self, model, payoff, nb_epochs=20, nb_batches=None,
                 hidden_size=10, use_path=False, eps=0.001, lr = 0.001, copy=True, values=[1, 2, 3, 4, 5, 6], storage_loc="neural_networks_4", start_const = True):
        """init class

        Args:
            model (object): object for creating the path samples 
            payoff (object): payoff function of the algorithm
            nb_epochs (int, optional): number of epochs in the game. Defaults to 20.
            nb_batches (int, optional): number of batches, used in the algorithm. Defaults to None.
            hidden_size (int, optional): size of the hidden layer of the neural network. Defaults to 10.
            use_path (bool, optional): wheater or not, using the whole trajectory. Defaults to False.
            eps (float, optional): bound for the termination criterion. Defaults to 0.001.
            copy (bool, optional): whether using the weights of the last period or not. Defaults to True.
            values (list, optional): possible values of the markov decision process. Defaults to [1, 2, 3, 4, 5, 6].
            storage_loc (str, optional): location, where the information of the algorithm beeing saved. Defaults to "neural_networks_4".
        """
        del nb_batches
        super().__init__(model, payoff, use_path=use_path,
                         copy=copy, values=values, storage_loc=storage_loc)
        self.hidden_size = hidden_size
        self.eps = eps
        self.storage_loc = storage_loc
        self.nb_epochs = nb_epochs
        self.start_const = start_const
        self.lr = lr
        if self.use_path:
            self.state_size = model.nb_stocks * (model.nb_dates+1)
        else:
            self.state_size = model.nb_stocks
        # if the model should be copied, the weights of the last period are used
        if self.copy:
            self.neural_stopping = OptimalStoppingOptimization(
                self.state_size, self.model.nb_paths, hidden_size=self.hidden_size,
                nb_iters=self.nb_epochs, eps=0.001, lr = self.lr, storage_loc=self.storage_loc, start_const= self.start_const)

    def stop(self, step, stock_values, immediate_exercise_values,
             discounted_next_values, copy=True, h=None):
        """

        Args:
            step (int): step in the game 
            stock_values (numpy.array):  values of the all stocks at timestep 'step' 
            immediate_exercise_values (numpy.array): value of the option at timestep 'step'
            discounted_next_values (numpy.array): continuation values 
            copy (bool, optional): whether weights of the last model should be used for initializing the network. Defaults to True.
            h (bool, optional): bool, whether recurrent network should be used. Defaults to None.

        Returns:
            numpy.array: array of the form {0,1}^n, where 1 means continue and 0 means stop
        """
        writer = SummaryWriter(f'runs_dos/{self.storage_loc}_{step}_{int(time.time())}')
        # if the weights are not copied, then a new neural network is initialized at each step in time
        if not copy:
            self.neural_stopping = OptimalStoppingOptimization(
                self.state_size, self.model.nb_paths, hidden_size=self.hidden_size,
                nb_iters=self.nb_epochs, eps=0.001, lr = self.lr, storage_loc=self.storage_loc, start_const= self.start_const)
        # only needed in not markovian case
        if self.use_path:
            # shape [paths, stocks, dates up to now]
            stock_values = np.flip(stock_values, axis=2)
            # add zeros to get shape [paths, stocks, dates+1]
            stock_values = np.concatenate(
                [stock_values, np.zeros(
                    (stock_values.shape[0], stock_values.shape[1],
                     self.model.nb_dates + 1 - stock_values.shape[2]))], axis=-1)
            stock_values = stock_values.reshape((stock_values.shape[0], -1))
        # train the neural network on the training sample
        self.neural_stopping.train_network(step, writer,
                                           stock_values[:self.split],
                                           immediate_exercise_values.reshape(-1, 1)[
                                               :self.split],
                                           discounted_next_values[:self.split])
        # create stopping policy for all paths
        inputs = stock_values
        stopping_rule = self.neural_stopping.evaluate_network(inputs)
        S = np.reshape([1, 2, 3, 4, 5, 6], (6, 1))
        try:
            logger.debug(np.round(self.neural_stopping.evaluate_network(S)))
        except Exception:
            print(traceback.format_exc())

        return stopping_rule


class OptimalStoppingOptimization(object):
    """Train/evaluation of the neural network used for the stopping decision"""

    def __init__(self, nb_stocks, nb_paths, hidden_size=10, nb_iters=20,
                 batch_size=2000, eps=0.001, lr =0.01, storage_loc="neural_networks_4", start_const = True):
        """Optimization class of the neural net

        Args:
            nb_stocks (int): number of stocks, generated by the path model 
            nb_paths (int): number of paths, generated by the path model
            hidden_size (int, optional): size of the hidden layer of the neural network. Defaults to 10.
            nb_iters (int, optional): number if iteration in training step. Defaults to 20.
            batch_size (int, optional): size of batches for training. Defaults to 2000.
            eps (float, optional): bound of the tetermination criteria of the learning phase. Defaults to 0.001.
            storage_loc (str, optional): location, where the information of the model are saved. Defaults to "neural_networks_4".
        """
        self.eps = eps
        self.nb_stocks = nb_stocks
        self.nb_paths = nb_paths
        self.nb_iters = nb_iters
        self.storage_loc = storage_loc
        self.batch_size = batch_size
        self.lr = lr
        self.network = networks.NetworkDOS(
            self.nb_stocks, hidden_size=hidden_size).double()
        if not start_const:
            self.network.apply(init_weights)
        else:
            fpath = f"./output/weights_const/model_epoch_final.pt"
            checkpoint = torch.load(fpath)
            self.network.load_state_dict(checkpoint['model_state_dict'])

    def _Loss(self, X):
        return -torch.mean(X)

    def train_network(self, step, writer, stock_values, immediate_exercise_value,
                      discounted_next_values):
        """training of the neural network

        Args:
            step (int): step in the backward recursion 
            stock_values (numpy.array): stockvalues of all trainings path
            immediate_exercise_value (numpy.array): value for exercising the option 
            discounted_next_values numpy.array): continuation values 
        """
        # initializing the optimzier
        optimizer = optim.Adam(self.network.parameters(), lr = self.lr)
        # prepare data (cast to torch tensor) for the training
        discounted_next_values = torch.from_numpy(
            discounted_next_values).double()
        immediate_exercise_value = torch.from_numpy(
            immediate_exercise_value).double()
        X_inputs = torch.from_numpy(stock_values).double()
        self.network.train(True)
        ones = torch.ones(len(discounted_next_values))
        writer.add_graph(self.network, X_inputs)
        count = 0
        # start with the training
        for i in range(self.nb_iters):
            for batch in tdata.BatchSampler(
                    tdata.RandomSampler(
                        range(len(X_inputs)), replacement=False),
                    batch_size=self.batch_size, drop_last=False):
                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    outputs = self.network(X_inputs[batch])
                    values = (immediate_exercise_value[batch] * outputs +
                              discounted_next_values[batch] * (ones[batch] - outputs))
                    loss = self._Loss(values)
                    loss.backward()
                    optimizer.step()
                
                count += 1
                if count % 10 == 0:
                    # add information to tensorboard file
                    writer.add_scalar("Loss/train", loss, count)
                    for name, layer in self.network.named_modules():
                        if isinstance(layer, torch.nn.Linear):
                            writer.add_histogram(f"{name}.weight", rgetattr(
                                self.network, f"{name}.weight"), count)
                            writer.add_histogram(f"{name}.weight.grad", rgetattr(
                                self.network, f"{name}.weight.grad"), count)
                            writer.add_histogram(f"{name}.bias", rgetattr(
                                self.network, f"{name}.bias"), count)
                            writer.add_histogram(f"{name}.bias.grad", rgetattr(
                                self.network, f"{name}.bias.grad"), count)

            # save the information of the model
            fpath = os.path.join(os.path.dirname(
                __file__), f"../output/{self.storage_loc}/phase_{step}")
            os.makedirs(fpath, exist_ok=True)
            tmp_path = fpath + f"/model_epoch_{i}.pt"
            torch.save({
                'epoch': i,
                'model_state_dict': self.network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, tmp_path)

    def log_train_network(self, stock_values, immediate_exercise_value,
                          discounted_next_values):
        optimizer = optim.Adam(self.network.parameters())
        discounted_next_values = torch.from_numpy(
            discounted_next_values).double()
        immediate_exercise_value = torch.from_numpy(
            immediate_exercise_value).double()
        X_inputs = torch.from_numpy(stock_values).double()

        self.network.train(True)
        ones = torch.ones(len(discounted_next_values))
        for _ in range(self.nb_iters):
            for batch in tdata.BatchSampler(
                    tdata.RandomSampler(
                        range(len(X_inputs)), replacement=False),
                    batch_size=self.batch_size, drop_last=False):

                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    outputs = self.network(X_inputs[batch]).reshape(-1)
                    values = (immediate_exercise_value[batch].reshape(-1) * torch.log(outputs+torch.finfo(torch.float32).eps) +
                              discounted_next_values[batch] * torch.log((ones[batch] - outputs)+torch.finfo(torch.float32).eps))
                    loss = self._Loss(values)
                    loss.backward()
                    optimizer.step()
    # some other optimization algorithm, based on tetermination criteria

    def train_network_abbruch(self, stock_values, immediate_exercise_value, discounted_next_values):
        eps = torch.tensor(self.eps, dtype=torch.float64)
        optimizer = optim.Adam(self.network.parameters())
        discounted_next_values = torch.from_numpy(
            discounted_next_values).double()
        immediate_exercise_value = torch.from_numpy(
            immediate_exercise_value).double()
        inputs = stock_values
        X_inputs = torch.from_numpy(inputs).double()
        self.network.train(True)
        ones = torch.ones(len(discounted_next_values))
        T = True
        while T:
            for batch in tdata.BatchSampler(tdata.RandomSampler(range(len(X_inputs)), replacement=False), batch_size=self.batch_size, drop_last=False):
                OSV_old = torch.tensor(-1000., dtype=torch.float64)
                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    outputs = self.network(X_inputs[batch]).reshape(-1)
                    values = (immediate_exercise_value[batch].reshape(
                        -1) * outputs + discounted_next_values[batch] * (ones[batch] - outputs))
                    loss = self._Loss(values)
                    loss.backward()
                    optimizer.step()
                    if torch.abs(loss-OSV_old) < eps:
                        T = False
                        break
                    else:
                        OSV_old = loss

    def log_train_network_abbruch(self, stock_values, immediate_exercise_value, discounted_next_values):
        eps = torch.tensor(self.eps, dtype=torch.float64)
        optimizer = optim.Adam(self.network.parameters())
        discounted_next_values = torch.from_numpy(
            discounted_next_values).double()
        immediate_exercise_value = torch.from_numpy(
            immediate_exercise_value).double()
        inputs = stock_values
        X_inputs = torch.from_numpy(inputs).double()
        self.network.train(True)
        ones = torch.ones(len(discounted_next_values))
        T = True
        while T:
            for batch in tdata.BatchSampler(tdata.RandomSampler(range(len(X_inputs)), replacement=False), batch_size=self.batch_size, drop_last=False):
                OSV_old = torch.tensor(-1000., dtype=torch.float64)
                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    outputs = self.network(X_inputs[batch]).reshape(-1)
                    values = (immediate_exercise_value[batch].reshape(
                        -1) * outputs + discounted_next_values[batch] * (ones[batch] - outputs))
                    loss = self._Loss(values)
                    loss.backward()
                    optimizer.step()
                    if torch.abs(loss-OSV_old) < eps:
                        T = False
                        break
                    else:
                        OSV_old = loss

    def evaluate_network(self, X_inputs):
        self.network.train(False)
        X_inputs = torch.from_numpy(X_inputs).double()
        outputs = self.network(X_inputs)
        return outputs.view(len(X_inputs)).detach().numpy()
