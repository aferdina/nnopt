""" Computes the American option price using the deep optimal stopping (DOS).

Using Variance reducing methods from MPD theory
"""
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
import configs.corr_stopp as corr_stopp
import backward_induction
import utils.networks as networks
import traceback
import sys


def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.manual_seed(42)
        # torch.nn.init.zeros_(m.weight)
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class DeepOptimalStopping(backward_induction.AmericanOptionPricer):
    """Computes the American option price using the deep optimal stopping (DOS)
    """

    def __init__(self, model, payoff, nb_epochs=20, nb_batches=None,
                 hidden_size=10, use_path=False, eps=0.001, copy=True, values=[1, 2, 3, 4, 5, 6], storage_loc="neural_networks_4", start_const=False):
        del nb_batches
        super().__init__(model, payoff, use_path=use_path,
                         copy=copy, values=values, storage_loc=storage_loc)
        self.hidden_size = hidden_size
        self.eps = eps
        self.storage_loc = storage_loc
        self.nb_epochs = nb_epochs
        self.start_const = start_const
        if self.use_path:
            self.state_size = model.nb_stocks * (model.nb_dates+1)
        else:
            self.state_size = model.nb_stocks
        if self.copy:
            self.neural_stopping = OptimalStoppingOptimization(
                self.state_size, self.model.nb_paths, hidden_size=self.hidden_size,
                nb_iters=self.nb_epochs, eps=0.001, storage_loc=self.storage_loc, start_const=self.start_const)

    def stop(self, step, stock_values, immediate_exercise_values,
             discounted_next_values, copy=True, h=None, new_init=False):
        """ see base class """
        if not copy:
            self.neural_stopping = OptimalStoppingOptimization(
                self.state_size, self.model.nb_paths, hidden_size=self.hidden_size,
                nb_iters=self.nb_epochs, eps=0.001, storage_loc=self.storage_loc, start_const=self.start_const)
        if self.use_path:
            # shape [paths, stocks, dates up to now]
            stock_values = np.flip(stock_values, axis=2)
            # add zeros to get shape [paths, stocks, dates+1]
            stock_values = np.concatenate(
                [stock_values, np.zeros(
                    (stock_values.shape[0], stock_values.shape[1],
                     self.model.nb_dates + 1 - stock_values.shape[2]))], axis=-1)
            stock_values = stock_values.reshape((stock_values.shape[0], -1))

        self.neural_stopping.train_network(step,
                                           stock_values[:self.split],
                                           immediate_exercise_values[
                                               :self.split],
                                           discounted_next_values[:self.split])
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
                 batch_size=2000, eps=0.001, storage_loc="neural_networks_4", start_const=True):
        self.eps = eps
        self.nb_stocks = nb_stocks
        self.nb_paths = nb_paths
        self.nb_iters = nb_iters
        self.storage_loc = storage_loc
        self.batch_size = batch_size
        self.network = networks.NetworksoftlogDOS(
            self.nb_stocks, hidden_size=hidden_size).double()
        if not start_const:
            self.network.apply(init_weights)
        else:
            fpath = f"./output/weights_log_const/model_epoch_final.pt"
            checkpoint = torch.load(fpath)
            self.network.load_state_dict(checkpoint['model_state_dict'])

    def _Loss(self, X):
        return -torch.mean(X)*2

    def train_network(self, step, stock_values, immediate_exercise_value,
                      discounted_next_values):
        # initialize optimizer for the game
        optimizer = optim.Adam(self.network.parameters())
        # create matrix to use pointwise multiplication to calculate the loss
        mat = np.concatenate((immediate_exercise_value.reshape(len(immediate_exercise_value), 1),
                             discounted_next_values.reshape(len(discounted_next_values), 1)), axis=1)
        mat = torch.from_numpy(mat).double()
        # cast input of neural network
        X_inputs = torch.from_numpy(stock_values).double()
        self.network.train(True)
        for i in range(self.nb_iters):
            for batch in tdata.BatchSampler(
                    tdata.RandomSampler(
                        range(len(X_inputs)), replacement=False),
                    batch_size=self.batch_size, drop_last=False):
                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    outputs = self.network(X_inputs[batch])
                    values = torch.mul(mat[batch, :], outputs)
                    loss = self._Loss(values)
                    loss.backward()
                    optimizer.step()
            fpath = os.path.join(os.path.dirname(
                __file__), f"./output/{self.storage_loc}/phase_{step}")
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
        outputs = self.network(X_inputs)[:, 0]
        return np.exp(outputs.view(len(X_inputs)).detach().numpy())
