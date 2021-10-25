import torch.nn as nn
import torch 
import sys 
import os
from utils.breludef import boundedrelu

class NetworkNLSM(nn.Module):
  def __init__(self, nb_stocks, hidden_size=10):
    super(NetworkNLSM, self).__init__()
    H = hidden_size
    self.layer1 = nn.Linear(nb_stocks+1, H)
    self.leakyReLU = nn.LeakyReLU(0.5)
    self.sigmoid = nn.Sigmoid()
    self.bn1 = nn.BatchNorm1d(num_features=H)
    self.layer2 = nn.Linear(H, H)
    self.bn2 = nn.BatchNorm1d(num_features=H)
    self.layer3 = nn.Linear(H, 1)
    self.bn3 = nn.BatchNorm1d(num_features=1)

  def forward(self, x):
    x = self.layer1(x)
    x = self.leakyReLU(x)
    x = self.layer3(x)
    return x

def brelu(input):
  """bounded Relu function

  Args:
      input (torch.tensor): input of activation function

  Returns:
      [torch.tensor]: bounded relu output
  """
  return torch.minimum(torch.relu(input),torch.tensor(1))

class UBRelu(nn.Module):

    def __init__(self):
        '''
        Init method.
        '''
        super().__init__()

    def forward(self, input):
          
      return brelu(input) 

class NetworkDOS(nn.Module):
  def __init__(self, nb_stocks, hidden_size=10):
    super(NetworkDOS, self).__init__()
    H = hidden_size
    self.bn0 = nn.BatchNorm1d(num_features=nb_stocks)
    self.layer1 = nn.Linear(nb_stocks, H)
    self.leakyReLU = nn.LeakyReLU(0.5)
    self.sigmoid = nn.Sigmoid()
    self.tanh = nn.Tanh()
    self.relu = nn.ReLU()
    self.bn1 = nn.BatchNorm1d(num_features=H)
    #self.layer2 = nn.Linear(H, H)
    self.bn2 = nn.BatchNorm1d(num_features=H)
    self.layer3 = nn.Linear(H, 1)
    self.bn3 = nn.BatchNorm1d(num_features=1)

  def forward(self, x):
    x = self.bn0(x)
    x = self.layer1(x)
    x = self.leakyReLU(x)
    x = self.bn2(x)
    x = self.layer3(x)
    x = self.sigmoid(x)
    return x

  
class NetworkeasyDOS(nn.Module):
  def __init__(self, nb_stocks, hidden_size=10):
    super(NetworkeasyDOS, self).__init__()
    H = hidden_size
    self.bn0 = nn.BatchNorm1d(num_features=nb_stocks)
    self.layer1 = nn.Linear(nb_stocks, H)
    self.leakyReLU = nn.LeakyReLU(0.5)
    self.sigmoid = nn.Sigmoid()
    self.tanh = nn.Tanh()
    self.relu = nn.ReLU()
    self.brelu = UBRelu()
    self.bn1 = nn.BatchNorm1d(num_features=H)
    self.layer2 = nn.Linear(H, H)
    self.bn2 = nn.BatchNorm1d(num_features=H)
    self.layer3 = nn.Linear(H, 1)
    self.bn3 = nn.BatchNorm1d(num_features=1)
    self.bbrelu = boundedrelu.apply

  def forward(self, x):
    x = self.bn0(x)
    x = self.layer1(x)
    x = self.relu(x)
    x = self.bn2(x)
    x = self.layer3(x)
    x = self.bbrelu(x)
    return x
  
  

class NetworklogDOS(nn.Module):
  def __init__(self, nb_stocks, hidden_size=10):
    super(NetworklogDOS, self).__init__()
    H = hidden_size
    self.bn0 = nn.BatchNorm1d(num_features=nb_stocks)
    self.layer1 = nn.Linear(nb_stocks, H)
    self.leakyReLU = nn.LeakyReLU(0.5)
    self.sigmoid = nn.Sigmoid()
    self.logsigmoid = nn.LogSigmoid()
    self.tanh = nn.Tanh()
    self.relu = nn.ReLU()
    self.bn1 = nn.BatchNorm1d(num_features=H)
    self.layer2 = nn.Linear(H, H)
    self.bn2 = nn.BatchNorm1d(num_features=H)
    self.layer3 = nn.Linear(H, 1)
    self.bn3 = nn.BatchNorm1d(num_features=1)

  def forward(self, x):
    x = self.bn0(x)
    x = self.layer1(x)
    x = self.leakyReLU(x)
    x = self.bn1(x)
    x = self.layer2(x)
    x = self.leakyReLU(x)
    x = self.bn2(x)
    x = self.layer3(x)
    x = self.logsigmoid(x)
    return x

class NetworksoftlogDOS(nn.Module):
  def __init__(self, nb_stocks, hidden_size=10):
    super(NetworksoftlogDOS, self).__init__()
    H = hidden_size
    self.bn0 = nn.BatchNorm1d(num_features=nb_stocks)
    self.layer1 = nn.Linear(nb_stocks, H)
    self.leakyReLU = nn.LeakyReLU(0.5)
    self.sigmoid = nn.Sigmoid()
    self.logsoft = nn.LogSoftmax(dim=1)
    self.tanh = nn.Tanh()
    self.relu = nn.ReLU()
    self.bn1 = nn.BatchNorm1d(num_features=H)
    #self.layer2 = nn.Linear(H, H)
    self.bn2 = nn.BatchNorm1d(num_features=H)
    self.layer3 = nn.Linear(H, 2)
    self.bn3 = nn.BatchNorm1d(num_features=2)

  def forward(self, x):
    x = self.bn0(x)
    x = self.layer1(x)
    x= self.leakyReLU(x)
    x = self.bn1(x)
    x = self.layer3(x)
    x = self.logsoft(x)
    return x


class NetworkDDOS(nn.Module):
  def __init__(self, nb_stocks, hidden_size=10, hidden_size2 = 10):
    super(NetworkDDOS, self).__init__()
    H = hidden_size
    H2 = hidden_size2
    self.bn0 = nn.BatchNorm1d(num_features=nb_stocks)
    self.layer1 = nn.Linear(nb_stocks, H)
    self.leakyReLU = nn.LeakyReLU(0.5)
    self.sigmoid = nn.Sigmoid()
    self.tanh = nn.Tanh()
    self.relu = nn.ReLU()
    self.bn1 = nn.BatchNorm1d(num_features=H)
    self.layer2 = nn.Linear(H, H)
    self.bn2 = nn.BatchNorm1d(num_features=H)
    self.layer4 = nn.Linear(H2, 1)
    self.bn4 = nn.BatchNorm1d(num_features=1)
    self.layer3 = nn.Linear(H,H2)
    self.bn3 = nn.BatchNorm1d(num_features=H2)


  def forward(self, x):
    x = self.bn0(x)
    x = self.layer1(x)
    x = self.leakyReLU(x)
    x = self.bn2(x)
    x= self.layer3(x)
    x = self.leakyReLU(x)
    x = self.bn3(x)
    x = self.layer4(x)
    x = self.sigmoid(x)
    return x
