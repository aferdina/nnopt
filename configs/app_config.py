import utils.networks as networks
import re
from os import listdir
from os.path import isfile, join
import itertools
import os 
#sys.path.insert(0, os.path.dirname("."))
# includes the configurations of the web application

# bool, if logmodel (logsoftmax) is used; these functions have two outputs (log(p),log(1-p)) and therefore needed a specific treatment
LOG_MODEL_TWO_OUT = True

# paths, where algorithms are stored
PATH_ONE = "neural_networks4_copy"
PATH_TWO = "neural_networks4_copy_const"

# network, of the algorithm
network = networks.NetworksoftlogDOS(nb_stocks=1, hidden_size=4)

# number of epochs, considered in the algorithm
NB_EPOCHS_START = 1
NB_EPOCHS = 9

epoch = NB_EPOCHS_START
step = 1


############################################################################################
#configs from weight_app


LOG_MODEL_TWO_OUT_WEIGHT = True

path = "./output/weights_log_const"
weight_files = [f for f in listdir(path) if isfile(join(path, f))]
r = re.compile("\d+")
list_of_numbers = list(map(r.findall, weight_files))
list_of_numbers_merged = list(itertools.chain(*list_of_numbers))
list_of_numbers_merged = list(map(int,list_of_numbers_merged))
list_of_numbers_merged.sort()

network_weight = networks.NetworksoftlogDOS(nb_stocks=1, hidden_size=4)

epoch_weight = 0