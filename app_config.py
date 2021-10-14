import networks

# includes the configurations of the web application

# bool, if logmodel (logsoftmax) is used; these functions have two outputs (log(p),log(1-p)) and therefore needed a specific treatment
LOG_MODEL_TWO_OUT = True

# paths, where algorithms are stored
PATH_ONE = "neural_networks4"
PATH_TWO = "neural_networks4_copy"

# network, of the algorithm
network = networks.NetworksoftlogDOS(nb_stocks=1, hidden_size=4)

# number of epochs, considered in the algorithm
NB_EPOCHS = 59
