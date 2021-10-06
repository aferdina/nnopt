#Calculating True Q-Values
import numpy as np

def Q_values(N, P, S):
    """Calculating correct Q values given number of timesteps and a transition matrix

    Args:
        N ([int]): nummber of time steps in the game
        P ([np.matrix]): probabilities for transition
        S ([np.matrix]): state space

    Returns:
        [type]: np.array with q value
    """
    result = np.asmatrix(S)
    for i in range(N-1):
        Q = np.transpose(np.matmul(P,np.transpose(result[-1,:])))
        result = np.concatenate((result, np.maximum(result[-1,:],Q)),axis = 0)
    return result

#Getting True stopping times
def getstoppingtimes(qvalues, N, S):
    """getting correct stopping values given Q-values, timesteps and statespace

    Args:
        qvalues ([np.array]): given q values
        N ([int]): number of time steps
        S ([np.array]): state space

    Returns:
        [np.array]: array with true and false whether stopp or not stopping
    """
    result = (qvalues <= np.array([S]*(N),dtype=np.float32))
    result = result.astype("float32")
    return result

N = 30
S = np.array([1,2,3,4,5,6])
P = np.array([[0.167,0.166,0.167,0.166,0.167,0.167] for _ in range(6)])

qvalues = Q_values(N,P,S)
#print(qvalues)
stoppingtimes = getstoppingtimes(qvalues,N,S)
#print(stoppingtimes)
mistakes = 0
mistakes2 = 0 
#print(stoppingtimes[1,:])

