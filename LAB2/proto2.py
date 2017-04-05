import numpy as np
from tools2 import *

def gmmloglik(log_emlik, weights):
    """Log Likelihood for a GMM model based on Multivariate Normal Distribution.

    Args:
        log_emlik: array like, shape (N, K).
            contains the log likelihoods for each of N observations and
            each of K distributions
        weights:   weight vector for the K components in the mixture

    Output:
        gmmloglik: scalar, log likelihood of data given the GMM model.
    """
    weights_mat = np.tile(np.log(weights), (log_emlik.shape[0],1))
    return np.sum(logsumexp(weights_mat + log_emlik, axis=1))

def forward(log_emlik, log_startprob, log_transmat):
    """Forward probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: log transition probability from state i to j

    Output:
        forward_prob: NxM array of forward log probabilities for each of the M states in the model
    """
    forward_prob = np.zeros(log_emlik.shape)
    forward_prob[0,:] = log_startprob + log_emlik[0,:]  # Fill first row, initial emissions
    for n in range(1,log_emlik.shape[0]):
        aux_previous_prob = np.tile(forward_prob[n-1,:], (log_emlik.shape[1], 1))  # Clone row from previous state into matrix
        forward_prob[n, :] = logsumexp(aux_previous_prob + log_transmat.T, axis=1) + log_emlik[n,:]  # Update
    return forward_prob


def backward(log_emlik, log_startprob, log_transmat):
    """Backward probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j

    Output:
        backward_prob: NxM array of backward log probabilities for each of the M states in the model
    """

def viterbi(log_emlik, log_startprob, log_transmat):
    """Viterbi path.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j

    Output:
        viterbi_loglik: log likelihood of the best path
        viterbi_path: best path
    """
    [N,M] = log_emlik.shape

    logV = np.zeros(log_emlik.shape)
    B = np.zeros(logV.shape)
    logV[0,:] = log_startprob + log_emlik[0,:]
    for n in range(1,N):
        for j in range(M):
            logV[n,j] = np.max(logV[n-1,:] + log_transmat[:,j]) + log_emlik[n,j]
            B[n, j] = np.argmax(logV[n - 1, :] + log_transmat[:, j])

    return [logV[-1,-1], B[:,-1]]

