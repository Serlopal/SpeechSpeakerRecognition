import proto2
from tools2 import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import log_multivariate_normal_density

plt.rcParams['image.cmap'] = 'jet'



tidigits = np.load('lab2_tidigits.npz', encoding='latin1')['tidigits']
models = np.load('lab2_models.npz', encoding='latin1')['models']

example = np.load('lab2_example.npz', encoding='latin1')['example'].item()
# plt.pcolormesh(example['hmm_obsloglik'].T)
# plt.show()

# (( 4 ))

# example using hmm: probabilities of gaussianans in mixture given samples
hmm_obsloglik = log_multivariate_normal_density(example['mfcc'], models[0]['hmm']['means'],models[0]['hmm']['covars'])
print(np.sum(hmm_obsloglik - example['hmm_obsloglik']))
# example using gmm: probabilities of gaussianans in mixture given samples
gmm_obsloglik = log_multivariate_normal_density(example['mfcc'], models[0]['gmm']['means'],models[0]['gmm']['covars'])
print(np.sum(gmm_obsloglik - example['gmm_obsloglik']))


# print(tidigits[0].keys())
# print(tidigits[10]['digit'])
# print(models[5]['digit'])
# Utterances and model corresponding to digit 'Four' : probabilities of gaussianans in mixture given samples
# hmm_obsloglik4 = log_multivariate_normal_density(tidigits[10]['mfcc'], models[5]['hmm']['means'],models[5]['hmm']['covars'])
# gmm_obsloglik4 = log_multivariate_normal_density(tidigits[10]['mfcc'], models[5]['gmm']['means'],models[5]['gmm']['covars'])
#
# print(hmm_obsloglik4.shape)
# print(models[5]['hmm']['transmat'].shape)

# plt.pcolormesh(hmm_obsloglik4.T)
# plt.xlim(0, hmm_obsloglik4.shape[0])
# plt.ylim(0, hmm_obsloglik4.shape[1])
# plt.title('Log Lik. HMM, Digit \'Four\'')
# plt.show()
# plt.pcolormesh(gmm_obsloglik4.T)
# plt.xlim(0, gmm_obsloglik4.shape[0])
# plt.ylim(0, gmm_obsloglik4.shape[1])
# plt.title('Log Lik. GMM, Digit \'Four\'')
# plt.show()

# (( 4 END ))

# (( 5 END))
gmm_loglik = proto2.gmmloglik(gmm_obsloglik, models[0]['gmm']['weights'])
# print(example['gmm_loglik'] - gmm_loglik)

gmm_global_loglik = np.zeros([len(models), len(tidigits)])
counter = 0
for j, utterance in enumerate(tidigits):
    for i, model in enumerate(models):
        gmm_obsloglik_aux = log_multivariate_normal_density(utterance['mfcc'], model['gmm']['means'], model['gmm']['covars'])
        gmm_global_loglik[i, j] = proto2.gmmloglik(gmm_obsloglik_aux, model['gmm']['weights'])
    # calculation of correct guesses
    model_likelihoods = gmm_global_loglik[:,j]
    winner = np.argmax(model_likelihoods)
    if models[winner]['digit'] == utterance['digit']:
        counter = counter + 1

print (counter*100 / (len(tidigits)),'% correctly guessed utterances')

#normalization
column_totals = np.sum(gmm_global_loglik,0)
gmm_global_loglik = -1*(gmm_global_loglik/column_totals)


# print(gmm_global_loglik)
# plt.pcolormesh(gmm_global_loglik)
# plt.title('Total normalized GMM-log-likelihoods for each pair utterance-model')
# plt.ylim(0, gmm_global_loglik.shape[0])
# plt.xlim(0, gmm_global_loglik.shape[1])
# plt.xlabel('Utterances')
# plt.ylabel('Models')
# plt.show()

# (( 5 END ))
# (( 6 BEGIN ))


# Computing forward for example
hmm_logalpha = proto2.forward(hmm_obsloglik, np.log(models[0]['hmm']['startprob']), np.log(models[0]['hmm']['transmat']))
print(hmm_logalpha)
print(example['hmm_logalpha'])

print(hmm_logalpha - example['hmm_logalpha'])

# Computing probability of whole sequence for example (Using HMM)
hmm_loglik = logsumexp(hmm_logalpha[-1,:])
print(hmm_loglik)
print(example['hmm_loglik'])

print(hmm_loglik - example['hmm_loglik'])

# Computing probability of whole sequence for example for each utterance-model pair (using HMM)


hmm_global_loglik = np.zeros([len(models), len(tidigits)])
counter = 0
for j, utterance in enumerate(tidigits):
    for i, model in enumerate(models):
        hmm_obsloglik_aux = log_multivariate_normal_density(utterance['mfcc'], model['hmm']['means'], model['hmm']['covars'])
        alpha_lattice = proto2.forward(hmm_obsloglik_aux, np.log(model['hmm']['startprob']), np.log(model['hmm']['transmat']))
        hmm_global_loglik[i, j] = logsumexp(alpha_lattice[-1,:])
    # calculation of correct guesses
    model_likelihoods = hmm_global_loglik[:,j]
    winner = np.argmax(model_likelihoods)
    if models[winner]['digit'] == utterance['digit']:
        counter = counter + 1
    else:
        print('The utterance', j, 'that corresponds to the digit', utterance['digit'], 'is mistaken by the digit',
              models[winner]['digit'], '(model', winner, ')')

# print (counter*100 / (len(tidigits)),'% correctly guessed utterances')

#normalization (for suitable printing)
column_totals = np.sum(hmm_global_loglik,0)
hmm_global_loglik = -1*(hmm_global_loglik/column_totals)

# print(hmm_global_loglik)
# plt.pcolormesh(hmm_global_loglik)
# plt.title('Total normalized HHM-log-likelihoods for each pair utterance-model')
# plt.ylim(0, hmm_global_loglik.shape[0])
# plt.xlim(0, hmm_global_loglik.shape[1])
# plt.xlabel('Utterances')
# plt.ylabel('Models')
# plt.show()

# alpha plotting
# alpha_lattice[alpha_lattice == -np.inf] = np.min(alpha_lattice[-1,:])
# alpha_lattice = alpha_lattice.T
# plt.pcolormesh(alpha_lattice)
# plt.title('alpha matrix')
# plt.ylim(0, alpha_lattice.shape[0])
# plt.xlim(0, alpha_lattice.shape[1])
# plt.xlabel('time')
# plt.ylabel('states')
# plt.show()


# GMM using HMM gaussians
ghmm_global_loglik = np.zeros([len(models), len(tidigits)])
counter = 0
for j, utterance in enumerate(tidigits):
    for i, model in enumerate(models):
        ghmm_obsloglik_aux = log_multivariate_normal_density(utterance['mfcc'], model['hmm']['means'], model['hmm']['covars'])
        ghmm_global_loglik[i, j] = proto2.gmmloglik(ghmm_obsloglik_aux, np.ones([1,len( model['hmm']['means'])])/len(model['hmm']['means']))
    # calculation of correct guesses
    model_likelihoods_ghmm = ghmm_global_loglik[:,j]
    winner = np.argmax(model_likelihoods_ghmm)
    if models[winner]['digit'] == utterance['digit']:
        counter = counter + 1

# print (counter*100 / (len(tidigits)),'% correctly guessed utterances using hmm gaussians for gmm')

#normalization
column_totals = np.sum(ghmm_global_loglik,0)
ghmm_global_loglik = -1*(ghmm_global_loglik/column_totals)


# print using hmm gaussians for gmm
# plt.pcolormesh(ghmm_global_loglik)
# plt.title('Total normalized GMM-log-likelihoods for each pair utterance-model using HMM gaussians')
# plt.ylim(0, ghmm_global_loglik.shape[0])
# plt.xlim(0, ghmm_global_loglik.shape[1])
# plt.xlabel('Utterances')
# plt.ylabel('Models')
# plt.show()


# print('digit of model is ', models[-1]['digit'])
# print('digit of utterance is ', tidigits[-1]['digit'])


# (( 6.2 VITERBI ))

# print(example['hmm_vloglik'])
[a,b] = proto2.viterbi(hmm_obsloglik, np.log(models[0]['hmm']['startprob']), np.log(models[0]['hmm']['transmat']))

# print(len(example['hmm_vloglik'][1]))
# print(len(b))

# Viterbi with GMM
hmm_global_vloglik = np.zeros([len(models), len(tidigits)])
counter = 0
for j, utterance in enumerate(tidigits):
    for i, model in enumerate(models):
        hmm_vloglik_aux = log_multivariate_normal_density(utterance['mfcc'], model['hmm']['means'], model['hmm']['covars'])
        (hmm_global_vloglik[i, j], viterbi_path) = proto2.viterbi(hmm_vloglik_aux, np.log(model['hmm']['startprob']), np.log(model['hmm']['transmat']))
    # calculation of correct guesses
    model_likelihoods_vhmm = hmm_global_vloglik[:,j]
    winner = np.argmax(model_likelihoods_vhmm)
    if models[winner]['digit'] == utterance['digit']:
        counter = counter + 1
    else:
        print('The utterance', j, 'that corresponds to the digit', utterance['digit'], 'is mistaken by the digit',
              models[winner]['digit'], '(model', winner, ')')

print (counter*100 / (len(tidigits)),'% correctly guessed utterances using hmm viterbi maximum likelihood')

#normalization
column_totals = np.sum(hmm_global_vloglik,0)
hmm_global_vloglik = -1*(hmm_global_vloglik/column_totals)

# print viterbi using hmm
# plt.pcolormesh(hmm_global_vloglik)
# plt.title('Total normalized HMM-viterbi-log-likelihoods for each pair utterance-model')
# plt.ylim(0, hmm_global_vloglik.shape[0])
# plt.xlim(0, hmm_global_vloglik.shape[1])
# plt.xlabel('Utterances')
# plt.ylabel('Models')
# plt.show()

# alpha plotting with viterbi path
# alpha_lattice[alpha_lattice == -np.inf] = np.min(alpha_lattice[-1,:])
# alpha_lattice = alpha_lattice.T
# plt.pcolormesh(alpha_lattice)
# plt.plot(viterbi_path)
# plt.title('Alpha matrix with Viterbi path')
# plt.ylim(0, alpha_lattice.shape[0])
# plt.xlim(0, alpha_lattice.shape[1])
# plt.xlabel('time')
# plt.ylabel('states')
# plt.show()


# (( 6.3 OPTIONAL ))

# Beta (backward pass) computation
hmm_logbeta = proto2.backward(hmm_obsloglik, np.log(models[0]['hmm']['startprob']), np.log(models[0]['hmm']['transmat']))
print(hmm_logbeta)
print(example['hmm_logbeta'])

print(hmm_logbeta - example['hmm_logbeta'])