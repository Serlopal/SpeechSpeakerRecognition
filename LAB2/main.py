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
print(example['gmm_loglik'] - gmm_loglik)

gmm_global_loglik = np.zeros([len(models), len(tidigits)])
counter = 0
for j, utterance in enumerate(tidigits):
    for i, model in enumerate(models):
        gmm_obsloglik_aux = log_multivariate_normal_density(utterance['mfcc'], model['gmm']['means'], model['gmm']['covars'])
        gmm_global_loglik[i, j] = proto2.gmmloglik(gmm_obsloglik_aux, model['gmm']['weights'])
    model_likelihoods = gmm_global_loglik[:,j]
    winner = np.argmax(model_likelihoods)
    if models[winner]['digit'] == utterance['digit']:
        counter = counter +1

print (counter / (gmm_global_loglik.shape[1]) , ' %')

#normalization
column_totals = np.sum(gmm_global_loglik,0)
gmm_global_loglik = -1*(gmm_global_loglik/column_totals)

#calculation of correct guesses





#print(gmm_global_loglik)
plt.pcolormesh(gmm_global_loglik)
plt.show()





