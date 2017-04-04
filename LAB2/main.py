import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import log_multivariate_normal_density

tidigits = np.load('lab2_tidigits.npz', encoding='latin1')['tidigits']
models = np.load('lab2_models.npz', encoding='latin1')['models']

print(models[0].keys())
# for element in models:
#     print(element['digit'])
#     print(element['pron'])
#     print(element['hmm']['transmat'])
#     print(element['hmm']['startprob'])
#     print(element['gmm'])

example = np.load('lab2_example.npz',encoding='latin1')['example'].item()
# plt.pcolormesh(example['hmm_obsloglik'].T)
# plt.show()

hmm_obsloglik = log_multivariate_normal_density(example['mfcc'], models[0]['hmm']['means'],models[0]['hmm']['covars'])
print(np.sum(hmm_obsloglik - example['hmm_obsloglik']))
gmm_obsloglik = log_multivariate_normal_density(example['mfcc'], models[0]['gmm']['means'],models[0]['gmm']['covars'])
print(np.sum(gmm_obsloglik - example['gmm_obsloglik']))


# print(tidigits[0].keys())
print(tidigits[10]['digit'])
print(models[5]['digit'])
# Utterances and model corresponding to digit 'Four'
hmm_obsloglik4 = log_multivariate_normal_density(tidigits[10]['mfcc'], models[5]['hmm']['means'],models[5]['hmm']['covars'])
gmm_obsloglik4 = log_multivariate_normal_density(tidigits[10]['mfcc'], models[5]['gmm']['means'],models[5]['gmm']['covars'])

print(hmm_obsloglik4.shape)
print(models[5]['hmm']['transmat'].shape)

plt.pcolormesh(hmm_obsloglik4.T)
plt.xlim(0, hmm_obsloglik4.shape[0])
plt.ylim(0, hmm_obsloglik4.shape[1])
plt.title('Log Lik. HMM, Digit \'Four\'')
plt.show()
plt.pcolormesh(gmm_obsloglik4.T)
plt.xlim(0, gmm_obsloglik4.shape[0])
plt.ylim(0, gmm_obsloglik4.shape[1])
plt.title('Log Lik. GMM, Digit \'Four\'')
plt.show()

