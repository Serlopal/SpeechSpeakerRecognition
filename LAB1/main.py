import proto
import tools
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.cluster.hierarchy import linkage, dendrogram


example = np.load('example_python3.npz')['example'].item()
tidigits = np.load('tidigits_python3.npz')['tidigits']
# print(example['samples'])
# frames = proto.enframe(example['samples'], int(20000*0.02), int(20000*0.01))
# preemph = proto.preemp(frames)
# windowed = proto.windowing(preemph)
# spec = proto.powerSpectrum(windowed, 512)
# mspec = proto.logMelSpectrum(spec, 20000)
# mfcc = proto.cepstrum(mspec,13)
# lmfcc = tools.lifter(mfcc)
#
# lmfcc = proto.mfcc(example['samples'])
# print(lmfcc.shape)
# matrix_tidigits = proto.correlation_mfcc(tidigits,40)
# correlation_matrix = np.corrcoef(matrix_tidigits)
#
# plt.pcolormesh(correlation_matrix)
#plt.show()

x=proto.mfcc(tidigits[0]['samples'])
y=proto.mfcc(tidigits[2]['samples'])
# result1 = (proto.dtw(x, y, distance.euclidean))
# np.save('result1',result1)

# glob_mat = proto.distances_global(tidigits)
# np.save('matrix_D', glob_mat)

D = np.load('matrix_D.npy')
plt.pcolormesh(D)
plt.show()

dendrogram(linkage(D, method='complete'))

# plt.pcolormesh(glob_mat)
# plt.show()

# lmfcc_tidigits = proto.mfcc(tidigits[3]['samples'])
# print(tidigits[3]['digit'])
#
# plt.pcolormesh(lmfcc_tidigits)
# plt.show()
#
# plt.pcolormesh(lmfcc)
# plt.show()


