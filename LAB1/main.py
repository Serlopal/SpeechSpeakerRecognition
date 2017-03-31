import proto
import tools
import numpy as np
import matplotlib.pyplot as plt

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
matrix_tidigits = proto.correlation_mfcc(tidigits,40)
correlation_matrix = np.corrcoef(matrix_tidigits)

plt.pcolormesh(correlation_matrix)
plt.show()

# lmfcc_tidigits = proto.mfcc(tidigits[3]['samples'])
# print(tidigits[3]['digit'])
#
# plt.pcolormesh(lmfcc_tidigits)
# plt.show()
#
# plt.pcolormesh(lmfcc)
# plt.show()


