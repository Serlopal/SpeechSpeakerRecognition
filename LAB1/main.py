import proto
import tools
import numpy as np
import matplotlib.pyplot as plt

example = np.load('example_python3.npz')['example'].item()
tidigits = np.load('tidigits_python3.npz')['tidigits']
# print(example['samples'])
frames = proto.enframe(example['samples'], int(20000*0.02), int(20000*0.01))
preemph = proto.preemp(frames)
windowed = proto.windowing(preemph)
spec = proto.powerSpectrum(windowed, 512)
mspec = proto.logMelSpectrum(spec, 20000)
mfcc = proto.cepstrum(mspec,13)
lmfcc = tools.lifter(mfcc)

print(np.sum(lmfcc-example['lmfcc']))

plt.pcolormesh(lmfcc)
plt.show()


