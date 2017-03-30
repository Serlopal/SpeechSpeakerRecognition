import numpy as np
import proto
import matplotlib.pyplot as plt

example = np.load('example_python3.npz')['example'].item()
tidigits = np.load('tidigits_python3.npz')['tidigits']
# print(example['samples'])
frames = proto.enframe(example['samples'], int(20000*0.02), int(20000*0.01))
preemph = proto.preemp(frames)

# print(preemph-example['preemph'])

plt.pcolormesh(preemph)
plt.show()


