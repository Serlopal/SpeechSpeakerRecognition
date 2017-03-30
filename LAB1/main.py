import numpy as np
import proto

example = np.load('example_python3.npz')['example'].item()
tidigits = np.load('tidigits_python3.npz')['tidigits']
# print(example['samples'])
frames = proto.enframe(example['samples'], int(16000*0.02), int(16000*0.01))
