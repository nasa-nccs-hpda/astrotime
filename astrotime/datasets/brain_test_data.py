import numpy as np
import matplotlib.pyplot as plt

data=np.load('jordan_data.npz',allow_pickle=True)
signals = data['signals']
times = data['times']
binary_times = data['binary_times']
index=1
X = binary_times[index].astype(np.float32)
T = times[index]
Y = signals[index]
validation_split = int(0.8*X.shape[0])

plt.figure(figsize=(15,5))
plt.plot(T,Y,label='truth')

plt.savefig('timepred.png')
plt.close()