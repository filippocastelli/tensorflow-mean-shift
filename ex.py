from pathlib import Path
import numpy as np
import matplotlib.pylab as plt

from meanshift_tf2 import MeanShiftTF, MeanShiftTF2

def get_cmap(n, name="hsv"):
    return plt.cm.get_cmap(name, n)

n_points = 100000
n_features = 2
mean_shift_tf = MeanShiftTF()

data = np.random.random((n_points, n_features)).astype(np.float32)
mean_shift_tf2 = MeanShiftTF2(data=data, chunk_size=4, radius=0.1)

#plt.scatter(data[:,0], data[:,1])
peaks, labels = mean_shift_tf.apply(data, radius=0.4, chunk_size=4)
#peaks, labels = mean_shift_tf2.run()
print(peaks) 
print(labels)
unique_labels = np.unique(labels)
plt.figure()
cmap = get_cmap(len(peaks))
for label in unique_labels:
    # plot the cluster peak
    plt.scatter(peaks[label][0], peaks[label][1], c=cmap(label), marker='x')
    # plot the cluster members
    plt.scatter(data[labels == label][:,0], data[labels == label][:,1], c=cmap(label), marker='.')
plt.show()
print("ciaos")