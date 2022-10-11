# Quick test to compare rust with matlab results

import cluster_streaming

import numpy as np
import scipy.io

labels_matlab = scipy.io.matlab.loadmat('tests/matlab_labels.mat')['labels'].flatten()
#voxels = np.load('tests/subset.npy')
voxels = np.load('tests/simulated_dataset_03.npy')

labels = cluster_streaming.get_labels(voxels[:, :4])
diff_labels = labels - labels_matlab
print("min", diff_labels.min())
print("argmin", diff_labels.argmin())
print("argmax", diff_labels.argmax())
print("sum", (diff_labels>0).sum())
print(np.where(diff_labels != 0)[0])
#print(np.int_(voxels[:20,4]))
print(labels[:140])