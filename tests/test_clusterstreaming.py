import cluster_streaming

from clustering import ClusterStreamingClustering

import numpy as np


def test_clustering():
        voxels = np.load('subset.npy')

        clustering = ClusterStreamingClustering(max_dist_tof=110)
        result = clustering.perform(voxels[:,:4])

        assert voxels.all() == result.all()

        #fig = plt.figure()
        #ax = fig.add_subplot(111, projection='3d')

        #ax.scatter(result[:, 0], result[:, 1], result[:, 2], c=result[:, 4], cmap=cc.cm.glasbey)

        #plt.show()

        print(np.unique(result[:, 4]).shape[0])

'''
def test_centroiding():
    voxels = np.genfromtxt('simulated_dataset_03.csv', delimiter=',')
    voxels[:, 2] = voxels[:, 2] * 1000
    centroiding = PeersCentroiding(max_dist_tof=110)
    pipeline = Pipeline(pipeline_steps=[centroiding])
    result = pipeline.run(voxels)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(voxels[:, 0], voxels[:, 1], voxels[:, 2], c=voxels[:, 3], cmap='hot')
    ax.scatter(result[:, 0], result[:, 1], result[:, 2], color='blue', marker='|', s=1000)

    plt.show()
    print(result.shape)
'''