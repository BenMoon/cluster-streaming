import numpy as np

class PeersCentroiding():

    def __init__(self, dim=256, max_dist_tof=0.11, min_cluster_size=5, *args, **kwargs):
        self.dim = dim
        self.max_dist_tof = max_dist_tof
        self.min_cluster_size = min_cluster_size
        super(PeersCentroiding, self).__init__(*args, **kwargs)

    def perform(self, data):
        temp_data = np.full((data.shape[0], data.shape[1] + 1), -1.0)
        temp_data[:, :-1] = data
        data = temp_data
        
        image = np.full((self.dim+6, self.dim+6), -1) # Auf allen vier Seiten wird ein Rand von 3 hinzugefügt, damit Nachbarn immer abfragbar sind
        deltaPos = [
            [(-1, 0), (1, 0), (0, -1), (0, 1)],
            [(-1, -1), (1, -1), (-1, 1), (1, 1)],
            [(-2, 0), (2, 0), (0, -2), (0, 2)],
            [(-2, 1), (-1, 2), (1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1)],
            [(-2, 2), (2, 2), (2, -2), (-2, -2)],
            [(-3, 0), (0, 3), (3, 0), (0, -3)],
            [(-3, 1), (-1, 3), (1, 3), (3, 1), (3, -1), (1, -3), (-1, -3), (-3, -1)],
            [(-3, -2), (3, -2), (-3, 2), (3, 2), (-2, -3), (2, -3), (-2, 3), (2, 3)]
        ] # Gruppen von Nachbarschaften für Abstand <4
        
        """Hauptalgorithmus:
    
        Die Datenpunkte werden nacheinander mit nahen Nachbarpunkten verglichen
        (effizient über map abfragbar). Dabei wird ein hinreichend naher
        Nachbarpunkt (Abstand kleiner als 4 Pixel) als Vorgänger markiert, wenn
        er der nächstliegende Nachbarpunkt ist, der zeitlich vor dem aktuellen
        Punkt und mit höherer Intensität gemessen wurde. Wenn es mehrere
        derartige Nachbarpunkte gibt, wird der zeitlich früheste gewählt.
    
        Ein Punkt, zu dem es keinen derartigen Vorgänger gibt, wird als
        Clusterentrum markiert (es wird der Punkt selbst als eigener Vorgänger
        eingetragen).
    
        Liegen zwei Clusterzentren sehr dicht beieinander (Abstand kleiner als 4
        Pixel), werden diese zu einem Clusterzentrum zusammengefasst."""
    
        for i in range(0, data.shape[0]):
            data[i, 4] = i
            curr = data[i]
            x, y, tof, tot, _ = curr
            x, y = int(x), int(y)
            image[x + 3, y + 3] = i
            prev = i
            neighbClusters = [i]

            for grp in deltaPos:
                for pos in grp:
                    i2 = image[x + pos[0] + 3, y + pos[1] + 3]
                    if i2 >= 0:
                        nb = data[i2]
                        if (tof - nb[2] < self.max_dist_tof) and (tof - nb[2] >= 0) and (nb[3] - tot >= 0): # Der Nachbarpunkt muss 0 bis < max_dist_tof Einheiten jünger sein und eine höhere Intensität aufweisen
                            prev = min(prev, i2) # übernehmen, wenn bisheriger Vorgänger in der Liste später auftauchte
                        if (data[i2, 4] == i2) and (tof - nb[2] < self.max_dist_tof): # Merken von Clusterzentren in der Nähe
                            neighbClusters.append(i2)

                if prev < i:
                    while data[prev, 4] != data[int(data[prev, 4]), 4]:
                        data[prev, 4] = data[int(data[prev, 4]), 4]
                    data[i, 4] = data[prev, 4]
                    data[i, 4] = data[prev, 4]
                    break

        """Aufbereitung 1:
    
        Alle Clusterzentren (d.h. Datenpunkte, die auf sich selbst als Vorgänger
        verweisen) werden in eine separate Liste "cluster_centers" extrahiert."""
        # This solution is without a threshold in terms of the size of the cluster.
        # cluster_centers = data[np.unique(data[:, 4]).astype(int)]

        
        """Aufbereitung 2:
    
        Zu jedem Clusterzentrum wird ermittelt, wie viele Punkte das Cluster
        umfasst. Sind dies mindestens min_cluster_size viele, wird das Cluster
        einer neuen Liste "cleanedFinal" von hinreichend großen Clustern
        hinzugefügt."""
        un, counts = np.unique(data[:, 4], return_counts=True)
        labels_with_cluster_size = np.column_stack((un, counts))
        cluster_centers_with_min_size = data[labels_with_cluster_size[labels_with_cluster_size[:, 1] >=
                                                                      self.min_cluster_size][:, 0].astype(int)]

        return cluster_centers_with_min_size[:, :4]
