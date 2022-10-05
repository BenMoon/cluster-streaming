use std::{cmp::min_by, cmp::min};

use numpy::{PyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::{pymodule, types::PyModule, PyResult, Python};

struct DeltaPos(
    [(i32, i32); 4],
    [(i32, i32); 4],
    [(i32, i32); 4],
    [(i32, i32); 8],
    [(i32, i32); 4],
    [(i32, i32); 4],
    [(i32, i32); 8],
    [(i32, i32); 8],
);

static DELTA_POS: [(i32, i32); 44] = 
    [(-1, 0), (1, 0), (0, -1), (0, 1),  // 0..3 group 1
     (-1, -1), (1, -1), (-1, 1), (1, 1), // 4..7 group 2
     (-2, 0), (2, 0), (0, -2), (0, 2),   // 8..11
     (-2, 1), (-1, 2), (1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), // 12..19
     (-2, 2), (2, 2), (2, -2), (-2, -2), // 20..23
     (-3, 0), (0, 3), (3, 0), (0, -3),   // 24..27
     (-3, 1), (-1, 3), (1, 3), (3, 1), (3, -1), (1, -3), (-1, -3), (-3, -1), // 28..35
     (-3, -2), (3, -2), (-3, 2), (3, 2), (-2, -3), (2, -3), (-2, 3), (2, 3)] // 36..43
    ; // Gruppen von Nachbarschaften für Abstand <4       ;


const DIM: usize = 256; // number of pixels of detector
static MAX_DIST_TOF:f64 = 0.11;



/// cluster streaming
/// find clusters in data stream from TimePix data in single trigger frame
#[pymodule]
fn cluster_streaming(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    


    /// find clusters and return labels
    #[pyfn(m)]
    fn get_labels<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<&'py PyArray1<usize>> {

        let data = x.as_array();
        let mut labels = vec![0usize; data.shape()[0]];

        // Auf allen vier Seiten wird ein Rand von 3 hinzugefügt, damit Nachbarn immer abfragbar sind
        //image = np.full((self.dim + 6, self.dim + 6), -1);
        let mut image = [[-1; DIM + 6]; DIM + 6];
        

        /*
        Hauptalgorithmus:
    
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
        Pixel), werden diese zu einem Clusterzentrum zusammengefasst.
         */
        let mut x:i32;
        let mut y:i32;
        let mut tof:f64;
        let mut tot:f64;
        let mut prev:usize;
        let mut i2:i32;
        let mut idx1:usize;
        let mut idx2:usize;
        let mut neighb_clusters = vec![1usize; 128];

        for i in 0..data.shape()[0] {
            x = data[[i, 0]] as i32;
            y = data[[i, 1]] as i32;
            tof = data[[i, 2]];
            tot = data[[i, 3]];

            labels[i] = i+1;
            prev = i;
            neighb_clusters.clear();
            neighb_clusters.push(i);

            image[x as usize + 3][y as usize + 3] = i as i32;
    
            for (j, pos) in DELTA_POS.iter().enumerate() {
                idx1 = (x + pos.0 + 3) as usize;
                idx2 = (y + pos.1 + 3) as usize;
                i2 = image[idx1][idx2];

                if i2 >= 0 {
                    // Der Nachbarpunkt muss 0 bis < max_dist_tof Einheiten jünger sein und eine höhere Intensität aufweisen
                    if (tof - data[[i2 as usize, 2]]) < MAX_DIST_TOF && (data[[i2 as usize, 3]] - tot) >= 0.0 {
                        prev = min(prev, i2 as usize);
                    }
                    // Merken von Clusterzentren in der Nähe
                    if (labels[i2 as usize] == i2 as usize) && (tof - data[[i2 as usize, 2]] < MAX_DIST_TOF) {  
                        neighb_clusters.push(i2 as usize);
                    }
                }

                // determine if group in DELTA_POS is iterated and next group is about to start
                if j == 3 || j == 7 || j == 11 || j == 19 || j == 23 || j == 27 || j == 35 {
                    if prev < i {
                        while labels[prev] != labels[labels[prev]-1] {
                            labels[prev] = labels[labels[prev]-1];
                        }
                        labels[i] = labels[prev];
                        break;
                    }
                }

            }
        
        }


        //let arr = PyArray2::<f64>::zeros(py, [3, 5], false);
        Ok(labels.to_pyarray(py))
    }
    Ok(())
}
