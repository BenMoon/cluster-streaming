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


const XYDIM: usize = 256; // number of pixels of detector
static MAX_DIST_TOF:f64 = 50.;



/// cluster streaming
/// find clusters in data stream from TimePix data in single trigger frame
#[pymodule]
fn cluster_streaming(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    


    /// find clusters and return labels
    #[pyfn(m)]
    fn get_labels<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<&'py PyArray1<i32>> {

        let data = x.as_array();


        let mut labels = vec![-1i32; data.shape()[0]];
        let mut image = [[-1i32; XYDIM + 6]; XYDIM + 6];

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

            labels[i] = i as i32;
            prev = i;
            neighb_clusters.clear();
            neighb_clusters.push(i);

            image[x as usize + 3][y as usize + 3] = i as i32;

            for (j, pos) in DELTA_POS.iter().enumerate() {
                idx1 = (x + pos.0 + 3) as usize;
                idx2 = (y + pos.1 + 3) as usize;
                i2 = image[idx1][idx2];

                // image is inititialized with -1
                if image[idx1][idx2] >= 0 {
                    // Der Nachbarpunkt muss 0 bis < max_dist_tof Einheiten jünger sein und eine höhere Intensität aufweisen
                    if (tof - data[[i2 as usize, 2]]) < MAX_DIST_TOF && (tof - data[[i2 as usize, 2]]) >= 0. && (data[[i2 as usize, 3]] - tot) >= 0.0 {
                        // übernehmen, wenn bisheriger Vorgänger in der Liste später auftauchte
                        prev = min(prev, i2 as usize);
                    }
                    // Merken von Clusterzentren in der Nähe
                    if (labels[i2 as usize] == i2) && (tof - data[[i2 as usize, 2]] < MAX_DIST_TOF) {
                        neighb_clusters.push(i2 as usize);
                    }
                }

                // determine if group in DELTA_POS is iterated and next group is about to start
                if j == 3 || j == 7 || j == 11 || j == 19 || j == 23 || j == 27 || j == 35 {
                    if prev < i {
                        while labels[prev] != labels[labels[prev] as usize] {
                            labels[prev] = labels[labels[prev] as usize];
                        }
                        labels[i] = labels[prev];
                        break;
                    }
                }

            }
            if prev == i {
                let mn = neighb_clusters
                    .iter()
                    .min().unwrap();
                for a in &neighb_clusters {
                    labels[*a] = *mn as i32;
                }
            }


        }

        //let arr = PyArray2::<f64>::zeros(py, [3, 5], false);
        Ok(labels.iter().map(|x| x+1).collect::<Vec<i32>>().to_pyarray(py))
    }

    Ok(())
}
