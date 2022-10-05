use std::{cmp::min};

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

static MAX_DIST_TOF:f64 = 50.;


fn get_labels() {
    let data = 
       [[109.        ,  33.        ,   2.42949637,   3.        ],
       [108.        ,  41.        ,   2.28945704, 261.        ],
       [108.        ,  45.        ,   2.39657285,  10.        ],
       [108.        ,  44.        ,   2.32681352,  32.        ],
       [108.        ,  43.        ,   2.30101608,  86.        ],
       [108.        ,  42.        ,   2.29273729, 179.        ],
       [109.        ,  32.        ,   2.43380449,   2.        ],
       [108.        ,  37.        ,   2.29244872, 185.        ],
       [108.        ,  39.        ,   2.28763282, 321.        ],
       [108.        ,  38.        ,   2.28836234, 298.        ],
       [109.        ,  34.        ,   2.35621329,  20.        ],
       [108.        ,  36.        ,   2.29977436,  93.        ],
       [108.        ,  35.        ,   2.32374397,  34.        ],
       [108.        ,  34.        ,   2.38751318,  12.        ],
       [108.        ,  33.        ,   2.42949637,   3.        ],
       [108.        ,  32.        ,   2.43807537,   1.        ],
       [107.        ,  46.        ,   2.42483764,   4.        ],
       [109.        ,  35.        ,   2.30935087,  52.        ],
       [109.        ,  37.        ,   2.28899671, 278.        ],
       [110.        ,  34.        ,   2.35621329,  20.        ]];
    let mut labels = vec![0i32; data.len()];
    
    let mut image = [[-1i32; 256 + 6]; 256 + 6];

    let mut x:i32;
    let mut y:i32;
    let mut tof:f64;
    let mut tot:f64;
    let mut prev:usize;
    let mut i2:i32;
    let mut idx1:usize;
    let mut idx2:usize;
    let mut neighb_clusters = vec![1usize; 128];

    for i in 0..data.len() {
        x = data[i][0] as i32;
        y = data[i][1] as i32;
        tof = data[i][2];
        tot = data[i][3];

        labels[i] = i as i32+1;
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
                if (tof - data[i2 as usize][2]) < MAX_DIST_TOF && (tof - data[i2 as usize][2]) >= 0. && (data[i2 as usize][3] - tot) >= 0.0 {
                    // übernehmen, wenn bisheriger Vorgänger in der Liste später auftauchte
                    prev = min(prev, i2 as usize);
                }
                // Merken von Clusterzentren in der Nähe
                if (labels[i2 as usize] == i2) && (tof - data[i2 as usize][2] < MAX_DIST_TOF) {
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
    println!("{:?}", labels.iter().map(|x| x+1).collect::<Vec<i32>>());


}

fn main() {
    get_labels();
}
