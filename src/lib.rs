use numpy::{PyArray2, PyReadonlyArray2, ToPyArray};
use pyo3::{pymodule, types::PyModule, PyResult, Python};


/// cluster streaming
/// find clusters in data stream from TimePix data
#[pymodule]
fn cluster_streaming(_py: Python<'_>, m: &PyModule) -> PyResult<()> {

    /// find clusters
    #[pyfn(m)]
    fn get_labels<'py>(py: Python<'py>, x: PyReadonlyArray2<'py, f64>) 
    -> PyResult<&'py PyArray2<f64>> {
        let data = x.as_array();

        let arr = PyArray2::<f64>::zeros(py, [3, 5], false); 
        Ok(arr)
    }
    Ok(())
}