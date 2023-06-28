/// # Database entry  
/// This stores a database entry, including the expected output of the evaluation,  
/// and the given inputs  
#[derive(Clone, Copy, Debug)]
pub struct Entry<const N: usize> {
    /// The inputs to the perceptron  
    pub inputs: [f64; N],
    /// The expected output from the perceptron (usually translated to a WDL score, see examples)  
    pub expected: f64,
}

/// # Database trait  
/// This is wrapped around a database, which the data loader will pull from.  
pub trait Database<const N: usize> {
    /// Returns the number of database entries, which is basically the size of the database.  
    fn size(&self) -> usize;
    /// Get an entry from the database given the index. This should be deterministic.  
    fn get(&self, idx: usize) -> Entry<N>;
}
