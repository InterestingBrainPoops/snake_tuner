/// # Entry trait
/// Implement this on your data base entries.
pub trait Entry<const N: usize> {
    /// Get the input array from the entry
    fn get_inputs(&self) -> [f64; N];
    /// Get the expected output from the entry
    fn get_expected_output(&self) -> f64;
}
/// # Database trait  
/// This is wrapped around a database, which the data loader will pull from.  
pub trait Database<const N: usize, E: Entry<N>> {
    /// Returns the number of database entries, which is basically the size of the database.  
    fn size(&self) -> usize;
    /// Get an entry from the database given the index. This should be deterministic.  
    fn get(&self, idx: usize) -> E;
}
