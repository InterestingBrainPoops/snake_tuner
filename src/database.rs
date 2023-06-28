//! Database wrapper  
//!  
//! This moduled provides:  
//! *   [`Entry`] A wrapper around database entries
//! *   [`Database`] A wrapper around a database

/// Wrapper trait around your entry, providing a way for the tuner to get the inputs and data label from the entry.
pub trait Entry<const N: usize> {
    /// Returns the input array from the entry.
    fn get_inputs(&self) -> [f64; N];
    /// Returns the expected output from the entry.
    fn get_expected_output(&self) -> f64;
}
/// Wrapper trait around your database, providing functions to get the size and an element from the database.
pub trait Database<const N: usize, E: Entry<N>> {
    /// Returns the number of database entries.  
    fn size(&self) -> usize;
    /// Returns the entry at index `idx` in the database.  
    fn get(&self, idx: usize) -> E;
}
