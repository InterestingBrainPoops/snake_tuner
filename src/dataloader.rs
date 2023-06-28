//! Dataloader struct
//!
//! This module provides a [`DataLoader`]
use rand::{seq::SliceRandom, thread_rng};

use crate::database::{Database, Entry};

/// Wrapper around a [`crate::database::Database`], providing shuffling and batching utilities.
#[derive(Clone, Debug)]
pub struct DataLoader<const N: usize, E: Entry<N>> {
    queue: Vec<Vec<E>>,
    idx: usize,
}

impl<const N: usize, E: Entry<N> + Clone> DataLoader<N, E> {
    /// Returns a new dataloader given the following input parameters.    
    /// *   `database` is not stored, only used to initalize the internal buffer  
    /// *   `batch_size` is the batch size of the data, and cannot be modified once created  
    /// *   `shuffle` sets whether or not to shuffle the dataset  
    pub fn new<D: Database<N, E>>(
        database: D,
        batch_size: usize,
        shuffle: bool,
    ) -> DataLoader<N, E> {
        let mut rng = thread_rng();
        let mut numbers = (0..database.size()).collect::<Vec<usize>>();
        if shuffle {
            numbers.shuffle(&mut rng);
        }
        let numbers = numbers
            .chunks(batch_size)
            .map(|s| s.iter().map(|item| database.get(*item)).collect())
            .collect();
        DataLoader {
            queue: numbers,
            idx: 0,
        }
    }

    /// Sample a batch of entries from the database.
    pub fn sample(&mut self) -> Vec<E> {
        if self.idx == self.queue.len() {
            self.idx = 0;
        }
        let out = self.queue[self.idx].clone();
        self.idx += 1;
        out
    }
}
