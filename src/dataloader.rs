use rand::{seq::SliceRandom, thread_rng};

use crate::database::{Database, Entry};

/// # Dataloader
/// A data loader for more convinient access to batches and shuffling.
/// Abstracts away the database queries, and only does them during the new function
#[derive(Clone)]
pub struct DataLoader<const N: usize> {
    queue: Vec<Vec<Entry<N>>>,
    idx: usize,
}

impl<const N: usize> DataLoader<N> {
    /// Makes a new data loader, given input options, and the database to query from.
    /// `database` is not stored, only used to initalize the internal buffer
    /// `batch_size` is the batch size of the data, and cannot be modified once created.
    /// `shuffle` sets whether or not to shuffle the dataset. (heavily reccomended)
    pub fn new<D: Database<N>>(database: &D, batch_size: usize, shuffle: bool) -> DataLoader<N> {
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

    pub fn sample(&mut self) -> Vec<Entry<N>> {
        if self.idx == self.queue.len() {
            self.idx = 0;
        }
        let out = self.queue[self.idx].clone();
        self.idx += 1;
        return out;
    }
}
