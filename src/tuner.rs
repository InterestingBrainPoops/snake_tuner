//! Main tuner struct  
//!   
//! This module provides the primary [`Tuner`] struct.  
use nalgebra::SVector;

use crate::{activation::ActivationFunction, database::Entry, dataloader::DataLoader, net::Net};

/// Holds the dataloader and other information for the tuner to function.  
pub struct Tuner<const N: usize, E: Entry<N> + Clone> {
    data_loader: DataLoader<N, E>,
    learning_rate: f64,
}

impl<const N: usize, E: Entry<N> + Clone> Tuner<N, E> {
    /// Returns a new tuner given a dataloader and learning rate.  
    /// A reccomended value for the learning rate is 0.005.  
    pub fn new(data_loader: DataLoader<N, E>, learning_rate: f64) -> Self {
        Tuner {
            data_loader,
            learning_rate,
        }
    }

    /// Returns the weights with after running one iteration of batch Stochastic Gradient Descent (SGD)  
    pub fn step<A: ActivationFunction>(&mut self, net: &mut Net) {
        let entries = self.data_loader.sample();
        for entry in entries {
            net.backward::<A>(
                entry.get_inputs(),
                entry.get_expected_output(),
                self.learning_rate,
            );
        }
    }
}
