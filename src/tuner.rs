//! Main tuner struct  
//!   
//! This module provides the primary [`Tuner`] struct.  
use nalgebra::SVector;

use crate::{activation::ActivationFunction, database::Entry, dataloader::DataLoader};

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
    pub fn step<A: ActivationFunction>(&mut self, mut weights: SVector<f64, N>) -> SVector<f64, N> {
        let entries = self.data_loader.sample();
        let sample_size = entries.len();
        let mut gradient_accumulator = SVector::from([0.0; N]);
        for entry in entries {
            let guess = A::evaluate(weights.dot(&entry.get_inputs()));
            let error = entry.get_expected_output() - guess;

            // Delta rule in accordance with https://en.wikipedia.org/wiki/Delta_rule
            gradient_accumulator = gradient_accumulator
                + (&entry.get_inputs())
                    * A::derivative(weights.dot(&entry.get_inputs()))
                    * self.learning_rate
                    * error;
        }

        gradient_accumulator = gradient_accumulator.map(|x| x / sample_size as f64);
        weights += gradient_accumulator;
        weights
    }
}
