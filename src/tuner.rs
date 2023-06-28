//! Main tuner struct  
//!   
//! This module provides the primary [`Tuner`] struct.  
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

    fn dot(x1: [f64; N], x2: [f64; N]) -> f64 {
        x1.iter()
            .enumerate()
            .map(|(idx, item)| item * x2[idx])
            .sum()
    }

    /// Returns the weights with after running one iteration of batch Stochastic Gradient Descent (SGD)  
    pub fn step<A: ActivationFunction>(&mut self, mut weights: [f64; N]) -> [f64; N] {
        let entries = self.data_loader.sample();
        let sample_size = entries.len();
        let mut gradient_accumulator = [0.0; N];

        for entry in entries {
            let guess = A::evaluate(Self::dot(weights, entry.get_inputs()));
            let error = entry.get_expected_output() - guess;
            for (idx, item) in gradient_accumulator.iter_mut().enumerate() {
                // Delta rule in accordance with https://en.wikipedia.org/wiki/Delta_rule
                *item += A::derivative(weights[idx] * entry.get_inputs()[idx])
                    * entry.get_inputs()[idx]
                    * self.learning_rate
                    * error;
            }
        }

        gradient_accumulator
            .iter_mut()
            .for_each(|x| *x /= sample_size as f64);
        weights
            .iter_mut()
            .enumerate()
            .for_each(|(idx, item)| *item += gradient_accumulator[idx]);
        weights
    }
}
