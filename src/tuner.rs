//! Main tuner struct  
//!   
//! This module provides the primary [`Tuner`] struct.  
use nalgebra::SVector;

use crate::{
    activation::ActivationFunction, database::Entry, dataloader::DataLoader, evaluation::Eval,
};

/// Holds the dataloader and other information for the tuner to function.  
pub struct Tuner<const W: usize, const C: usize, E: Entry<C> + Clone, V: Eval<C, W>> {
    data_loader: DataLoader<C, E>,
    learning_rate: f64,
    pub evaluation: V,
}

impl<const W: usize, const C: usize, E: Entry<C> + Clone, V: Eval<C, W>> Tuner<W, C, E, V> {
    /// Returns a new tuner given a dataloader and learning rate.  
    /// A recommended value for the learning rate is 0.005.  
    pub fn new(data_loader: DataLoader<C, E>, learning_rate: f64, evaluation: V) -> Self {
        Tuner {
            data_loader,
            learning_rate,
            evaluation,
        }
    }

    /// Returns the weights with after running one iteration of batch Stochastic Gradient Descent (SGD)  
    pub fn step<A: ActivationFunction>(&mut self) {
        let entries = self.data_loader.sample();
        let sample_size = entries.len() as f64;
        let mut gradient_accumulator = SVector::from([0.0; W]);
        for entry in entries {
            let guess = A::evaluate(self.evaluation.forward(entry.get_inputs()));
            let error = entry.get_expected_output() - guess;

            // Delta rule in accordance with https://en.wikipedia.org/wiki/Delta_rule
            gradient_accumulator += self.evaluation.derivative_vector(entry.get_inputs())
                * A::derivative(self.evaluation.forward(entry.get_inputs()))
                * error;
        }

        gradient_accumulator = gradient_accumulator / sample_size * self.learning_rate;
        self.evaluation.nudge_weights(gradient_accumulator);
    }
}
