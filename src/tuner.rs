use std::f64::consts::E;

use crate::dataloader::DataLoader;

/// The Tuner
/// This can be passed around, since it holds all the information it needs to be consistent.
pub struct Tuner<const N: usize> {
    data_loader: DataLoader<N>,
    learning_rate: f64,
}

impl<const N: usize> Tuner<N> {
    /// Make a new tuner from an input database and learning rate.
    /// Reccomended learning rate is 0.005
    pub fn new(data_loader: &DataLoader<N>, learning_rate: f64) -> Self {
        Tuner {
            data_loader: data_loader.clone(),
            learning_rate,
        }
    }
    /// Run a training step on the weights given.
    /// It will grab a sample from the dataloader and process it.
    pub fn step(&mut self, mut weights: [f64; N]) -> [f64; N] {
        let entries = self.data_loader.sample();
        let sample_size = entries.len();
        let mut gradient_accumulator = [0.0; N];

        for entry in entries {
            let guess = sigmoid(Self::dot(weights, entry.inputs));
            let error = entry.expected - guess;
            for (idx, item) in gradient_accumulator.iter_mut().enumerate() {
                // Delta rule in accordance with https://en.wikipedia.org/wiki/Delta_rule
                *item += d_sigmoid(weights[idx] * entry.inputs[idx])
                    * entry.inputs[idx]
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

    fn dot(x1: [f64; N], x2: [f64; N]) -> f64 {
        x1.iter()
            .enumerate()
            .map(|(idx, item)| item * x2[idx])
            .sum()
    }
}

fn sigmoid(n: f64) -> f64 {
    1.0 / (1.0 + E.powf(-n))
}

fn d_sigmoid(n: f64) -> f64 {
    sigmoid(n) * (1.0 - sigmoid(n))
}
