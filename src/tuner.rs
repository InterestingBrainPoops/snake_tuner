use crate::{activation::ActivationFunction, database::Entry, dataloader::DataLoader};

/// The Tuner  
/// This can be passed around, since it holds all the information it needs to be consistent.  
pub struct Tuner<const N: usize, E: Entry<N> + Clone> {
    data_loader: DataLoader<N, E>,
    learning_rate: f64,
}

impl<const N: usize, E: Entry<N> + Clone> Tuner<N, E> {
    /// Make a new tuner from an input database and learning rate.  
    /// Reccomended learning rate is 0.005  
    pub fn new(data_loader: &DataLoader<N, E>, learning_rate: f64) -> Self {
        Tuner {
            data_loader: data_loader.clone(),
            learning_rate,
        }
    }

    fn dot(x1: [f64; N], x2: [f64; N]) -> f64 {
        x1.iter()
            .enumerate()
            .map(|(idx, item)| item * x2[idx])
            .sum()
    }

    /// Run a training step on the weights given.  
    /// It will grab a sample from the dataloader and process it.  
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
