use std::f64::consts::E;

use crate::database::Database;
use rand::prelude::*;
pub struct Tuner<const N: usize, D: Database<N>> {
    database: D,
    learning_rate: f64,
    rng: ThreadRng,
}

impl<const N: usize, D: Database<N>> Tuner<N, D> {
    pub fn new(database: D, learning_rate: f64) -> Self {
        Tuner {
            database,
            learning_rate,
            rng: thread_rng(),
        }
    }

    pub fn step(&mut self, weights: [f64; N]) -> [f64; N] {
        let db_item = self
            .database
            .get(self.rng.gen_range(0..self.database.size()));
        let guess = sigmoid(Self::dot(weights, db_item.inputs));
        let error = db_item.expected - guess;
        let mut out = weights.clone();
        for (idx, item) in out.iter_mut().enumerate() {
            // Delta rule in accordance with https://en.wikipedia.org/wiki/Delta_rule
            *item += d_sigmoid(weights[idx] * db_item.inputs[idx])
                * db_item.inputs[idx]
                * self.learning_rate
                * error;
        }

        return out;
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
