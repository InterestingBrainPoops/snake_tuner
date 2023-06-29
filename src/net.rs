use std::ops::Mul;

use nalgebra::{DMatrix, DVector, SMatrix, SVector};

use crate::activation::ActivationFunction;

pub struct Layer {
    pub weights_io: DMatrix<f64>,
    pub bias_o: DVector<f64>,
}

impl Layer {
    pub fn new(inputs: usize, outputs: usize) -> Layer {
        Layer {
            weights_io: DMatrix::zeros(outputs, inputs),
            bias_o: DVector::zeros(outputs),
        }
    }
    pub fn backprop<A: ActivationFunction>(
        &mut self,
        out_errors: DVector<f64>,
        output: DVector<f64>,
        input: DVector<f64>,
        learning_rate: f64,
    ) -> DVector<f64> {
        let gradients = output.map(|x| A::derivative(x)).component_mul(&out_errors) * learning_rate;

        let input_t = input.transpose();
        let weights_io_deltas = gradients.clone().mul(input_t);

        self.weights_io += weights_io_deltas;

        self.bias_o += gradients;
        let wio_t = self.weights_io.transpose();
        wio_t.mul(out_errors)
    }

    pub fn forward<A: ActivationFunction>(&self, input: DVector<f64>) -> DVector<f64> {
        (self.weights_io.clone().mul(input) + self.bias_o.clone()).map(|x| A::evaluate(x))
    }
}

pub struct Net {
    layers: Vec<Layer>,
}

impl Net {
    pub fn new(layer_sizes: Vec<usize>) -> Net {
        let mut layers = vec![];
        for (idx, in_size) in layer_sizes[..(layer_sizes.len() - 1)].iter().enumerate() {
            layers.push(Layer::new(*in_size, layer_sizes[idx + 1]));
        }
        Net { layers }
    }

    pub fn forward<A: ActivationFunction>(&self, input: DVector<f64>) -> DVector<f64> {
        let mut input = input;
        for x in &self.layers {
            input = x.forward::<A>(input.clone());
        }
        input
    }

    pub fn backward<A: ActivationFunction>(
        &mut self,
        input: DVector<f64>,
        expected: DVector<f64>,
        learning_rate: f64,
    ) {
        let mut layer_intermmediates = vec![input];
        for (idx, layer) in self.layers.iter().enumerate() {
            layer_intermmediates.push(layer.forward::<A>(layer_intermmediates[idx].clone()));
        }
        let mut error = expected - layer_intermmediates.last().unwrap().clone();
        for (idx, layer) in self.layers.iter_mut().enumerate().rev() {
            error = layer.backprop::<A>(
                error,
                layer_intermmediates[idx + 1].clone(),
                layer_intermmediates[idx].clone(),
                learning_rate,
            );
        }
    }
}
