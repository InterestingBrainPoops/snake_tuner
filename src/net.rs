use std::ops::Mul;

use nalgebra::{SMatrix, SVector};

use crate::activation::ActivationFunction;

pub struct Net<const I: usize, const H: usize, const O: usize> {
    pub weights_ih: SMatrix<f64, H, I>,
    pub weights_ho: SMatrix<f64, O, H>,
    pub bias_h: SMatrix<f64, H, 1>,
    pub bias_o: SMatrix<f64, O, 1>,
    pub lr: f64,
}

pub struct Layer<const I: usize, const O: usize> {
    pub weights_io: SMatrix<f64, O, I>,
    pub bias_o: SMatrix<f64, O, 1>,
}

impl<const I: usize, const O: usize> Layer<I, O> {
    pub fn backprop<A: ActivationFunction>(
        &mut self,
        out_errors: SVector<f64, O>,
        output: SVector<f64, O>,
        input: SVector<f64, I>,
        learning_rate: f64,
    ) -> SVector<f64, I> {
        let gradients = output.map(|x| A::derivative(x)).component_mul(&out_errors) * learning_rate;

        let input_t = input.transpose();
        let weights_io_deltas = gradients.mul(input_t);

        self.weights_io += weights_io_deltas;

        self.bias_o += gradients;
        let wio_t = self.weights_io.transpose();
        wio_t.mul(out_errors)
    }
}
impl<const I: usize, const H: usize, const O: usize> Net<I, H, O> {
    pub fn forward<A: ActivationFunction>(&self, input: SVector<f64, I>) -> SVector<f64, O> {
        let mut hidden = self.weights_ih.mul(input);
        hidden += self.bias_h;
        hidden = hidden.map(|x| A::evaluate(x));

        let mut output = self.weights_ho.mul(hidden);
        output += self.bias_o;
        output = output.map(|x| A::evaluate(x));

        output
    }

    pub fn train<A: ActivationFunction>(
        &mut self,
        input: SVector<f64, I>,
        target: SVector<f64, O>,
    ) {
        let mut hidden = self.weights_ih.mul(input);
        hidden += self.bias_h;
        hidden = hidden.map(|x| A::evaluate(x));

        let mut output = self.weights_ho.mul(hidden);
        output += self.bias_o;
        output = output.map(|x| A::evaluate(x));

        let output_error = target - output;
        // begin H->O layer
        let gradients = output
            .map(|x| A::derivative(x))
            .component_mul(&output_error)
            * self.lr;

        let hidden_t = hidden.transpose();
        let weight_ho_deltas = gradients.mul(hidden_t);

        self.weights_ho += weight_ho_deltas;

        self.bias_o += gradients;
        let who_t = self.weights_ho.transpose();
        let hidden_errors = who_t.mul(output_error);

        // end H->0 layer

        let hidden_gradient = hidden
            .map(|x| A::derivative(x))
            .component_mul(&hidden_errors)
            * self.lr;

        let inputs_t = input.transpose();
        let weights_ih_deltas = hidden_gradient.mul(inputs_t);

        self.weights_ih += weights_ih_deltas;

        self.bias_h += hidden_gradient;
    }
}
