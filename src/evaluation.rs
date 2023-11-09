use nalgebra::SVector;

use crate::activation::ActivationFunction;
/// Eval trait  
/// C is the number of coefficients it takes as an input
/// W is the number of weights the evaluation contains
pub trait Eval<const C: usize, const W: usize, A: ActivationFunction> {
    fn activation_fn(&self) -> A;
    fn forward(&self, coefficients: SVector<f64, C>) -> f64;
    fn nudge_weights(&mut self, new_weights: SVector<f64, W>);
    fn derivative_vector(&self, coefficients: SVector<f64, C>) -> SVector<f64, W>;
}

pub mod evaluations {
    use nalgebra::SVector;

    use crate::activation::ActivationFunction;

    use super::Eval;
    /// Linear evaluation
    #[derive(Clone, Debug)]
    pub struct Linear<const N: usize, A: ActivationFunction + Clone> {
        weights: SVector<f64, N>,
        activate_fn: A,
    }

    impl<const N: usize, A: ActivationFunction + Clone> Linear<N, A> {
        /// Create a new set of Linear weights randomly
        pub fn new<const S: usize>(activate_fn: A) -> Linear<S, A> {
            Linear {
                weights: SVector::new_random(),
                activate_fn,
            }
        }

        pub fn from_weights(weights: SVector<f64, N>, activate_fn: A) -> Linear<N, A> {
            Linear {
                weights,
                activate_fn,
            }
        }
    }
    impl<const N: usize, A: ActivationFunction + Clone> Eval<N, N, A> for Linear<N, A> {
        fn forward(&self, coefficients: SVector<f64, N>) -> f64 {
            coefficients.dot(&self.weights)
        }

        fn nudge_weights(&mut self, updates: SVector<f64, N>) {
            self.weights += updates;
        }

        fn derivative_vector(&self, coefficients: SVector<f64, N>) -> SVector<f64, N> {
            coefficients
        }

        fn activation_fn(&self) -> A {
            self.activate_fn.clone()
        }
    }
}
