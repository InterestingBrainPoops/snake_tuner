use nalgebra::SVector;
/// Eval trait  
/// C is the number of coefficients it takes as an input
/// W is the number of weights the evaluation contains
pub trait Eval<const C: usize, const W: usize> {
    fn forward(&self, coefficients: SVector<f64, C>) -> f64;
    fn nudge_weights(&mut self, new_weights: SVector<f64, W>);
    fn derivative_vector(&self, coefficients: SVector<f64, C>) -> SVector<f64, W>;
}

pub mod evaluations {
    use nalgebra::SVector;

    use super::Eval;

    /// Linear evaluation
    pub struct Linear<const N: usize> {
        weights: SVector<f64, N>,
    }

    impl<const N: usize> Linear<N> {
        /// Create a new set of Linear weights from
        pub fn new<const S: usize>() -> Linear<S> {
            Linear {
                weights: SVector::new_random(),
            }
        }
    }
    impl<const N: usize> Eval<N, N> for Linear<N> {
        fn forward(&self, coefficients: SVector<f64, N>) -> f64 {
            coefficients.dot(&self.weights)
        }

        fn nudge_weights(&mut self, updates: SVector<f64, N>) {
            self.weights += updates;
        }

        fn derivative_vector(&self, coefficients: SVector<f64, N>) -> SVector<f64, N> {
            coefficients
        }
    }
}
