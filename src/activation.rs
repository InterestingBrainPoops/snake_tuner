//! Activation functions and utilities  
//!   
//! This module provides:  
//! *   [`ActivationFunction`] A trait to make your own activation functions
//! *   [`functions`] A module which contains a collection of activation functions to use

/// Wrapper trait around Activation functions which includes the derivative and normal calculation.
pub trait ActivationFunction {
    /// Evaluate the function.  
    fn evaluate(value: f64) -> f64;

    /// Evaluate the derivative of the function.
    fn derivative(value: f64) -> f64;
}

/// A collection of activation functions for use in your code
pub mod functions {
    use std::f64::consts::E;

    use super::ActivationFunction;

    /// Sigmoid Activation function
    pub struct Sigmoid;
    impl ActivationFunction for Sigmoid {
        fn evaluate(value: f64) -> f64 {
            1.0 / (1.0 + E.powf(-value))
        }

        fn derivative(value: f64) -> f64 {
            value * (1.0 - value)
        }
    }

    /// ReLu Activation function
    pub struct ReLu;
    impl ActivationFunction for ReLu {
        fn evaluate(value: f64) -> f64 {
            value.max(0.0)
        }

        fn derivative(value: f64) -> f64 {
            if value >= 0.0 {
                1.0
            } else {
                0.0
            }
        }
    }
}
