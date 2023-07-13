use crate::{activation::ActivationFunction, database::Entry, evaluation::Eval};

pub trait Optimizer<A: ActivationFunction, const W: usize, const C: usize, E: Eval<C, W, A>> {
    fn step<T: Entry<C>>(&mut self, eval: &mut E, entries: Vec<T>);
    fn reset(&mut self);
}

pub mod optimizers {
    use nalgebra::SVector;

    use crate::{activation::ActivationFunction, evaluation::Eval};

    use super::Optimizer;

    pub struct SGD {
        lr: f64,
    }
    impl SGD {
        pub fn new(learning_rate: f64) -> SGD {
            SGD { lr: learning_rate }
        }
    }

    impl<A: ActivationFunction, const W: usize, const C: usize, E: Eval<C, W, A>>
        Optimizer<A, W, C, E> for SGD
    {
        fn step<T: crate::database::Entry<C>>(&mut self, eval: &mut E, entries: Vec<T>) {
            let sample_size = entries.len() as f64;
            let mut gradient_accumulator = SVector::from([0.0; W]);
            for entry in entries {
                let guess = eval
                    .activation_fn()
                    .evaluate(eval.forward(entry.get_inputs()));
                let error = entry.get_expected_output() - guess;

                // Delta rule in accordance with https://en.wikipedia.org/wiki/Delta_rule
                gradient_accumulator += eval.derivative_vector(entry.get_inputs())
                    * eval
                        .activation_fn()
                        .derivative(eval.forward(entry.get_inputs()))
                    * error;
            }

            gradient_accumulator = gradient_accumulator / sample_size * self.lr;
            eval.nudge_weights(gradient_accumulator);
        }

        fn reset(&mut self) {}
    }
}
