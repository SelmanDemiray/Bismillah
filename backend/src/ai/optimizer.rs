// For simplicity, the optimizer logic is integrated directly into the
// layer's backward pass in this implementation (learning_rate parameter).
// This file is a placeholder for more advanced optimizers like Adam or RMSprop.

pub trait Optimizer {
    fn update_weights(&self);
}

pub struct SGD {
    pub learning_rate: f64,
}
