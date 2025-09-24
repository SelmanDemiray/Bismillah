use super::tensor::Tensor;
use super::math_utils::{sigmoid, sigmoid_derivative, relu, relu_derivative};

#[derive(Clone, Copy)]
pub enum ActivationType {
    Sigmoid,
    ReLU,
}

#[derive(Clone)]
pub struct Activation {
    pub activation_type: ActivationType,
    output: Option<Tensor>,
}

impl Activation {
    pub fn new(activation_type: ActivationType) -> Self {
        Self { activation_type, output: None }
    }

    pub fn forward(&mut self, input: Tensor) -> Tensor {
        let output = match self.activation_type {
            ActivationType::Sigmoid => input.map(sigmoid),
            ActivationType::ReLU => input.map(relu),
        };
        self.output = Some(output.clone());
        output
    }

    pub fn backward(&self, output_gradient: Tensor) -> Tensor {
        let output = self.output.as_ref().expect("Output not set for backward pass.");
        let derivative = match self.activation_type {
            ActivationType::Sigmoid => output.map(sigmoid_derivative),
            ActivationType::ReLU => output.map(relu_derivative),
        };
        output_gradient * derivative
    }
}
