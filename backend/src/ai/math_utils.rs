use std::f64::consts::E;

// Sigmoid activation function
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + E.powf(-x))
}

// Derivative of sigmoid
pub fn sigmoid_derivative(x: f64) -> f64 {
    x * (1.0 - x)
}

// ReLU activation function
pub fn relu(x: f64) -> f64 {
    x.max(0.0)
}

// Derivative of ReLU
pub fn relu_derivative(x: f64) -> f64 {
    if x > 0.0 { 1.0 } else { 0.0 }
}
