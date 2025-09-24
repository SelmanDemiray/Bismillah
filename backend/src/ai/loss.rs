use super::tensor::Tensor;

// Trait for loss functions
pub trait Loss {
    fn forward(&self, y_true: &Tensor, y_pred: &Tensor) -> f64;
    fn backward(&self, y_true: &Tensor, y_pred: &Tensor) -> Tensor;
}

// Mean Squared Error loss
pub struct MSE;

impl Loss for MSE {
    fn forward(&self, y_true: &Tensor, y_pred: &Tensor) -> f64 {
        let diff = y_pred.clone() - y_true.clone();
        let squared_error: f64 = diff.data.iter().map(|&e| e.powi(2)).sum();
        squared_error / y_true.data.len() as f64
    }

    fn backward(&self, y_true: &Tensor, y_pred: &Tensor) -> Tensor {
        let diff = y_pred.clone() - y_true.clone();
        let gradient_data: Vec<f64> = diff.data.iter().map(|&e| 2.0 * e / y_true.data.len() as f64).collect();
        Tensor::from(gradient_data, y_pred.shape.clone())
    }
}
