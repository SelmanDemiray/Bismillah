use std::ops::{Add, Mul, Sub};
use rand::Rng;

// A basic multi-dimensional array (Tensor) implementation
#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Vec<f64>,
    pub shape: Vec<usize>,
}

impl Tensor {
    // Creates a new Tensor filled with zeros
    pub fn new(shape: Vec<usize>) -> Self {
        let size = shape.iter().product::<usize>();
        Tensor {
            data: vec![0.0; size],
            shape,
        }
    }

    // Creates a Tensor with random values (e.g., for weight initialization)
    pub fn random(shape: Vec<usize>) -> Self {
        let size = shape.iter().product::<usize>();
        let mut rng = rand::thread_rng();
        let data = (0..size).map(|_| rng.gen_range(-0.5..0.5)).collect();
        Tensor { data, shape }
    }
    
    // Creates a tensor from a vector of data and a shape
    pub fn from(data: Vec<f64>, shape: Vec<usize>) -> Self {
        // Explicitly define the product as usize to resolve ambiguity
        assert_eq!(data.len(), shape.iter().product::<usize>(), "Data size does not match shape product");
        Tensor { data, shape }
    }

    // Dot product for 2D matrices
    pub fn dot(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape.len(), 2, "Dot product only implemented for 2D tensors");
        assert_eq!(other.shape.len(), 2, "Dot product only implemented for 2D tensors");
        assert_eq!(self.shape[1], other.shape[0], "Matrix dimensions are incompatible for dot product");

        let mut result_data = vec![0.0; self.shape[0] * other.shape[1]];
        for i in 0..self.shape[0] {
            for j in 0..other.shape[1] {
                let mut sum = 0.0;
                for k in 0..self.shape[1] {
                    sum += self.data[i * self.shape[1] + k] * other.data[k * other.shape[1] + j];
                }
                result_data[i * other.shape[1] + j] = sum;
            }
        }
        Tensor {
            data: result_data,
            shape: vec![self.shape[0], other.shape[1]],
        }
    }

    // Transpose a 2D matrix
    pub fn t(&self) -> Tensor {
        assert_eq!(self.shape.len(), 2, "Transpose only implemented for 2D tensors");
        let mut transposed_data = vec![0.0; self.data.len()];
        for i in 0..self.shape[0] {
            for j in 0..self.shape[1] {
                transposed_data[j * self.shape[0] + i] = self.data[i * self.shape[1] + j];
            }
        }
        Tensor {
            data: transposed_data,
            shape: vec![self.shape[1], self.shape[0]],
        }
    }

    // Apply a function to each element
    pub fn map(&self, f: fn(f64) -> f64) -> Tensor {
        Tensor {
            data: self.data.iter().map(|&x| f(x)).collect(),
            shape: self.shape.clone(),
        }
    }
}

// Element-wise addition
impl Add for Tensor {
    type Output = Tensor;
    fn add(self, other: Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "Tensor shapes must match for addition");
        let data = self.data.iter().zip(other.data.iter()).map(|(a, b)| a + b).collect();
        Tensor { data, shape: self.shape }
    }
}

// Element-wise subtraction
impl Sub for Tensor {
    type Output = Tensor;
    fn sub(self, other: Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "Tensor shapes must match for subtraction");
        let data = self.data.iter().zip(other.data.iter()).map(|(a, b)| a - b).collect();
        Tensor { data, shape: self.shape }
    }
}


// Element-wise multiplication
impl Mul for Tensor {
    type Output = Tensor;
    fn mul(self, other: Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "Tensor shapes must match for element-wise multiplication");
        let data = self.data.iter().zip(other.data.iter()).map(|(a, b)| a * b).collect();
        Tensor { data, shape: self.shape }
    }
}

// Scalar multiplication
impl Mul<f64> for Tensor {
    type Output = Tensor;
    fn mul(self, scalar: f64) -> Tensor {
        let data = self.data.iter().map(|&val| val * scalar).collect();
        Tensor { data, shape: self.shape }
    }
}

