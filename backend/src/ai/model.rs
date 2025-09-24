use super::layers::Layer;
use super::tensor::Tensor;
use super::loss::Loss;

pub struct Model {
    pub layers: Vec<Box<dyn Layer>>,
}

pub enum TransformerType {
    EncoderOnly,
    DecoderOnly,
    EncoderDecoder,
    DeepSeek,
    Custom(String),
}

pub struct TransformerConfig {
    pub transformer_type: TransformerType,
    pub num_layers: usize,
    pub num_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub ff_dim: usize,
    pub input_dim: usize,
    pub activation: super::activations::Activation,
}

impl Model {
    pub fn new() -> Self {
        Model { layers: Vec::new() }
    }

    pub fn from_transformer_config(config: TransformerConfig) -> Self {
        let mut layers: Vec<Box<dyn Layer>> = Vec::new();
        match config.transformer_type {
            TransformerType::EncoderOnly => {
                for _ in 0..config.num_layers {
                    layers.push(Box::new(super::layers::TransformerEncoder::new(
                        config.num_heads,
                        config.key_dim,
                        config.value_dim,
                        config.ff_dim,
                        config.input_dim,
                        config.activation.clone(),
                    )));
                }
            }
            TransformerType::DecoderOnly => {
                for _ in 0..config.num_layers {
                    layers.push(Box::new(super::layers::TransformerDecoder::new(
                        config.num_heads,
                        config.key_dim,
                        config.value_dim,
                        config.ff_dim,
                        config.input_dim,
                        config.activation.clone(),
                    )));
                }
            }
            TransformerType::EncoderDecoder => {
                // Encoder stack
                for _ in 0..config.num_layers {
                    layers.push(Box::new(super::layers::TransformerEncoder::new(
                        config.num_heads,
                        config.key_dim,
                        config.value_dim,
                        config.ff_dim,
                        config.input_dim,
                        config.activation.clone(),
                    )));
                }
                // Decoder stack
                for _ in 0..config.num_layers {
                    layers.push(Box::new(super::layers::TransformerDecoder::new(
                        config.num_heads,
                        config.key_dim,
                        config.value_dim,
                        config.ff_dim,
                        config.input_dim,
                        config.activation.clone(),
                    )));
                }
            }
            TransformerType::DeepSeek => {
                // Placeholder: DeepSeek-specific logic can be added here
                for _ in 0..config.num_layers {
                    layers.push(Box::new(super::layers::TransformerDecoder::new(
                        config.num_heads,
                        config.key_dim,
                        config.value_dim,
                        config.ff_dim,
                        config.input_dim,
                        config.activation.clone(),
                    )));
                }
            }
            TransformerType::Custom(_) => {
                // Placeholder for custom transformer types
            }
        }
        Model { layers }
    }

    pub fn add(&mut self, layer: Box<dyn Layer>) {
        self.layers.push(layer);
    }

    pub fn predict(&mut self, input: Tensor) -> Tensor {
        let mut output = input;
        for layer in self.layers.iter_mut() {
            output = layer.forward(output);
        }
        output
    }
    
    pub fn fit(&mut self, x_train: Vec<Tensor>, y_train: Vec<Tensor>, epochs: usize, learning_rate: f64, loss_fn: &dyn Loss) {
        for epoch in 0..epochs {
            let mut total_error = 0.0;
            for (x, y) in x_train.iter().zip(y_train.iter()) {
                // Forward pass
                let output = self.predict(x.clone());
                
                // Calculate error
                total_error += loss_fn.forward(y, &output);
                
                // Backward pass
                let mut gradient = loss_fn.backward(y, &output);
                for layer in self.layers.iter_mut().rev() {
                    gradient = layer.backward(gradient, learning_rate);
                }
            }
            println!("Epoch {}/{}, Error={}", epoch + 1, epochs, total_error / x_train.len() as f64);
        }
    }
}
