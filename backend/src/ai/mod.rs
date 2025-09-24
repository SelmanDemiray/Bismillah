// This file makes the 'ai' directory a module.
// It publicly exports the components of our neural network library.

pub mod activations;
pub mod layers;
pub mod loss;
pub mod math_utils;
pub mod model;
pub mod optimizer;
pub mod tensor;

// Publicly use the primary structs for easier access
pub use model::Model;
pub use tensor::Tensor;
pub use layers::{Layer, Dense};
pub use activations::Activation;
pub use loss::Loss;
pub use optimizer::Optimizer;

pub use layers::{MultiHeadAttention, TransformerEncoder, TransformerDecoder};
pub use model::{TransformerType, TransformerConfig};
pub use layers::{PositionalEncoding, LayerNorm, Dropout};
pub use layers::{Residual, RotaryPositionalEmbedding, GLU, LayerStack};
pub use layers::{SparseAttention, MixtureOfExperts, ParameterSharing, ReversibleLayer, LayerOutputExtractor};
pub use layers::{TokenEmbedding, SegmentEmbedding, PositionEmbedding, TransformerBuilder};
pub use layers::{EmbeddingDropout, OutputHead, OutputHeadType, LayerConfig, TransformerTemplates};
