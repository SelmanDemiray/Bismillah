use super::tensor::Tensor;
use super::activations::Activation;

// Trait for any neural network layer
pub trait Layer {
    fn forward(&mut self, input: Tensor) -> Tensor;
    fn backward(&mut self, output_gradient: Tensor, learning_rate: f64) -> Tensor;
}

// A fully connected (dense) layer
pub struct Dense {
    pub weights: Tensor,
    pub biases: Tensor,
    pub activation: Activation,
    pub input: Option<Tensor>, // Store input for backpropagation
}

impl Dense {
    pub fn new(input_size: usize, output_size: usize, activation: Activation) -> Self {
        Dense {
            weights: Tensor::random(vec![input_size, output_size]),
            biases: Tensor::random(vec![1, output_size]),
            activation,
            input: None,
        }
    }
}

impl Layer for Dense {
    fn forward(&mut self, input: Tensor) -> Tensor {
        self.input = Some(input.clone());
        let output = input.dot(&self.weights);
        // Broadcasting bias addition - simplified for this example
        let biased_output_data: Vec<f64> = output.data.chunks(self.biases.shape[1]).flat_map(|row| {
            row.iter().zip(self.biases.data.iter()).map(|(x, b)| x + b).collect::<Vec<f64>>()
        }).collect();
        let biased_output = Tensor::from(biased_output_data, output.shape.clone());

        self.activation.forward(biased_output)
    }

    fn backward(&mut self, output_gradient: Tensor, learning_rate: f64) -> Tensor {
        let input = self.input.as_ref().expect("Input not set for backward pass.");
        
        let activation_gradient = self.activation.backward(output_gradient);
        let weights_gradient = input.t().dot(&activation_gradient);
        
        // This is a simplification for bias gradient
        let biases_gradient_data: Vec<f64> = (0..self.biases.shape[1])
            .map(|j| activation_gradient.data.chunks(activation_gradient.shape[1]).map(|row| row[j]).sum())
            .collect();
        let biases_gradient = Tensor::from(biases_gradient_data, self.biases.shape.clone());


        let input_gradient = activation_gradient.dot(&self.weights.t());
        
        // Update weights and biases
        self.weights = self.weights.clone() - (weights_gradient * learning_rate);
        self.biases = self.biases.clone() - (biases_gradient * learning_rate);

        input_gradient
    }
}

// --- Transformer Components ---

// Multi-Head Attention Layer
pub struct MultiHeadAttention {
    pub num_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    // Add weights, biases, etc. as needed
    pub custom_attention: Option<Box<dyn Fn(&Tensor) -> Tensor>>, // For DeepSeek/custom
}

impl MultiHeadAttention {
    pub fn new(num_heads: usize, key_dim: usize, value_dim: usize) -> Self {
        MultiHeadAttention { num_heads, key_dim, value_dim, custom_attention: None }
    }

    pub fn with_custom_attention<F: 'static + Fn(&Tensor) -> Tensor>(mut self, f: F) -> Self {
        self.custom_attention = Some(Box::new(f));
        self
    }
}

impl Layer for MultiHeadAttention {
    fn forward(&mut self, input: Tensor) -> Tensor {
        if let Some(ref custom) = self.custom_attention {
            return custom(&input);
        }
        // TODO: Implement multi-head attention logic
        input // Placeholder
    }
    fn backward(&mut self, output_gradient: Tensor, _learning_rate: f64) -> Tensor {
        // TODO: Implement backward pass for attention
        output_gradient // Placeholder
    }
}

// Positional Encoding Layer
pub struct PositionalEncoding {
    pub max_len: usize,
    pub dim: usize,
    pub encoding: Tensor,
}

impl PositionalEncoding {
    pub fn new(max_len: usize, dim: usize) -> Self {
        // TODO: Implement sinusoidal or learned positional encoding
        let encoding = Tensor::random(vec![max_len, dim]); // Placeholder
        PositionalEncoding { max_len, dim, encoding }
    }
}

impl Layer for PositionalEncoding {
    fn forward(&mut self, input: Tensor) -> Tensor {
        // TODO: Add positional encoding to input
        input // Placeholder
    }
    fn backward(&mut self, output_gradient: Tensor, _learning_rate: f64) -> Tensor {
        output_gradient // No gradients for fixed encoding
    }
}

// Layer Normalization
pub struct LayerNorm {
    pub dim: usize,
    // Add learnable parameters if needed
}

impl LayerNorm {
    pub fn new(dim: usize) -> Self {
        LayerNorm { dim }
    }
}

impl Layer for LayerNorm {
    fn forward(&mut self, input: Tensor) -> Tensor {
        // TODO: Implement layer normalization
        input // Placeholder
    }
    fn backward(&mut self, output_gradient: Tensor, _learning_rate: f64) -> Tensor {
        output_gradient // Placeholder
    }
}

// Dropout Layer
pub struct Dropout {
    pub rate: f64,
}

impl Dropout {
    pub fn new(rate: f64) -> Self {
        Dropout { rate }
    }
}

impl Layer for Dropout {
    fn forward(&mut self, input: Tensor) -> Tensor {
        // TODO: Implement dropout
        input // Placeholder
    }
    fn backward(&mut self, output_gradient: Tensor, _learning_rate: f64) -> Tensor {
        output_gradient // Placeholder
    }
}

// Residual Connection Layer
pub struct Residual {
    pub layer: Box<dyn Layer>,
}

impl Residual {
    pub fn new(layer: Box<dyn Layer>) -> Self {
        Residual { layer }
    }
}

impl Layer for Residual {
    fn forward(&mut self, input: Tensor) -> Tensor {
        let out = self.layer.forward(input.clone());
        out + input // Skip connection
    }
    fn backward(&mut self, output_gradient: Tensor, learning_rate: f64) -> Tensor {
        let grad = self.layer.backward(output_gradient.clone(), learning_rate);
        grad + output_gradient // Backprop through skip
    }
}

// Rotary Positional Embedding Layer
pub struct RotaryPositionalEmbedding {
    pub dim: usize,
}

impl RotaryPositionalEmbedding {
    pub fn new(dim: usize) -> Self {
        RotaryPositionalEmbedding { dim }
    }
}

impl Layer for RotaryPositionalEmbedding {
    fn forward(&mut self, input: Tensor) -> Tensor {
        // TODO: Implement rotary positional embedding
        input // Placeholder
    }
    fn backward(&mut self, output_gradient: Tensor, _learning_rate: f64) -> Tensor {
        output_gradient // Placeholder
    }
}

// Gated Linear Unit (GLU) Layer
pub struct GLU {
    pub dense_a: Dense,
    pub dense_b: Dense,
}

impl GLU {
    pub fn new(input_size: usize, output_size: usize, activation: Activation) -> Self {
        GLU {
            dense_a: Dense::new(input_size, output_size, activation.clone()),
            dense_b: Dense::new(input_size, output_size, activation),
        }
    }
}

impl Layer for GLU {
    fn forward(&mut self, input: Tensor) -> Tensor {
        let a = self.dense_a.forward(input.clone());
        let b = self.dense_b.forward(input);
        a * b // Element-wise gating
    }
    fn backward(&mut self, output_gradient: Tensor, learning_rate: f64) -> Tensor {
        let grad_a = self.dense_a.backward(output_gradient.clone(), learning_rate);
        let grad_b = self.dense_b.backward(output_gradient, learning_rate);
        grad_a + grad_b // Simplified
    }
}

// Utility: Stack Layers with Custom Composition
pub struct LayerStack {
    pub layers: Vec<Box<dyn Layer>>,
}

impl LayerStack {
    pub fn new(layers: Vec<Box<dyn Layer>>) -> Self {
        LayerStack { layers }
    }
}

impl Layer for LayerStack {
    fn forward(&mut self, mut input: Tensor) -> Tensor {
        for layer in self.layers.iter_mut() {
            input = layer.forward(input);
        }
        input
    }
    fn backward(&mut self, mut output_gradient: Tensor, learning_rate: f64) -> Tensor {
        for layer in self.layers.iter_mut().rev() {
            output_gradient = layer.backward(output_gradient, learning_rate);
        }
        output_gradient
    }
}

// --- Embedding Dropout & Normalization ---
pub struct EmbeddingDropout {
    pub rate: f64,
}

impl EmbeddingDropout {
    pub fn new(rate: f64) -> Self {
        EmbeddingDropout { rate }
    }
}

impl Layer for EmbeddingDropout {
    fn forward(&mut self, input: Tensor) -> Tensor {
        // TODO: Implement dropout for embeddings
        input // Placeholder
    }
    fn backward(&mut self, output_gradient: Tensor, _learning_rate: f64) -> Tensor {
        output_gradient // Placeholder
    }
}

// --- Output Heads ---
pub enum OutputHeadType {
    Classification,
    Regression,
    LanguageModeling,
}

pub struct OutputHead {
    pub head_type: OutputHeadType,
    pub dense: Dense,
}

impl OutputHead {
    pub fn new(head_type: OutputHeadType, input_dim: usize, output_dim: usize, activation: Activation) -> Self {
        OutputHead { head_type, dense: Dense::new(input_dim, output_dim, activation) }
    }
}

impl Layer for OutputHead {
    fn forward(&mut self, input: Tensor) -> Tensor {
        self.dense.forward(input)
    }
    fn backward(&mut self, output_gradient: Tensor, learning_rate: f64) -> Tensor {
        self.dense.backward(output_gradient, learning_rate)
    }
}

// --- Layer-wise Configuration ---
pub struct LayerConfig {
    pub activation: Option<Activation>,
    pub dropout: Option<f64>,
    pub normalization: bool,
}

impl LayerConfig {
    pub fn new(activation: Option<Activation>, dropout: Option<f64>, normalization: bool) -> Self {
        LayerConfig { activation, dropout, normalization }
    }
}

// --- Transformer Templates Utility ---
pub struct TransformerTemplates;

impl TransformerTemplates {
    pub fn bert(vocab_size: usize, max_len: usize, embed_dim: usize, num_layers: usize, num_heads: usize, ff_dim: usize, activation: Activation) -> LayerStack {
        let mut builder = TransformerBuilder::new()
            .add_embedding(Box::new(TokenEmbedding::new(vocab_size, embed_dim)))
            .add_embedding(Box::new(PositionEmbedding::new(max_len, embed_dim)));
        for _ in 0..num_layers {
            builder = builder.add_encoder_block(Box::new(TransformerEncoder::new(num_heads, embed_dim, embed_dim, ff_dim, embed_dim, activation.clone())));
        }
        builder.build()
    }

    pub fn gpt(vocab_size: usize, max_len: usize, embed_dim: usize, num_layers: usize, num_heads: usize, ff_dim: usize, activation: Activation) -> LayerStack {
        let mut builder = TransformerBuilder::new()
            .add_embedding(Box::new(TokenEmbedding::new(vocab_size, embed_dim)))
            .add_embedding(Box::new(PositionEmbedding::new(max_len, embed_dim)));
        for _ in 0..num_layers {
            builder = builder.add_decoder_block(Box::new(TransformerDecoder::new(num_heads, embed_dim, embed_dim, ff_dim, embed_dim, activation.clone())));
        }
        builder.build()
    }

    pub fn t5(vocab_size: usize, max_len: usize, embed_dim: usize, num_layers: usize, num_heads: usize, ff_dim: usize, activation: Activation) -> LayerStack {
        let mut builder = TransformerBuilder::new()
            .add_embedding(Box::new(TokenEmbedding::new(vocab_size, embed_dim)))
            .add_embedding(Box::new(PositionEmbedding::new(max_len, embed_dim)));
        for _ in 0..num_layers {
            builder = builder.add_encoder_block(Box::new(TransformerEncoder::new(num_heads, embed_dim, embed_dim, ff_dim, embed_dim, activation.clone())));
        }
        for _ in 0..num_layers {
            builder = builder.add_decoder_block(Box::new(TransformerDecoder::new(num_heads, embed_dim, embed_dim, ff_dim, embed_dim, activation.clone())));
        }
        builder.build()
    }

    // Add DeepSeek and other advanced templates as needed
}

// Sparse Attention Layer (for long context)
pub struct SparseAttention {
    pub num_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
}

impl SparseAttention {
    pub fn new(num_heads: usize, key_dim: usize, value_dim: usize) -> Self {
        SparseAttention { num_heads, key_dim, value_dim }
    }
}

impl Layer for SparseAttention {
    fn forward(&mut self, input: Tensor) -> Tensor {
        // TODO: Implement sparse attention logic
        input // Placeholder
    }
    fn backward(&mut self, output_gradient: Tensor, _learning_rate: f64) -> Tensor {
        output_gradient // Placeholder
    }
}

// Mixture of Experts (MoE) Layer
pub struct MixtureOfExperts {
    pub experts: Vec<Dense>,
    pub gating: Dense,
}

impl MixtureOfExperts {
    pub fn new(input_size: usize, output_size: usize, num_experts: usize, activation: Activation) -> Self {
        let experts = (0..num_experts).map(|_| Dense::new(input_size, output_size, activation.clone())).collect();
        let gating = Dense::new(input_size, num_experts, activation);
        MixtureOfExperts { experts, gating }
    }
}

impl Layer for MixtureOfExperts {
    fn forward(&mut self, input: Tensor) -> Tensor {
        // TODO: Implement MoE logic (gating + expert selection)
        input // Placeholder
    }
    fn backward(&mut self, output_gradient: Tensor, learning_rate: f64) -> Tensor {
        // TODO: Implement backward for MoE
        output_gradient // Placeholder
    }
}

// Parameter Sharing Utility
pub struct ParameterSharing {
    pub shared_layer: Box<dyn Layer>,
}

impl ParameterSharing {
    pub fn new(shared_layer: Box<dyn Layer>) -> Self {
        ParameterSharing { shared_layer }
    }
}

impl Layer for ParameterSharing {
    fn forward(&mut self, input: Tensor) -> Tensor {
        self.shared_layer.forward(input)
    }
    fn backward(&mut self, output_gradient: Tensor, learning_rate: f64) -> Tensor {
        self.shared_layer.backward(output_gradient, learning_rate)
    }
}

// Reversible Layer (for memory-efficient transformers)
pub struct ReversibleLayer {
    pub layer_f: Box<dyn Layer>,
    pub layer_g: Box<dyn Layer>,
}

impl ReversibleLayer {
    pub fn new(layer_f: Box<dyn Layer>, layer_g: Box<dyn Layer>) -> Self {
        ReversibleLayer { layer_f, layer_g }
    }
}

impl Layer for ReversibleLayer {
    fn forward(&mut self, input: Tensor) -> Tensor {
        // TODO: Implement reversible logic
        input // Placeholder
    }
    fn backward(&mut self, output_gradient: Tensor, learning_rate: f64) -> Tensor {
        // TODO: Implement reversible backward
        output_gradient // Placeholder
    }
}

// Layer Output Extractor (for probing/intermediate supervision)
pub struct LayerOutputExtractor {
    pub layer: Box<dyn Layer>,
    pub last_output: Option<Tensor>,
}

impl LayerOutputExtractor {
    pub fn new(layer: Box<dyn Layer>) -> Self {
        LayerOutputExtractor { layer, last_output: None }
    }
    pub fn get_last_output(&self) -> Option<&Tensor> {
        self.last_output.as_ref()
    }
}

impl Layer for LayerOutputExtractor {
    fn forward(&mut self, input: Tensor) -> Tensor {
        let out = self.layer.forward(input);
        self.last_output = Some(out.clone());
        out
    }
    fn backward(&mut self, output_gradient: Tensor, learning_rate: f64) -> Tensor {
        self.layer.backward(output_gradient, learning_rate)
    }
}

// Transformer Encoder Block
pub struct TransformerEncoder {
    pub attention: MultiHeadAttention,
    pub feed_forward: Dense,
}

impl TransformerEncoder {
    pub fn new(num_heads: usize, key_dim: usize, value_dim: usize, ff_dim: usize, input_dim: usize, activation: Activation) -> Self {
        TransformerEncoder {
            attention: MultiHeadAttention::new(num_heads, key_dim, value_dim),
            feed_forward: Dense::new(input_dim, ff_dim, activation),
        }
    }
}

impl Layer for TransformerEncoder {
    fn forward(&mut self, input: Tensor) -> Tensor {
        let attn_out = self.attention.forward(input.clone());
        self.feed_forward.forward(attn_out)
    }
    fn backward(&mut self, output_gradient: Tensor, learning_rate: f64) -> Tensor {
        let grad_ff = self.feed_forward.backward(output_gradient, learning_rate);
        self.attention.backward(grad_ff, learning_rate)
    }
}

// Transformer Decoder Block
pub struct TransformerDecoder {
    pub self_attention: MultiHeadAttention,
    pub cross_attention: MultiHeadAttention,
    pub feed_forward: Dense,
}

impl TransformerDecoder {
    pub fn new(num_heads: usize, key_dim: usize, value_dim: usize, ff_dim: usize, input_dim: usize, activation: Activation) -> Self {
        TransformerDecoder {
            self_attention: MultiHeadAttention::new(num_heads, key_dim, value_dim),
            cross_attention: MultiHeadAttention::new(num_heads, key_dim, value_dim),
            feed_forward: Dense::new(input_dim, ff_dim, activation),
        }
    }
}

impl Layer for TransformerDecoder {
    fn forward(&mut self, input: Tensor) -> Tensor {
        let self_attn_out = self.self_attention.forward(input.clone());
        let cross_attn_out = self.cross_attention.forward(self_attn_out);
        self.feed_forward.forward(cross_attn_out)
    }
    fn backward(&mut self, output_gradient: Tensor, learning_rate: f64) -> Tensor {
        let grad_ff = self.feed_forward.backward(output_gradient, learning_rate);
        let grad_cross = self.cross_attention.backward(grad_ff, learning_rate);
        self.self_attention.backward(grad_cross, learning_rate)
    }
}

// --- End Transformer Components ---
// --- Embedding Layers ---

// Token Embedding Layer
pub struct TokenEmbedding {
    pub vocab_size: usize,
    pub embed_dim: usize,
    pub embeddings: Tensor,
}

impl TokenEmbedding {
    pub fn new(vocab_size: usize, embed_dim: usize) -> Self {
        let embeddings = Tensor::random(vec![vocab_size, embed_dim]);
        TokenEmbedding { vocab_size, embed_dim, embeddings }
    }
}

impl Layer for TokenEmbedding {
    fn forward(&mut self, input: Tensor) -> Tensor {
        // TODO: Implement token lookup
        input // Placeholder
    }
    fn backward(&mut self, output_gradient: Tensor, _learning_rate: f64) -> Tensor {
        output_gradient // Placeholder
    }
}

// Segment Embedding Layer
pub struct SegmentEmbedding {
    pub num_segments: usize,
    pub embed_dim: usize,
    pub embeddings: Tensor,
}

impl SegmentEmbedding {
    pub fn new(num_segments: usize, embed_dim: usize) -> Self {
        let embeddings = Tensor::random(vec![num_segments, embed_dim]);
        SegmentEmbedding { num_segments, embed_dim, embeddings }
    }
}

impl Layer for SegmentEmbedding {
    fn forward(&mut self, input: Tensor) -> Tensor {
        // TODO: Implement segment lookup
        input // Placeholder
    }
    fn backward(&mut self, output_gradient: Tensor, _learning_rate: f64) -> Tensor {
        output_gradient // Placeholder
    }
}

// Position Embedding Layer
pub struct PositionEmbedding {
    pub max_len: usize,
    pub embed_dim: usize,
    pub embeddings: Tensor,
}

impl PositionEmbedding {
    pub fn new(max_len: usize, embed_dim: usize) -> Self {
        let embeddings = Tensor::random(vec![max_len, embed_dim]);
        PositionEmbedding { max_len, embed_dim, embeddings }
    }
}

impl Layer for PositionEmbedding {
    fn forward(&mut self, input: Tensor) -> Tensor {
        // TODO: Implement position lookup
        input // Placeholder
    }
    fn backward(&mut self, output_gradient: Tensor, _learning_rate: f64) -> Tensor {
        output_gradient // Placeholder
    }
}

// --- Transformer Builder Utility ---
pub struct TransformerBuilder {
    pub layers: Vec<Box<dyn Layer>>,
}

impl TransformerBuilder {
    pub fn new() -> Self {
        TransformerBuilder { layers: Vec::new() }
    }

    pub fn add_embedding(mut self, embedding: Box<dyn Layer>) -> Self {
        self.layers.push(embedding);
        self
    }

    pub fn add_encoder_block(mut self, encoder: Box<dyn Layer>) -> Self {
        self.layers.push(encoder);
        self
    }

    pub fn add_decoder_block(mut self, decoder: Box<dyn Layer>) -> Self {
        self.layers.push(decoder);
        self
    }

    pub fn add_layer(mut self, layer: Box<dyn Layer>) -> Self {
        self.layers.push(layer);
        self
    }

    pub fn build(self) -> LayerStack {
        LayerStack::new(self.layers)
    }
}
