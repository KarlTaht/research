from typing import Optional, Dict, Any
import torch
import torch.nn as nn

from ..base import BaseLanguageModel


class CustomTransformer:
    def __init__(self, config):
        """
        Initialize model with specified layer sizes.
        """

        self.d_model = 128
        self.vocab_size = 128
        self.max_seq_len = 128

        self._initialize_weights_and_biases()


    # === Architecture & Initialization ===
    def _initialize_weights_and_biases(self):
        """Initialize weights for all layers. Xavier initialization optimized for ReLU or GELU"""

        # Create input embedding matrix using Xavier-style initialization
        self.vocab_embedding = torch.randn(
            self.vocab_size,
            self.d_model
        ) * (1.0 / self.d_model) ** 0.5
        self.pos_embedding = torch.randn(
            self.max_seq_len,
            self.d_model
        ) * (1.0 / self.d_model) ** 0.5

        # Hidden layers: Xavier-style initiatlization
        self.hidden_layers_weights = np.random.randn(
            self.num_hidden_layers,
            self.hidden_dimension,
            self.hidden_dimension
        ) * np.sqrt(1.0 / self.hidden_dimension)
        self.hidden_layer_biases = np.zeros(
            (self.num_hidden_layers, self.hidden_dimension)
        )

        # Output layer: Xavier-style initiatlization
        self.output_layer_weights = np.random.randn(
            self.hidden_dimension,
            self.output_dimension,
        ) * np.sqrt(1.0 / self.hidden_dimension)
        self.output_layer_biases = np.zeros(self.output_dimension)


    # === Forward Pass ===
    def forward(self, batched_tokens):
        """
        Forward pass through the network. Accepts sequence tokens. 

        Args:
            tokens: Tensor of size [batch_size, seq_len] populated with integer
                    indices of the tokens. 

        Returns:
            Output predictions
        """
        
        # Step 1: Embed the tokens into the input
        # shape = [batch_size, seq_len, d_model]
        embedding = self.embed(batched_tokens)

        # Step 2: 

    def embed(self, batched_tokens):
        """
        Embeds the input tokens to a learned vector of size d_model
        This indexes into the weight matrix of shape [vocab_size, d_model]
        Note the dimension is consistent between embedding, attention, and FCNN

        Each row in the embedding matrix embeds a particular token
        The embedding for a particular token is the size d_model

        For example, if we have seq_len = 4 and d_model = 3,
        -> The embedding matrix has # rows equal to vocab_ize 
        -> The embedding matrix has columns equal to d_model (3)
        -> The resultant embed input will have number of rows equal to seq_len (4)

        Args:
            token_ids: [batch, seq_len]

        Returns:
            embedding: [batch, seq_len, d_model]
        """
        # Grabbing the token vocab embeddings rows
        tokens = self.vocab_embedding[batched_tokens] 
        # shape = [batch_size, seq_len, d_model]

        # Grab the positional encoding for the sequence length in this batch
        # Note that all sequences in a batch must be the same length 
        # This is handled by the input, which produces a padding
        sequence_length = batched_tokens.shape[1]
        positions = self.pos_embedding[:sequence_length]
        # shape = [seq_len, d_model]

        # Design Choice: Compressed representation 
        # Token vocabulary + token position combined via element-wise addition
        # Sequence positions broadcast across each seq. in batch
        return tokens + positions

    
    def gelu_derivative(self, x):
        """Derivative of GELU activation function."""
        pass

    # === Backward Pass ===
    def backward(self, loss_gradient):
        """
        Backward pass to compute gradients.

        Args:
            loss_gradient: Gradient of loss with respect to output activations

        Returns:
            Nothing
        """
        pass

    def compute_gradient_norms(self):
        """
        Compute L2 norms of gradients for each layer.

        Returns:
            Dictionary mapping layer names to L2 norms
        """
        pass

    # === Parameter Updates ===
    def update_parameters(self, learning_rate):
        """Update weights and biases using computed gradients."""

       pass


    # === Prediction ===
    def predict(self, X, store_activations=False):
        """Make predictions on new data."""
        pass

    # === Visualization Support ===
    def get_state(self):
        """
        Return model state for visualization.

        Returns:
            Dictionary containing weights, activations, gradients, architecture
        """
        pass

    def get_layer_info(self, layer_idx):
        """Get detailed info about a specific layer."""
        pass