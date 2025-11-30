from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseLanguageModel


class CustomTransformer:
    def __init__(self, config):
        """
        Initialize model with specified layer sizes.
        """
        self.vocab_size = 128
        self.max_seq_len = 128
        
        self.n_blocks = 8
        self.n_heads = 4
        self.d_model = 128
        self.d_ffn = 128
        
        self.d_head = self.d_model // self.n_heads

        self.device = self._resolve_device(config)
        self.dtype = config.get_dtype() or torch.bfloat16

        self._initialize_weights_and_biases()
        self._init_causal_mask()

    @staticmethod
    def _resolve_device(config) -> str:
        if config.get_device():
            return config.get_device()
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    # === Architecture & Initialization ===
    def _initialize_weights_and_biases(self):
        """Initialize weights for all layers. Xavier initialization optimized for ReLU or GELU"""

        # Create input embedding matrix using Xavier-style initialization
        self.vocab_embedding = torch.randn(
            self.vocab_size,
            self.d_model,
            device=self.device,
            dtype=self.dtype,
        ) * (1.0 / self.d_model) ** 0.5
        self.pos_embedding = torch.randn(
            self.max_seq_len,
            self.d_model,
            device=self.device,
            dtype=self.dtype,
        ) * (1.0 / self.d_model) ** 0.5

        # Create the attention matrix [n_blocks, d_model, d_model]
        self.Q = torch.randn(
            self.n_blocks,
            self.d_model,
            self.d_model,
            device=self.device,
            dtype=self.dtype,
        ) * (1.0 / self.d_model) ** 0.5
        self.K = torch.randn(
            self.n_blocks,
            self.d_model,
            self.d_model,
            device=self.device,
            dtype=self.dtype,
        ) * (1.0 / self.d_model) ** 0.5
        self.V = torch.randn(
            self.n_blocks,
            self.d_model,
            self.d_model,
            device=self.device,
            dtype=self.dtype,
        ) * (1.0 / self.d_model) ** 0.5
        # Output Projection matrix
        self.W_o = torch.randn(
            self.n_blocks,
            self.d_model,
            self.d_model,
            device=self.device,
            dtype=self.dtype,
        ) * (1.0 / self.d_model) ** 0.5

        # Attention layer normalization weight/bias [n_blocks, d_model]
        self.attention_gamma = torch.ones(
            (self.n_blocks, self.d_model),
            device=self.device, dtype=self.dtype
        )
        self.attention_beta = torch.zeros(
            (self.n_blocks, self.d_model),
            device=self.device, dtype=self.dtype
        )

        # Create the FFN (d_model -> d_ffn -> d_model)
        self.W1 = torch.randn(
            self.n_blocks,
            self.d_model,
            self.d_ffn,
            device=self.device,
            dtype=self.dtype,
        ) * (1.0 / self.d_model) ** 0.5
        self.W2 = torch.randn(
            self.n_blocks,
            self.d_ffn,
            self.d_model,
            device=self.device,
            dtype=self.dtype,
        ) * (1.0 / self.d_model) ** 0.5
        # Attention layer normalization weight/bias
        self.ffn_gamma = torch.ones(
            (self.n_blocks, self.d_model),
            device=self.device, dtype=self.dtype
        )
        self.ffn_beta = torch.zeros(
            (self.n_blocks, self.d_model),
            device=self.device, dtype=self.dtype
        )

        # Create the output embedding matrix
        self.output_projection = torch.randn(
            self.d_model,
            self.vocab_size,
            device=self.device,
            dtype=self.dtype,
        ) * (1.0 / self.d_model) ** 0.5

        
    def _init_causal_mask(self):
        self.causal_mask = torch.triu(
            torch.full(
                (self.max_seq_len, self.max_seq_len), 
                float('-inf'), 
                device=self.device,
                dtype=self.dtype,
            ),
            diagonal=1,
        )
    
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

        # Step 2: Decoder Block (repeat N-times)
        latent_result = self.decoder(embedding)
        
        # Step 5: Final Output
        logits = latent_result @ self.output_projection

        return logits


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
        # shape = [seq_len, d_model]
        sequence_length = batched_tokens.shape[1]
        positions = self.pos_embedding[:sequence_length]
        
        # Design Choice: Compressed representation 
        # Token vocabulary + token position combined via element-wise addition
        # Sequence positions broadcast across each seq. in batch
        return tokens + positions

    def decoder(self, X):
        """
        Compute attention, then compute FFN. 
        Layer normalization forces scales the values such that mean=0, std=1
        """
        latent_result = X

        # Each block receives the previous blocks output 
        for block_step in range(self.n_blocks):
            latent_result = latent_result + self.attention(latent_result, block_step)
            latent_result = self.normalize_attention(latent_result, block_step)

            latent_result = latent_result + self.feed_forward_network(latent_result, block_step)
            latent_result = self.normalize_ffn(latent_result, block_step)
        return latent_result


    def attention(self, X, block_step):
        """
        Args:
            X:  [batch_size, seq_len, d_model]

        Returns:
            Decoded output: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = X.shape

        # Three linear projections, [batch_size, seq_len, d_model]
        # Same mathematically for multi-head attention
        Q = X @ self.Q[block_step]
        K = X @ self.K[block_step]
        V = X @ self.V[block_step]

        # Projections per-head start
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1,2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1,2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1,2)
       
        # Compute attention scores with 
        # Need to make the dimensions match: 
        # Q [batch_size, seq_len, d_model] * K [batch, d_model, seq_len]

        # With multi-head attention, this becomes
        # Q @ K.T: [batch, n_heads, seq, d_head] @ [batch, n_heads, d_head, seq]
        #        → [batch, n_heads, seq, seq]
        scores = self.stabilize_scores(Q @ K.transpose(-2, -1))
        # This removes tokens from seeing the "future"
        # We generated a global mask once, then slice the matrix for current seq_len
        scores += self.causal_mask[:seq_len, :seq_len]
 
        # Compute (token, head) probabilities using softmax 
        # How relevant is each query to each key? 
        probabilities = F.softmax(scores, dim=-1)

        # Considering the probabilities, what knowledge should be retrieved from the values?
        # [batch, seq_len, d_model]
        # probabilities @ V: [batch, n_heads, seq, seq] @ [batch, n_heads, seq, d_head]
        #          → [batch, n_heads, seq, d_head]
        retrieved_knowledge = (probabilities @ V)

        # Next, we need to actually distill the knowledge. Each attention head has produced
        # It's own independent representation, but we need to "mix" it together
        # This step allows for x-head concatiation
        return retrieved_knowledge.transpose(1,2).contiguous().view(batch_size, seq_len, self.d_model) @ self.W_o[block_step]

    
    def feed_forward_network(self, X, block_step):
        """
        Computes a 2-layer fully-connected neural network with activations 
        only after the first layer (in this case, gelu)
        """
        return F.gelu(X @ self.W1[block_step]) @ self.W2[block_step]

    def stabilize_scores(self, scores):
        """
        In multi-head attention, this becomes d_head
        If it was single-head, it would be d_model (math works the same)
        """
        return scores / (self.d_head ** 0.5)
    
    def normalize_attention(self, X, block_step):
        X = F.layer_norm(X, normalized_shape=(self.d_model,))
        return (self.attention_gamma[block_step] * X) + self.attention_beta[block_step]
    
    def normalize_ffn(self, X, block_step):
        X = F.layer_norm(X, normalized_shape=(self.d_model,))
        return (self.ffn_gamma[block_step] * X) + self.ffn_beta[block_step]

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