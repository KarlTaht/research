from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ..base import BaseLanguageModel


class CustomTransformer:

    class BackpropCache:
        def __init__(self):
            self.activations = {}
            self.gradients = {}

        def store_activation(self, *args):
            """Last arg is value, everything before is the key tuple"""
            key = args[:-1] if len(args) > 2 else args[0]
            value = args[-1]
            self.activations[key] = value

        def get_activation(self, *args):
            """All args form the key tuple"""
            key = args if len(args) > 1 else args[0]
            return self.activations[key]

        def store_gradient(self, *args):
            """Last arg is value, everything before is the key tuple"""
            key = args[:-1] if len(args) > 2 else args[0]
            value = args[-1]
            self.gradients[key] = value

        def get_gradient(self, *args):
            key = args if len(args) > 1 else args[0]
            return self.gradients[key]

    
    def __init__(self, config):
        """
        Initialize model with specified layer sizes.

        Args:
            config: Configuration object or dict with optional keys:
                - vocab_size (default: 128)
                - max_seq_len (default: 128)
                - n_blocks (default: 8)
                - n_heads (default: 4)
                - d_model (default: 128)
                - d_ffn (default: 128)
                - device (default: auto-detect)
                - dtype (default: bfloat16)
        """
        # Support both dict-style and object-style config access
        def get_config(key, default):
            if hasattr(config, 'get'):
                return config.get(key, default)
            return getattr(config, key, default)

        self.vocab_size = get_config('vocab_size', 128)
        self.max_seq_len = get_config('max_seq_len', 128)

        self.n_blocks = get_config('n_blocks', 8)
        self.n_heads = get_config('n_heads', 4)
        self.d_model = get_config('d_model', 128)
        self.d_ffn = get_config('d_ffn', 128)

        self.d_head = self.d_model // self.n_heads

        self.device = self._resolve_device(config)
        self.dtype = get_config('dtype', None) or torch.bfloat16

        self.cache = self.BackpropCache()

        self._initialize_weights_and_biases()
        self._init_causal_mask()

    @staticmethod
    def _resolve_device(config) -> str:
        # Support both dict-style and object-style config access
        device = None
        if hasattr(config, 'get'):
            device = config.get('device', None)
        elif hasattr(config, 'get_device'):
            device = config.get_device()
        elif hasattr(config, 'device'):
            device = config.device

        if device:
            return device
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
        
        # Step 5: Final Output projecting back to vocabulary
        logits = latent_result @ self.output_projection

        return logits

    def backward(self, loss_gradient):
        """
        Backward pass to compute gradients. 
        
        In back propagation, we need to be able to compute the "sensitivity" of
        how each input maps to each output so we can make the necessary corrections
        in our computation. 

        Internally, the network knows it's computations, so we can compute the sensitivies
        via the derivatives corresponding to that computation. 

        However, to start the backpropagation process we need to compute the sensitivity
        with respect to the final outputs. The sensisitivity in this case depends on what
        our target objective is. Most generally, that "objective" is to minimize the loss
        and depends on the loss function itself (MSE, Cross-entropy). By passing 
        in the loss gradient we are able to separate how we define the objective and 
        measure success from the internal mechanics of the network.

        Args:
            loss_gradient: dL/dLogits
            "How sensitive is loss with respect to each output?"
            -> [batch_size, seq_len, vocab_size]

        Returns:
            Nothing
        """

        # Starting from the "back" of the network, the input to the output projection
        # was the output of the last transformer block, the decoder's latent_result
        output_projection_input = self.cache.get_activation('latent_result')

        # Weight Gradient for output_projection weight matrix
        # dL/dW - Gradient of each weight, conceptually the weight's contribution to the output
        # dL/dW = Input activations * output loss sensitivity 
        # dL/dW - Input @ dL/dY  [d_model, vocab_size]
        self.cache.store_gradient(
            'output_projection', 
            (output_projection_input.transpose(-2, -1) @ loss_gradient).sum(dim=0)
        )

        # Gradient to pass backwards 
        # How should the upstream layer have changed its output? 
        # dL/dX = dL/dY @ W.T [batch_size, seq_len, d_model]
        # derivative of the loss wrt the output project's input (decoder output)
        grad = loss_gradient @ self.output_projection.T

        # Next, backpropgate through attention blocks
        for block_step in range(self.n_blocks -1, -1, -1):

            # === Backward through FFN layer norm ===
            # Forward was: ffn_out = normalize_ffn(ffn_pre_norm)
            grad_ffn_pre_norm = self.backward_normalize_ffn(grad, block_step)

            self.cache.store_gradient(
                (block_step, 'ffn_pre_norm'),
                grad_ffn_pre_norm
            )

            # === Backward through residual: ffn_pre_norm = ffn_input + ffn_res ===
            # Gradient splits to both paths
            grad_ffn_res = grad_ffn_pre_norm     # Goes into FFN
            grad_ffn_input_skip = grad_ffn_pre_norm   # Skip connection

            grad_ffn_input = self.backward_ffn(grad_ffn_res, block_step)

            # Combine skip and FFN gradients
            grad_post_attn_norm = grad_ffn_input_skip + grad_ffn_input

            # === Backward through attention layer norm ===
            # Forward was: post_attn_norm = normalize_attention(attn_pre_norm)
            grad_attn_pre_norm = self.backward_normalize_attention(grad_post_attn_norm, block_step)

            # === Backward through residual: attn_pre_norm = block_input + attn_res ===
            grad_attn_res = grad_attn_pre_norm  # Goes into Attention
            grad_skip_attn = grad_attn_pre_norm     # Skip connection

            grad_attn_input = self.backward_attention(grad_attn_res, block_step)

            # Combine skip and attention gradients
            grad = grad_skip_attn + grad_attn_input

        self.backward_embed(grad)

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
        self.cache.store_activation('token_indices', batched_tokens)
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

    def backward_embed(self, grad):
        # Vocabulary embedding 
        token_incides = self.cache.get_activation('token_indices')
        grad_vocab_embedding = torch.zeros_like(self.vocab_embedding)
        grad_vocab_embedding.index_add_(
            0,
            token_incides.view(-1),
            grad.view(-1, self.d_model)
        )
        self.cache.store_gradient(
            'vocab_embedding',
            grad_vocab_embedding
        )

        # Positional embedding
        # dL / dP needs to accumulate across the batch
        # pos_embedding[:seq_len] was broadcast across batch
        seq_len = grad.shape[1]
        grad_pos_embedding = torch.zeros_like(self.pos_embedding)
        grad_pos_embedding[:seq_len] = grad.sum(dim=0) # [seq_len, d_model]
        self.cache.store_gradient(
            'pos_embedding',
            grad_pos_embedding
        )


    def decoder(self, X):
        """
        Compute attention, then compute FFN. 
        Layer normalization forces scales the values such that mean=0, std=1
        """
        latent_result = X

        # Each block receives the previous blocks output 
        for block_step in range(self.n_blocks):
            # Cache input to block
            self.cache.store_activation('block', block_step, 'input', latent_result)

            # Cache attention, pre-norm latent result, then normalize
            attn_res = self.attention(latent_result, block_step)
            self.cache.store_activation('block', block_step, 'attn_res', attn_res)
            latent_result += attn_res
            self.cache.store_activation('block', block_step, 'attn_pre_norm', latent_result)
            latent_result = self.normalize_attention(latent_result, block_step)

            # Cache pre-normalization ffn, then normalize
            self.cache.store_activation('block', block_step, 'ffn_input', latent_result)
            ffn_res = self.feed_forward_network(latent_result, block_step)
            self.cache.store_activation('block', block_step, 'ffn_res', ffn_res)
            latent_result += ffn_res
            self.cache.store_activation('block', block_step, 'ffn_pre_norm', latent_result)
            latent_result = self.normalize_ffn(latent_result, block_step)

        self.cache.store_activation('latent_result', latent_result)
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
       
        # Note: Caching happens after re-shape
        self.cache.store_activation('block', block_step, 'Q', Q)
        self.cache.store_activation('block', block_step, 'K', K)
        self.cache.store_activation('block', block_step, 'V', V)
       
        # Compute attention scores with w/ multi-headed attention
        # Q [batch_size, seq_len, d_model] * K [batch, d_model, seq_len]
        # Need to make the dimensions match: 
        # Q @ K.T: [batch, n_heads, seq, d_head] @ [batch, n_heads, d_head, seq]
        #        → [batch, n_heads, seq, seq]
        scores = self.stabilize_scores(Q @ K.transpose(-2, -1))
        # This removes tokens from seeing the "future"
        # We generated a global mask once, then slice the matrix for current seq_len
        scores += self.causal_mask[:seq_len, :seq_len]
 
        # Compute (token, head) probabilities using softmax
        # How relevant is each query to each key?
        # PRECISION MIXING: Cast to float32 for numerical stability, clamp to prevent overflow
        if self.dtype == torch.bfloat16:
            scores_fp32 = scores.to(torch.float32)
            scores_fp32 = torch.clamp(scores_fp32, min=-30.0, max=30.0)
            probabilities = F.softmax(scores_fp32, dim=-1).to(self.dtype)
        else:
            probabilities = F.softmax(scores, dim=-1)
        self.cache.store_activation('block', block_step, 'probabilities', probabilities)

        # Considering the probabilities, what knowledge should be retrieved from the values?
        # [batch, seq_len, d_model]
        # probabilities @ V: [batch, n_heads, seq, seq] @ [batch, n_heads, seq, d_head]
        #          → [batch, n_heads, seq, d_head]
        retrieved_knowledge = (probabilities @ V)
        # This step re-combines the separate heads into a single view in memory
        #  → [batch, d_model, seq]
        retrieved_knowledge = retrieved_knowledge.transpose(1,2).contiguous().view(batch_size, seq_len, self.d_model)
        self.cache.store_activation('block', block_step, 'retrieved_knowledge', retrieved_knowledge)

        # Next, we need to actually distill the knowledge. Each attention head has produced
        # It's own independent representation, but we need to "mix" it together
        # This step allows for x-head concatiation
        return retrieved_knowledge  @ self.W_o[block_step]
    
    def backward_attention(self, grad, block_step):
        batch_size, seq_len, _ = grad.shape
        retrieved_knowledge = self.cache.get_activation(
            'block', block_step, 'retrieved_knowledge'
        )

        # === 7. Backward through W_o
        # dL / dW - Loss gradient with respect to weights
        # = input (retrieved knowledge) * loss_gradient
        # retrieved_knowledge: [batch, d_model, seq_len] -> t -> [batch, seq_len, d_model]
        # grad: [batch_size, seq_len, d_model]
        self.cache.store_gradient(
            ('W_o', block_step),
            (retrieved_knowledge.transpose(-2,-1) @ grad).sum(dim=0)
        )

        # === 6. Backward through reshaped retrieved knowledge
        # Backwards through W_o, then unpack attention heads
        grad_retrieved_knowledge_concat = grad @ self.W_o[block_step].T
        grad_retrieved_knowledge = grad_retrieved_knowledge_concat.view(
            batch_size, seq_len, self.n_heads, self.d_head
        ).transpose(1,2) # [batch, n_heads, seq_len, d_head]

        # === 5. Backward through probabilities @ V
        # "R = prob @ V (let R be "retrieved knowledge")
        # dL / dP = dR / dL @ V.T
        # dL / dV = prob.T @ dR / dL
        probabilities = self.cache.get_activation('block', block_step, 'probabilities')
        V_act = self.cache.get_activation('block', block_step, 'V')
        # dL / dP = loss wrt. probability (what did probability interact with?)
        grad_probabilities = grad_retrieved_knowledge @ V_act.transpose(-2,-1)
        # dL / dV = loss wrt. V (what did V interact with?)
        grad_V = probabilities.transpose(-2,-1) @ grad_retrieved_knowledge  

        # === 4. Backward through softmax
        grad_scores = self.backward_softmax(grad_probabilities, probabilities)

        # === 3. Backward through mask
        # Mask is a constant, so the derivative is 1 (passes straight through)

        # === 2. Backward through scaling (stabilizing the scores)
        # (Q @ K.T) / sqrt(d_head)
        grad_scores = grad_scores / (self.d_head ** 0.5)

        # === 1C. Backward through Q @ K.T
        # Get cached values, # [n_head, seq_len, d_model]
        # We already got V_act
        Q_act = self.cache.get_activation('block', block_step, 'Q') 
        K_act = self.cache.get_activation('block', block_step, 'K')
        # scores = Q @ K.T
        # dL / dK = dL/dS * K
        # dL / dQ = Q.T @ dL/dS
        grad_K = grad_scores @ K_act
        grad_Q = grad_scores.transpose(-2,-1) @ Q_act

        # === 1B. Backward through re-shape QKV
        grad_Q = grad_Q.transpose(1,2).contiguous().view(batch_size, seq_len, self.d_model)
        grad_K = grad_K.transpose(1,2).contiguous().view(batch_size, seq_len, self.d_model)
        grad_V = grad_V.transpose(1,2).contiguous().view(batch_size, seq_len, self.d_model)

        # === 1A. Backward through QKV
        attn_input = self.cache.get_activation('block', block_step, 'input')
        self.cache.store_gradient(
            ('W_Q', block_step),
            (attn_input.transpose(-2, -1) @ grad_Q).sum(dim=0)
        )
        self.cache.store_gradient(
            ('W_K', block_step),
            (attn_input.transpose(-2, -1) @ grad_K).sum(dim=0)
        )
        self.cache.store_gradient(
            ('W_V', block_step),
            (attn_input.transpose(-2, -1) @ grad_V).sum(dim=0)
        )

        # Combined Gradients
        return (
            grad_Q @ self.Q[block_step].T +
            grad_K @ self.K[block_step].T +
            grad_V @ self.V[block_step].T
        )

    def backward_softmax(self, grad_prob, prob):
        """
        Args:
            grad_prob: dL/dY [batch, n_heads, seq_len, seq_len]
            prob: [batch, n_heads, seq_len, seq_len]

        Returns
            
            dL/dX : [batch, n_heads, seq_len, seq_len]
        """

        summation = (grad_prob * prob).sum(dim=-1, keepdim=True)
        return prob * (grad_prob - summation)

    def feed_forward_network(self, X, block_step):
        """
        Computes a 2-layer fully-connected neural network with activations 
        only after the first layer (in this case, gelu)
        """
        # Input is cached a layer up 
        # self.cache.store_activation('ffn_input', block_step, X)

        pre_activation = X @ self.W1[block_step]
        self.cache.store_activation('block', block_step, 'ffn_pre_activation', pre_activation)

        post_activation = F.gelu(pre_activation)
        self.cache.store_activation('block', block_step, 'ffn_post_activation', post_activation)

        # Output is already cached
        # self.cache.store_activation('block', block_step, 'ffn_res', ffn_res)
        return post_activation @ self.W2[block_step]
    
    def backward_ffn(self, grad, block_step):
        """
        Backwards pass through the FFN. Since it's only a 2-layer network, 
        we don't use a loop. If you wanted a multi-layered FFN, you'd need to
        re-write this function
        """

        ffn_input = self.cache.get_activation('block', block_step, 'ffn_input')
        pre_activation = self.cache.get_activation(
            'block', block_step, 'ffn_pre_activation'
        )
        post_activation = self.cache.get_activation(
            'block', block_step, 'ffn_post_activation'
        )

        # Backward pass through W2 (no GeLU)
        # dL / dW = input * loss_gradient
        self.cache.store_gradient(
            ('W2', block_step),
            (post_activation.transpose(-2, -1) @ grad).sum(dim=0)
        )

        # Propagate through the activations
        # dL/dX (post-activation) = grad * W2
        grad_post_activation = grad @ self.W2[block_step].T
        # dL / dX (pre-activiation) = dL / dX (post-activtion) * gelu'(pre-activation)
        grad_pre_activation = grad_post_activation * self.gelu_derivative(pre_activation)

        # Backwards pass through W1
        # dL / dW = input * pre-activation loss gradient
        self.cache.store_gradient(
            ('W1', block_step),
            (ffn_input.transpose(-2, -1) @ grad_pre_activation).sum(dim=0)
        )

        # dL / dX (ffn input) = grad_pre_activation * W1
        grad_input = grad_pre_activation @ self.W1[block_step].T
        return grad_input

    def stabilize_scores(self, scores):
        """
        In multi-head attention, this becomes d_head
        If it was single-head, it would be d_model (math works the same)
        """
        return scores / (self.d_head ** 0.5)
    
    def normalize_attention(self, X, block_step):
        # PRECISION MIXING: Perform layer norm in float32 for numerical stability
        eps = 1e-5
        if self.dtype == torch.bfloat16:
            X_fp32 = X.to(torch.float32)
            mean = X_fp32.mean(dim=-1, keepdim=True)
            var = X_fp32.var(dim=-1, keepdim=True, unbiased=False)
            std = torch.sqrt(var + eps)
            X_norm = ((X_fp32 - mean) / std).to(self.dtype)
            std = std.to(self.dtype)
        else:
            mean = X.mean(dim=-1, keepdim=True)
            var = X.var(dim=-1, keepdim=True, unbiased=False)
            std = torch.sqrt(var + eps)
            X_norm = (X - mean) / std
        # Cache normalized value and std for backward pass
        self.cache.store_activation('block', block_step, 'attn_x_norm', X_norm)
        self.cache.store_activation('block', block_step, 'attn_std', std)
        return (self.attention_gamma[block_step] * X_norm) + self.attention_beta[block_step]

    def normalize_ffn(self, X, block_step):
        # PRECISION MIXING: Perform layer norm in float32 for numerical stability
        eps = 1e-5
        if self.dtype == torch.bfloat16:
            X_fp32 = X.to(torch.float32)
            mean = X_fp32.mean(dim=-1, keepdim=True)
            var = X_fp32.var(dim=-1, keepdim=True, unbiased=False)
            std = torch.sqrt(var + eps)
            X_norm = ((X_fp32 - mean) / std).to(self.dtype)
            std = std.to(self.dtype)
        else:
            mean = X.mean(dim=-1, keepdim=True)
            var = X.var(dim=-1, keepdim=True, unbiased=False)
            std = torch.sqrt(var + eps)
            X_norm = (X - mean) / std
        # Cache normalized value and std for backward pass
        self.cache.store_activation('block', block_step, 'ffn_x_norm', X_norm)
        self.cache.store_activation('block', block_step, 'ffn_std', std)
        return (self.ffn_gamma[block_step] * X_norm) + self.ffn_beta[block_step]

    def backward_normalize_ffn(self, grad, block_step):
        """Backward pass through FFN layer normalization."""
        # Get cached values
        x_norm = self.cache.get_activation('block', block_step, 'ffn_x_norm')
        std = self.cache.get_activation('block', block_step, 'ffn_std')
        gamma = self.ffn_gamma[block_step]

        # Gradient through affine transform: y = gamma * x_norm + beta
        # dL/dgamma = sum(dL/dy * x_norm) over batch and seq dimensions
        self.cache.store_gradient(
            ('ffn_gamma', block_step),
            (grad * x_norm).sum(dim=(0, 1))  # [d_model]
        )
        # dL/dbeta = sum(dL/dy)
        self.cache.store_gradient(
            ('ffn_beta', block_step),
            grad.sum(dim=(0, 1))  # [d_model]
        )

        # Gradient through layer norm: x_norm = (x - mean) / std
        # dL/dx_norm = dL/dy * gamma
        grad_x_norm = grad * gamma

        # Compute in float32 for numerical stability
        if self.dtype == torch.bfloat16:
            grad_x_norm = grad_x_norm.to(torch.float32)
            x_norm = x_norm.to(torch.float32)
            std = std.to(torch.float32)

        # Layer norm backward formula:
        # dL/dx = (1/std) * (dL/dx_norm - mean(dL/dx_norm) - x_norm * mean(dL/dx_norm * x_norm))
        mean_grad = grad_x_norm.mean(dim=-1, keepdim=True)
        mean_grad_x = (grad_x_norm * x_norm).mean(dim=-1, keepdim=True)
        grad_input = (1.0 / std) * (grad_x_norm - mean_grad - x_norm * mean_grad_x)

        if self.dtype == torch.bfloat16:
            grad_input = grad_input.to(self.dtype)

        return grad_input

    def backward_normalize_attention(self, grad, block_step):
        """Backward pass through attention layer normalization."""
        # Get cached values
        x_norm = self.cache.get_activation('block', block_step, 'attn_x_norm')
        std = self.cache.get_activation('block', block_step, 'attn_std')
        gamma = self.attention_gamma[block_step]

        # Gradient through affine transform: y = gamma * x_norm + beta
        self.cache.store_gradient(
            ('attention_gamma', block_step),
            (grad * x_norm).sum(dim=(0, 1))  # [d_model]
        )
        self.cache.store_gradient(
            ('attention_beta', block_step),
            grad.sum(dim=(0, 1))  # [d_model]
        )

        # Gradient through layer norm
        grad_x_norm = grad * gamma

        # Compute in float32 for numerical stability
        if self.dtype == torch.bfloat16:
            grad_x_norm = grad_x_norm.to(torch.float32)
            x_norm = x_norm.to(torch.float32)
            std = std.to(torch.float32)

        # Layer norm backward formula
        mean_grad = grad_x_norm.mean(dim=-1, keepdim=True)
        mean_grad_x = (grad_x_norm * x_norm).mean(dim=-1, keepdim=True)
        grad_input = (1.0 / std) * (grad_x_norm - mean_grad - x_norm * mean_grad_x)

        if self.dtype == torch.bfloat16:
            grad_input = grad_input.to(self.dtype)

        return grad_input

    def update_parameters(self, learning_rate):
        """
        Update weights / biases using gradient
        """
        lr = learning_rate

        self.output_projection -= lr * self.cache.get_gradient('output_projection')

        for block_step in range(self.n_blocks):
            # FFN Normalization
            self.ffn_gamma[block_step] -= lr * self.cache.get_gradient(('ffn_gamma', block_step))
            self.ffn_beta[block_step] -= lr * self.cache.get_gradient(('ffn_beta', block_step))

            # FFN
            self.W1[block_step] -= lr * self.cache.get_gradient(('W1', block_step))
            self.W2[block_step] -= lr * self.cache.get_gradient(('W2', block_step))

            # Attention Normalization
            self.attention_gamma[block_step] -= lr * self.cache.get_gradient(('attention_gamma', block_step))
            self.attention_beta[block_step] -= lr * self.cache.get_gradient(('attention_beta', block_step))

            # Attention Output
            self.W_o[block_step] -= lr * self.cache.get_gradient(('W_o', block_step))

            # QKV
            self.Q[block_step] -= lr * self.cache.get_gradient(('W_Q', block_step))
            self.K[block_step] -= lr * self.cache.get_gradient(('W_K', block_step))
            self.V[block_step] -= lr * self.cache.get_gradient(('W_V', block_step))

        # Embedding
        self.vocab_embedding -= lr * self.cache.get_gradient('vocab_embedding')
        self.pos_embedding -= lr * self.cache.get_gradient('pos_embedding')

    # === Serialization ===
    def state_dict(self) -> dict:
        """
        Return all model parameters as a dictionary.

        Similar to nn.Module.state_dict() but for raw tensors.
        All tensors are detached and moved to CPU for safe serialization.

        Returns:
            Dictionary mapping parameter names to tensors
        """
        return {
            # Embeddings
            'vocab_embedding': self.vocab_embedding.detach().cpu(),
            'pos_embedding': self.pos_embedding.detach().cpu(),
            # Attention weights [n_blocks, d_model, d_model]
            'Q': self.Q.detach().cpu(),
            'K': self.K.detach().cpu(),
            'V': self.V.detach().cpu(),
            'W_o': self.W_o.detach().cpu(),
            # Attention layer norm [n_blocks, d_model]
            'attention_gamma': self.attention_gamma.detach().cpu(),
            'attention_beta': self.attention_beta.detach().cpu(),
            # FFN weights [n_blocks, ...]
            'W1': self.W1.detach().cpu(),
            'W2': self.W2.detach().cpu(),
            # FFN layer norm [n_blocks, d_model]
            'ffn_gamma': self.ffn_gamma.detach().cpu(),
            'ffn_beta': self.ffn_beta.detach().cpu(),
            # Output projection [d_model, vocab_size]
            'output_projection': self.output_projection.detach().cpu(),
        }

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        """
        Load model parameters from a state dictionary.

        Args:
            state_dict: Dictionary mapping parameter names to tensors
            strict: If True, raise error on missing/unexpected keys

        Returns:
            Tuple of (missing_keys, unexpected_keys)
        """
        expected_keys = {
            'vocab_embedding', 'pos_embedding',
            'Q', 'K', 'V', 'W_o',
            'attention_gamma', 'attention_beta',
            'W1', 'W2',
            'ffn_gamma', 'ffn_beta',
            'output_projection',
        }

        missing_keys = expected_keys - set(state_dict.keys())
        unexpected_keys = set(state_dict.keys()) - expected_keys

        if strict and missing_keys:
            raise KeyError(f"Missing keys in state_dict: {missing_keys}")
        if strict and unexpected_keys:
            raise KeyError(f"Unexpected keys in state_dict: {unexpected_keys}")

        # Load each parameter, moving to correct device and dtype
        if 'vocab_embedding' in state_dict:
            self.vocab_embedding = state_dict['vocab_embedding'].to(self.device, self.dtype)
        if 'pos_embedding' in state_dict:
            self.pos_embedding = state_dict['pos_embedding'].to(self.device, self.dtype)
        if 'Q' in state_dict:
            self.Q = state_dict['Q'].to(self.device, self.dtype)
        if 'K' in state_dict:
            self.K = state_dict['K'].to(self.device, self.dtype)
        if 'V' in state_dict:
            self.V = state_dict['V'].to(self.device, self.dtype)
        if 'W_o' in state_dict:
            self.W_o = state_dict['W_o'].to(self.device, self.dtype)
        if 'attention_gamma' in state_dict:
            self.attention_gamma = state_dict['attention_gamma'].to(self.device, self.dtype)
        if 'attention_beta' in state_dict:
            self.attention_beta = state_dict['attention_beta'].to(self.device, self.dtype)
        if 'W1' in state_dict:
            self.W1 = state_dict['W1'].to(self.device, self.dtype)
        if 'W2' in state_dict:
            self.W2 = state_dict['W2'].to(self.device, self.dtype)
        if 'ffn_gamma' in state_dict:
            self.ffn_gamma = state_dict['ffn_gamma'].to(self.device, self.dtype)
        if 'ffn_beta' in state_dict:
            self.ffn_beta = state_dict['ffn_beta'].to(self.device, self.dtype)
        if 'output_projection' in state_dict:
            self.output_projection = state_dict['output_projection'].to(self.device, self.dtype)

        return missing_keys, unexpected_keys

    def get_config(self) -> dict:
        """Return model configuration as dictionary."""
        return {
            'vocab_size': self.vocab_size,
            'max_seq_len': self.max_seq_len,
            'n_blocks': self.n_blocks,
            'n_heads': self.n_heads,
            'd_model': self.d_model,
            'd_ffn': self.d_ffn,
            'd_head': self.d_head,
            'device': self.device,
            'dtype': str(self.dtype),  # Store as string for serialization
        }

    def save_checkpoint(self, path: str, **metadata):
        """
        Save model checkpoint to file.

        The checkpoint contains:
        - state_dict: All model parameters
        - config: Model architecture configuration
        - metadata: Any additional info (epoch, loss, etc.)

        Args:
            path: Path to save checkpoint
            **metadata: Additional metadata to save (epoch, loss, optimizer_state, etc.)

        Example:
            model.save_checkpoint('checkpoint.pt', epoch=5, train_loss=1.23)
        """
        checkpoint = {
            'state_dict': self.state_dict(),
            'config': self.get_config(),
            'metadata': metadata,
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")

    @classmethod
    def from_checkpoint(cls, path: str, config=None, device: str = None):
        """
        Load model from checkpoint file.

        Args:
            path: Path to checkpoint file
            config: Optional config override (uses saved config if None)
            device: Device to load model onto (auto-detect if None)

        Returns:
            Tuple of (model, metadata)

        Example:
            model, meta = CustomTransformer.from_checkpoint('checkpoint.pt')
            print(f"Loaded from epoch {meta.get('epoch')}")
        """
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)

        # Use saved config or override
        saved_config = checkpoint.get('config', {})
        if config is None:
            # Create a config dict from saved values
            config = saved_config.copy()
            if device:
                config['device'] = device

        # Create model with config
        model = cls(config)

        # Load state
        model.load_state_dict(checkpoint['state_dict'])

        print(f"Checkpoint loaded: {path}")
        return model, checkpoint.get('metadata', {})

    # === Visualization Support ===
    def get_state(self):
        """
        Return model state for visualization.

        Returns:
            Dictionary containing weights, activations, gradients, architecture
        """
        return {
            'weights': self.state_dict(),
            'activations': dict(self.cache.activations),
            'gradients': dict(self.cache.gradients),
            'config': self.get_config(),
        }

    def get_layer_info(self, layer_idx):
        """Get detailed info about a specific layer/block."""
        if layer_idx < 0 or layer_idx >= self.n_blocks:
            raise IndexError(f"Block index {layer_idx} out of range [0, {self.n_blocks})")

        return {
            'block_idx': layer_idx,
            'attention': {
                'Q': self.Q[layer_idx],
                'K': self.K[layer_idx],
                'V': self.V[layer_idx],
                'W_o': self.W_o[layer_idx],
                'gamma': self.attention_gamma[layer_idx],
                'beta': self.attention_beta[layer_idx],
            },
            'ffn': {
                'W1': self.W1[layer_idx],
                'W2': self.W2[layer_idx],
                'gamma': self.ffn_gamma[layer_idx],
                'beta': self.ffn_beta[layer_idx],
            },
        }

    # === Math Functions (to be refactored) ===
    def gelu_derivative(self, x):
        """Derivative of GELU activation function."""
        # Approximate derivative using the tanh formulation
        sqrt_2_pi = math.sqrt(2 / math.pi)
        cubic_term = 0.044715 * x**3
        tanh_arg = sqrt_2_pi * (x + cubic_term)
        tanh_out = torch.tanh(tanh_arg)
        
        # d/dx[0.5 * x * (1 + tanh(...))]
        sech2 = 1 - tanh_out**2
        dtanh = sqrt_2_pi * (1 + 3 * 0.044715 * x**2) * sech2
        
        return 0.5 * (1 + tanh_out) + 0.5 * x * dtanh