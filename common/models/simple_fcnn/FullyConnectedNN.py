import numpy as np

class FullyConnectedNN:
    """
    A neural network model built from scratch for learning backpropagation.
    Stores weights, activations, gradients, and layer configurations.
    """

    def __init__(self, config):
        """
        Initialize model with specified layer sizes.
        """

        self.input_dimension = config['input_dimension']
        self.hidden_dimension = config['hidden_dimension']
        self.num_hidden_layers = config['hidden_layers']
        self.output_dimension = config['output_dimension']

        self.activations = {}
        self.gradients = {}

        self._initialize_weights_and_biases()


    # === Architecture & Initialization ===
    def _initialize_weights_and_biases(self):
        """Initialize weights for all layers. Xavier initialization optimized for ReLU or GELU"""

        # Input layer: Xavier-style initialization
        self.input_layer_weights = np.random.randn(
            self.input_dimension,
            self.hidden_dimension
        ) * np.sqrt(1.0 / self.input_dimension)
        self.input_layer_biases = np.zeros(self.hidden_dimension)

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
    def forward(self, X):
        """
        Forward pass through the network. 

        Args:
            X: Input data

        Returns:
            Output predictions
        """

        # Input layer, Z = pre-activation, h = hidden state
        z1 = X @ self.input_layer_weights + self.input_layer_biases
        h1 = self.gelu(z1)

        # Store the input + first layer state
        self.activations['input'] = X
        self.activations['hidden_0_pre'] = z1  # Pre-activation
        self.activations['hidden_0'] = h1      # Post-activation
    
        # Hidden layers
        h = h1
        for i in range(self.num_hidden_layers):
            z = h @ self.hidden_layers_weights[i] + self.hidden_layer_biases[i]
            h = self.gelu(z)

            # Store the pre+post hidden states
            self.activations[f'hidden_{i+1}_pre'] = z
            self.activations[f'hidden_{i+1}'] = h
    
        # Output layer (no activation - raw logits)
        output = h @ self.output_layer_weights + self.output_layer_biases
        self.activations['output'] = output
    
        return output

    def gelu(self, x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
    
    def gelu_derivative(self, x):
        """Derivative of GELU activation function."""
        # Approximate derivative using the tanh formulation
        sqrt_2_pi = np.sqrt(2 / np.pi)
        cubic_term = 0.044715 * x**3
        tanh_arg = sqrt_2_pi * (x + cubic_term)
        tanh_out = np.tanh(tanh_arg)
        
        # d/dx[0.5 * x * (1 + tanh(...))]
        sech2 = 1 - tanh_out**2
        dtanh = sqrt_2_pi * (1 + 3 * 0.044715 * x**2) * sech2
        
        return 0.5 * (1 + tanh_out) + 0.5 * x * dtanh

    # === Backward Pass ===
    def backward(self, loss_gradient):
        """
        Backward pass to compute gradients.

        Args:
            loss_gradient: Gradient of loss with respect to output activations

        Returns:
            Nothing
        """

        self.gradients = {}

        # Output Layer
        last_hidden = self.activations[f'hidden_{self.num_hidden_layers}']
        self.gradients['output_weights'] = last_hidden.T @ loss_gradient
        self.gradients['output_biases'] = np.sum(loss_gradient, axis=0)
        gradient_flow = loss_gradient @ self.output_layer_weights.T

        # Hidden Layers
        for i in range(self.num_hidden_layers -1, -1, -1):
            # Gradient through GELU
            layer_Z = self.activations[f'hidden_{i+1}_pre']
            dZ = gradient_flow * self.gelu_derivative(layer_Z)

            # Get the previous activation
            A_prev = self.activations[f'hidden_{i}']

            # Compute Gradients
            self.gradients[f'hidden_{i}_weights'] = A_prev.T @ dZ
            self.gradients[f'hidden_{i}_biases'] = np.sum(dZ, axis=0)
            gradient_flow = dZ @ self.hidden_layers_weights[i].T

        # Input Layer Backward
        X = self.activations['input']
        layer_Z = self.activations['hidden_0_pre']
        dZ = gradient_flow * self.gelu_derivative(layer_Z)

        self.gradients['input_weights'] = X.T @ dZ
        self.gradients['input_biases'] = np.sum(dZ, axis=0)

    def compute_gradient_norms(self):
        """
        Compute L2 norms of gradients for each layer.

        Returns:
            Dictionary mapping layer names to L2 norms
        """
        norms = {}

        # Input layer
        if 'input_weights' in self.gradients:
            input_grad = self.gradients['input_weights']
            norms['Input Layer'] = float(np.linalg.norm(input_grad))

        # Hidden layers
        for i in range(self.num_hidden_layers):
            grad_key = f'hidden_{i}_weights'
            if grad_key in self.gradients:
                hidden_grad = self.gradients[grad_key]
                norms[f'Hidden Layer {i+1}'] = float(np.linalg.norm(hidden_grad))

        # Output layer
        if 'output_weights' in self.gradients:
            output_grad = self.gradients['output_weights']
            norms['Output Layer'] = float(np.linalg.norm(output_grad))

        return norms

    # === Parameter Updates ===
    def update_parameters(self, learning_rate):
        """Update weights and biases using computed gradients."""

        # Input Layer
        self.input_layer_weights -= learning_rate * self.gradients['input_weights']
        self.input_layer_biases -= learning_rate * self.gradients['input_biases']

        # Hidden Layers
        for i in range(self.num_hidden_layers):
            self.hidden_layers_weights[i] -= learning_rate * self.gradients[f'hidden_{i}_weights']
            self.hidden_layer_biases[i] -= learning_rate * self.gradients[f'hidden_{i}_biases']

        # Output Layer
        self.output_layer_weights -= learning_rate * self.gradients['output_weights']
        self.output_layer_biases -= learning_rate * self.gradients['output_biases']


    # === Prediction ===
    def predict(self, X, store_activations=False):
        """Make predictions on new data."""
        if store_activations:
            return self.forward(X)

        # Input layer
        z1 = X @ self.input_layer_weights + self.input_layer_biases
        h1 = self.gelu(z1)
        
        # Hidden layers
        h = h1
        for i in range(self.num_hidden_layers):
            z = h @ self.hidden_layers_weights[i] + self.hidden_layer_biases[i]
            h = self.gelu(z)
        
        # Output layer (no activation - raw logits)
        output = h @ self.output_layer_weights + self.output_layer_biases
        return output

    # === Visualization Support ===
    def get_state(self):
        """
        Return model state for visualization.

        Returns:
            Dictionary containing weights, activations, gradients, architecture
        """
        model_state = {
            'weights': {},
            'activations': {},
            'gradients': {}
        }

        # State of network
        model_state['weights']['input_weights'] = self.input_layer_weights
        model_state['weights']['input_biases'] = self.input_layer_biases
        model_state['weights']['hidden_weights'] = self.hidden_layers_weights
        model_state['weights']['hidden_biases'] = self.hidden_layer_biases
        model_state['weights']['output_weights'] = self.output_layer_weights
        model_state['weights']['output_biases'] = self.output_layer_biases

        # State from data processing
        model_state['activations'] = self.activations
        model_state['gradients'] = self.gradients
        
        return model_state

    def get_layer_info(self, layer_idx):
        """Get detailed info about a specific layer."""
        pass