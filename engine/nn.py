"""A Minimal Neural Network Module."""
from __future__ import annotations

import numpy as np


class NeuralNetworkLayer(object):
    """
    Feed-Forward Neural Network Layer

    Parameters
    ----------
        d_in: int
            Number of input features.
        d_out: int
            Number of output features.
        name: str
            Name of the layer.
        alpha: float
            Learning rate for weight updates.
        theta: float
            Learning rate decay rate (1 - theta).
            Set to 0 for no decay.
        prev: NeuralNetworkLayer | None
            Previous layer in the network.
        next: NeuralNetworkLayer | None
            Next layer in the network.
    
    Example
    -------
        >>> lr = 0.01
        >>> layer1 = NeuralNetworkLayer(d_in=10, d_out=5, alpha=lr)
        >>> layer2 = NeuralNetworkLayer(d_in=5, d_out=2, name="output_layer", alpha=lr, prev=layer1)
    """
    def __init__(
        self,
        d_in: int,
        d_out: int,
        alpha: float,
        theta: float = 0.,  # 1 - learning_rate_decay_rate
        prev: NeuralNetworkLayer | list[NeuralNetworkLayer] = None,
        next: NeuralNetworkLayer | list[NeuralNetworkLayer] = None,
        name: str = "NeuralNetworkLayer"
    ):

        # Allow prev/next to be lists for branching/merging
        self.prev = []
        self.next = []
        if prev is not None:
            prev = [prev] if not isinstance(prev, list) else prev
            for p in prev:
                self.add_prev(p)
        if next is not None:
            next = [next] if not isinstance(next, list) else next
            for n in next:
                self.add_next(n)

        self.d_in = d_in
        self.d_out = d_out
        self.w = 0.1 * np.random.randn(self.d_in, self.d_out)
        self.b = np.zeros((1, self.d_out))
        self.dw = None
        self.db = None
        self.trainable = True
        self.inference = False  # switch for inference mode
        self.name = name

        # set learning rate to freeze
        # set to very small value to "soft freeze"
        self._alpha = alpha
        self.theta = theta


    @property
    def alpha(self):
        if self.trainable:
            return self._alpha
        return 0.
    
    def set_alpha(self, rate: float):
        self._alpha = rate
    
    def set_theta(self, rate: float):
        self.theta = rate

    def freeze(self):
        """Freeze the layer's weights."""
        self.trainable = False
    
    def soft_freeze(self, rate: float = 1e-6):
        """Soft freeze the layer's weights by setting a very small learning rate."""
        self.trainable = True
        self.set_alpha(rate)

    def unfreeze(self):
        """Unfreeze the layer's weights."""
        self.trainable = True

    def __repr__(self):
        return f"{self.name}(d_in={self.d_in}, d_out={self.d_out}, trainable={self.trainable})"

    def __call__(self, inputs: np.ndarray):
        return self.forward(inputs)

    def forward(self, X: np.ndarray, inference: bool = False) -> np.ndarray:
        if self.trainable and not self.inference and not inference:
            self.X = X  # cache for backprop
        return np.dot(X, self.w) + self.b

    def backward(self, dvalues: np.ndarray):
        if not self.trainable:
            return dvalues
        self.d_w = np.dot(self.X.T, dvalues) 
        self.d_b = np.sum(dvalues, axis=0, keepdims=True) # Gradient on values
        self.dinputs = np.dot(dvalues, self.w.T)
        return self.dinputs

    def update_params(self):
        # vanilla SGD update
        self.w -= self.alpha * self.d_w
        self.b -= self.alpha * self.d_b
        self._alpha *= (1. - self.theta)
    

    def add_prev(self, prev_layer):
        if prev_layer is not None:
            self.prev.append(prev_layer)

            if self not in prev_layer.next:
                prev_layer.next.append(self)

    def add_next(self, next_layer):
        if next_layer is not None:
            self.next.append(next_layer)
            # Ensure bidirectional link
            if self not in next_layer.prev:
                next_layer.prev.append(self)

    def set_prev(self, prev_layer):
        self.prev = []
        self.add_prev(prev_layer)

    def set_next(self, next_layer):
        self.next = []
        self.add_next(next_layer)

    def get_next(self):
        return self.next

    def get_prev(self):
        return self.prev

    def set_params(self, w: np.ndarray, b: np.ndarray):
        self.w = w
        self.b = b
    
    def mutate(self, scale: float):
        """Mutate the layer's weights and biases in-place."""
        self.w += np.random.normal(0, scale, self.w.shape)
        self.b += np.random.normal(0, scale, self.b.shape)


class MSELoss:
    """
    Mean Squared Error
    -----------------------
    Calculation:
        1. Forward
            MEAN( y[0] - yhat[0]) ** 2 + (y[1] - yhat[1]) ** 2 + ... + (y[n] - yhat[n]) ** 2 )
        2. Backward:
            partial_deriv is:
                (-2(y - yhat))/num_outputs (to normalize)
    -----------------------
    Note:
        axis=-1 tells numpy to calculate mean across outputs, for each sample separately
    """
    @staticmethod
    def forward(yhat: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.mean((y - yhat)**2, axis=-1)

    @staticmethod
    def backward(yhat: np.ndarray, y: np.ndarray):

        num_samples, num_outputs = len(yhat), len(yhat[0])
        dinputs = -2 * (y - yhat) / num_outputs #gradient
        dinputs = dinputs / num_samples #normalize
        return dinputs
