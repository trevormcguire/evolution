import numpy as np

from engine.nn import NeuralNetworkLayer


class ConcatenateLayer(NeuralNetworkLayer):
    """
    Concatenates outputs from multiple previous layers along axis=1 (feature axis).

    Assumes inputs are 2D arrays: (batch_size, features)
    """
    def __init__(self, name="ConcatenateLayer", next=None):
        super().__init__(d_in=None, d_out=None, alpha=0., name=name, prev=[], next=next)
        self.trainable = False  # No weights

    def forward(self, X_list: list[np.ndarray], inference=False) -> np.ndarray:
        self.Xs = X_list
        return np.concatenate(X_list, axis=1)

    def backward(self, d_out: np.ndarray) -> list[np.ndarray]:
        splits = [x.shape[1] for x in self.Xs]
        return np.split(d_out, np.cumsum(splits)[:-1], axis=1)

    def update_params(self):
        pass  # No weights to update


class Conv1DLayer(NeuralNetworkLayer):
    """
    1D Convolutional Layer

    Parameters
    ----------
        in_channels: int
        out_channels: int
        kernel_size: int
        stride: int
        padding: int
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        alpha: float = 0.01,
        theta: float = 0.0,
        name: str = "Conv1DLayer",
        prev=None,
        next=None
    ):
        super().__init__(d_in=None, d_out=None, alpha=alpha, theta=theta, prev=prev, next=next, name=name)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.kernels = 0.1 * np.random.randn(out_channels, in_channels, kernel_size)
        self.biases = np.zeros((out_channels, 1))
        self.trainable = True

    def forward(self, X: np.ndarray, inference=False) -> np.ndarray:
        # X shape: (batch, channels, length)
        self.X = X
        b, c, l = X.shape
        k = self.kernel_size
        s = self.stride
        p = self.padding

        # Output sequence length after convolution
        l_out = (l + 2*p - k) // s + 1
        # Pad input along the time dimension
        X_padded = np.pad(X, ((0, 0), (0, 0), (p, p)), mode="constant")
        self.X_padded = X_padded
        out = np.zeros((b, self.out_channels, l_out))

        # Loop over batch, filters, and sequence positions
        for i in range(b):
            for oc in range(self.out_channels):
                for j in range(l_out):
                    start = j * s
                    region = X_padded[i, :, start:start+k]
                    out[i, oc, j] = np.sum(region * self.kernels[oc]) + self.biases[oc]
        return out

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        X = self.X
        X_p = self.X_padded
        b, c, l = X.shape
        k = self.kernel_size
        s = self.stride
        p = self.padding
        l_out = d_out.shape[2]

        dX_p = np.zeros_like(X_p)
        dK = np.zeros_like(self.kernels)
        dB = np.zeros_like(self.biases)

        # Loop over batch, filters, and output positions
        for i in range(b):
            for oc in range(self.out_channels):
                for j in range(l_out):
                    start = j * s
                    region = X_p[i, :, start:start+k]
                    dK[oc] += region * d_out[i, oc, j]
                    dB[oc] += d_out[i, oc, j]
                    dX_p[i, :, start:start+k] += self.kernels[oc] * d_out[i, oc, j]

        # Normalize gradients by batch size
        self.d_kernels = dK / b
        self.d_biases = dB / b

        # Remove padding from gradient w.r.t. input
        if p > 0:
            dX = dX_p[:, :, p:-p]
        else:
            dX = dX_p
        return dX

    def update_params(self):
        self.kernels -= self.alpha * self.d_kernels
        self.biases -= self.alpha * self.d_biases
        self._alpha *= (1. - self.theta)



class Conv2DLayer(NeuralNetworkLayer):
    """
    2D Convolutional Layer.

    Parameters
    ----------
        in_channels: int
            Number of input channels (e.g., 3 for RGB).
        out_channels: int
            Number of output channels (filters).
        kernel_size: int
            Size of the convolutional kernel (assumes square).
        stride: int
            Stride of the convolution.
        padding: int
            Padding added to the input.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        alpha: float = 0.01,
        theta: float = 0.0,
        name: str = "Conv2DLayer",
        prev=None,
        next=None
    ):
        super().__init__(d_in=None, d_out=None, alpha=alpha, theta=theta, prev=prev, next=next, name=name)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.kernels = 0.1 * np.random.randn(out_channels, in_channels, kernel_size, kernel_size)
        self.biases = np.zeros((out_channels, 1))

        self.trainable = True

    def forward(self, X: np.ndarray, inference=False) -> np.ndarray:
        """
        Parameters
        ----------
        X : shape (batch_size, in_channels, height, width)

        Returns
        -------
        Output : shape (batch_size, out_channels, H_out, W_out)
        """
        self.X = X  # Cache for backward

        batch_size, _, h_in, w_in = X.shape
        k = self.kernel_size
        s = self.stride
        p = self.padding

        h_out = (h_in + 2 * p - k) // s + 1
        w_out = (w_in + 2 * p - k) // s + 1

        # Pad input
        X_padded = np.pad(X, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

        self.X_padded = X_padded
        output = np.zeros((batch_size, self.out_channels, h_out, w_out))

        for b in range(batch_size):
            for oc in range(self.out_channels):
                for i in range(h_out):
                    for j in range(w_out):
                        region = X_padded[b, :, i*s:i*s+k, j*s:j*s+k]
                        output[b, oc, i, j] = np.sum(region * self.kernels[oc]) + self.biases[oc]

        return output

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        d_out : shape (batch_size, out_channels, H_out, W_out)

        Returns
        -------
        d_input : shape (batch_size, in_channels, height, width)
        """
        X = self.X
        X_padded = self.X_padded
        batch_size, _, h_in, w_in = X.shape
        k = self.kernel_size
        s = self.stride
        p = self.padding

        h_out = (h_in + 2 * p - k) // s + 1
        w_out = (w_in + 2 * p - k) // s + 1

        dX_padded = np.zeros_like(X_padded)
        dKernels = np.zeros_like(self.kernels)
        dBiases = np.zeros_like(self.biases)

        for b in range(batch_size):
            for oc in range(self.out_channels):
                for i in range(h_out):
                    for j in range(w_out):
                        i0, j0 = i * s, j * s
                        region = X_padded[b, :, i0:i0+k, j0:j0+k]

                        dKernels[oc] += region * d_out[b, oc, i, j]
                        dBiases[oc] += d_out[b, oc, i, j]
                        dX_padded[b, :, i0:i0+k, j0:j0+k] += self.kernels[oc] * d_out[b, oc, i, j]

        if self.trainable:
            self.d_kernels = dKernels / batch_size
            self.d_biases = dBiases / batch_size

        # Remove padding from dX
        if p > 0:
            dX = dX_padded[:, :, p:-p, p:-p]
        else:
            dX = dX_padded

        return dX

    def update_params(self):
        self.kernels -= self.alpha * self.d_kernels
        self.biases -= self.alpha * self.d_biases
        self._alpha *= (1. - self.theta)
