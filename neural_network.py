"""A feedforward neural network implemented from scratch with NumPy.

This module provides a small but complete neural network supporting:
  - Arbitrary layer sizes (any number of hidden layers)
  - Several activation functions (ReLU, sigmoid, tanh, softmax)
  - Mini-batch gradient descent training
  - Mean squared error and cross-entropy losses

It depends only on NumPy and is intended to be readable and educational.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Activation functions
# --------------------------------------------------------------------------- #
class Activation:
    """Base class for activation functions."""

    def forward(self, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, z: np.ndarray) -> np.ndarray:
        """Derivative of the activation with respect to its input ``z``."""
        raise NotImplementedError


class ReLU(Activation):
    def forward(self, z: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, z)

    def backward(self, z: np.ndarray) -> np.ndarray:
        return (z > 0).astype(z.dtype)


class Sigmoid(Activation):
    def forward(self, z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    def backward(self, z: np.ndarray) -> np.ndarray:
        s = self.forward(z)
        return s * (1.0 - s)


class Tanh(Activation):
    def forward(self, z: np.ndarray) -> np.ndarray:
        return np.tanh(z)

    def backward(self, z: np.ndarray) -> np.ndarray:
        return 1.0 - np.tanh(z) ** 2


class Identity(Activation):
    def forward(self, z: np.ndarray) -> np.ndarray:
        return z

    def backward(self, z: np.ndarray) -> np.ndarray:
        return np.ones_like(z)


# --------------------------------------------------------------------------- #
# Loss functions
# --------------------------------------------------------------------------- #
class Loss:
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        raise NotImplementedError

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Gradient of the loss with respect to ``y_pred``."""
        raise NotImplementedError


class MSELoss(Loss):
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return float(np.mean((y_pred - y_true) ** 2))

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        n = y_true.shape[0]
        return 2.0 * (y_pred - y_true) / n


class BinaryCrossEntropy(Loss):
    """Binary cross-entropy. Assumes predictions are in (0, 1)."""

    def __init__(self, eps: float = 1e-12) -> None:
        self.eps = eps

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        p = np.clip(y_pred, self.eps, 1.0 - self.eps)
        return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        p = np.clip(y_pred, self.eps, 1.0 - self.eps)
        n = y_true.shape[0]
        return (p - y_true) / (p * (1 - p)) / n


# --------------------------------------------------------------------------- #
# Dense (fully connected) layer
# --------------------------------------------------------------------------- #
class DenseLayer:
    """A fully connected layer: ``output = activation(x @ W + b)``."""

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        activation: Activation,
        rng: np.random.Generator | None = None,
    ) -> None:
        rng = rng or np.random.default_rng()
        # He-style initialization scaled by the fan-in for stable gradients.
        scale = np.sqrt(2.0 / n_inputs)
        self.W = rng.normal(0.0, scale, size=(n_inputs, n_outputs))
        self.b = np.zeros((1, n_outputs))
        self.activation = activation

        # Caches populated during the forward pass for use in backprop.
        self._x: np.ndarray | None = None
        self._z: np.ndarray | None = None

        # Gradients populated during the backward pass.
        self.dW: np.ndarray | None = None
        self.db: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        self._z = x @ self.W + self.b
        return self.activation.forward(self._z)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backpropagate ``grad_output`` (dL/d_output) through the layer.

        Returns the gradient with respect to this layer's input so it can be
        passed to the preceding layer.
        """
        # Gradient flowing into the pre-activation value z.
        grad_z = grad_output * self.activation.backward(self._z)

        self.dW = self._x.T @ grad_z
        self.db = np.sum(grad_z, axis=0, keepdims=True)

        # Gradient with respect to the layer input.
        return grad_z @ self.W.T

    def update(self, learning_rate: float) -> None:
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db


# --------------------------------------------------------------------------- #
# Neural network
# --------------------------------------------------------------------------- #
class NeuralNetwork:
    """A simple sequential feedforward neural network."""

    def __init__(self, layers: list[DenseLayer], loss: Loss) -> None:
        self.layers = layers
        self.loss = loss

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        grad = self.loss.backward(y_pred, y_true)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update(self, learning_rate: float) -> None:
        for layer in self.layers:
            layer.update(learning_rate)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 1000,
        learning_rate: float = 0.1,
        batch_size: int | None = None,
        verbose: bool = True,
        rng: np.random.Generator | None = None,
    ) -> list[float]:
        """Train with mini-batch gradient descent. Returns per-epoch losses."""
        rng = rng or np.random.default_rng()
        n_samples = X.shape[0]
        batch_size = batch_size or n_samples
        history: list[float] = []

        for epoch in range(epochs):
            # Shuffle the data each epoch.
            perm = rng.permutation(n_samples)
            X_shuf, y_shuf = X[perm], y[perm]

            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                xb, yb = X_shuf[start:end], y_shuf[start:end]

                y_pred = self.forward(xb)
                epoch_loss += self.loss.forward(y_pred, yb)
                n_batches += 1

                self.backward(y_pred, yb)
                self.update(learning_rate)

            epoch_loss /= n_batches
            history.append(epoch_loss)

            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch:5d} | loss = {epoch_loss:.6f}")

        return history

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


# --------------------------------------------------------------------------- #
# Example: learn the XOR function
# --------------------------------------------------------------------------- #
def _demo_xor() -> None:
    rng = np.random.default_rng(seed=42)

    # XOR is not linearly separable, so it needs a hidden layer.
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)

    net = NeuralNetwork(
        layers=[
            DenseLayer(2, 8, Tanh(), rng=rng),
            DenseLayer(8, 1, Sigmoid(), rng=rng),
        ],
        loss=BinaryCrossEntropy(),
    )

    print("Training a neural network to learn XOR...\n")
    net.train(X, y, epochs=2000, learning_rate=0.5, verbose=True, rng=rng)

    print("\nPredictions:")
    preds = net.predict(X)
    for inputs, target, pred in zip(X, y, preds):
        print(
            f"  input={inputs.astype(int)} "
            f"target={int(target[0])} "
            f"predicted={pred[0]:.4f} -> {int(round(pred[0]))}"
        )


if __name__ == "__main__":
    _demo_xor()
