"""Classical neural network components used by the hybrid model.

This module defines a small multilayer perceptron (MLP) that processes
input features and produces parameters for the quantum circuit. Keeping
the network compact ensures the example runs quickly on CPU while still
showing how classical layers can prepare meaningful quantum gate angles.
"""

import torch
from torch import nn


class ClassicalEncoder(nn.Module):
    """Encode input features into parameters for the quantum circuit."""

    def __init__(self, input_dim: int = 2, hidden_dim: int = 8, output_dim: int = 2):
        """Initialize the encoder with simple linear layers.

        Args:
            input_dim: Number of input features (2 for moons dataset).
            hidden_dim: Size of the hidden layer.
            output_dim: Number of parameters to send to the quantum circuit
                (one per qubit rotation angle).
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute quantum parameters from inputs."""
        return self.net(x)


class ClassicalClassifier(nn.Module):
    """Final classifier that consumes quantum outputs."""

    def __init__(self, input_dim: int = 2):
        """Initialize a simple linear classifier.

        Args:
            input_dim: Number of features coming from the quantum layer.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Produce logits for binary classification."""
        return self.net(x)
