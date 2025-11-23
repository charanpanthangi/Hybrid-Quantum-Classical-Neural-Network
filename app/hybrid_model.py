"""Definition of the end-to-end hybrid quantum-classical model."""

import torch
from torch import nn

from app.classical_model import ClassicalClassifier, ClassicalEncoder
from app.quantum_layer import QuantumLayer


class HybridModel(nn.Module):
    """Full hybrid pipeline: classical encoder -> quantum circuit -> classifier."""

    def __init__(self, n_qubits: int = 2):
        """Create the model components.

        Args:
            n_qubits: Number of qubits and therefore size of quantum outputs.
        """
        super().__init__()
        self.encoder = ClassicalEncoder(output_dim=n_qubits)
        self.quantum = QuantumLayer(n_qubits=n_qubits)
        self.classifier = ClassicalClassifier(input_dim=n_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through classical and quantum parts.

        The gradients propagate through the quantum layer thanks to
        PennyLane's PyTorch integration, meaning the entire hybrid system
        can be trained end-to-end with standard optimizers.
        """
        # Classical encoder maps inputs to circuit angles
        quantum_params = self.encoder(x)
        # Quantum layer processes each set of angles to expectation values
        quantum_outputs = self.quantum(quantum_params)
        # Classical classifier converts quantum outputs to logits
        logits = self.classifier(quantum_outputs)
        return logits
