"""Quantum layer implemented with PennyLane.

The quantum layer receives parameters from the classical encoder and
applies them as rotation angles on a small set of qubits. Measurements of
the qubits provide classical numbers that can be fed into the following
classical classifier. PennyLane handles automatic differentiation so
gradients flow seamlessly through this layer during training.
"""

from typing import Callable
import pennylane as qml
import torch
from torch import nn


class QuantumLayer(nn.Module):
    """Quantum circuit layer that outputs expectation values."""

    def __init__(self, n_qubits: int = 2, n_shots: int | None = None):
        """Create a PennyLane QNode wrapped for PyTorch.

        Args:
            n_qubits: Number of qubits used in the circuit.
            n_shots: Optional finite shots for sampling; ``None`` uses
                analytic mode for faster demos.
        """
        super().__init__()
        self.n_qubits = n_qubits
        # Simple default device using the built-in simulator.
        self.dev = qml.device("default.qubit", wires=n_qubits, shots=n_shots)

        # Define the quantum node using PennyLane's QNode decorator.
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs: torch.Tensor) -> torch.Tensor:
            # Encode classical outputs as rotation angles on each qubit.
            for idx in range(n_qubits):
                angle = inputs[idx]
                qml.RX(angle, wires=idx)
                qml.RY(angle, wires=idx)
            # Add simple entanglement to correlate qubits.
            if n_qubits > 1:
                qml.CZ(wires=[0, 1])
            # Measure expectation values of PauliZ on each qubit to return classical values.
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit: Callable[[torch.Tensor], torch.Tensor] = circuit

    def forward(self, quantum_params: torch.Tensor) -> torch.Tensor:
        """Run the quantum circuit for each sample in the batch.

        Args:
            quantum_params: Tensor of shape (batch_size, n_qubits) containing
                rotation angles produced by the classical encoder.

        Returns:
            Tensor of shape (batch_size, n_qubits) with expectation values in
            the range [-1, 1].
        """
        # Ensure the parameters are processed per sample.
        outputs = []
        for params in quantum_params:
            outputs.append(self.circuit(params))
        return torch.stack(outputs)
