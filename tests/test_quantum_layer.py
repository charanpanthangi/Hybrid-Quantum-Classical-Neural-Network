"""Tests for the quantum layer outputs."""

import torch

from app.quantum_layer import QuantumLayer


def test_quantum_layer_output_range():
    layer = QuantumLayer(n_qubits=2)
    params = torch.zeros((3, 2), dtype=torch.float32)
    output = layer(params)
    assert output.shape == (3, 2)
    assert torch.all(output <= 1.0)
    assert torch.all(output >= -1.0)
