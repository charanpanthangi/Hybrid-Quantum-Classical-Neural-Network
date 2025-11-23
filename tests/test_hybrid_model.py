"""Tests for the full hybrid model forward pass."""

import torch

from app.hybrid_model import HybridModel


def test_hybrid_forward_shapes():
    model = HybridModel(n_qubits=2)
    sample = torch.randn(4, 2)
    logits = model(sample)
    assert logits.shape == (4, 1)
