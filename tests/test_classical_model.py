"""Tests for the classical encoder and classifier."""

import torch

from app.classical_model import ClassicalEncoder, ClassicalClassifier


def test_encoder_output_shape():
    encoder = ClassicalEncoder(input_dim=2, hidden_dim=4, output_dim=2)
    sample = torch.randn(5, 2)
    output = encoder(sample)
    assert output.shape == (5, 2)


def test_classifier_output_shape():
    classifier = ClassicalClassifier(input_dim=2)
    sample = torch.randn(5, 2)
    logits = classifier(sample)
    assert logits.shape == (5, 1)
