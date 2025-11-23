"""Tests for the training loop to ensure loss decreases."""

import torch

from app.dataset import load_moons
from app.hybrid_model import HybridModel
from app.trainer import Trainer


def test_training_reduces_loss():
    train_loader, test_loader, _, _, _, _ = load_moons(
        n_samples=80, noise=0.05, batch_size=16
    )
    model = HybridModel()
    trainer = Trainer(model, lr=0.05)
    history = trainer.train(train_loader, test_loader, epochs=2)
    assert history["train_loss"][0] >= history["train_loss"][-1]
