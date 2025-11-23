"""Tests for dataset loading and normalization."""

import pytest

from app.dataset import load_moons


def test_load_moons_shapes():
    train_loader, test_loader, X_train, y_train, X_test, y_test = load_moons(
        n_samples=100, noise=0.05, batch_size=16
    )
    X_batch, y_batch = next(iter(train_loader))
    assert X_batch.shape[1] == 2
    assert y_batch.shape[1] == 1
    assert X_train.mean() == pytest.approx(0.0, abs=1e-6)
    assert X_train.std() == pytest.approx(1.0, rel=0.2)
    # Ensure test split exists
    assert X_test.shape[1] == 2
    assert y_train.shape[0] + y_test.shape[0] == 100
