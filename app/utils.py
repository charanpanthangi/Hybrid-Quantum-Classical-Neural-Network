"""Utility helpers for the HQNN project."""

from typing import Tuple

import torch
from torch.utils.data import DataLoader


def calculate_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute binary accuracy from logits and labels."""
    preds = (torch.sigmoid(logits) > 0.5).float()
    correct = (preds == labels).float().mean().item()
    return correct


def get_dataloaders(train_loader: DataLoader, test_loader: DataLoader) -> Tuple[DataLoader, DataLoader]:
    """Simple passthrough to highlight dataloader usage in other modules."""
    return train_loader, test_loader
