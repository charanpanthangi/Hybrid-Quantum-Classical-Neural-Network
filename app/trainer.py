"""Training utilities for the hybrid model."""

from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader


class Trainer:
    """Simple trainer to optimize the hybrid model."""

    def __init__(self, model: nn.Module, lr: float = 0.01):
        """Initialize optimizer and loss function.

        Args:
            model: The hybrid model to train.
            lr: Learning rate for Adam optimizer.
        """
        self.model = model
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def train(self, train_loader: DataLoader, test_loader: DataLoader, epochs: int = 10) -> Dict[str, List[float]]:
        """Run the training loop.

        Args:
            train_loader: DataLoader for training data.
            test_loader: DataLoader for evaluation data.
            epochs: Number of passes over the training set.

        Returns:
            Dictionary containing loss and accuracy history.
        """
        history = {"train_loss": [], "test_loss": [], "test_acc": []}
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            for X_batch, y_batch in train_loader:
                self.optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            avg_train_loss = total_loss / len(train_loader)

            # Evaluate on the test set
            self.model.eval()
            with torch.no_grad():
                test_loss = 0.0
                correct = 0
                total = 0
                for X_batch, y_batch in test_loader:
                    logits = self.model(X_batch)
                    loss = self.criterion(logits, y_batch)
                    preds = torch.sigmoid(logits) > 0.5
                    correct += (preds.float() == y_batch).sum().item()
                    total += y_batch.numel()
                    test_loss += loss.item()
                avg_test_loss = test_loss / len(test_loader)
                accuracy = correct / total

            history["train_loss"].append(avg_train_loss)
            history["test_loss"].append(avg_test_loss)
            history["test_acc"].append(accuracy)

        return history
