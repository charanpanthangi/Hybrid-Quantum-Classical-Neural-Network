"""Dataset utilities for the hybrid quantum-classical neural network.

This module focuses on generating and preparing the two-moons dataset,
which is a classic non-linear binary classification problem. The curved
structure makes it ideal for demonstrating hybrid models that combine
classical feature extraction with quantum layers, as the additional
non-linear quantum step can help separate the intertwined classes.
"""

from typing import Tuple

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader


def load_moons(
    n_samples: int = 300, noise: float = 0.1, batch_size: int = 32
) -> Tuple[DataLoader, DataLoader, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and preprocess the moons dataset.

    The moons dataset contains two interleaving half circles. Because the
    classes are non-linearly separable, it is a natural fit for a hybrid
    quantum-classical model, which can introduce richer feature mappings
    via the quantum circuit.

    Args:
        n_samples: Total number of data points to generate.
        noise: Amount of Gaussian noise to add to the data.
        batch_size: Batch size for the data loaders.

    Returns:
        A tuple containing training and test DataLoaders as well as the
        raw training and test feature arrays for plotting.
    """

    # Generate the dataset with reproducibility for tests and demos.
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)

    # Normalize the features to zero mean and unit variance for stable training.
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split into training and testing sets to evaluate generalization.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Convert to PyTorch tensors for integration with the training loop.
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, X_train, y_train, X_test, y_test
