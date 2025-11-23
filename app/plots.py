"""SVG plotting utilities for the hybrid model."""

from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch import nn


plt.switch_backend("Agg")  # Ensure plots work in headless environments.


def _ensure_svg():
    """Configure Matplotlib to emit SVG outputs only."""
    plt.rcParams["savefig.format"] = "svg"


def plot_decision_boundary(model: nn.Module, X: np.ndarray, y: np.ndarray, output_path: str) -> None:
    """Plot the decision boundary of a trained model and save as SVG."""
    _ensure_svg()
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    with torch.no_grad():
        logits = model(torch.tensor(grid, dtype=torch.float32)).reshape(xx.shape)
        probs = torch.sigmoid(logits).numpy()
    plt.figure(figsize=(6, 4))
    plt.contourf(xx, yy, probs, levels=50, cmap="RdBu", alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolors="k")
    plt.title("Decision boundary of hybrid model")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close()


def plot_training_loss(history: Dict[str, Iterable[float]], output_path: str) -> None:
    """Plot training and test loss curves."""
    _ensure_svg()
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["train_loss"], label="Train loss")
    plt.plot(epochs, history["test_loss"], label="Test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Binary cross-entropy loss")
    plt.title("Training progress")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close()


def plot_latent_space(encoder_outputs: np.ndarray, output_path: str) -> None:
    """Visualize encoder outputs in 2D using PCA."""
    _ensure_svg()
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(encoder_outputs)
    plt.figure(figsize=(6, 4))
    plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.7)
    plt.title("Latent space of classical encoder outputs")
    plt.xlabel("PCA component 1")
    plt.ylabel("PCA component 2")
    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close()
