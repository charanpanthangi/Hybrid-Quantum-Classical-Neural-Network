"""CLI entrypoint for training and visualizing the hybrid model."""

import argparse
from pathlib import Path

import torch

from app.dataset import load_moons
from app.hybrid_model import HybridModel
from app.plots import plot_decision_boundary, plot_training_loss, plot_latent_space
from app.trainer import Trainer


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for quick experimentation."""
    parser = argparse.ArgumentParser(description="Train a hybrid quantum-classical neural network on the moons dataset.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate")
    return parser.parse_args()


def main() -> None:
    """End-to-end pipeline: data, model, training, and plots."""
    args = parse_args()
    train_loader, test_loader, X_train, y_train, X_test, y_test = load_moons(batch_size=args.batch)
    model = HybridModel()
    trainer = Trainer(model, lr=args.learning_rate)
    history = trainer.train(train_loader, test_loader, epochs=args.epochs)

    # Collect encoder outputs for latent space visualization.
    with torch.no_grad():
        encoder_outputs = model.encoder(torch.tensor(X_train, dtype=torch.float32)).numpy()

    examples_dir = Path("examples")
    examples_dir.mkdir(exist_ok=True)
    decision_path = examples_dir / "decision_boundary_hybrid.svg"
    loss_path = examples_dir / "training_loss.svg"
    latent_path = examples_dir / "latent_space_visualization.svg"

    plot_decision_boundary(model, X_train, y_train, str(decision_path))
    plot_training_loss(history, str(loss_path))
    plot_latent_space(encoder_outputs, str(latent_path))

    print("Training complete. SVG plots saved to the examples/ directory.")
    print(f"Decision boundary: {decision_path}")
    print(f"Training loss: {loss_path}")
    print(f"Latent space: {latent_path}")


if __name__ == "__main__":
    main()
