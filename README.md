# Hybrid Quantum–Classical Neural Network (HQNN)

## What This Project Does
- Combines classical neural networks with a quantum circuit layer.
- Performs binary classification on the moons dataset.
- Demonstrates hybrid backpropagation across classical and quantum components.

## Why Hybrid Models Are Interesting
- Quantum circuits inject non-linear geometry that can help separate tricky datasets.
- Classical layers handle versatile feature extraction and scaling.
- Hybrid architectures are realistic for NISQ devices where small circuits are paired with classical compute.

## Why SVG (Not PNG)
> CODEX cannot preview PNG/JPG and shows
> “Binary files are not supported.”
> All images in this repository use SVG for safe rendering and easy diffs.

## How It Works
- Classical layers preprocess data and generate rotation angles.
- Quantum layer performs parameterized rotations and an entangling gate, then measures qubit expectations.
- Measurements feed into a final classical classifier to produce logits.
- The entire system is trained end-to-end with PyTorch optimizers and PennyLane's differentiable circuits.

## Project Structure
```
hybrid-quantum-classical-neural-network/
├── app/
│   ├── dataset.py                 # generate moons dataset
│   ├── classical_model.py         # classical MLP layers
│   ├── quantum_layer.py           # PQC layer (PennyLane QNode)
│   ├── hybrid_model.py            # full hybrid HQNN model
│   ├── trainer.py                 # training loop
│   ├── plots.py                   # SVG-only plot utilities
│   ├── utils.py                   # small helpers
│   ├── main.py                    # CLI entrypoint
├── notebooks/
│   ├── hqnn_demo.ipynb
├── examples/
│   ├── decision_boundary_hybrid.svg
│   ├── training_loss.svg
│   ├── latent_space_visualization.svg
├── tests/
│   ├── test_dataset.py
│   ├── test_classical_model.py
│   ├── test_quantum_layer.py
│   ├── test_hybrid_model.py
│   ├── test_trainer.py
├── requirements.txt
├── Dockerfile
├── .gitignore
└── LICENSE
```

## How to Run
```bash
pip install -r requirements.txt
python app/main.py --epochs 10 --batch 32
```

### What You Should See
- Console output indicating training progress.
- Decision boundary plot saved to `examples/decision_boundary_hybrid.svg`.
- Training loss curve saved to `examples/training_loss.svg`.
- Latent space projection saved to `examples/latent_space_visualization.svg`.

## Future Ideas
- More qubits or deeper entangling layers.
- Larger classical encoders for richer feature extraction.
- Running the quantum circuit on real hardware via PennyLane plugins.
