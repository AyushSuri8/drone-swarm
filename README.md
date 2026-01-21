# HOCBF-GNN-Swarm: Safe Swarm Orchestration

### Official PyTorch Implementation

**Safety Under Saturation: Differentiable Control Barriers and Phase Transitions in High-Density Swarm Reinforcement Learning** | **Ayush Suri**

## 📌 Abstract

This repository contains the simulation environment and training code for the paper *"Safety Under Saturation."* It implements a hybrid architecture for drone swarms ( to ) using:

* **Graph Neural Networks (GNN)** for decentralized policy.
* **High-Order Control Barrier Functions (HOCBF)** for safety guarantees.
* **PyBullet** for GPU-accelerated physics (127 FPS).

## 🚀 Key Features

* **GPU-Resident Simulation:** Custom PyBullet wrapper optimized for CUDA.
* **Differentiable Safety Layer:** Solves QP optimization inside the PyTorch graph.
* **Sim-to-Real:** Includes `AsyncDelayWrapper` to simulate network latency.

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/hocbf-swarm.git
cd hocbf-swarm
pip install -r requirements.txt

```

## 💻 Usage

To train the swarm with the Existence Penalty:

```bash
python train.py --n_agents 8 --algo hocbf --penalty_liveness 0.05

```

To visualize the "Jamming Phase" at N=64:

```bash
python evaluate.py --load_model models/jamming_n64.pt --render

```

## 📊 Results

![Result Plot](assets/results_plot.png)

## 📜 Citation

If you use this code, please cite:

```bibtex
@article{yourname2026swarm,
  title={Safety Under Saturation},
  author={Your Name},
  year={2026}
}

```
