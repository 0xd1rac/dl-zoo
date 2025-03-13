# Neural Architecture Search (NAS)

## **Beginner Projects**
- [ ] **Grid Search vs. Random Search for CNNs**
  - Implement **grid search and random search** to find the best CNN architecture on MNIST.
  - Compare model accuracy, parameter count, and training time.
  - Visualize the search space.

- [ ] **Reinforcement Learning-Based NAS for Small Networks**
  - Implement a **basic RL-based NAS** using a policy gradient agent.
  - Train on a small dataset like **Fashion-MNIST**.
  - Compare the best-discovered architecture with a standard CNN.

- [ ] **Evolutionary Algorithm for Neural Architecture Search**
  - Implement **genetic algorithms (GA)** for NAS.
  - Train on CIFAR-10 with a population of CNNs.
  - Compare the best model with a hand-designed architecture.

- [ ] **Differentiable Architecture Search (DARTS) on a Simple Dataset**
  - Implement **DARTS** (Differentiable Architecture Search).
  - Train on CIFAR-10 and analyze the discovered architecture.
  - Compare **first-order vs. second-order** DARTS optimization.

- [ ] **Hyperparameter Optimization for NAS**
  - Use **Hyperopt or Optuna** to optimize NAS search parameters.
  - Apply on **MobileNet or ResNet**.
  - Compare **early stopping vs. full training** for efficiency.

## **Intermediate Projects**
- [ ] **Efficient Neural Architecture Search (ENAS)**
  - Implement **ENAS** (Efficient NAS) with a shared weight strategy.
  - Train on CIFAR-100 to generate an efficient CNN.
  - Compare performance with standard NAS and manual architectures.

- [ ] **Neural Predictor-Based NAS**
  - Train a small **ML model (MLP, GNN, or Transformer)** to predict CNN performance.
  - Use it to guide NAS instead of full model training.
  - Compare with brute-force NAS search.

- [ ] **Hardware-Aware NAS for Edge Devices**
  - Implement **hardware-aware NAS** (e.g., latency-aware search).
  - Target mobile deployment using **TensorFlow Lite or NVIDIA Jetson**.
  - Optimize both **model accuracy and inference speed**.

## **Advanced Projects**
- [ ] **NAS for Transformer Architectures**
  - Apply NAS to **search Transformer hyperparameters** (e.g., number of heads, depth, hidden size).
  - Use datasets like **GLUE or ImageNet**.
  - Compare NAS-discovered Transformers with BERT and ViTs.

- [ ] **Multi-Objective NAS (Accuracy vs. Energy Consumption)**
  - Implement **multi-objective NAS** with constraints like **accuracy, latency, FLOPs, or energy use**.
  - Train on CIFAR-10 and measure energy consumption using **NVML or Edge TPU profiling**.
  - Compare **Pareto-optimal architectures**.
