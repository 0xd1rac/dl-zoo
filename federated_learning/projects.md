# Federated Learning (FL) Projects Checklist

## **Beginner Projects**
- [ ] **Basic Federated Learning with MNIST**
  - Implement **Federated Averaging (FedAvg)** with multiple clients.
  - Train a CNN on **MNIST** using federated learning.
  - Compare centralized vs. federated training.

- [ ] **Federated Learning with Non-IID Data**
  - Simulate **non-IID (heterogeneous) client datasets**.
  - Train a federated model on **Fashion-MNIST**.
  - Analyze performance vs. IID data.

- [ ] **Privacy-Preserving Federated Learning**
  - Implement **differential privacy** in federated training.
  - Train on CIFAR-10 and analyze privacy-utility trade-offs.
  - Compare different noise levels in DP-SGD.

- [ ] **Federated Learning with Secure Aggregation**
  - Implement **secure aggregation** to prevent server-side data leaks.
  - Train a CNN on CIFAR-10 with simulated clients.
  - Compare training with and without secure aggregation.

## **Intermediate Projects**
- [ ] **Federated Learning on Edge Devices**
  - Deploy a federated model on **Raspberry Pi or Jetson Nano**.
  - Train a lightweight CNN model on a real-world dataset.
  - Optimize for **low power consumption and latency**.

- [ ] **Federated Learning for Medical Imaging**
  - Train a federated model on **chest X-ray datasets (e.g., NIH or CheXpert)**.
  - Ensure **data remains on separate hospitals (clients)**.
  - Compare centralized vs. federated medical image classification.

- [ ] **Federated Learning for NLP**
  - Train a federated **LSTM or Transformer** on a **text dataset (e.g., IMDb, Reddit comments)**.
  - Implement **FedText** for personalized NLP models.
  - Compare performance vs. fine-tuned BERT.

- [ ] **Heterogeneous Federated Learning with Varying Compute Power**
  - Simulate clients with **different compute capabilities**.
  - Use a CNN for CIFAR-100 and adjust client update strategies.
  - Compare adaptive learning rates for slow vs. fast clients.

## **Advanced Projects**
- [ ] **Personalized Federated Learning (Per-FedAvg)**
  - Implement **Per-FedAvg**, where clients learn personalized models.
  - Train on **heterogeneous CIFAR-10 splits**.
  - Compare with standard FedAvg.

- [ ] **Federated Learning for Smart IoT Devices**
  - Deploy a **federated ML model on IoT sensors**.
  - Train on **smart home activity datasets**.
  - Optimize for **low bandwidth and on-device inference**.

- [ ] **Federated Learning with Knowledge Distillation**
  - Train a **teacher model centrally** and **distill knowledge to federated clients**.
  - Apply on **image or text datasets**.
  - Compare distillation-based FL with standard FL.

- [ ] **Blockchain-Based Federated Learning**
  - Implement **blockchain to track model updates**.
  - Train on a **financial fraud detection dataset**.
  - Compare with traditional server-client federated learning.

- [ ] **Adversarial Robustness in Federated Learning**
  - Simulate **poisoned clients** in federated training.
  - Train a model on CIFAR-10 with Byzantine attacks.
  - Implement **Byzantine-resilient aggregation strategies**.

