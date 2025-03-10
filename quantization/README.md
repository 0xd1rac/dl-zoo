# Quantization 

## **Beginner Projects**
- [ ] **Post-Training Quantization (PTQ) on MNIST**
  - Train a simple CNN on MNIST.
  - Apply **Post-Training Quantization (PTQ)** using TensorFlow Lite or PyTorch.
  - Compare accuracy before and after quantization.

- [ ] **Quantized Inference with ONNX Runtime**
  - Convert a trained **ResNet model** to ONNX format.
  - Apply quantization using **ONNX Runtime’s quantization toolkit**.
  - Compare inference speed on CPU vs. GPU.

- [ ] **Integer-Only Quantization for Mobile Deployment**
  - Train a MobileNet model on CIFAR-10.
  - Use TensorFlow Lite’s **integer-only quantization** for ARM processors.
  - Deploy on a **Raspberry Pi or Android**.

- [ ] **Dynamic Quantization for NLP Models**
  - Apply **dynamic quantization** to a Transformer-based model (e.g., BERT).
  - Compare accuracy vs. latency for **FP32 vs. INT8 inference**.
  - Evaluate on **Hugging Face’s benchmark datasets**.

- [ ] **K-Means-Based Weight Quantization**
  - Implement **K-means clustering** to quantize neural network weights.
  - Train on a simple dataset (e.g., Fashion-MNIST).
  - Compare accuracy vs. compression ratio.

## **Intermediate Projects**
- [ ] **Per-Channel vs. Per-Tensor Quantization**
  - Train an EfficientNet model on CIFAR-10.
  - Compare **per-channel quantization** (individual weight tensors) vs. **per-tensor quantization** (entire layer).
  - Analyze model accuracy and size reduction.

- [ ] **Training-Aware Quantization (QAT) for Edge AI**
  - Train a model using **Quantization-Aware Training (QAT)** on ImageNet.
  - Deploy on NVIDIA Jetson or Coral Edge TPU.
  - Compare performance against PTQ.

- [ ] **Quantized Object Detection with YOLOv8**
  - Convert **YOLOv8 to a quantized version** (FP32 → INT8).
  - Test on the **COCO dataset**.
  - Measure latency improvements for real-time inference.

## **Advanced Projects**
- [ ] **Adaptive Rounding for Weight Quantization**
  - Implement **adaptive rounding techniques** for quantization (instead of simple rounding).
  - Test on **ResNet-50 or ViT models**.
  - Compare accuracy vs. computational efficiency.

- [ ] **Sparse Quantization for Transformer Models**
  - Apply **quantization with structured sparsity** to a **Transformer model**.
  - Compare performance with **pruned models**.
  - Test on **Hugging Face’s GLUE benchmark**.
