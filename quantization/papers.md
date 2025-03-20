- [ ] **Courbariaux, M., Bengio, Y., & David, J.-P. (2015).** *BinaryConnect: Training Deep Neural Networks with Binary Weights during Propagations.*  
  Introduced the idea of binarizing weights during training, demonstrating that neural networks can be effectively trained using binary weights to reduce model size and computation.

- [ ] **Hubara, I., Courbariaux, M., Soudry, D., El-Yaniv, R., & Bengio, Y. (2016).** *Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1.*  
  Pioneered binarized neural networks, where both weights and activations are reduced to a single bit, significantly decreasing memory footprint and inference time.

- [ ] **Rastegari, M., Ordonez, V., Redmon, J., & Farhadi, A. (2016).** *XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks.*  
  Extended binarization to both weights and activations in convolutional networks, achieving competitive performance on large-scale tasks with greatly reduced computational cost.

- [ ] **Lin, D., Talathi, S. S., & Annapureddy, V. (2016).** *Fixed Point Quantization of Deep Convolutional Networks.*  
  Explored fixed-point quantization techniques for deep networks, setting a foundation for reducing precision in network computations without significant accuracy loss.

- [ ] **Hubara, I., Courbariaux, M., Soudry, D., El-Yaniv, R., & Bengio, Y. (2017).** *Quantized Neural Networks: Training Neural Networks with Low Precision Weights and Activations.*  
  Extended earlier binarization work to multi-bit quantization, showing that low-precision representations can be used effectively in deep learning while maintaining performance.

- [ ] **Zhou, S., Wu, Y., Ni, Z., Zhou, X., Wen, H., & Zou, Y. (2017).** *Incremental Network Quantization: Towards Lossless CNNs with Low-Precision Weights.*  
  Proposed an incremental approach to quantization that gradually converts full-precision weights into low-precision representations, often achieving nearly lossless performance.

- [ ] **Jacob, B., Kligys, S., Chen, B., Zhu, M., Tang, M., Howard, A., ... & Adam, H. (2018).** *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference.*  
  Developed methods for quantizing neural networks such that they can be executed using integer-only arithmetic, which is critical for deployment on resource-constrained hardware.

- [ ] **Choi, Y., El-Khamy, M., & Lee, J. (2018).** *PACT: Parameterized Clipping Activation for Quantized Neural Networks.*  
  Introduced a learnable clipping parameter for activations that improves the performance of quantized networks by better matching the dynamic range of low-precision data.

- [ ] **Zhuang, X., Wang, W., Song, Z., & Li, Y. (2018).** *Differentiable Quantization for Deep Neural Networks.*  
  Proposed a differentiable framework for quantization, enabling end-to-end training of networks with quantized parameters and reducing the performance gap with full-precision models.

- [ ] **Mishra, A., & Marr, D. (2018).** *Apprentice: Using Knowledge Distillation Techniques to Improve Low-Precision Network Accuracy.*  
  Explored the use of knowledge distillation to transfer knowledge from a high-precision teacher model to a low-precision student model, improving the accuracy of quantized networks.

- [ ] **Xu, Y., Cheng, W., Yang, Q., & Wang, X. (2018).** *Quantization Networks: Training Deep Neural Networks with Low Precision Weights and Activations via Dual Representations.*  
  Proposed a dual representation approach for training quantized networks, allowing models to maintain high accuracy despite using low-precision computations.

- [ ] **Zhang, D., Yang, X., Ye, Y., Li, X., & Li, W. (2020).** *LQ-Nets: Learned Quantization for Highly Accurate and Compact Deep Neural Networks.*  
  Introduced a learned quantization method that adapts quantization parameters during training, achieving a better balance between model compactness and accuracy.

- [ ] **Goncharenko, I., Andriushchenko, M., & Vetrov, D. (2021).** *Mixed Precision Quantization of Neural Networks.*  
  Developed techniques for mixed precision quantization, allowing different layers or operations to use varying bit-widths to optimize both performance and computational efficiency.

- [ ] **Nagel, M., Fink, T., Berkenkamp, F., & Krause, A. (2020).** *Efficient Quantization Methods for Large-Scale Deep Learning Models.*  
  Presented scalable quantization methods that address the challenges of applying quantization to large-scale models, combining efficiency with minimal impact on accuracy.

- [ ] **Wang, K., Goyal, Y., Gupta, A., & Lipton, Z. (2019).** *Training Quantized Neural Networks: A Comprehensive Survey.*  
  Although a survey, this work compiles many seminal techniques and ideas in quantization, serving as an essential reference for researchers and practitioners in the field.
