# Variational Autoencoders
## **Beginner Projects**

- [ ] **Basic VAE on MNIST**
    - Train a simple **Variational Autoencoder (VAE)** on the MNIST dataset.
    - Visualize the latent space and generate new digits.
    - Compare with a standard autoencoder.

- [ ] **Conditional VAE (cVAE) for Handwritten Digits**
  - Implement a **Conditional VAE (cVAE)** for digit generation.
  - Condition the model on class labels to generate specific digits.
  - Compare image quality with an unconditional VAE.

- [ ] **VAE for Image Denoising**
  - Train a VAE to remove noise from images.
  - Apply on **Fashion-MNIST or CIFAR-10**.
  - Compare with traditional **denoising autoencoders (DAE)**.

- [ ] **Latent Space Interpolation with VAEs**
  - Train a VAE on MNIST or CelebA.
  - Generate **smooth interpolations** between different images in the latent space.
  - Compare results with GAN-based interpolation.

- [ ] **VAE for Outlier Detection**
  - Train a VAE on **normal samples** from CIFAR-10.
  - Detect out-of-distribution (OOD) samples using **reconstruction loss**.
  - Compare with PCA-based anomaly detection.

## **Intermediate Projects**
- [ ] **β-VAE for Disentangled Representations**
  - Implement **β-VAE** and increase the **β hyperparameter** to encourage disentanglement.
  - Train on **dSprites or CelebA**.
  - Visualize latent dimensions and assess feature separation.

- [ ] **VAE for Text Generation**
  - Implement a **Variational Recurrent Autoencoder (VRAE)** for text modeling.
  - Train on a dataset like **Shakespeare or movie reviews**.
  - Compare with LSTMs and Transformer-based text generation.

- [ ] **VQVAE (Vector Quantized VAE) for Discrete Latents**
  - Implement a **VQ-VAE** for **discrete latent representations**.
  - Train on **CIFAR-10** and visualize quantized latent space.
  - Compare with standard VAEs.

## **Advanced Projects**
- [ ] **Hierarchical VAEs for Complex Data**
  - Implement a **Hierarchical VAE (HVAE)** with multiple latent layers.
  - Train on **CelebA-HQ** for high-resolution face generation.
  - Compare with a single-layer VAE.

- [ ] **VAE-GAN Hybrid Model**
  - Combine **VAE and GAN** for high-quality image generation.
  - Use **CelebA** or **LSUN Bedrooms** dataset.
  - Compare **VAE’s smooth latent space** with **GAN’s sharp image outputs**.
