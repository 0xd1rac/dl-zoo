# EBM
## Beginner Projects
- [ ] **Binary Classification with EBMs**
  - Train an EBM as a binary classifier on MNIST.
  - Compare it with logistic regression or neural networks.
  - Experiment with different energy functions.

- [ ] **Denoising with EBMs**
  - Train an EBM for image denoising.
  - Add noise to MNIST/CIFAR-10 images and train the model to reconstruct the clean image.
  - Compare with traditional denoising autoencoders.

- [ ] **Unsupervised Representation Learning**
  - Train an EBM to learn feature representations for clustering.
  - Use CIFAR-10 and analyze the learned feature space.
  - Compare with PCA and contrastive learning (SimCLR).

- [ ] **Simple Image Generation with EBMs**
  - Train an EBM as a generative model on MNIST.
  - Sample new digits using Langevin sampling.
  - Compare sample quality with GANs and VAEs.

- [ ] **EBM for Anomaly Detection**
  - Train an EBM on normal samples (e.g., normal digits from MNIST).
  - Use the model’s energy function to detect anomalies (e.g., out-of-distribution images).
  - Compare with autoencoder-based anomaly detection.

## Intermediate Projects
- [ ] **Contrastive Divergence Training on CIFAR-10**
  - Implement Contrastive Divergence (CD-k) to train an EBM on CIFAR-10.
  - Use Gibbs sampling to refine the generated images.
  - Compare CD-1 vs. CD-10 for training stability.

- [ ] **EBMs for Image Inpainting**
  - Train an EBM for image inpainting.
  - Mask out parts of CIFAR-10 images and use Langevin dynamics to complete them.
  - Compare with GAN-based inpainting methods.

- [ ] **EBM for Multi-Modal Data Modeling**
  - Train an EBM on paired datasets (e.g., text-image pairs).
  - Learn a shared energy function for joint modeling of text and images.
  - Compare with contrastive learning (e.g., CLIP).

## Advanced Projects
- [ ] **3D Shape Generation with EBMs**
  - Train an EBM to generate 3D voxel representations of objects.
  - Use ShapeNet or ModelNet datasets.
  - Sample 3D objects using MCMC.
