# Energy Based Models
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
     
# Score Matching 
## Beginner Projects
- [ ] **Score Matching on 1D Gaussian Data**
  - Implement score matching to estimate the score function of a 1D Gaussian distribution.
  - Compare it with exact gradients of the log-density.
  - Use denoising score matching (DSM) and visualize the estimated scores.

- [ ] **Denoising Score Matching on MNIST**
  - Train a score-based model on MNIST using denoising score matching (DSM).
  - Corrupt images with Gaussian noise and estimate the denoised gradients.
  - Compare with standard autoencoders.

- [ ] **Score-Based Generative Model for CIFAR-10**
  - Train a score model on CIFAR-10 using denoising score matching.
  - Generate images via Langevin sampling.
  - Compare with GANs and VAEs.

- [ ] **Noise Conditioned Score Networks (NCSN)**
  - Implement NCSN for learning multi-scale score functions.
  - Train on CIFAR-10 or CelebA.
  - Visualize learned scores at different noise levels.

- [ ] **Score Matching for Anomaly Detection**
  - Train a score model on normal images (e.g., CIFAR-10).
  - Use the energy function from the score model to detect anomalies (e.g., out-of-distribution images).
  - Compare with standard anomaly detection methods.

## Intermediate Projects
- [ ] **Langevin Dynamics for Image Generation**
  - Use a pre-trained score model to generate images via Langevin dynamics.
  - Train on MNIST or CelebA.
  - Experiment with different noise schedules and sampling steps.

- [ ] **Diffusion Models via Score Matching**
  - Implement a diffusion probabilistic model trained using score matching.
  - Compare training with score matching loss vs. KL-based loss.
  - Use DDPM (Denoising Diffusion Probabilistic Models) as a baseline.

- [ ] **Conditional Score-Based Generative Models**
  - Train a conditional score model for class-conditional generation.
  - Use classifier guidance to steer sampling toward specific image categories.
  - Compare it with conditional GANs.

## Advanced Projects
- [ ] **3D Shape Generation with Score Matching**
  - Train a score-based model on 3D voxelized objects (e.g., from ShapeNet).
  - Use Langevin dynamics to generate 3D objects.
  - Compare with GAN-based 3D generation.

- [ ] **Audio Generation with Score-Based Models**
  - Train a score-based model on waveform data for speech synthesis.
  - Generate high-quality speech using Langevin sampling.
  - Compare with GAN-based and diffusion-based speech models.

