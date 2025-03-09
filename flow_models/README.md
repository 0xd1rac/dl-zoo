# Normalization Flow Models
## Beginner Projects
- [ ] **Normalizing Flows on 1D Gaussian Data**
  - Implement a simple RealNVP or Planar Flow model.
  - Train on a 1D Gaussian mixture model.
  - Visualize how the flow transforms simple distributions into complex ones.

- [ ] **Density Estimation with Normalizing Flows**
  - Train a RealNVP model to learn the density of 2D synthetic datasets (e.g., Swiss Roll, Moons, or Circles).
  - Compare learned densities with kernel density estimation (KDE).

- [ ] **Flow-based Image Generation (MNIST)**
  - Train RealNVP or Glow on MNIST.
  - Sample new digits and visualize latent space interpolations.
  - Compare results with GAN-based models.

- [ ] **Conditional Normalizing Flow for Image Generation**
  - Train a conditional RealNVP model on MNIST.
  - Generate digits conditioned on class labels.
  - Compare with conditional GANs.

- [ ] **Flow-based Anomaly Detection**
  - Train RealNVP on normal samples (e.g., CIFAR-10).
  - Use the model's likelihood estimates for out-of-distribution detection.
  - Compare with other anomaly detection methods.

## Intermediate Projects
- [ ] **Glow for Image Generation**
  - Train Glow on CIFAR-10 or CelebA.
  - Experiment with temperature scaling for sampling.
  - Compare with VAE and GAN-based models.

- [ ] **Neural ODEs for Continuous Normalizing Flows**
  - Implement Continuous Normalizing Flows (CNFs) using Neural ODEs.
  - Train on 2D toy datasets and visualize flow trajectories.
  - Compare training efficiency with discrete flow models.

- [ ] **Invertible Flow Models for Super-Resolution**
  - Train a Glow-based model for image super-resolution.
  - Compare performance with SRGAN and diffusion models.
  - Use conditional flows to enhance control over outputs.

## Advanced Projects
- [ ] **Flow-Based Speech Synthesis (WaveGlow)**
  - Implement WaveGlow for text-to-speech (TTS).
  - Train on the LJSpeech dataset.
  - Compare performance with WaveNet and diffusion-based speech models.

- [ ] **Flow-based Molecular Generation**
  - Train a Graph Normalizing Flow (GraphNVP) for molecule generation.
  - Use the ZINC molecular dataset.
  - Compare with GAN and diffusion-based molecular models.
