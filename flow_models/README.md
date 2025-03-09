# Beginner Projects
- [ ] **Normalizing Flows on 1D Gaussian Data** <br/>
      - Implement a simple RealNVP or Planar Flow model.<br/>
      - Train on a 1D Gaussian mixture model.<br/>
      - Visualize how the flow transforms simple distributions into complex ones.<br/>
- [ ] Density Estimation with Normalizing Flows <br/>
      - Train a RealNVP model to learn the density of 2D synthetic datasets (e.g., Swiss Roll, Moons, or Circles).<br/>
      - Compare learned densities with kernel density estimation (KDE).<br/>
- [ ] Flow-based Image Generation (MNIST) <br/>
      - Train RealNVP or Glow on MNIST.<br/>
      - Sample new digits and visualize latent space interpolations.<br/>
      - Compare results with GAN-based models.<br/>
- [ ] Conditional Normalizing Flow for Image Generation<br/>
      - Train a conditional RealNVP model on MNIST. <br/>
      - Generate digits conditioned on class labels. <br/>
      - Compare with conditional GANs. <br/>
- [ ] Flow-based Anomaly Detection
      - Train RealNVP on normal samples (e.g., CIFAR-10). <br/>
      - Use the model's likelihood estimates for out-of-distribution detection. <br/>
      - Compare with other anomaly detection methods. <br/>
      
# Intermediate Projects
- [ ] Glow for Image Generation  <br/>
      - Train Glow on CIFAR-10 or CelebA.  <br/>
      - Experiment with temperature scaling for sampling.  <br/>
      - Compare with VAE and GAN-based models.  <br/>
- [ ] Neural ODEs for Continuous Normalizing Flows  <br/>
      - Implement Continuous Normalizing Flows (CNFs) using Neural ODEs.  <br/>
      - Train on 2D toy datasets and visualize flow trajectories.  <br/>
      - Compare training efficiency with discrete flow models.  <br/>
- [ ] Invertible Flow Models for Super-Resolution  <br/>
      - Train a Glow-based model for image super-resolution.  <br/>
      - Compare performance with SRGAN and diffusion models.  <br/>
      - Use conditional flows to enhance control over outputs. <br/>

# Advanced Projects
- [ ] Flow-Based Speech Synthesis (WaveGlow) <br/>
      - Implement WaveGlow for text-to-speech (TTS). <br/>
      - Train on the LJSpeech dataset. <br/>
      - Compare performance with WaveNet and diffusion-based speech models. <br/>
- [ ] Flow-based Molecular Generation <br/>
      - Train a Graph Normalizing Flow (GraphNVP) for molecule generation. <br/>
      - Use the ZINC molecular dataset. <br/>
      - Compare with GAN and diffusion-based molecular models.<br/>
