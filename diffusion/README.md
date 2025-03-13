## DDPM
**Training algorithm**
1. Sample an image, $x_0$ from dataset
2. Sample a time step, t randomly from [0,T]
3. Sample noise, $\epsilon_t$ from $N(O,I)$
4. Add sampled noise the image using the precomputed forward diffusion formula: $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon_t$ where $\bar{\alpha}$ is a known constant derived from the diffusion noise schedule $\beta_t$
5. Use a neural network, $net_{\theta}(x_t,t)$ to predict the added noise, $\hat{\epsilon}$
6. Compute loss between the actual sampled noise, $\epsilon_t$ and predicted noise $\hat{\epsilon_t}$.
   $$L(\theta) = ||\epsilon_t - \hat{\epsilon_t}||^2$$
7. Backprop

**Image Generation Algorithm**
1. Sample noise image, $x_T$ from $N(0,I)$
2. Perform iterative denoising/reverse diffusion
     - for t in T -> 1:
          - sample noise, $z$ from $N(0,I)$ if t > 1, else $z = 0$
          - Using the trained model predict noise: $\hat{\epsilon} = net_{\theta}(x_t, t)$
          - Update noisy sample $x_t$ to $x_{t-1}$ using the closed form equation:
               $$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}$$

# Diffusion Models
## Beginner Projects
- [ ] **Denoising Diffusion Model on MNIST**
  - Train a simple denoising diffusion model on MNIST digits.
  - Experiment with different noise schedules.
  - Try generating digits from noise and interpolating between digits.

- [ ] **Conditional Image Generation**
  - Train a conditional diffusion model on Fashion-MNIST or CIFAR-10.
  - Use class labels as conditioning information to generate specific objects (e.g., only shoes, only shirts).
  - Experiment with classifier-free guidance.

- [ ] **Diffusion-Based Super-Resolution**
  - Use a pre-trained diffusion model for image super-resolution (e.g., upscale 32×32 CIFAR-10 images to 128×128).
  - Compare performance with traditional CNN-based super-resolution models like SRCNN.

## Intermediate Projects
- [ ] **Text-to-Image Diffusion Model**
  - Implement a latent diffusion model (LDM) like Stable Diffusion but trained on a small dataset.
  - Use CLIP embeddings for text conditioning.
  - Experiment with different guidance techniques (classifier guidance, classifier-free guidance).

- [ ] **Inpainting with Diffusion Models**
  - Train a model to fill missing image regions.
  - Use the CelebA dataset to inpaint missing parts of human faces.
  - Compare with GAN-based inpainting methods.

- [ ] **Diffusion-Based Style Transfer**
  - Modify a diffusion model to blend artistic styles into generated images.
  - Train on a dataset of paintings and transfer styles to real-world images.

## Advanced Projects
- [ ] **3D Object Generation with Diffusion Models**
  - Implement Score Distillation Sampling (SDS) to generate 3D meshes from text prompts (similar to DreamFusion).
  - Experiment with diffusion in NeRF representations for scene synthesis.

- [ ] **Video Frame Interpolation with Diffusion Models**
  - Use diffusion to synthesize intermediate frames in a video.
  - Train on a dataset like UCF101 or DAVIS.
  - Compare results with optical flow-based interpolation.

- [ ] **Audio Diffusion Model for Speech Synthesis**
  - Train a diffusion model on speech waveforms for text-to-speech (TTS).
  - Compare with models like WaveNet or Tacotron.
  - Experiment with conditioning on phonemes or mel-spectrograms.

- [ ] **Protein Structure Generation with Diffusion Models**
  - Apply diffusion models to protein structure prediction (e.g., train on PDB datasets).
  - Generate 3D protein conformations conditioned on amino acid sequences.
  - Compare with AlphaFold and RoseTTAFold.
