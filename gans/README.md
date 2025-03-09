# Generative Adverserial Networks 
## Beginner Projects
- [ ] **Basic GAN on MNIST**
  - Train a vanilla GAN on the MNIST dataset.
  - Generate realistic-looking digits from noise.
  - Experiment with different loss functions (e.g., Binary Cross-Entropy vs. Wasserstein loss).

- [ ] **Conditional GAN (cGAN) for Handwritten Digits**
  - Train a cGAN on MNIST or Fashion-MNIST.
  - Condition the GAN on class labels to generate specific digits or clothing items.
  - Experiment with label conditioning techniques.

- [ ] **DCGAN on CIFAR-10**
  - Implement a Deep Convolutional GAN (DCGAN) for generating CIFAR-10 images.
  - Use convolutional layers instead of fully connected layers.
  - Compare performance with vanilla GANs.

- [ ] **Image-to-Image Translation (Pix2Pix)**
  - Train a GAN to map edge images to real images (e.g., edges → shoes using the Edges2Shoes dataset).
  - Experiment with L1 loss + adversarial loss.

## Intermediate Projects
- [ ] **CycleGAN for Style Transfer**
  - Implement CycleGAN to perform unpaired style transfer (e.g., photo → painting using Monet or Van Gogh styles).
  - Train on the Monet2Photo dataset.
  - Compare with other style transfer techniques.

- [ ] **Super-Resolution with SRGAN**
  - Implement SRGAN for super-resolution (e.g., upscale 32×32 CIFAR-10 images to 128×128).
  - Compare with traditional upscaling techniques.

- [ ] **Face Aging with GANs**
  - Train a StarGAN or CycleGAN to modify a person’s age in images.
  - Use the CelebA dataset with labeled ages.
  - Experiment with multi-domain transformations (e.g., adding facial expressions).

- [ ] **Text-to-Image GAN (AttnGAN or StackGAN)**
  - Implement a Text-to-Image GAN that generates images from captions.
  - Use the CUB-200 or MS-COCO dataset.
  - Train a StackGAN or AttnGAN and compare the results.

## Advanced Projects
- [ ] **3D Object Generation with GANs**
  - Train a 3D GAN (e.g., GANformer-3D) to generate 3D models.
  - Use datasets like ShapeNet or ModelNet.
  - Visualize 3D objects using Voxel representation or Point Clouds.

- [ ] **Music Generation with GANs**
  - Train a GAN to generate music sequences (e.g., MIDI files).
  - Use a dataset like Lakh MIDI.
  - Experiment with GANs in spectrogram space vs. raw audio.

- [ ] **StyleGAN for High-Resolution Faces**
  - Train StyleGAN2 on CelebA-HQ for photorealistic face generation.
  - Experiment with style mixing and truncation tricks.
  - Modify the model for cartoon or anime-style face generation.

- [ ] **GAN-Based Anomaly Detection**
  - Train a GAN anomaly detector for detecting outliers in datasets (e.g., fraud detection).
  - Use an Autoencoder-GAN (AnoGAN) on datasets like MNIST or medical images.

- [ ] **Image Inpainting with GANs**
  - Train a GAN to restore missing parts of images.
  - Use datasets like Places2 for natural scene inpainting.
  - Compare with diffusion-based inpainting techniques.

- [ ] **GAN for Video Generation**
  - Implement MoCoGAN or TGAN for video frame synthesis.
  - Train on UCF-101 or VoxCeleb for talking head generation.
  - Experiment with temporal coherence.

- [ ] **Adversarial Attacks with GANs**
  - Train a GAN to generate adversarial examples that fool deep networks.
  - Implement AdvGAN and test it on classifiers like ResNet or MobileNet.
  - Analyze robustness to different perturbation strengths.

