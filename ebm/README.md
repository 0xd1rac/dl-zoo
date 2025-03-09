
# Beginner Projects 
- [ ] Binary Classifications with EBMs<br /> 
      - Train an EBM as a binary classifier on MNIST.<br /> 
      - Compare it with logistic regression or neural networks. <br /> 
      - Experiment with different energy functions. <br /> 
- [ ] Denoising with EBMs<br /> 
      - Train an EBM for image denoising.<br /> 
      - Add noise to MNIST/CIFAR-10 images and train the model to reconstruct the clean image. <br/> 
      - Compare with traditional denoising autoencoders.<br/> 
- [ ] Unsupervised Representation Learning <br/> 
      - Train an EBM to learn feature representations for clustering. <br/> 
      - Use CIFAR-10 and analyze the learned feature space. <br/> 
      - Compare with PCA and contrastive learning (SimCLR).<br/> 
- [ ] Simple Image Generation with EBMs <br/> 
      - Train an EBM as a generative model on MNIST. <br/> 
      - Sample new digits using Langevin sampling. <br/> 
      - Compare sample quality with GANs and VAEs. <br/> 
- [ ] EBM for Anomaly Detection <br/> 
      - Train an EBM on normal samples (e.g., normal digits from MNIST).<br/> 
      - Use the model’s energy function to detect anomalies (e.g., out-of-distribution images).<br/> 
      - Compare with autoencoder-based anomaly detection.<br/> 

# Intermediate Projects 
- [ ] Contrastive Divergence Training on CIFAR-10     <br/> 
      - Implement Contrastive Divergence (CD-k) to train an EBM on CIFAR-10. <br/> 
      - Use Gibbs sampling to refine the generated images.      <br/> 
      - Compare CD-1 vs. CD-10 for training stability. <br/> 
- [ ] EBMs for Image Inpainting <br/> 
      - Train an EBM for image inpainting. <br/> 
      - Mask out parts of CIFAR-10 images and use Langevin dynamics to complete them. <br/> 
      - Compare with GAN-based inpainting methods. <br/> 
- [ ] EBM for Multi-Modal Data Modeling <br/> 
      - Train an EBM on paired datasets (e.g., text-image pairs). <br/> 
      - Learn a shared energy function for joint modeling of text and images. <br/> 
      - Compare with contrastive learning (e.g., CLIP). <br/> 

# Advanced Projects
- [ ] 3D Shape Generation with EBMs <br/> 
      - Train an EBM to generate 3D voxel representations of objects. <br/> 
      - Use ShapeNet or ModelNet datasets. <br/> 
      - Sample 3D objects using MCMC. <br/> 

- [ ] EBM for Video Prediction <br/> 
      - Train an EBM to model the energy of video frames. <br/> 
      - Generate future frames by minimizing energy differences. <br/> 
      - Compare with RNN and transformer-based video prediction. <br/> 
