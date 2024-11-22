# Tutorial: Understanding and Implementing Text-Guided Stable Diffusion

## Introduction

**Stable Diffusion** is a powerful generative model that synthesizes high-quality images guided by textual descriptions. It leverages the strengths of **Variational Autoencoders (VAEs)** and **Denoising Diffusion Probabilistic Models (DDPMs)** to produce images efficiently and effectively.

In this tutorial, we'll delve deep into the components of Stable Diffusion, focusing on:

- The **connection between the VAE encoder (`vae_encoder`), VAE decoder (`vae_decoder`), and the Conditional UNet (`conditional_unet`)**.
- **How the VAE encoder and decoder are trained**.
- **Why we need VAE encoder and decoders**, and why they are called **VAE** and not just **AE**.
- The **motivation behind each part** of the model.
- Detailed **training and inference code** with inline comments and tensor sizes.
- **How ground truth data is generated** for training.

---

## Table of Contents

1. [Understanding Stable Diffusion](#1-understanding-stable-diffusion)
2. [Motivation Behind Each Component](#2-motivation-behind-each-component)
   - [Why Use a VAE?](#why-use-a-vae)
   - [Why a Variational Autoencoder (VAE) and Not an Autoencoder (AE)?](#why-a-variational-autoencoder-vae-and-not-an-autoencoder-ae)
   - [Connection Between VAE and Conditional UNet](#connection-between-vae-and-conditional-unet)
3. [Training the VAE Encoder and Decoder](#3-training-the-vae-encoder-and-decoder)
4. [Generating Ground Truth Data](#4-generating-ground-truth-data)
5. [Training the Conditional UNet](#5-training-the-conditional-unet)
6. [Inference with Stable Diffusion](#6-inference-with-stable-diffusion)
7. [Code Implementation](#7-code-implementation)
   - [VAE Encoder and Decoder](#vae-encoder-and-decoder)
   - [Conditional UNet](#conditional-unet-1)
   - [Text Encoder](#text-encoder)
   - [Noise Scheduler](#noise-scheduler)
   - [Training the VAE](#training-the-vae)
   - [Generating Ground Truth Data](#generating-ground-truth-data)
   - [Training the Conditional UNet](#training-the-conditional-unet)
   - [Inference Code](#inference-code)
8. [Conclusion](#8-conclusion)
9. [References](#9-references)

---

## 1. Understanding Stable Diffusion

Stable Diffusion is a generative model that produces images by reversing a diffusion process in a latent space. The key components are:

- **VAE Encoder (`vae_encoder`)**: Compresses input images into a lower-dimensional latent space.
- **VAE Decoder (`vae_decoder`)**: Reconstructs images from the latent representations.
- **Conditional UNet (`conditional_unet`)**: Performs denoising in the latent space, guided by text embeddings.
- **Text Encoder**: Converts textual descriptions into embeddings.
- **Noise Scheduler**: Manages the addition and removal of noise during the diffusion process.

---

## 2. Motivation Behind Each Component

### Why Use a VAE?

- **Dimensionality Reduction**: Images are high-dimensional data. Operating directly on pixel space is computationally expensive. The VAE reduces the dimensionality, enabling efficient computation.
- **Latent Space Manipulation**: By encoding images into a latent space, we can perform diffusion in a space that captures the essential features of the data.
- **Reconstruction Capability**: The VAE ensures that images can be accurately reconstructed from the latent space after the diffusion process.

### Why a Variational Autoencoder (VAE) and Not an Autoencoder (AE)?

- **Probabilistic Framework**: A VAE models the data distribution probabilistically, learning the parameters of a latent distribution (mean and variance) rather than deterministic encodings.
- **Continuous and Smooth Latent Space**: The VAE enforces a smooth latent space where similar latent vectors correspond to similar images, which is beneficial for generating coherent outputs.
- **Regularization**: The KL-divergence loss in VAEs regularizes the latent space, preventing overfitting and ensuring that the latent variables follow a standard normal distribution.

### Connection Between VAE and Conditional UNet

- The **VAE Encoder** encodes the input image into a latent representation.
- The **Conditional UNet** operates on this latent representation, denoising it at each timestep of the diffusion process.
- The **VAE Decoder** reconstructs the final image from the denoised latent representation.
- **Motivation**: By working in the latent space, the Conditional UNet can focus on the essential features, making the diffusion process more efficient and scalable for high-resolution images.

---

## 3. Training the VAE Encoder and Decoder

### Objectives

- **Reconstruction Loss**: Ensure that the VAE can accurately reconstruct the input image from the latent representation.
- **KL-Divergence Loss**: Regularize the latent space to follow a standard normal distribution, promoting smoothness and continuity.

### Process

1. **Forward Pass**:
   - The input image is passed through the VAE Encoder to obtain the mean (`μ`) and log-variance (`logσ²`) of the latent distribution.
   - A latent vector is sampled from this distribution using the reparameterization trick.
   - The latent vector is passed through the VAE Decoder to reconstruct the image.

2. **Loss Computation**:
   - **Reconstruction Loss**: Measures the difference between the input image and the reconstructed image (e.g., using Mean Squared Error).
   - **KL-Divergence Loss**: Measures how much the learned latent distribution diverges from a standard normal distribution.

3. **Backpropagation**:
   - The total loss (sum of reconstruction and KL-divergence losses) is backpropagated to update the VAE Encoder and Decoder weights.

---

## 4. Generating Ground Truth Data

For training the Conditional UNet:

1. **Encode Images**:
   - Use the trained VAE Encoder to encode images into latent representations.

2. **Add Noise**:
   - At each timestep `t`, add Gaussian noise to the latent representation based on the noise schedule.

3. **Ground Truth Noise**:
   - The noise added at each timestep is the ground truth that the Conditional UNet aims to predict during training.

---

## 5. Training the Conditional UNet

### Objectives

- **Noise Prediction**: Train the Conditional UNet to predict the noise added to the latent representations at each timestep, conditioned on the text embeddings.
- **Text Guidance**: Ensure that the model generates images coherent with the input text descriptions.

### Process

1. **Forward Pass**:
   - Noisy latent representations and corresponding time steps are input to the Conditional UNet along with text embeddings.
   - The model predicts the noise added to the latent representations.

2. **Loss Computation**:
   - **MSE Loss**: Measures the difference between the predicted noise and the actual noise added.

3. **Backpropagation**:
   - The loss is backpropagated to update the Conditional UNet weights.

---

## 6. Inference with Stable Diffusion

### Process

1. **Initialize**:
   - Start with a random latent vector sampled from a standard normal distribution.

2. **Iterative Denoising**:
   - For each timestep from `T` down to `1`, use the Conditional UNet to predict the noise and update the latent vector accordingly.

3. **Decode Image**:
   - After denoising, use the VAE Decoder to reconstruct the image from the final latent representation.

---

## 7. Code Implementation

Now, let's implement each component with code blocks, including inline comments and tensor sizes.

### Prerequisites

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import math
```

---

### VAE Encoder and Decoder

#### VAE Encoder

```python
class VAEEncoder(nn.Module):
    def __init__(self, latent_dim=256):
        super(VAEEncoder, self).__init__()
        # Define convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)    # [batch_size, 64, H/2, W/2]
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # [batch_size, 128, H/4, W/4]
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1) # [batch_size, 256, H/8, W/8]
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1) # [batch_size, 512, H/16, W/16]
        self.conv_mu = nn.Conv2d(512, latent_dim, kernel_size=3, padding=1)    # [batch_size, latent_dim, H/16, W/16]
        self.conv_logvar = nn.Conv2d(512, latent_dim, kernel_size=3, padding=1)# [batch_size, latent_dim, H/16, W/16]
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: [batch_size, 3, H, W]
        x = self.relu(self.conv1(x))    # [batch_size, 64, H/2, W/2]
        x = self.relu(self.conv2(x))    # [batch_size, 128, H/4, W/4]
        x = self.relu(self.conv3(x))    # [batch_size, 256, H/8, W/8]
        x = self.relu(self.conv4(x))    # [batch_size, 512, H/16, W/16]
        mu = self.conv_mu(x)            # [batch_size, latent_dim, H/16, W/16]
        logvar = self.conv_logvar(x)    # [batch_size, latent_dim, H/16, W/16]
        return mu, logvar               # Mean and log-variance for the latent distribution
```

#### VAE Decoder

```python
class VAEDecoder(nn.Module):
    def __init__(self, latent_dim=256):
        super(VAEDecoder, self).__init__()
        # Define transposed convolutional layers
        self.deconv1 = nn.ConvTranspose2d(latent_dim, 512, kernel_size=4, stride=2, padding=1)  # [batch_size, 512, H/8, W/8]
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)         # [batch_size, 256, H/4, W/4]
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)         # [batch_size, 128, H/2, W/2]
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)          # [batch_size, 64, H, W]
        self.deconv5 = nn.Conv2d(64, 3, kernel_size=3, padding=1)                               # [batch_size, 3, H, W]
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, z):
        # z: [batch_size, latent_dim, H/16, W/16]
        x = self.relu(self.deconv1(z))  # [batch_size, 512, H/8, W/8]
        x = self.relu(self.deconv2(x))  # [batch_size, 256, H/4, W/4]
        x = self.relu(self.deconv3(x))  # [batch_size, 128, H/2, W/2]
        x = self.relu(self.deconv4(x))  # [batch_size, 64, H, W]
        x = self.tanh(self.deconv5(x))  # [batch_size, 3, H, W] with values in [-1, 1]
        return x                        # Reconstructed image
```

---

### Conditional UNet

```python
class ConditionalUNet(nn.Module):
    def __init__(self, latent_dim=256, text_embed_dim=512, time_embed_dim=256):
        super(ConditionalUNet, self).__init__()
        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        # Text embedding projection
        self.text_proj = nn.Linear(text_embed_dim, time_embed_dim)

        # Encoder (Downsampling)
        self.down1 = self.conv_block(latent_dim, 256)   # [batch_size, 256, H/16, W/16]
        self.down2 = self.conv_block(256, 512)          # [batch_size, 512, H/32, W/32]
        self.down3 = self.conv_block(512, 1024)         # [batch_size, 1024, H/64, W/64]
        # Bottleneck
        self.bottleneck = self.conv_block(1024, 1024)
        # Decoder (Upsampling)
        self.up3 = self.up_conv(1024, 512)
        self.conv3 = self.conv_block(1024, 512)
        self.up2 = self.up_conv(512, 256)
        self.conv2 = self.conv_block(512, 256)
        self.up1 = self.up_conv(256, latent_dim)
        self.conv1 = self.conv_block(512, latent_dim)

    def conv_block(self, in_channels, out_channels):
        # Convolutional block with GroupNorm and ReLU activation
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
        )

    def up_conv(self, in_channels, out_channels):
        # Transposed convolution for upsampling
        return nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)

    def forward(self, x, t, text_embed):
        # x: [batch_size, latent_dim, H/16, W/16]
        # t: [batch_size, 1] (normalized timestep)
        # text_embed: [batch_size, text_embed_dim]
        time_embed = self.time_mlp(t)               # [batch_size, time_embed_dim]
        text_embed = self.text_proj(text_embed)     # [batch_size, time_embed_dim]
        emb = time_embed + text_embed               # [batch_size, time_embed_dim]
        emb = emb[:, :, None, None]                 # [batch_size, time_embed_dim, 1, 1]

        # Encoder
        d1 = self.down1(x + emb)                    # [batch_size, 256, H/16, W/16]
        d2 = self.down2(F.avg_pool2d(d1, 2) + emb)  # [batch_size, 512, H/32, W/32]
        d3 = self.down3(F.avg_pool2d(d2, 2) + emb)  # [batch_size, 1024, H/64, W/64]
        # Bottleneck
        b = self.bottleneck(F.avg_pool2d(d3, 2) + emb)  # [batch_size, 1024, H/128, W/128]
        # Decoder
        u3 = self.up3(b)                            # [batch_size, 512, H/64, W/64]
        u3 = self.conv3(torch.cat([u3, d3], dim=1) + emb)  # [batch_size, 512, H/64, W/64]
        u2 = self.up2(u3)                           # [batch_size, 256, H/32, W/32]
        u2 = self.conv2(torch.cat([u2, d2], dim=1) + emb)  # [batch_size, 256, H/32, W/32]
        u1 = self.up1(u2)                           # [batch_size, latent_dim, H/16, W/16]
        u1 = self.conv1(torch.cat([u1, d1], dim=1) + emb)  # [batch_size, latent_dim, H/16, W/16]
        return u1  # Predicted noise
```

---

### Text Encoder

We'll use a pre-trained CLIP text encoder.

```python
import clip

class TextEncoder(nn.Module):
    def __init__(self, device='cuda'):
        super(TextEncoder, self).__init__()
        self.model, _ = clip.load('ViT-B/32', device=device)
        self.device = device

    def forward(self, text):
        # text: List of strings [batch_size]
        tokens = clip.tokenize(text).to(self.device)     # [batch_size, token_length]
        text_embeddings = self.model.encode_text(tokens) # [batch_size, text_embed_dim]
        return text_embeddings
```

---

### Noise Scheduler

```python
class NoiseScheduler:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        self.beta = torch.linspace(beta_start, beta_end, timesteps)  # [T]
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)            # [T]
        self.sqrt_alpha_hat = torch.sqrt(self.alpha_hat)             # [T]
        self.sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat)  # [T]

    def get_parameters(self, t):
        # t: [batch_size]
        sqrt_alpha_hat_t = self.sqrt_alpha_hat[t].view(-1, 1, 1, 1)          # [batch_size, 1, 1, 1]
        sqrt_one_minus_alpha_hat_t = self.sqrt_one_minus_alpha_hat[t].view(-1, 1, 1, 1)  # [batch_size, 1, 1, 1]
        return sqrt_alpha_hat_t, sqrt_one_minus_alpha_hat_t
```

---

### Training the VAE

```python
# Initialize VAE components
vae_encoder = VAEEncoder(latent_dim=256).to(device)
vae_decoder = VAEDecoder(latent_dim=256).to(device)
# Optimizer
vae_optimizer = torch.optim.Adam(list(vae_encoder.parameters()) + list(vae_decoder.parameters()), lr=1e-4)
# Training loop
for epoch in range(num_epochs):
    for images, _ in dataloader:
        images = images.to(device)  # [batch_size, 3, H, W]
        # Forward pass
        mu, logvar = vae_encoder(images)         # Both: [batch_size, latent_dim, H/16, W/16]
        std = torch.exp(0.5 * logvar)            # [batch_size, latent_dim, H/16, W/16]
        eps = torch.randn_like(std)              # [batch_size, latent_dim, H/16, W/16]
        z = mu + eps * std                       # [batch_size, latent_dim, H/16, W/16]
        recon_images = vae_decoder(z)            # [batch_size, 3, H, W]
        # Compute losses
        recon_loss = F.mse_loss(recon_images, images)  # Reconstruction loss
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())  # KL divergence loss
        vae_loss = recon_loss + kl_loss
        # Backpropagation
        vae_optimizer.zero_grad()
        vae_loss.backward()
        vae_optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs}, VAE Loss: {vae_loss.item():.4f}")
```

---

### Generating Ground Truth Data

```python
def generate_ground_truth_data(images, t, vae_encoder, noise_scheduler):
    # images: [batch_size, 3, H, W]
    # t: [batch_size], random timesteps
    with torch.no_grad():
        mu, logvar = vae_encoder(images)    # Both: [batch_size, latent_dim, H/16, W/16]
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)  # [batch_size, latent_dim, H/16, W/16]
    # Get noise parameters
    sqrt_alpha_hat_t, sqrt_one_minus_alpha_hat_t = noise_scheduler.get_parameters(t)
    # Sample noise
    epsilon = torch.randn_like(z)  # [batch_size, latent_dim, H/16, W/16]
    # Create noisy latent
    z_noisy = sqrt_alpha_hat_t * z + sqrt_one_minus_alpha_hat_t * epsilon  # [batch_size, latent_dim, H/16, W/16]
    return z_noisy, epsilon  # Noisy latent and ground truth noise
```

---

### Training the Conditional UNet

```python
# Initialize components
text_encoder = TextEncoder(device).to(device)
unet = ConditionalUNet(latent_dim=256, text_embed_dim=512).to(device)
noise_scheduler = NoiseScheduler()

# Optimizer
unet_optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4)

# Training loop
for epoch in range(num_epochs):
    for images, texts in dataloader:
        images = images.to(device)   # [batch_size, 3, H, W]
        texts = list(texts)
        batch_size = images.size(0)
        # Sample random timesteps
        t = torch.randint(0, noise_scheduler.timesteps, (batch_size,), device=device).long()  # [batch_size]
        # Generate ground truth data
        z_noisy, epsilon = generate_ground_truth_data(images, t, vae_encoder, noise_scheduler)
        # Get text embeddings
        text_embeddings = text_encoder(texts)  # [batch_size, text_embed_dim]
        # Normalize time steps
        t_normalized = t.float() / noise_scheduler.timesteps  # [batch_size]
        t_normalized = t_normalized.unsqueeze(1)              # [batch_size, 1]
        # Predict noise
        epsilon_pred = unet(z_noisy, t_normalized.to(device), text_embeddings.to(device))
        # Compute loss
        loss = F.mse_loss(epsilon_pred, epsilon)
        # Backpropagation
        unet_optimizer.zero_grad()
        loss.backward()
        unet_optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs}, UNet Loss: {loss.item():.4f}")
```

---

### Inference Code

```python
@torch.no_grad()
def generate_image(text_prompt, num_steps=50):
    # Prepare text embedding
    text_embeddings = text_encoder([text_prompt])  # [1, text_embed_dim]
    # Start from random noise in latent space
    z = torch.randn(1, 256, image_size//16, image_size//16).to(device)  # [1, 256, H/16, W/16]
    for i in reversed(range(num_steps)):
        t = torch.full((1,), i, device=device, dtype=torch.long)  # [1]
        sqrt_alpha_hat_t, sqrt_one_minus_alpha_hat_t = noise_scheduler.get_parameters(t)
        t_normalized = t.float() / noise_scheduler.timesteps  # [1]
        t_normalized = t_normalized.unsqueeze(1)              # [1, 1]
        # Predict noise
        epsilon_pred = unet(z, t_normalized.to(device), text_embeddings.to(device))
        # Update latent
        beta_t = noise_scheduler.beta[t].view(-1, 1, 1, 1)
        z = (z - beta_t / sqrt_one_minus_alpha_hat_t * epsilon_pred) / noise_scheduler.alpha[t].sqrt().view(-1, 1, 1, 1)
        if i > 0:
            noise = torch.randn_like(z)
            z += noise * beta_t.sqrt()
    # Decode image
    generated_image = vae_decoder(z)
    # Rescale to [0, 1]
    generated_image = (generated_image + 1) / 2.0
    return generated_image  # [1, 3, H, W]
```

---

## 8. Conclusion

In this tutorial, we explored the components of Stable Diffusion, understanding the motivations and connections between each part:

- **VAE Encoder and Decoder**: Compress and reconstruct images, enabling efficient diffusion in latent space.
- **Conditional UNet**: Predicts noise added during diffusion, guided by text embeddings.
- **Text Encoder**: Converts text prompts into embeddings that guide image generation.
- **Noise Scheduler**: Manages the addition and removal of noise in the diffusion process.

By training the VAE separately, we ensure that the latent space is meaningful and reconstructible. Training the Conditional UNet with ground truth noise enables the model to learn the reverse diffusion process, conditioned on textual input.

---

## 9. References

- **Stable Diffusion Paper**: Rombach, R., Blattmann, A., Lorenz, D., et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. *arXiv preprint arXiv:2112.10752*.
- **Denoising Diffusion Probabilistic Models**: Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. *Advances in Neural Information Processing Systems*, 33, 6840-6851.
- **CLIP**: Radford, A., Kim, J. W., Hallacy, C., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. *Proceedings of the International Conference on Machine Learning*.

---

By following this tutorial, you should now have a solid understanding of how Stable Diffusion works, the role of each component, and how to implement and train the model from scratch.