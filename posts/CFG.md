[![Home](https://img.shields.io/badge/Home-Click%20Here-blue?style=flat&logo=homeadvisor&logoColor=white)](../)

# Classifier-Free Diffusion Models: A Comprehensive Guide

Classifier-free diffusion models have revolutionized generative modeling by enabling high-quality image generation, super-resolution, inpainting, and text-to-image synthesis without relying on external classifiers. This guide delves deep into the mechanics of classifier-free diffusion, exploring its training and inference processes, mathematical foundations, practical applications, and implementation details with code examples.

---

### Table of Contents

1. [Introduction to Diffusion Models](#introduction-to-diffusion-models)
2. [Classifier-Free Guidance: Concept and Intuition](#classifier-free-guidance-concept-and-intuition)
3. [Mathematical Foundations](#mathematical-foundations)
4. [Model Architecture: U-Net in Diffusion Models](#model-architecture-unet-in-diffusion-models)
5. [Training Classifier-Free Diffusion Models](#training-classifier-free-diffusion-models)
6. [Inference with Classifier-Free Guidance](#inference-with-classifier-free-guidance)
7. [Applications](#applications)
    - [Image Inpainting](#image-inpainting)
    - [Image Super-Resolution](#image-super-resolution)
    - [Text-to-Image Generation](#text-to-image-generation)
8. [Pseudocode for Training and Inference](#pseudocode-for-training-and-inference)
9. [Implementation Examples with Code](#implementation-examples-with-code)
    - [U-Net Structure](#unet-structure)
    - [Training Code](#training-code)
    - [Inference Code](#inference-code)
10. [References](#references)

---

## Introduction to Diffusion Models

**Diffusion models** are a class of generative models that iteratively transform noise into data by reversing a diffusion (noise addition) process. They have gained prominence due to their ability to generate high-fidelity images and other data types.

**Key Components:**
- **Forward Diffusion Process:** Gradually adds noise to data over a series of timesteps.
- **Reverse Diffusion Process:** Learns to remove noise step-by-step to generate data from noise.

**Advantages:**
- High-quality and diverse generation.
- Stable training compared to other generative models like GANs.

## Classifier-Free Guidance: Concept and Intuition

**Classifier-Free Guidance** is a technique to control the generation process in diffusion models without relying on external classifiers. Instead, the generative model itself is trained to handle both conditional and unconditional generation, allowing guidance by interpolating between these two modes during inference.

**Intuitions:**
- **Conditional Mode:** Generates data based on specific conditions (e.g., text prompts).
- **Unconditional Mode:** Generates data without any conditions.
- **Guidance:** Enhances adherence to conditions by amplifying the conditional signals over the unconditional ones.

This method simplifies the architecture, reduces computational overhead, and mitigates biases introduced by external classifiers.

## Mathematical Foundations

### Forward Diffusion Process

Given data $x_0$ from the data distribution $q(x_0)$, the forward process adds Gaussian noise over $T$ timesteps:

```math
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t \mathbf{I})
```

Where $\beta_t$ is the noise variance at timestep $t$.

An analytical expression for $q(x_t | x_0)$ is:

```math
q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) \mathbf{I})
```

With $\bar{\alpha}_t = \prod_{s=1}^t (1 - \beta_s)$.

### Reverse Diffusion Process

The model learns to approximate the reverse process $p_\theta(x_{t-1} | x_t, c)$, where $c$ is the condition (e.g., text).

The objective is to minimize the variational bound or, equivalently, use denoising score matching to train the model to predict the noise $\epsilon$:

```math
\mathcal{L} = \mathbb{E}_{x_0, \epsilon, t} \left[ \| \epsilon - \epsilon_\theta(x_t, t, c) \|^2 \right]
```

### Classifier-Free Guidance

During training, the model is occasionally trained without conditions by dropping them with probability $p$. Let $c$ be the condition and $\emptyset$ represent no condition.

The model predicts noise as:

```math
\epsilon_\theta(x_t, t, c) = \epsilon_\theta(x_t, t, c) \quad \text{(conditional)}
```
```math
\epsilon_\theta(x_t, t, \emptyset) \quad \text{(unconditional)}
```

During inference, guidance is performed by:

```math
\epsilon_{\text{guided}} = \epsilon_\theta(x_t, t, c) + w \cdot (\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \emptyset))
```

Where $w$ is the guidance scale. This amplifies the conditional signal, guiding the generation towards desired conditions.

## Model Architecture: U-Net in Diffusion Models

The **U-Net** architecture is pivotal in diffusion models due to its ability to capture multi-scale features through its encoder-decoder structure with skip connections.

**Key Components:**
- **Encoder:** Downsamples the input to capture high-level features.
- **Bottleneck:** Processes the compressed representation.
- **Decoder:** Upsamples to reconstruct the output.
- **Skip Connections:** Facilitate information flow between corresponding encoder and decoder layers.

In classifier-free diffusion models, the condition $c$ (e.g., text embeddings) is incorporated into the U-Net, typically via Adaptive Normalization layers (e.g., FiLM layers) or concatenation with feature maps.

**Illustrative U-Net Structure:**

```
Input: x_t
Condition: c (if conditioned)

Encoder:
  - Conv
  - Downsample
  - Residual Blocks (with condition c)

Bottleneck:
  - Residual Blocks (with condition c)

Decoder:
  - Upsample
  - Residual Blocks (with condition c)
  - Conv

Output: Predicted noise \epsilon_\theta(x_t, t, c)
```

## Training Classifier-Free Diffusion Models

### Training Objectives

The primary objective is to train the model to predict the added noise $\epsilon$ at each timestep $t$:

```math
\mathcal{L} = \mathbb{E}_{x_0, t, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t, c) \|^2 \right]
```

### Incorporating Guidance During Training

To enable classifier-free guidance, the model is trained in both conditional and unconditional modes:

1. **With Condition:** With probability $1 - p$, provide the condition $c$ to the model.
2. **Without Condition:** With probability $p$, replace $c$ with $\emptyset$ (e.g., zero vectors).

This dual training allows the model to learn to generate both conditionally and unconditionally, facilitating guidance during inference.

### Training Steps

1. **Sample a Data Point:** $x_0 \sim q(x_0)$.
2. **Sample Timestep:** $t \sim \text{Uniform}(\{1, 2, ..., T\})$.
3. **Sample Noise:** $\epsilon \sim \mathcal{N}(0, \mathbf{I})$.
4. **Generate Noisy Input:** $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$.
5. **Decide Condition:** With probability $p$, set $c = \emptyset$; else, use the actual condition.
6. **Predict Noise:** $\epsilon_\theta(x_t, t, c)$.
7. **Compute Loss:** $\| \epsilon - \epsilon_\theta(x_t, t, c) \|^2$.
8. **Backpropagate and Update Model Parameters.**

## Inference with Classifier-Free Guidance

During inference, classifier-free guidance modifies the model's predictions to steer the generation towards the desired conditions.

### Guided Prediction

```math
\epsilon_{\text{guided}} = \epsilon_\theta(x_t, t, c) + w \cdot (\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \emptyset))
```

Where:
- $\epsilon_\theta(x_t, t, c)$: Conditional prediction.
- $\epsilon_\theta(x_t, t, \emptyset)$: Unconditional prediction.
- $w$: Guidance scale (controls the strength of conditioning).

### Sampling Process

1. **Initialize:** Start with $x_T \sim \mathcal{N}(0, \mathbf{I})$.
2. **Iteratively Denoise:**
   - For each timestep $t = T, T-1, ..., 1$:
     - Compute $\epsilon_{\text{guided}}$.
     - Estimate $x_{t-1}$ using the guided prediction.
3. **Output:** $x_0$ as the generated data.

## Applications

### Image Inpainting

**Objective:** Fill in missing regions of an image coherently based on the surrounding context.

**Condition $c$:** Masked image where missing regions are marked.

**Process:**
- The model is conditioned on the masked image.
- During generation, the model focuses on reconstructing the missing parts while keeping the known regions intact.

### Image Super-Resolution

**Objective:** Enhance the resolution of a low-resolution image.

**Condition $c$:** Low-resolution image.

**Process:**
- The model is conditioned on the low-resolution input.
- Generates a high-resolution version, maintaining consistency with the low-res input.

### Text-to-Image Generation

**Objective:** Generate images based on textual descriptions.

**Condition $c$:** Text embeddings (e.g., from a transformer model like CLIP).

**Process:**
- The model is conditioned on the text embeddings.
- Generates images that semantically align with the input text.

## Pseudocode for Training and Inference

### General Training Pseudocode

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        x0, c = batch  # x0: images, c: conditions (e.g., masks, low-res images, text)
        t = sample_timesteps(batch_size)
        epsilon = sample_noise(batch_size)
        xt = sqrt_alpha_bar[t] * x0 + sqrt_one_minus_alpha_bar[t] * epsilon
        # Decide whether to use condition or not
        if random() < p_drop_condition:
            c = None
        epsilon_pred = model(xt, t, c)
        loss = mse_loss(epsilon, epsilon_pred)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

### General Inference Pseudocode with Guidance

```python
def sample(model, c, guidance_scale, T, device):
    x = torch.randn(batch_size, channels, height, width).to(device)
    for t in reversed(range(1, T + 1)):
        eps_cond = model(x, t, c)
        eps_uncond = model(x, t, None)
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        x = sample_step(x, eps, t)
    return x
```

### Application-Specific Considerations

- **Image Inpainting:**
  - Condition $c$: Masked image.
  - Ensure masked regions are updated, while known regions are preserved.

- **Image Super-Resolution:**
  - Condition $c$: Low-resolution image.
  - Align high-resolution output with low-res input via upsampling layers.

- **Text-to-Image:**
  - Condition $c$: Text embeddings.
  - Integrate text features into the U-Net via cross-attention or FiLM layers.

## Implementation Examples with Code

Below are simplified PyTorch implementations illustrating the key components of classifier-free diffusion models for different applications. Each example includes inline comments and tensor size annotations for clarity.

### U-Net Structure

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, condition_emb_dim=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.activation = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        if condition_emb_dim:
            self.condition_mlp = nn.Linear(condition_emb_dim, out_channels)
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t, c=None):
        h = self.activation(self.conv1(x))  # [B, C, H, W]
        h = h + self.time_mlp(t).unsqueeze(-1).unsqueeze(-1)  # Add time embedding
        if c is not None:
            h = h + self.condition_mlp(c).unsqueeze(-1).unsqueeze(-1)  # Add condition embedding
        h = self.activation(h)
        h = self.conv2(h)
        return h + self.residual(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=256, condition_emb_dim=None):
        super(UNet, self).__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        if condition_emb_dim:
            self.condition_emb_dim = condition_emb_dim
        else:
            self.condition_emb_dim = 0
        self.encoder1 = ResidualBlock(in_channels, 64, time_emb_dim, self.condition_emb_dim)
        self.encoder2 = ResidualBlock(64, 128, time_emb_dim, self.condition_emb_dim)
        self.bottleneck = ResidualBlock(128, 256, time_emb_dim, self.condition_emb_dim)
        self.decoder1 = ResidualBlock(256 + 128, 128, time_emb_dim, self.condition_emb_dim)
        self.decoder2 = ResidualBlock(128 + 64, 64, time_emb_dim, self.condition_emb_dim)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x, t, c=None):
        t = self.time_mlp(t)  # [B, time_emb_dim]
        # Encoder
        e1 = self.encoder1(x, t, c)  # [B, 64, H, W]
        e2 = self.encoder2(F.max_pool2d(e1, 2), t, c)  # [B, 128, H/2, W/2]
        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e2, 2), t, c)  # [B, 256, H/4, W/4]
        # Decoder
        d1 = self.decoder1(F.interpolate(b, scale_factor=2) + e2, t, c)  # [B, 128, H/2, W/2]
        d2 = self.decoder2(F.interpolate(d1, scale_factor=2) + e1, t, c)  # [B, 64, H, W]
        out = self.final_conv(d2)  # [B, out_channels, H, W]
        return out
```

**Tensor Sizes Example:**

- Input $x$: `[B, 3, 64, 64]`
- Time embedding $t$: `[B, 1]`
- Condition $c$: `[B, C]` (varies per application)
- Output: `[B, 3, 64, 64]` (predicting noise)

### Training Code

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Assume dataset is defined appropriately
# For example, for image inpainting, dataset provides (masked_image, condition)

def train_model(model, dataloader, optimizer, device, num_epochs, T, p_drop_condition=0.1):
    mse_loss = nn.MSELoss()
    model.to(device)
    model.train()
    
    for epoch in range(num_epochs):
        for batch in dataloader:
            x0, c = batch  # x0: [B, C, H, W], c: [B, C_cond]
            x0 = x0.to(device)
            c = c.to(device)
            B = x0.size(0)
            
            # Sample timesteps
            t = torch.randint(1, T+1, (B,), device=device).unsqueeze(-1).float()  # [B, 1]
            
            # Sample noise
            epsilon = torch.randn_like(x0)  # [B, C, H, W]
            
            # Generate noisy inputs
            sqrt_alpha_bar = torch.sqrt(alpha_bar[t]).view(B, 1, 1, 1)
            sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar[t]).view(B, 1, 1, 1)
            xt = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * epsilon  # [B, C, H, W]
            
            # Decide to drop condition
            drop = torch.rand(B, 1, 1, 1, device=device) < p_drop_condition
            c_input = c.clone()
            c_input[drop.squeeze()] = 0  # Set condition to zero where dropped
            
            # Predict noise
            epsilon_pred = model(xt, t.squeeze(), c_input)  # [B, C, H, W]
            
            # Compute loss
            loss = mse_loss(epsilon, epsilon_pred)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# Hyperparameters and other definitions would be required here
```

### Inference Code

```python
def sample_image(model, c, guidance_scale, T, device):
    model.eval()
    with torch.no_grad():
        x = torch.randn(c.size(0), 3, 64, 64).to(device)  # Starting from noise
        for t in reversed(range(1, T + 1)):
            t_tensor = torch.full((c.size(0), 1), t, device=device).float()
            # Predict noise with condition
            eps_cond = model(x, t_tensor, c)  # [B, C, H, W]
            # Predict noise without condition
            eps_uncond = model(x, t_tensor, torch.zeros_like(c))  # [B, C, H, W]
            # Guided noise
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            # Update x
            x = sample_step(x, eps, t)
        return x

def sample_step(x, eps, t):
    # This function implements the reverse diffusion step
    # For simplicity, assume fixed beta schedule and variance
    alpha = alpha[t]
    alpha_bar = alpha_bar[t]
    sqrt_recip_alpha = 1 / torch.sqrt(alpha)
    sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar)
    x_prev = sqrt_recip_alpha * (x - (beta[t] / sqrt_one_minus_alpha) * eps)
    x_prev = x_prev + torch.sqrt(beta[t]) * torch.randn_like(x)  # Add noise
    return x_prev
```

### Image Inpainting Example

**Training Considerations:**
- **Condition $c$:** Masked image where masked regions are set to zero or a constant value.
- **Loss:** Only compute loss on masked regions to focus on inpainting.

**Code Adjustments:**

- Modify the loss to focus on masked regions.
- Ensure the model preserves unmasked regions.

```python
def train_inpainting(model, dataloader, optimizer, device, num_epochs, T, p_drop_condition=0.1):
    mse_loss = nn.MSELoss()
    model.to(device)
    model.train()
    
    for epoch in range(num_epochs):
        for batch in dataloader:
            x0, mask = batch  # x0: [B, C, H, W], mask: [B, 1, H, W]
            x0 = x0.to(device)
            mask = mask.to(device)
            c = x0 * mask  # Condition is masked image
            B = x0.size(0)
            
            # Sample timesteps
            t = torch.randint(1, T+1, (B,), device=device).unsqueeze(-1).float()
            
            # Sample noise
            epsilon = torch.randn_like(x0)
            
            # Generate noisy inputs
            sqrt_alpha_bar = torch.sqrt(alpha_bar[t]).view(B, 1, 1, 1)
            sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar[t]).view(B, 1, 1, 1)
            xt = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * epsilon
            
            # Decide to drop condition
            drop = torch.rand(B, 1, 1, 1, device=device) < p_drop_condition
            c_input = c.clone()
            c_input[drop.squeeze()] = 0
            
            # Predict noise
            epsilon_pred = model(xt, t.squeeze(), c_input)
            
            # Compute loss only on masked regions
            loss = mse_loss(epsilon * mask, epsilon_pred * mask)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Inpainting Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
```

### Image Super-Resolution Example

**Training Considerations:**
- **Condition $c$:** Low-resolution (LR) image.
- **Target $x_0$:** High-resolution (HR) image.
- **Loss:** MSE between predicted noise and actual noise.

**Code Adjustments:**

- Upsample LR images before conditioning.
- Ensure the model can handle different resolutions.

```python
def train_super_resolution(model, dataloader, optimizer, device, num_epochs, T, p_drop_condition=0.1):
    mse_loss = nn.MSELoss()
    model.to(device)
    model.train()
    
    for epoch in range(num_epochs):
        for batch in dataloader:
            lr, hr = batch  # lr: [B, C, H, W], hr: [B, C, H*2, W*2]
            hr = hr.to(device)
            lr = lr.to(device)
            c = lr  # Condition is low-res image
            B = hr.size(0)
            
            # Sample timesteps
            t = torch.randint(1, T+1, (B,), device=device).unsqueeze(-1).float()
            
            # Sample noise
            epsilon = torch.randn_like(hr)
            
            # Generate noisy inputs
            sqrt_alpha_bar = torch.sqrt(alpha_bar[t]).view(B, 1, 1, 1)
            sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar[t]).view(B, 1, 1, 1)
            xt = sqrt_alpha_bar * hr + sqrt_one_minus_alpha_bar * epsilon
            
            # Decide to drop condition
            drop = torch.rand(B, 1, 1, 1, device=device) < p_drop_condition
            c_input = c.clone()
            c_input[drop.squeeze()] = 0
            
            # Predict noise
            epsilon_pred = model(xt, t.squeeze(), c_input)
            
            # Compute loss
            loss = mse_loss(epsilon, epsilon_pred)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Super-Resolution Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
```

### Text-to-Image Generation Example

**Training Considerations:**
- **Condition $c$:** Text embeddings (e.g., from a pre-trained transformer).
- **Loss:** Standard MSE loss between predicted and actual noise.

**Code Adjustments:**

- Integrate text embeddings into the model, possibly via cross-attention layers.
- Ensure text embeddings are properly aligned with the U-Net.

```python
def train_text_to_image(model, dataloader, optimizer, device, num_epochs, T, p_drop_condition=0.1):
    mse_loss = nn.MSELoss()
    model.to(device)
    model.train()
    
    for epoch in range(num_epochs):
        for batch in dataloader:
            x0, text = batch  # x0: [B, C, H, W], text: [B, D]
            x0 = x0.to(device)
            text = text.to(device)
            c = text  # Condition is text embeddings
            B = x0.size(0)
            
            # Sample timesteps
            t = torch.randint(1, T+1, (B,), device=device).unsqueeze(-1).float()
            
            # Sample noise
            epsilon = torch.randn_like(x0)
            
            # Generate noisy inputs
            sqrt_alpha_bar = torch.sqrt(alpha_bar[t]).view(B, 1, 1, 1)
            sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar[t]).view(B, 1, 1, 1)
            xt = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * epsilon
            
            # Decide to drop condition
            drop = torch.rand(B, device=device) < p_drop_condition
            c_input = c.clone()
            c_input[drop] = 0  # Zero out text embeddings where dropped
            
            # Predict noise
            epsilon_pred = model(xt, t.squeeze(), c_input)
            
            # Compute loss
            loss = mse_loss(epsilon, epsilon_pred)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Text-to-Image Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
```

## What is Guidance and How is the Condition Generation Model Trained?

**Guidance** in classifier-free diffusion refers to the technique of steering the generative process towards desired conditions by amplifying the conditional signals. This is achieved by interpolating between the conditional and unconditional predictions during inference.

### Condition Generation Model

The **condition generation model** is essentially the same diffusion model trained to handle both conditional and unconditional scenarios. During training, the model learns to generate data with and without conditions, enabling flexibility during inference.

**Training Steps:**
1. **Conditional Training:** Model learns to generate data based on conditions.
2. **Unconditional Training:** Model learns to generate data without any conditions (i.e., generating from noise).

This dual training empowers the model to understand and separate the influence of conditions from the inherent data distribution, facilitating effective guidance.

## Structure of the Diffusion Model and Guidance Model

### Diffusion Model Structure

- **Input:** Noisy data $x_t$, timestep $t$, condition $c$ (optional).
- **Architecture:** U-Net with:
  - Time embeddings integrated into residual blocks.
  - Condition embeddings integrated via Adaptive Normalization or cross-attention.
- **Output:** Predicted noise $\epsilon_\theta(x_t, t, c)$.

### Guidance Model Structure

The guidance mechanism leverages both the conditional and unconditional predictions from the same diffusion model:

- **Conditional Prediction:** $\epsilon_\theta(x_t, t, c)$.
- **Unconditional Prediction:** $\epsilon_\theta(x_t, t, \emptyset)$.
- **Guided Prediction:** Combination of the two using a guidance scale.

This approach eliminates the need for a separate guidance network, simplifying the architecture.

## Pseudocode for Training and Inference

### Training Pseudocode for Image Inpainting

```python
def train_image_inpainting(model, dataloader, optimizer, device, num_epochs, T, p_drop=0.1):
    model.train()
    for epoch in range(num_epochs):
        for x0, mask in dataloader:
            x0, mask = x0.to(device), mask.to(device)
            c = x0 * mask  # Condition is masked image
            t = sample_timesteps(x0.size(0), T, device)
            epsilon = torch.randn_like(x0)
            xt = compute_xt(x0, epsilon, t)
            c_input = drop_condition(c, p_drop)
            epsilon_pred = model(xt, t, c_input)
            loss = compute_loss(epsilon, epsilon_pred, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

### Inference Pseudocode for Image Inpainting

```python
def inference_image_inpainting(model, masked_image, mask, guidance_scale, T, device):
    model.eval()
    with torch.no_grad():
        c = masked_image * mask
        x = torch.randn_like(masked_image).to(device)
        for t in reversed(range(1, T + 1)):
            t_tensor = torch.tensor([t], device=device).float().expand(masked_image.size(0))
            eps_cond = model(x, t_tensor, c)
            eps_uncond = model(x, t_tensor, torch.zeros_like(c))
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            x = reverse_diffusion_step(x, eps, t)
            x = x * (1 - mask) + masked_image * mask  # Preserve known regions
        return x
```

### Training Pseudocode for Image Super-Resolution

```python
def train_super_resolution(model, dataloader, optimizer, device, num_epochs, T, p_drop=0.1):
    model.train()
    for epoch in range(num_epochs):
        for lr, hr in dataloader:
            lr, hr = lr.to(device), hr.to(device)
            c = lr  # Condition is low-res image
            t = sample_timesteps(hr.size(0), T, device)
            epsilon = torch.randn_like(hr)
            xt = compute_xt(hr, epsilon, t)
            c_input = drop_condition(c, p_drop)
            epsilon_pred = model(xt, t, c_input)
            loss = mse_loss(epsilon, epsilon_pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

### Inference Pseudocode for Image Super-Resolution

```python
def inference_super_resolution(model, lr_image, guidance_scale, T, device):
    model.eval()
    with torch.no_grad():
        c = lr_image
        x = torch.randn(upscale(lr_image)).to(device)  # Initialize with noise
        for t in reversed(range(1, T + 1)):
            t_tensor = torch.tensor([t], device=device).float().expand(lr_image.size(0))
            eps_cond = model(x, t_tensor, c)
            eps_uncond = model(x, t_tensor, torch.zeros_like(c))
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            x = reverse_diffusion_step(x, eps, t)
            x = align_with_lr(x, lr_image)  # Ensure consistency with low-res input
        return x
```

### Training Pseudocode for Text-to-Image

```python
def train_text_to_image(model, dataloader, optimizer, device, num_epochs, T, p_drop=0.1):
    model.train()
    for epoch in range(num_epochs):
        for x0, text in dataloader:
            x0, text = x0.to(device), text.to(device)
            c = text  # Condition is text embedding
            t = sample_timesteps(x0.size(0), T, device)
            epsilon = torch.randn_like(x0)
            xt = compute_xt(x0, epsilon, t)
            c_input = drop_condition(c, p_drop)
            epsilon_pred = model(xt, t, c_input)
            loss = mse_loss(epsilon, epsilon_pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

### Inference Pseudocode for Text-to-Image

```python
def inference_text_to_image(model, text_embedding, guidance_scale, T, device):
    model.eval()
    with torch.no_grad():
        c = text_embedding
        x = torch.randn(desired_image_shape).to(device)
        for t in reversed(range(1, T + 1)):
            t_tensor = torch.tensor([t], device=device).float().expand(text_embedding.size(0))
            eps_cond = model(x, t_tensor, c)
            eps_uncond = model(x, t_tensor, torch.zeros_like(c))
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            x = reverse_diffusion_step(x, eps, t)
        return x
```

## Implementation Examples with Code

### U-Net Structure

The U-Net architecture is central to diffusion models, enabling efficient multi-scale feature extraction and reconstruction.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super(TimeEmbedding, self).__init__()
        self.linear1 = nn.Linear(dim, dim * 4)
        self.linear2 = nn.Linear(dim * 4, dim * 4)

    def forward(self, t):
        # t: [B, 1]
        x = self.linear1(t)
        x = F.silu(x)
        x = self.linear2(x)
        return x  # [B, dim*4]

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, time_dim=256, condition_dim=0):
        super(UNet, self).__init__()
        self.time_mlp = TimeEmbedding(time_dim)
        self.condition_dim = condition_dim
        # Encoder
        self.enc1 = ResidualBlock(in_channels, 64, time_dim, condition_dim)
        self.enc2 = ResidualBlock(64, 128, time_dim, condition_dim)
        self.enc3 = ResidualBlock(128, 256, time_dim, condition_dim)
        # Bottleneck
        self.bottleneck = ResidualBlock(256, 512, time_dim, condition_dim)
        # Decoder
        self.dec3 = ResidualBlock(512 + 256, 256, time_dim, condition_dim)
        self.dec2 = ResidualBlock(256 + 128, 128, time_dim, condition_dim)
        self.dec1 = ResidualBlock(128 + 64, 64, time_dim, condition_dim)
        # Final layer
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x, t, c=None):
        t = self.time_mlp(t)  # [B, time_dim*4]
        # Encoder
        e1 = self.enc1(x, t, c)  # [B, 64, H, W]
        e2 = self.enc2(F.max_pool2d(e1, 2), t, c)  # [B, 128, H/2, W/2]
        e3 = self.enc3(F.max_pool2d(e2, 2), t, c)  # [B, 256, H/4, W/4]
        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e3, 2), t, c)  # [B, 512, H/8, W/8]
        # Decoder
        d3 = self.dec3(F.interpolate(b, scale_factor=2) + e3, t, c)  # [B, 256, H/4, W/4]
        d2 = self.dec2(F.interpolate(d3, scale_factor=2) + e2, t, c)  # [B, 128, H/2, W/2]
        d1 = self.dec1(F.interpolate(d2, scale_factor=2) + e1, t, c)  # [B, 64, H, W]
        out = self.final(d1)  # [B, out_channels, H, W]
        return out
```

**Tensor Sizes Example:**

- Input $x$: `[B, 3, 64, 64]`
- Time embedding $t$: `[B, 256*4]`
- Condition $c$: `[B, C_cond]` (if any)
- Output: `[B, 3, 64, 64]`

### Training Code

**Example for Text-to-Image Generation:**

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Assume TextToImageDataset is defined
# Each batch provides (images, text_embeddings)

def train_text_to_image(model, dataloader, optimizer, device, num_epochs, T, p_drop=0.1):
    mse_loss = nn.MSELoss()
    model.to(device)
    model.train()
    
    for epoch in range(num_epochs):
        for batch_idx, (images, text_embeddings) in enumerate(dataloader):
            images = images.to(device)  # [B, 3, H, W]
            text_embeddings = text_embeddings.to(device)  # [B, D]
            B = images.size(0)
            
            # Sample timesteps uniformly
            t = torch.randint(1, T+1, (B,), device=device).long()  # [B]
            
            # Sample noise
            epsilon = torch.randn_like(images)  # [B, 3, H, W]
            
            # Compute noisy images
            sqrt_alpha_bar = torch.sqrt(alpha_bar[t]).view(B, 1, 1, 1)
            sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar[t]).view(B, 1, 1, 1)
            xt = sqrt_alpha_bar * images + sqrt_one_minus_alpha_bar * epsilon  # [B, 3, H, W]
            
            # Decide to drop condition
            drop = torch.rand(B, device=device) < p_drop
            c_input = text_embeddings.clone()
            c_input[drop] = 0  # Zero out text embeddings where dropped
            
            # Predict noise
            epsilon_pred = model(xt, t.float(), c_input)  # [B, 3, H, W]
            
            # Compute loss
            loss = mse_loss(epsilon, epsilon_pred)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")
```

### Inference Code

**Example for Text-to-Image Generation:**

```python
def sample_text_to_image(model, text_embedding, guidance_scale, T, device):
    model.eval()
    with torch.no_grad():
        B = text_embedding.size(0)
        x = torch.randn(B, 3, 64, 64).to(device)  # Initialize with noise
        for t in reversed(range(1, T + 1)):
            t_tensor = torch.full((B,), t, device=device).float()  # [B]
            # Predict noise with condition
            eps_cond = model(x, t_tensor, text_embedding)  # [B, 3, H, W]
            # Predict noise without condition
            eps_uncond = model(x, t_tensor, torch.zeros_like(text_embedding))  # [B, 3, H, W]
            # Guided noise
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            # Reverse diffusion step (simplified)
            x = reverse_diffusion_step(x, eps, t)
        return x  # Generated images
```

**Note:** The `reverse_diffusion_step` function should implement the specific update rule based on the chosen noise schedule and variance. This implementation is typically framework-specific and requires careful numerical handling to ensure stability.

## References

1. **Ho, J., Jain, A., & Abbeel, P. (2020).** Denoising Diffusion Probabilistic Models. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
2. **Dhariwal, P., & Nichol, A. (2021).** Diffusion Models Beat GANs on Image Synthesis. [arXiv:2105.05233](https://arxiv.org/abs/2105.05233)
3. **Nichol, A., & Dhariwal, P. (2021).** Improved Denoising Diffusion Probabilistic Models. [arXiv:2102.09672](https://arxiv.org/abs/2102.09672)
4. **Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022).** High-Resolution Image Synthesis with Latent Diffusion Models. [arXiv:2112.10752](https://arxiv.org/abs/2112.10752)
5. **Karras, T., Aittala, M., Laine, S., & Lehtinen, J. (2022).** Alias-Free Generative Adversarial Networks. [arXiv:2106.12423](https://arxiv.org/abs/2106.12423)
6. **Saharia, C., Chan, W., Lee, H., Zhang, X., & Wang, X. (2022).** Palette: Image-to-Image Diffusion Models. [arXiv:2201.10010](https://arxiv.org/abs/2201.10010)
7. **Huang, X., Li, C., Chen, M., Xu, W., & Zhang, M. (2022).** Hierarchical Text-Conditional Image Generation with CLIP Latents. [arXiv:2204.06125](https://arxiv.org/abs/2204.06125)
8. **Ramesh, A., Pavlov, M., Goh, G., Gray, S., Voss, C., Radford, A., ... & Sutskever, I. (2021).** Zero-Shot Text-to-Image Generation. [arXiv:2102.12092](https://arxiv.org/abs/2102.12092)
9. **Kingma, D. P., & Welling, M. (2014).** Auto-Encoding Variational Bayes. [arXiv:1312.6114](https://arxiv.org/abs/1312.6114)
10. **Dhariwal, P., & Nichol, A. (2021).** Diffusion Models for Improved Text-to-Image Synthesis. [OpenAI Blog](https://openai.com/blog/improved-diffusion-models)

These references provide foundational and advanced insights into diffusion models, classifier-free guidance, and their applications in generative modeling.
