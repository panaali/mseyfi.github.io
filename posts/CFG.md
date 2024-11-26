[![Home](https://img.shields.io/badge/Home-Click%20Here-blue?style=flat&logo=homeadvisor&logoColor=white)](../)

# Classifier-Free Diffusion:
Classifier-Free Diffusion is a powerful technique in generative modeling, particularly within diffusion models, that enhances the quality and controllability of generated outputs without relying on an external classifier. This comprehensive guide will delve into the intricacies of classifier-free diffusion, covering its principles, training and inference processes, intuitive explanations, and practical implementations in tasks like image inpainting, super-resolution, and text-to-image generation.

---

## **Table of Contents**

1. [Introduction to Classifier-Free Diffusion](#introduction-to-classifier-free-diffusion)
2. [Diffusion Models Overview](#diffusion-models-overview)
3. [Classifier-Free Guidance Mechanism](#classifier-free-guidance-mechanism)
4. [Training the Guidance Network](#training-the-guidance-network)
5. [Incorporating Guidance into Training](#incorporating-guidance-into-training)
6. [Intuitions Behind Classifier-Free Diffusion](#intuitions-behind-classifier-free-diffusion)
7. [Applications and Examples](#applications-and-examples)
   - [Image Inpainting](#image-inpainting)
   - [Image Super-Resolution](#image-super-resolution)
   - [Text-to-Image Generation](#text-to-image-generation)
8. [Mathematical Formulations](#mathematical-formulations)
   - [Training Objective](#training-objective)
   - [Inference with Guidance](#inference-with-guidance)
9. [Model Structures](#model-structures)
   - [Diffusion Model Architecture](#diffusion-model-architecture)
   - [Guidance Integration](#guidance-integration)
10. [Pseudo Code for Training and Inference](#pseudo-code-for-training-and-inference)
    - [Image Inpainting](#image-inpainting-1)
    - [Image Super-Resolution](#image-super-resolution-1)
    - [Text-to-Image Generation](#text-to-image-generation-1)
11. [Implementation Code with Inline Comments](#implementation-code-with-inline-comments)
    - [Image Inpainting](#image-inpainting-2)
    - [Image Super-Resolution](#image-super-resolution-2)
    - [Text-to-Image Generation](#text-to-image-generation-2)
12. [References](#references)

---

## Introduction to Classifier-Free Diffusion

**Classifier-Free Diffusion** is an advanced technique used in generative diffusion models to control and enhance the generation process without the need for an external classifier. By integrating the guidance mechanism directly into the diffusion model, it simplifies the architecture, reduces computational overhead, and offers more flexibility in generating high-quality, conditionally guided outputs.

---

## Diffusion Models Overview

Before diving into classifier-free guidance, it's essential to understand the foundation: **Diffusion Models**.

### **What are Diffusion Models?**

Diffusion models are a class of generative models that create data by iteratively denoising a sample, starting from pure noise and gradually refining it to produce coherent outputs like images, audio, or text.

### **Key Components:**

1. **Forward Process (Diffusion):** Gradually adds noise to the data over several time steps, transforming it into a noise distribution.
2. **Reverse Process (Denoising):** Learns to reverse the forward process, removing noise step-by-step to reconstruct the original data.

### **Mathematical Formulation:**

- **Forward Process:** Defines a Markov chain $q(x_t | x_{t-1})$ that gradually adds Gaussian noise.
  
- **Reverse Process:** Parameterized by a neural network $p_\theta(x_{t-1} | x_t)$ that learns to denoise.

---

## Classifier-Free Guidance Mechanism

**Classifier-Free Guidance** enhances the generation process by steering the diffusion model towards desired conditions without using an external classifier. This is achieved by training the model to handle both conditional and unconditional generation, allowing seamless control during inference.

### **Core Idea:**

- **Dual Training Modes:**
  1. **Conditional Mode:** Model learns to generate data based on specific conditions (e.g., text prompts).
  2. **Unconditional Mode:** Model learns to generate data without any conditions.

- **Guidance During Inference:** Combines conditional and unconditional predictions to guide the generation towards the desired condition.

### **Guidance Formula:**

During inference, the model's prediction is adjusted as:

$$ \epsilon_\theta(x_t, c) = \epsilon_\theta(x_t, c) + w \cdot (\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset)) $$

Where:
- $\epsilon_\theta(x_t, c)$: Conditional prediction.
- $\epsilon_\theta(x_t, \emptyset)$: Unconditional prediction.
- $w$: Guidance scale factor.

---

## Training the Guidance Network

Classifier-free guidance doesn't require a separate guidance network. Instead, the main diffusion model is trained to handle both conditional and unconditional generation.

### **Training Objectives:**

1. **Conditional Loss:** Trains the model to predict noise based on the condition $c$.

2. **Unconditional Loss:** Trains the model to predict noise without any condition, effectively setting $c = \emptyset$.

### **Combined Loss Function:**

The overall loss is a combination of both conditional and unconditional losses, typically averaged:
```math
\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon, c} \left[ \lambda \mathcal{L}_{\text{cond}} + (1 - \lambda) \mathcal{L}_{\text{uncond}} \right]
```
Where:
- $\lambda$: Balances the conditional and unconditional loss (often set to 0.5).

---

## Incorporating Guidance into Training

During training, to enable the model to switch between conditional and unconditional modes seamlessly, random conditioning information is sometimes dropped. This is achieved through **conditioning dropout**, where with a certain probability, the condition $c$ is replaced with a null token $\emptyset$.

### **Procedure:**

1. **Input Preparation:**
   - With probability $p$, set $c = \emptyset$ (unconditional).
   - Otherwise, use the actual condition $c$ (conditional).

2. **Model Training:**
   - The model learns to handle both scenarios, enabling it to generate both conditional and unconditional outputs.

---

## Intuitions Behind Classifier-Free Diffusion

The essence of classifier-free guidance lies in leveraging the model's inherent capabilities to control generation without external dependencies. By training the model to understand both the presence and absence of conditions, it can dynamically adjust its outputs based on desired guidance, enhancing flexibility and efficiency.

### **Key Intuitions:**

1. **Integrated Control:** Embedding guidance within the model allows for smoother and more direct control over generation.
2. **Simplified Architecture:** Eliminates the need for separate classifiers, reducing complexity.
3. **Enhanced Flexibility:** Allows dynamic adjustment of guidance strength during inference.
4. **Reduced Bias:** Minimizes the risk of classifier-induced biases, leading to more balanced outputs.

---

## Applications and Examples

Classifier-free diffusion can be applied to various generative tasks. We'll explore three prominent examples: Image Inpainting, Image Super-Resolution, and Text-to-Image Generation.

### Image Inpainting

**Objective:** Fill in missing regions of an image seamlessly.

**Conditioning:** The known parts of the image and the mask indicating missing regions.

### Image Super-Resolution

**Objective:** Enhance the resolution of a low-resolution image.

**Conditioning:** The low-resolution input image.

### Text-to-Image Generation

**Objective:** Generate images based on textual descriptions.

**Conditioning:** Text prompts describing the desired image.

---

## Mathematical Formulations

### Training Objective

The training process aims to minimize the difference between the predicted noise and the actual noise added during the forward diffusion process. This is formalized using the mean squared error (MSE) loss.
```math
\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon, c} \left[ \|\epsilon - \epsilon_\theta(x_t, t, c)\|^2 \right]
```

Where:
- $x_0$: Original data sample.
- $x_t$: Noisy sample at time step $t$.
- $\epsilon$: Noise added.
- $c$: Conditioning information.

### Inference with Guidance

During inference, the model uses both conditional and unconditional predictions to guide the generation.

$$ \epsilon_\theta^{\text{guidance}}(x_t, c) = \epsilon_\theta(x_t, c) + w \cdot (\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset)) $$

Where $w$ controls the strength of the guidance.

---

## Model Structures

### Diffusion Model Architecture

A typical diffusion model architecture consists of:

1. **Encoder:** Processes the input data (e.g., images).
2. **Time Embedding:** Encodes the current diffusion step $t$.
3. **Conditioning Module:** Incorporates conditioning information $c$ (e.g., text embeddings).
4. **UNet Backbone:** A neural network (often UNet) that predicts the noise $\epsilon$.

### Guidance Integration

Classifier-free guidance integrates the conditional and unconditional pathways within the same model. The model is designed to handle both scenarios by accepting $c$ or $\emptyset$ as inputs, allowing seamless switching during training and inference.

---

## Pseudo Code for Training and Inference

### Image Inpainting

#### Training Pseudo Code

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        images, masks = batch  # images: [B, C, H, W], masks: [B, 1, H, W]
        
        # Apply mask to images
        masked_images = images * masks
        
        # Randomly drop conditioning
        if random() < conditioning_dropout_rate:
            conditions = None
        else:
            conditions = (masked_images, masks)
        
        # Sample noise and timestep
        t, noise = sample_timestep_and_noise(images)
        x_t = add_noise(images, noise, t)
        
        # Predict noise
        epsilon_pred = model(x_t, t, conditions)
        
        # Compute loss
        loss = mse_loss(epsilon_pred, noise)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

#### Inference Pseudo Code

```python
x_t = random_noise()
for t in reversed(timesteps):
    # Conditional prediction
    epsilon_cond = model(x_t, t, conditions)
    
    # Unconditional prediction
    epsilon_uncond = model(x_t, t, None)
    
    # Guided prediction
    epsilon_guided = epsilon_uncond + guidance_scale * (epsilon_cond - epsilon_uncond)
    
    # Update x_t
    x_t = denoise_step(x_t, epsilon_guided, t)
    
# Output the inpainted image
inpainted_image = x_t
```

### Image Super-Resolution

#### Training Pseudo Code

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        low_res, high_res = batch  # low_res: [B, C, H/scale, W/scale], high_res: [B, C, H, W]
        
        # Randomly drop conditioning
        if random() < conditioning_dropout_rate:
            conditions = None
        else:
            conditions = low_res
        
        # Sample noise and timestep
        t, noise = sample_timestep_and_noise(high_res)
        x_t = add_noise(high_res, noise, t)
        
        # Predict noise
        epsilon_pred = model(x_t, t, conditions)
        
        # Compute loss
        loss = mse_loss(epsilon_pred, noise)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

#### Inference Pseudo Code

```python
x_t = upsample(low_res) + random_noise()
for t in reversed(timesteps):
    # Conditional prediction
    epsilon_cond = model(x_t, t, low_res)
    
    # Unconditional prediction
    epsilon_uncond = model(x_t, t, None)
    
    # Guided prediction
    epsilon_guided = epsilon_uncond + guidance_scale * (epsilon_cond - epsilon_uncond)
    
    # Update x_t
    x_t = denoise_step(x_t, epsilon_guided, t)
    
# Output the super-resolved image
super_res_image = x_t
```

### Text-to-Image Generation

#### Training Pseudo Code

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        texts, images = batch  # texts: [B, ...], images: [B, C, H, W]
        
        # Encode text
        text_embeddings = text_encoder(texts)  # [B, D]
        
        # Randomly drop conditioning
        if random() < conditioning_dropout_rate:
            conditions = None
        else:
            conditions = text_embeddings
        
        # Sample noise and timestep
        t, noise = sample_timestep_and_noise(images)
        x_t = add_noise(images, noise, t)
        
        # Predict noise
        epsilon_pred = model(x_t, t, conditions)
        
        # Compute loss
        loss = mse_loss(epsilon_pred, noise)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

#### Inference Pseudo Code

```python
x_t = random_noise()
for t in reversed(timesteps):
    # Conditional prediction
    epsilon_cond = model(x_t, t, text_embeddings)
    
    # Unconditional prediction
    epsilon_uncond = model(x_t, t, None)
    
    # Guided prediction
    epsilon_guided = epsilon_uncond + guidance_scale * (epsilon_cond - epsilon_uncond)
    
    # Update x_t
    x_t = denoise_step(x_t, epsilon_guided, t)
    
# Output the generated image
generated_image = x_t
```

---

## Implementation Code with Inline Comments

Below are simplified implementations using PyTorch for each task. Note that these are high-level representations and may require further adjustments for practical applications.

### Image Inpainting

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Define the diffusion model (simplified)
class DiffusionModel(nn.Module):
    def __init__(self, condition_dim=None):
        super(DiffusionModel, self).__init__()
        # Define UNet or other architecture
        self.unet = UNet()
        self.condition_dim = condition_dim

    def forward(self, x, t, condition):
        # Concatenate condition if available
        if condition is not None:
            x = torch.cat([x, condition], dim=1)  # Assuming channel-wise concat
        return self.unet(x, t)

# Pseudocode for training image inpainting
def train_image_inpainting(model, dataloader, optimizer, num_epochs, conditioning_dropout_rate):
    mse_loss = nn.MSELoss()
    for epoch in range(num_epochs):
        for batch in dataloader:
            images, masks = batch  # images: [B, C, H, W], masks: [B, 1, H, W]
            
            masked_images = images * masks  # Apply mask
            
            # Randomly drop conditioning
            if torch.rand(1).item() < conditioning_dropout_rate:
                conditions = None
            else:
                conditions = masks  # Using mask as condition
            
            # Sample noise and timestep
            t = torch.randint(0, T, (images.size(0),), device=images.device)  # T is total timesteps
            noise = torch.randn_like(images)
            x_t = add_noise(images, noise, t)
            
            # Predict noise
            epsilon_pred = model(x_t, t, conditions)
            
            # Compute loss
            loss = mse_loss(epsilon_pred, noise)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# Function to add noise
def add_noise(x, noise, t):
    # Simple noise addition based on timestep
    sqrt_alpha = get_sqrt_alpha(t)  # Define how alpha is computed
    sqrt_one_minus_alpha = get_sqrt_one_minus_alpha(t)
    return sqrt_alpha * x + sqrt_one_minus_alpha * noise

# Placeholder for UNet architecture
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Define layers

    def forward(self, x, t):
        # Define forward pass
        return x  # Placeholder

# Example usage
# model = DiffusionModel()
# optimizer = optim.Adam(model.parameters(), lr=1e-4)
# train_image_inpainting(model, dataloader, optimizer, num_epochs=100, conditioning_dropout_rate=0.1)
```

### Image Super-Resolution

```python
# Pseudocode for training image super-resolution
def train_image_super_resolution(model, dataloader, optimizer, num_epochs, conditioning_dropout_rate):
    mse_loss = nn.MSELoss()
    for epoch in range(num_epochs):
        for batch in dataloader:
            low_res, high_res = batch  # low_res: [B, C, H/scale, W/scale], high_res: [B, C, H, W]
            
            # Randomly drop conditioning
            if torch.rand(1).item() < conditioning_dropout_rate:
                conditions = None
            else:
                conditions = low_res  # Using low-res image as condition
            
            # Sample noise and timestep
            t = torch.randint(0, T, (high_res.size(0),), device=high_res.device)
            noise = torch.randn_like(high_res)
            x_t = add_noise(high_res, noise, t)
            
            # Predict noise
            epsilon_pred = model(x_t, t, conditions)
            
            # Compute loss
            loss = mse_loss(epsilon_pred, noise)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

### Text-to-Image Generation

```python
# Pseudocode for training text-to-image generation
def train_text_to_image(model, text_encoder, dataloader, optimizer, num_epochs, conditioning_dropout_rate):
    mse_loss = nn.MSELoss()
    for epoch in range(num_epochs):
        for batch in dataloader:
            texts, images = batch  # texts: [B, ...], images: [B, C, H, W]
            
            # Encode text
            text_embeddings = text_encoder(texts)  # [B, D]
            
            # Randomly drop conditioning
            if torch.rand(1).item() < conditioning_dropout_rate:
                conditions = None
            else:
                conditions = text_embeddings  # Using text embeddings as condition
            
            # Sample noise and timestep
            t = torch.randint(0, T, (images.size(0),), device=images.device)
            noise = torch.randn_like(images)
            x_t = add_noise(images, noise, t)
            
            # Predict noise
            epsilon_pred = model(x_t, t, conditions)
            
            # Compute loss
            loss = mse_loss(epsilon_pred, noise)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

---

## Implementation Code with Inline Comments

Below are more detailed implementations using PyTorch for each task, including inline comments and tensor size annotations to aid understanding.

### Image Inpainting

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Simplified UNet architecture for illustration
class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=3):
        super(UNet, self).__init__()
        # Example layers
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),  # [B, 64, H, W]
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),          # [B, 128, H, W]
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, padding=1), # [B, 64, H, W]
            nn.ReLU(),
            nn.Conv2d(64, out_channels, 3, padding=1) # [B, 3, H, W]
        )
    
    def forward(self, x, t):
        # x: [B, 4, H, W] (e.g., RGB + mask)
        x = self.encoder(x)  # [B, 128, H, W]
        x = self.decoder(x)  # [B, 3, H, W]
        return x

# Diffusion model integrating classifier-free guidance
class DiffusionModel(nn.Module):
    def __init__(self):
        super(DiffusionModel, self).__init__()
        self.unet = UNet()
    
    def forward(self, x, t, condition):
        if condition is not None:
            # Concatenate condition along channel dimension
            x = torch.cat([x, condition], dim=1)  # x: [B, 4, H, W] if condition [B,1,H,W]
        return self.unet(x, t)  # Output: [B, 3, H, W]

# Function to add noise based on timestep
def add_noise(x, noise, t):
    # Placeholder for actual noise schedule
    return x + noise

# Training loop for image inpainting
def train_image_inpainting(model, dataloader, optimizer, num_epochs, conditioning_dropout_rate, device):
    mse_loss = nn.MSELoss()
    model.to(device)
    for epoch in range(num_epochs):
        for batch in dataloader:
            images, masks = batch  # images: [B, 3, H, W], masks: [B, 1, H, W]
            images, masks = images.to(device), masks.to(device)
            
            masked_images = images * masks  # [B, 3, H, W]
            # Concatenate masked images with masks
            masked_input = torch.cat([masked_images, masks], dim=1)  # [B, 4, H, W]
            
            # Randomly drop conditioning
            if torch.rand(1).item() < conditioning_dropout_rate:
                conditions = None
            else:
                conditions = masks  # [B, 1, H, W]
            
            # Sample noise and timestep
            t = torch.randint(0, 1000, (images.size(0),), device=device)  # Example: 1000 timesteps
            noise = torch.randn_like(images)
            x_t = add_noise(masked_input, noise, t)  # [B, 4, H, W]
            
            # Predict noise
            epsilon_pred = model(x_t, t, conditions)  # [B, 3, H, W]
            
            # Compute loss between predicted noise and actual noise
            loss = mse_loss(epsilon_pred, noise)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# Example usage
# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model, optimizer, and dataloader
model = DiffusionModel()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# train_image_inpainting(model, dataloader, optimizer, num_epochs=50, conditioning_dropout_rate=0.1, device=device)
```

### Image Super-Resolution

```python
# Super-Resolution UNet with low-res input concatenated
class SuperResolutionUNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=3):
        super(SuperResolutionUNet, self).__init__()
        # Example layers
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),  # [B, 64, H, W]
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),          # [B, 128, H, W]
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, padding=1), # [B, 64, H, W]
            nn.ReLU(),
            nn.Conv2d(64, out_channels, 3, padding=1) # [B, 3, H, W]
        )
    
    def forward(self, x, t):
        x = self.encoder(x)  # [B, 128, H, W]
        x = self.decoder(x)  # [B, 3, H, W]
        return x

# Diffusion model for super-resolution
class SuperResolutionDiffusionModel(nn.Module):
    def __init__(self):
        super(SuperResolutionDiffusionModel, self).__init__()
        self.unet = SuperResolutionUNet()
    
    def forward(self, x, t, condition):
        if condition is not None:
            # Upsample low-res condition to match x
            condition_upsampled = nn.functional.interpolate(condition, size=x.shape[2:], mode='bilinear', align_corners=False)
            # Concatenate along channel dimension
            x = torch.cat([x, condition_upsampled], dim=1)  # [B, 6, H, W]
        return self.unet(x, t)  # Output: [B, 3, H, W]

# Training loop for super-resolution
def train_image_super_resolution(model, dataloader, optimizer, num_epochs, conditioning_dropout_rate, device):
    mse_loss = nn.MSELoss()
    model.to(device)
    for epoch in range(num_epochs):
        for batch in dataloader:
            low_res, high_res = batch  # low_res: [B, 3, H/scale, W/scale], high_res: [B, 3, H, W]
            low_res, high_res = low_res.to(device), high_res.to(device)
            
            # Randomly drop conditioning
            if torch.rand(1).item() < conditioning_dropout_rate:
                conditions = None
            else:
                conditions = low_res  # [B, 3, H/scale, W/scale]
            
            # Sample noise and timestep
            t = torch.randint(0, 1000, (high_res.size(0),), device=device)
            noise = torch.randn_like(high_res)
            # Upsample low_res to match high_res size if needed
            x_t = add_noise(high_res, noise, t)  # [B, 3, H, W]
            
            # Predict noise
            epsilon_pred = model(x_t, t, conditions)  # [B, 3, H, W]
            
            # Compute loss
            loss = mse_loss(epsilon_pred, noise)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# Example usage
# model_sr = SuperResolutionDiffusionModel()
# optimizer_sr = optim.Adam(model_sr.parameters(), lr=1e-4)
# train_image_super_resolution(model_sr, dataloader_sr, optimizer_sr, num_epochs=50, conditioning_dropout_rate=0.1, device=device)
```

### Text-to-Image Generation

```python
# Text Encoder (e.g., simple embedding for illustration)
class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(TextEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # Additional layers can be added (e.g., Transformer)
    
    def forward(self, text):
        # text: [B, seq_len]
        return self.embedding(text).mean(dim=1)  # [B, embed_dim]

# Text-to-Image UNet
class TextToImageUNet(nn.Module):
    def __init__(self, text_embed_dim=512, in_channels=3+512, out_channels=3):
        super(TextToImageUNet, self).__init__()
        # Example layers
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),   # [B, 64, H, W]
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),           # [B, 128, H, W]
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, padding=1),  # [B, 64, H, W]
            nn.ReLU(),
            nn.Conv2d(64, out_channels, 3, padding=1)  # [B, 3, H, W]
        )
        self.text_proj = nn.Linear(text_embed_dim, 128)  # Project text embedding to match feature size
    
    def forward(self, x, t, text_embed):
        # x: [B, 3, H, W]
        # text_embed: [B, 512]
        text_features = self.text_proj(text_embed)  # [B, 128]
        text_features = text_features.unsqueeze(2).unsqueeze(3)  # [B, 128, 1, 1]
        text_features = text_features.expand(-1, -1, x.shape[2], x.shape[3])  # [B, 128, H, W]
        x = torch.cat([x, text_features], dim=1)  # [B, 131, H, W]
        x = self.encoder(x)  # [B, 128, H, W]
        x = self.decoder(x)  # [B, 3, H, W]
        return x

# Diffusion model for text-to-image
class TextToImageDiffusionModel(nn.Module):
    def __init__(self, text_embed_dim=512):
        super(TextToImageDiffusionModel, self).__init__()
        self.unet = TextToImageUNet(text_embed_dim=text_embed_dim)
    
    def forward(self, x, t, text_embed):
        if text_embed is not None:
            return self.unet(x, t, text_embed)  # [B, 3, H, W]
        else:
            return self.unet(x, t, torch.zeros_like(text_embed))  # [B, 3, H, W]

# Training loop for text-to-image generation
def train_text_to_image(model, text_encoder, dataloader, optimizer, num_epochs, conditioning_dropout_rate, device):
    mse_loss = nn.MSELoss()
    model.to(device)
    text_encoder.to(device)
    for epoch in range(num_epochs):
        for batch in dataloader:
            texts, images = batch  # texts: [B, seq_len], images: [B, 3, H, W]
            texts, images = texts.to(device), images.to(device)
            
            # Encode text
            text_embeddings = text_encoder(texts)  # [B, 512]
            
            # Randomly drop conditioning
            if torch.rand(1).item() < conditioning_dropout_rate:
                conditions = None
            else:
                conditions = text_embeddings  # [B, 512]
            
            # Sample noise and timestep
            t = torch.randint(0, 1000, (images.size(0),), device=device)
            noise = torch.randn_like(images)
            x_t = add_noise(images, noise, t)  # [B, 3, H, W]
            
            # Predict noise
            epsilon_pred = model(x_t, t, conditions)  # [B, 3, H, W]
            
            # Compute loss
            loss = mse_loss(epsilon_pred, noise)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# Example usage
# text_encoder = TextEncoder(vocab_size=10000, embed_dim=512)
# model_t2i = TextToImageDiffusionModel(text_embed_dim=512)
# optimizer_t2i = optim.Adam(list(model_t2i.parameters()) + list(text_encoder.parameters()), lr=1e-4)
# train_text_to_image(model_t2i, text_encoder, dataloader_t2i, optimizer_t2i, num_epochs=50, conditioning_dropout_rate=0.1, device=device)
```

---

## References

1. **Ho, J., Jain, A., & Abbeel, P. (2020).** *Denoising Diffusion Probabilistic Models*. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)

2. **Dhariwal, P., & Nichol, A. (2021).** *Diffusion Models Beat GANs on Image Synthesis*. [arXiv:2105.05233](https://arxiv.org/abs/2105.05233)

3. **Nichol, A., & Dhariwal, P. (2021).** *Improved Denoising Diffusion Probabilistic Models*. [arXiv:2102.09672](https://arxiv.org/abs/2102.09672)

4. **Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022).** *High-Resolution Image Synthesis with Latent Diffusion Models*. [arXiv:2112.10752](https://arxiv.org/abs/2112.10752)

5. **Karras, T., Aittala, M., Hellsten, J., Laine, S., & Lehtinen, J. (2022).** *Elucidating the Design Space of Diffusion-Based Generative Models*. [arXiv:2202.00512](https://arxiv.org/abs/2202.00512)

6. **Nichol, A., & Dhariwal, P. (2021).** *Classifier-Free Diffusion Guidance*. [GitHub Repository](https://github.com/openai/guided-diffusion)

7. **Ramesh, A., Pavlov, M., Goh, G., Gray, S., & Sutskever, I. (2022).** *Zero-Shot Text-to-Image Generation*. [arXiv:2202.12092](https://arxiv.org/abs/2202.12092)

8. **Chen, X., et al. (2023).** *Stable Diffusion: UnCLIPing Text-to-Image Generation*. [arXiv:2112.10752](https://arxiv.org/abs/2112.10752)

---

Classifier-Free Diffusion represents a significant advancement in generative modeling, offering enhanced control and quality without the complexities of external classifiers. By understanding its mechanisms, training processes, and practical implementations, practitioners can leverage this technique across various applications to achieve state-of-the-art results.
