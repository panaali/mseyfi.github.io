[![Home](https://img.shields.io/badge/Home-Click%20Here-blue?style=flat&logo=homeadvisor&logoColor=white)](../)

## [![CV](https://img.shields.io/badge/CV-Selected_Topics_in_Generative_AI-green?style=for-the-badge&logo=github)](../main_page/GenAI)

# Classifier-Free Diffusion Models: A Comprehensive Guide!

1. **Introduction to Classifier-Free Diffusion Models**
2. **Diffusion Model Structure**
3. **Classifier-Free Guidance: Concepts and Mathematics**
4. **Incorporating Guidance into Training**
5. **Applications and Unet Architecture Variations**
   - Image Super-Resolution
   - Image Inpainting
   - Text-to-Image Generation
6. **Training and Inference Formulations**
7. **Pseudo Code for Training and Inference**
8. **Detailed Code Examples with Inline Comments**
   - Unet Structure
   - Training and Inference for Each Application
9. **References**

---

## 1. Introduction to Classifier-Free Diffusion Models

**Classifier-Free Diffusion Models** are a class of generative models that utilize diffusion processes to generate high-quality data (e.g., images) without relying on external classifiers for guidance. Instead, the guidance mechanism is integrated directly into the diffusion model, enhancing controllability and simplifying the architecture.

Key Advantages:
- **Simplicity**: Eliminates the need for separate classifier models.
- **Efficiency**: Streamlines training and inference processes.
- **Flexibility**: Allows dynamic adjustment of guidance strength.
- **Reduced Bias**: Minimizes biases introduced by external classifiers.

---

## 2. Diffusion Model Structure

### **Overview**

Diffusion models generate data by reversing a gradual noising process. They consist of two main processes:

1. **Forward (Noising) Process**: Adds noise to data over a series of time steps.
2. **Reverse (Denoising) Process**: Removes noise to reconstruct the original data.

### **Mathematical Formulation**

Let $\mathbf{x}_0$ be the original data sample (e.g., an image). The forward process adds Gaussian noise over $T$ time steps:


$$ q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I}) $$

where $\beta_t$ is the noise variance schedule.

The reverse process aims to model $p_\theta(\mathbf{x}_{t-1}\|\mathbf{x}_t)$, typically parameterized as,:

$$ p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \mu_\theta(\mathbf{x}_t, t), \Sigma_\theta(\mathbf{x}_t, t)) $$

The model is trained to predict the noise $\epsilon$ added at each step.

### **Unet Architecture**

A **Unet** is commonly used as the backbone for diffusion models due to its ability to capture multi-scale features. It consists of:

- **Encoder**: Downsamples the input while increasing feature channels.
- **Bottleneck**: Processes the most compressed representation.
- **Decoder**: Upsamples back to the original resolution, integrating features from the encoder via skip connections.

---

## 3. Classifier-Free Guidance: Concepts and Mathematics

### **Conceptual Overview**

Classifier-Free Guidance enhances the generative process by conditioning the diffusion model on specific inputs (e.g., text prompts) without using an external classifier. It achieves this by training the model in both conditional and unconditional modes and then interpolating their outputs during inference.

### **Mathematical Formulation**

Let $\mathbf{c}$ represent the conditioning information (e.g., text prompt). The model is trained to predict the noise $\epsilon$ under two scenarios:

1. **Conditional Prediction**: $\epsilon_\theta(\mathbf{x}_t, t, \mathbf{c})$
2. **Unconditional Prediction**: $\epsilon_\theta(\mathbf{x}_t, t, \text{null})$

During inference, the guided prediction is:

$$ \epsilon_{\text{guided}} = \epsilon_\theta(\mathbf{x}_t, t, \mathbf{null}) + s \cdot \left( \epsilon_\theta(\mathbf{x}_t, t, \mathbf{c}) - \epsilon_\theta(\mathbf{x}_t, t, \text{null}) \right) $$

where $s$ is the guidance scale controlling the strength of conditioning.

### **Intuition Behind the Formula**

- **Unconditional Prediction**: Represents the general denoising without specific guidance.
- **Conditional Prediction**: Incorporates the desired attributes via $\mathbf{c}$.
- **Difference Term**: $$\epsilon_\theta(\mathbf{x}_t, t, \mathbf{c}) - \epsilon_\theta(\mathbf{x}_t, t, \text{null})$$ captures the influence of the condition.
- **Guided Prediction**: Enhances the conditional prediction by scaling the influence of the condition.

---

## 4. Incorporating Guidance into Training

### **Training Procedure**

To enable classifier-free guidance, the model is trained to handle both conditional and unconditional scenarios:

1. **Randomly Drop Conditions**: With a certain probability (e.g., 10%), the conditioning information $\mathbf{c}$ is replaced with a null value during training.
   
$$  \mathbf{c}' = \begin{cases} \mathbf{c} & \text{with probability } p \\ \text{null} & \text{with probability } 1 - p \end{cases} $$

2. **Objective Function**: The model minimizes the difference between the predicted noise and the actual noise added during the forward process, regardless of whether $\mathbf{c}$ is present.

   
$$ \mathcal{L} = \mathbb{E}_{t, \mathbf{x}_0, \epsilon, \mathbf{c}'} \left[ \|\epsilon - \epsilon_\theta(\mathbf{x}_t, t, \mathbf{c}')\|^2 \right] $$

### **Intuition**

By training the model to handle both conditional and unconditional inputs, it learns to separate the intrinsic data distribution from the influence of the condition. This separation is crucial for effective guidance during inference.

---

## 5. Applications and Unet Architecture Variations

Classifier-Free Diffusion Models can be adapted for various tasks by modifying the conditioning mechanism and the Unet architecture accordingly.

### **Common Steps Across Applications**

1. **Conditioning Integration**: Incorporate conditioning information into the Unet.
2. **Noise Addition**: Apply the forward diffusion process to the input data.
3. **Model Training**: Train the Unet to predict the noise based on the noisy input and conditioning.
4. **Guided Generation**: Use classifier-free guidance during inference to steer the generation.

### **A. Image Super-Resolution**

**Objective**: Enhance the resolution of a low-resolution (LR) image to a high-resolution (HR) image.

**Conditioning**: The LR image serves as the condition.

**Unet Integration**:
- Concatenate the LR image with the noisy HR image as input.
- Alternatively, inject the LR features at multiple Unet layers via cross-attention or feature concatenation.

**Example Structure**:

```plaintext
Input: Concatenated [Noisy HR Image, LR Image]
Encoder: Processes concatenated input
Bottleneck: Handles multi-scale features
Decoder: Reconstructs HR Image
Skip Connections: From encoder to decoder
```

### **B. Image Inpainting**

**Objective**: Fill in missing regions of an image based on the available context.

**Conditioning**: The masked image (with missing regions) serves as the condition.

**Unet Integration**:
- Provide the masked image as an additional channel.
- Use attention mechanisms to focus on unmasked regions.

**Example Structure**:

```plaintext
Input: Concatenated [Noisy Image, Masked Image]
Encoder: Processes concatenated input
Bottleneck: Integrates contextual information
Decoder: Reconstructs the complete image
Skip Connections: From encoder to decoder
```

### **C. Text-to-Image Generation**

**Objective**: Generate images based on textual descriptions.

**Conditioning**: Text embeddings derived from a language model (e.g., CLIP).

**Unet Integration**:
- Incorporate text embeddings via cross-attention layers.
- Embed text conditioning at multiple Unet layers.

**Example Structure**:

```plaintext
Input: Noisy Image
Condition: Text Embedding
Unet Layers: Integrate text via cross-attention
Decoder: Generates image conditioned on text
Skip Connections: From encoder to decoder
```

---

## 6. Training and Inference Formulations

### **Training Objective**

The training aims to minimize the mean squared error between the predicted noise and the actual noise added during the forward process.

$$ \mathcal{L} = \mathbb{E}_{t, \mathbf{x}_0, \epsilon, \mathbf{c}'} \left[ \|\epsilon - \epsilon_\theta(\mathbf{x}_t, t, \mathbf{c}')\|^2 \right] $$

### **Inference Process**

1. **Initialization**: Start with a sample of pure noise $\mathbf{x}_T \sim \mathcal{N}(0, \mathbf{I})$.

2. **Reverse Denoising**: Iteratively apply the denoising step from $t = T$ to $t = 1$:

   
   $$ \mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_{\text{guided}} \right) + \sigma_t \mathbf{z} $$

   where $\epsilon_{\text{guided}}$ is the guided noise prediction:


$$ \epsilon_{\text{guided}} = \epsilon_\theta(\mathbf{x}_t, t, \mathbf{c}) + s \cdot \left( \epsilon_\theta(\mathbf{x}_t, t, \mathbf{c}) - \epsilon_\theta(\mathbf{x}_t, t, \text{null}) \right) $$

   and $\mathbf{z} \sim \mathcal{N}(0, \mathbf{I})$.

3. **Termination**: The final $\mathbf{x}_0$ is the generated data sample.

### **Guidance Scale $s$**

- Controls the strength of the conditioning.
- Higher $s$ leads to outputs that more closely follow the condition but may reduce diversity.

---

## 7. Pseudo Code for Training and Inference

### **A. Training Pseudo Code**

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        # Sample data and condition
        x0, c = batch['image'], batch['condition']
        
        # Sample time step
        t = sample_uniformly(1, T)
        
        # Sample noise
        epsilon = sample_noise()
        
        # Forward diffusion
        xt = sqrt_alpha_cumprod[t] * x0 + sqrt_one_minus_alpha_cumprod[t] * epsilon
        
        # Randomly drop condition
        if random < drop_prob:
            c_prime = null_condition
        else:
            c_prime = c
        
        # Predict noise
        epsilon_pred = model(xt, t, c_prime)
        
        # Compute loss
        loss = MSE(epsilon_pred, epsilon)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### **B. Inference Pseudo Code**

```python
# Initialize with noise
xT = sample_noise()
xt = xT

for t in reversed(range(1, T+1)):
    # Predict noise
    epsilon_theta_c = model(xt, t, c)
    epsilon_theta_null = model(xt, t, null_condition)
    
    # Guided noise prediction
    epsilon_guided = epsilon_theta_c + s * (epsilon_theta_c - epsilon_theta_null)
    
    # Compute denoised sample
    x_prev = (xt - beta[t] / sqrt(1 - alpha_cumprod[t]) * epsilon_guided) / sqrt(alpha[t])
    
    # Add noise if not the last step
    if t > 1:
        x_prev += sigma[t] * sample_noise()
    
    xt = x_prev

# Output generated sample
x0 = xt
```

---

## 8. Detailed Code Examples with Inline Comments

We'll provide detailed PyTorch code examples for the Unet structure and training/inference processes for each application: image inpainting, super-resolution, and text-to-image generation.

### **A. Unet Structure**

Here's a simplified Unet implementation with support for conditioning via cross-attention. We'll focus on the text-to-image application, but the structure can be adapted for other tasks.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, cond_emb_dim=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_emb = nn.Linear(time_emb_dim, out_channels)
        self.cond_emb = nn.Linear(cond_emb_dim, out_channels) if cond_emb_dim else None
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x, t_emb, c_emb=None):
        out = self.conv1(x)
        out += self.time_emb(t_emb).unsqueeze(-1).unsqueeze(-1)
        if self.cond_emb and c_emb is not None:
            out += self.cond_emb(c_emb).unsqueeze(-1).unsqueeze(-1)
        out = self.relu(out)
        out = self.conv2(out)
        return self.relu(out + self.residual_conv(x))

class CrossAttention(nn.Module):
    def __init__(self, in_channels, cond_dim, num_heads=8):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.scale = (in_channels // num_heads) ** -0.5
        self.to_q = nn.Linear(in_channels, in_channels)
        self.to_k = nn.Linear(cond_dim, in_channels)
        self.to_v = nn.Linear(cond_dim, in_channels)
        self.proj = nn.Linear(in_channels, in_channels)
    
    def forward(self, x, c):
        batch, channels, height, width = x.size()
        q = self.to_q(x).view(batch, channels, height * width).permute(0, 2, 1)  # [B, HW, C]
        k = self.to_k(c).view(batch, channels, -1).permute(0, 2, 1)            # [B, C', C]
        v = self.to_v(c).view(batch, channels, -1).permute(0, 2, 1)            # [B, C', C]
        
        attn = torch.bmm(q, k.transpose(1, 2)) * self.scale                  # [B, HW, C']
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, v)                                             # [B, HW, C]
        out = out.permute(0, 2, 1).view(batch, channels, height, width)
        return self.proj(out)

class Unet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, time_emb_dim=256, cond_emb_dim=512, base_channels=64):
        super(Unet, self).__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        self.encoder1 = ResidualBlock(in_channels, base_channels, time_emb_dim, cond_emb_dim)
        self.encoder2 = ResidualBlock(base_channels, base_channels*2, time_emb_dim, cond_emb_dim)
        self.encoder3 = ResidualBlock(base_channels*2, base_channels*4, time_emb_dim, cond_emb_dim)
        
        self.cross_attn = CrossAttention(base_channels*4, cond_emb_dim)
        
        self.bottleneck = ResidualBlock(base_channels*4, base_channels*4, time_emb_dim, cond_emb_dim)
        
        self.decoder3 = ResidualBlock(base_channels*4 + base_channels*2, base_channels*2, time_emb_dim, cond_emb_dim)
        self.decoder2 = ResidualBlock(base_channels*2 + base_channels, base_channels, time_emb_dim, cond_emb_dim)
        self.decoder1 = ResidualBlock(base_channels + in_channels, base_channels, time_emb_dim, cond_emb_dim)
        
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)
    
    def forward(self, x, t, c):
        # Embed time
        t_emb = self.time_mlp(t.unsqueeze(-1))  # [B, time_emb_dim]
        
        # Encoder
        e1 = self.encoder1(x, t_emb, c)        # [B, base_channels, H, W]
        e2 = self.encoder2(F.max_pool2d(e1, 2), t_emb, c)  # [B, base_channels*2, H/2, W/2]
        e3 = self.encoder3(F.max_pool2d(e2, 2), t_emb, c)  # [B, base_channels*4, H/4, W/4]
        
        # Cross Attention
        e3 = self.cross_attn(e3, c)           # [B, base_channels*4, H/4, W/4]
        
        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e3, 2), t_emb, c)  # [B, base_channels*4, H/8, W/8]
        
        # Decoder
        d3 = F.interpolate(b, scale_factor=2, mode='nearest')  # [B, base_channels*4, H/4, W/4]
        d3 = torch.cat([d3, e3], dim=1)                      # [B, base_channels*8, H/4, W/4]
        d3 = self.decoder3(d3, t_emb, c)                     # [B, base_channels*2, H/4, W/4]
        
        d2 = F.interpolate(d3, scale_factor=2, mode='nearest')  # [B, base_channels*2, H/2, W/2]
        d2 = torch.cat([d2, e2], dim=1)                        # [B, base_channels*4, H/2, W/2]
        d2 = self.decoder2(d2, t_emb, c)                       # [B, base_channels, H/2, W/2]
        
        d1 = F.interpolate(d2, scale_factor=2, mode='nearest')  # [B, base_channels, H, W]
        d1 = torch.cat([d1, e1], dim=1)                        # [B, base_channels + in_channels, H, W]
        d1 = self.decoder1(d1, t_emb, c)                       # [B, base_channels, H, W]
        
        out = self.final_conv(d1)                               # [B, out_channels, H, W]
        return out
```

**Explanation and Tensor Sizes**:

- **Input**: `[B, C, H, W]` where `B` is batch size, `C` is channels, `H` and `W` are height and width.
- **Time Embedding**: `t` is a scalar representing the time step, embedded into a higher-dimensional space.
- **Residual Blocks**: Each block processes the input, integrates time and condition embeddings, and maintains residual connections.
- **Cross Attention**: Allows the model to focus on specific aspects of the condition (e.g., text embeddings) while processing the image.
- **Decoder**: Reconstructs the image by upsampling and integrating features from the encoder.

### **B. Training Code for Text-to-Image Generation**

Here's a simplified training loop for text-to-image generation using the above Unet.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Assume the following components are defined:
# - dataset: a PyTorch Dataset yielding {'image': image_tensor, 'text': text_embedding}
# - tokenizer: a tokenizer converting text to embeddings
# - model: an instance of the Unet class
# - optimizer: an optimizer instance
# - device: 'cuda' or 'cpu'

def train_diffusion_model(model, dataset, tokenizer, num_epochs=100, batch_size=32, lr=1e-4, device='cuda', drop_prob=0.1):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    
    for epoch in range(num_epochs):
        for batch in dataloader:
            # Move data to device
            x0 = batch['image'].to(device)             # [B, 3, H, W]
            text = batch['text']                       # [B, seq_len]
            c = tokenizer(text).to(device)             # [B, cond_dim]
            
            # Sample time step uniformly
            t = torch.randint(1, T+1, (x0.size(0),), device=device)  # [B]
            
            # Sample noise
            epsilon = torch.randn_like(x0)              # [B, 3, H, W]
            
            # Forward diffusion: x_t = sqrt_alpha_cumprod[t] * x0 + sqrt_one_minus_alpha_cumprod[t] * epsilon
            sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod[t]).view(-1, 1, 1, 1)
            sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alpha_cumprod[t]).view(-1, 1, 1, 1)
            xt = sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha_cumprod * epsilon  # [B, 3, H, W]
            
            # Randomly drop condition
            mask = torch.rand(x0.size(0), device=device) < drop_prob
            c_prime = c.clone()
            c_prime[mask] = 0  # Assuming 0 represents the null condition
            
            # Predict noise
            epsilon_pred = model(xt, t, c_prime)        # [B, 3, H, W]
            
            # Compute loss
            loss = mse_loss(epsilon_pred, epsilon)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# Hyperparameters and constants
T = 1000
alpha = 1.0 - torch.linspace(0, 1, T) ** 2  # Example schedule
alpha_cumprod = torch.cumprod(alpha, dim=0).to(device)
```

**Notes**:

- **Condition Dropping**: With probability `drop_prob`, the condition `c` is set to zero (assuming zero represents the unconditional case).
- **Time Steps**: Sampled uniformly from `[1, T]`.
- **Forward Diffusion**: Adds noise to the original image based on the time step.
- **Loss Computation**: Mean Squared Error between predicted noise and actual noise.
- **Optimization**: Standard gradient descent using Adam.

### **C. Inference Code for Text-to-Image Generation**

```python
@torch.no_grad()
def generate_images(model, text_embeddings, tokenizer, num_samples=1, guidance_scale=1.5, device='cuda'):
    model.eval()
    batch_size = text_embeddings.size(0)
    
    # Initialize with noise
    xt = torch.randn(batch_size, 3, H, W, device=device)  # [B, 3, H, W]
    
    for t in reversed(range(1, T+1)):
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)  # [B]
        
        # Predict noise with condition
        epsilon_theta_c = model(xt, t_tensor, text_embeddings)  # [B, 3, H, W]
        
        # Predict noise without condition (null)
        epsilon_theta_null = model(xt, t_tensor, torch.zeros_like(text_embeddings))  # [B, 3, H, W]
        
        # Guided noise prediction
        epsilon_guided = epsilon_theta_c + guidance_scale * (epsilon_theta_c - epsilon_theta_null)  # [B, 3, H, W]
        
        # Compute denoised sample
        sqrt_alpha = torch.sqrt(alpha[t-1]).to(device)
        sqrt_one_minus_alpha = torch.sqrt(1 - alpha[t-1]).to(device)
        xt = (xt - (beta[t-1] / sqrt_one_minus_alpha) * epsilon_guided) / sqrt_alpha
        
        # Add noise if not the last step
        if t > 1:
            sigma = beta[t-1] ** 0.5
            xt += sigma * torch.randn_like(xt)
    
    return xt
```

**Explanation**:

- **Initialization**: Start with pure Gaussian noise.
- **Reverse Process**: Iteratively denoise from `t = T` to `t = 1`.
- **Guided Prediction**: Combine conditional and unconditional predictions using the guidance scale.
- **Denoising Step**: Update `xt` based on the guided noise prediction.
- **Final Output**: `xt` at `t=0` is the generated image.

---

## 9. References

1. **Original Classifier-Free Guidance Paper**:
   - *"Classifier-Free Diffusion Guidance"*, Ho & Salimans, 2021.
   - [arXiv:2207.12598](https://arxiv.org/abs/2207.12598)

2. **Diffusion Models Survey**:
   - *"Diffusion Models Beat GANs on Image Synthesis"*, Dhariwal & Nichol, 2021.
   - [arXiv:2105.05233](https://arxiv.org/abs/2105.05233)

3. **Stable Diffusion**:
   - *"High-Resolution Image Synthesis with Latent Diffusion Models"*, Rombach et al., 2022.
   - [arXiv:2112.10752](https://arxiv.org/abs/2112.10752)

4. **DALL·E 2**:
   - *"Hierarchical Text-Conditional Image Generation with CLIP Latents"*, Ramesh et al., 2022.
   - [arXiv:2204.06125](https://arxiv.org/abs/2204.06125)

5. **Imagen by Google**:
   - *"Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding"*, Saharia et al., 2022.
   - [arXiv:2205.11487](https://arxiv.org/abs/2205.11487)

6. **Unet Architecture**:
   - *"U-Net: Convolutional Networks for Biomedical Image Segmentation"*, Ronneberger et al., 2015.
   - [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)

7. **Diffusers Library**:
   - Hugging Face's `diffusers` library provides implementations and tools for diffusion models.
   - [Hugging Face Diffusers](https://github.com/huggingface/diffusers)

---

## Conclusion

Classifier-Free Diffusion Models represent a powerful and flexible approach to generative modeling, eliminating the need for external classifiers and integrating guidance directly into the diffusion process. By training the model to handle both conditional and unconditional scenarios, and by effectively interpolating between them during inference, these models achieve high-quality, controllable generation across various applications such as image super-resolution, inpainting, and text-to-image synthesis.

Understanding the underlying mathematics, architecture, and training dynamics is crucial for leveraging these models effectively. The provided code examples offer a foundational starting point, which can be extended and customized based on specific application requirements.

For a deeper understanding and advanced implementations, refer to the listed references and explore open-source projects like Stable Diffusion and Hugging Face's `diffusers` library.
