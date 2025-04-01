[![Home](https://img.shields.io/badge/Home-Click%20Here-blue?style=flat&logo=homeadvisor&logoColor=white)](../)

## [![GENAI](https://img.shields.io/badge/CV-Selected_Topics_in_Generative_AI-green?style=for-the-badge&logo=github)](../main_page/GenAI)

Below is a comprehensive tutorial on **Noise Conditional Score Networks (NCSN)**. We’ll cover:

1. **High-Level Intuition**  
2. **Mathematical Foundations**  
3. **Training Procedure (Denoising Score Matching)**  
4. **Sampling Procedure** (Annealed Langevin Dynamics / SDE-based approaches)  
5. **Comparison with DDPM**  
6. **Practical Computer Vision Example** with code

---
## 0. Langevin dynamics:
Great question! This touches on a deep connection between **Langevin dynamics** and **probabilistic inference**, especially in the context of sampling from a complex probability distribution.

---

### 🔁 The Big Picture:
In Langevin dynamics (specifically *overdamped* Langevin), we aim to **sample from a target distribution** — usually a **posterior** in Bayesian inference or a Boltzmann distribution in physics.

The dynamics are governed by:

$$
dx_t = \frac{1}{2} \nabla \log p(x_t) \, dt + dW_t
$$

where:
- $ p(x) $ is the target distribution (typically a probability density),
- $ \nabla \log p(x) $ is the **score function**, and
- $ dW_t $ is Brownian motion (standard Gaussian noise).

---

### ✅ Why use the **gradient of the log-density** (score function) instead of the density itself?

Here are the key reasons:

---

#### 0.1. **Sampling Doesn’t Require Normalization**
We often work with **unnormalized densities**: $ \tilde{p}(x) = e^{-U(x)} $, where $ U(x) $ is a potential or energy function.

- We *don't* know the normalization constant $ Z = \int e^{-U(x)} dx $, so we can't evaluate the true $ p(x) = \frac{1}{Z} e^{-U(x)} $.
- However, when taking $ \nabla \log p(x) $, the normalization constant cancels out:

$$
\nabla \log p(x) = \nabla \log \left( \frac{1}{Z} e^{-U(x)} \right) = -\nabla U(x)
$$

So we can compute $ \nabla \log p(x) $ **even when we don't know the full pdf**!

---

#### 0.2. **Gradient Drives the Drift in Langevin Dynamics**
Langevin dynamics is derived from physics — particles moving in a potential field:

- The **drift** term $ \frac{1}{2} \nabla \log p(x) $ pushes the samples toward high-probability regions.
- The **noise** ensures exploration.

So, the gradient tells the particle **which direction increases the probability**, which is exactly what we want for sampling efficiently.

---

#### 0.3. **Mathematical Convenience & Stability**
- Gradients of log-probability give a **vector field** that guides dynamics in a stable way.
- Using the raw probability would lead to issues:
  - It’s scalar → not directly useful for dynamics.
  - It could vary wildly across space → numerically unstable.
  - It’s often hard to estimate or normalize in practice.

---

#### 0.4. **Connections to Score-Based Models and Diffusion**
Modern generative methods like **score-based diffusion models** and **denoising score matching** also use $ \nabla \log p(x) $ because:
- It's easier to learn or approximate.
- It provides meaningful information about the shape of the distribution.

---


## 1. High-Level Intuition

A **Score-based Generative Model** learns the gradient of the log-probability (the “score”) for a family of **noisy** versions of data. Instead of directly learning a generative model $ p(x) $, we train a network $ s_\theta(x, \sigma) $ that approximates:

$$
\nabla_x \log p_\sigma(x) \quad \text{where} \quad p_\sigma(x)
$$

is the distribution of data *corrupted* by noise of scale $\sigma$. Once we learn a good approximation of the score $\nabla_x \log p_\sigma(x)$, we can sample from the (clean) distribution by **progressively denoising** data using **Langevin dynamics** (or an equivalent Stochastic Differential Equation).

**Core idea**:  
- We add noise to the real data $\mathbf{x}$ at various noise levels $\sigma$.  
- We train a network to predict the **direction** that reduces noise (the gradient of the log-density).  
- Once trained, we can reverse the process: start from pure noise and iteratively “denoise” (i.e., move in the direction the score network suggests) to obtain samples.

---

## 2. Mathematical Foundations

### 2.1. Score Matching and Denoising Score Matching

A **score** of a distribution $p(x)$ is:
$$
\nabla_x \log p(x).
$$

- **Score Matching** aims to learn a function $ s_\theta(x) \approx \nabla_x \log p(x) $.  
- **Denoising Score Matching** extends the idea by adding noise: we corrupt data $\mathbf{x}$ to $\mathbf{x}^\sim$ with some noise scale $\sigma$, and we learn  
  $$
  s_\theta(\mathbf{x}^\sim, \sigma) \approx \nabla_{\mathbf{x}^\sim} \log p_\sigma(\mathbf{x}^\sim),
  $$
  where $p_\sigma(\mathbf{x}^\sim)$ is the distribution of $\mathbf{x}$ after corruption with noise $\sigma$.

#### Why add noise?
- It stabilizes training.
- We can train the network to handle multiple noise levels $\sigma$.
- We can then **anneal** from larger $\sigma$ to smaller $\sigma$ at sampling time, bridging coarse to fine denoising steps.

### 2.2. Gradient of a Gaussian Log Density

A common noise corruption is adding **Gaussian noise**:
$$
\mathbf{x}^\sim = \mathbf{x} + \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I}).
$$

Under this corruption model:
$$
p_\sigma(\mathbf{x}^\sim \mid \mathbf{x}) = \frac{1}{(\sqrt{2\pi}\sigma)^d} \exp\left( - \frac{ \|\mathbf{x}^\sim - \mathbf{x}\|^2 }{2\sigma^2} \right).
$$

**Key fact**: For a Gaussian w.r.t. $\mathbf{x}^\sim$,  
$$
\nabla_{\mathbf{x}^\sim} \log p_\sigma(\mathbf{x}^\sim \mid \mathbf{x}) 
\;=\; -\frac{ \mathbf{x}^\sim - \mathbf{x} }{ \sigma^2 }.
$$

That means if you want the gradient of the (log-)density at $\mathbf{x}^\sim$, it is **proportional** to $(\mathbf{x}-\mathbf{x}^\sim)$. Hence,

$$
\nabla_{\mathbf{x}^\sim} \log p_\sigma(\mathbf{x}^\sim) 
\;\propto\; (\mathbf{x} - \mathbf{x}^\sim)/\sigma^2.
$$

(In practice, we also consider the mixture of all data $\mathbf{x}$, but the local gradient structure remains the same.)

### 2.3. Loss Function

To make the network $s_\theta(\mathbf{x}^\sim, \sigma)$ close to the true $\nabla_{\mathbf{x}^\sim}\log p_\sigma(\mathbf{x}^\sim)$, we minimize a **mean-squared error** between the network’s prediction and the ground truth gradient:

$$
\mathcal{L}(\theta) \;=\; \mathbb{E}_{\mathbf{x}\sim p_\text{data}, \boldsymbol{\epsilon}\sim \mathcal{N}(0,\sigma^2 \mathbf{I})} 
\left[
\big\| s_\theta(\mathbf{x}+\boldsymbol{\epsilon}, \sigma) - 
\frac{\mathbf{x} - (\mathbf{x} + \boldsymbol{\epsilon})}{\sigma^2} \big\|^2
\right].
$$

Or more generally, we can do a weighted sum/integration over multiple $\sigma$-levels, e.g., $\sigma \in [\sigma_\text{max}, \sigma_\text{min}]$.

---

## 3. Training Procedure

Below is a **conceptual** step-by-step procedure for training an **NCSN**:

1. **Sample a real example** $\mathbf{x}$ from the dataset.  
2. **Sample a noise scale** $\sigma$ (often from a predefined schedule or distribution).  
3. **Corrupt** $\mathbf{x}$ with Gaussian noise to get $\mathbf{x}^\sim = \mathbf{x} + \boldsymbol{\epsilon}$.  
4. **Forward pass**: Evaluate $s_\theta(\mathbf{x}^\sim, \sigma)$.  
5. **Compute the target**: $\nabla_{\mathbf{x}^\sim}\log p_\sigma(\mathbf{x}^\sim)\approx \frac{\mathbf{x} - \mathbf{x}^\sim}{\sigma^2}$.  
6. **Compute the loss**: MSE between the prediction and target.  
7. **Backprop** and **update** $\theta$.

### Pseudocode (NCSN Training)

```
Given:
  - Dataset D of real samples x
  - Noise schedule { σ_i : i=1..N } or continuous range
  - Neural network sθ(x, σ)

repeat until converged:
    x ← sample_batch(D)
    σ ← sample a noise level (e.g., uniform in [σ_min, σ_max])
    ε ← Gaussian noise ~ N(0, σ^2 I)
    x_tilde = x + ε

    # Score matching target = (x - x_tilde)/σ^2
    score_target = (x - x_tilde) / σ^2

    # Network’s score prediction
    score_pred = sθ(x_tilde, σ)

    # Compute loss (MSE)
    loss = MeanSquaredError(score_pred, score_target)

    # Backprop & update network
    loss.backward()
    optimizer.step()

end
```

---

## 4. Sampling Procedure

After training, we have a network $ s_\theta(\mathbf{x}, \sigma) $ that approximates $\nabla_{\mathbf{x}} \log p_\sigma(\mathbf{x})$. We can generate samples from the **original** data distribution by starting with random noise and **gradually denoising** using an **annealed Langevin** approach or continuous SDE approach.

### 4.1. Annealed Langevin Dynamics (Discrete Version)

One popular procedure is:

1. Start with a random sample $\mathbf{x}_0\sim \mathcal{N}(0, I)$.  
2. For a sequence of noise levels $\sigma_\text{max} = \sigma_1 \;>\;\sigma_2\;>\;\dots\;>\;\sigma_L = \sigma_\text{min}$:  
   - Perform a few **Langevin** updates at fixed $\sigma_i$:
     $$
     \mathbf{x} \leftarrow \mathbf{x} + \alpha \, s_\theta(\mathbf{x}, \sigma_i) + \sqrt{2\alpha}\,\boldsymbol{\eta}, 
     \quad \boldsymbol{\eta} \sim \mathcal{N}(0,I),
     $$
     where $\alpha$ is a small step size.  
3. As $\sigma_i$ decreases, the sampling distribution transitions from “rough, large-scale” denoising to “fine-scale” denoising at the end.  
4. Finally, $\mathbf{x}_L$ is a sample approximating the **true** data distribution.

### Pseudocode (NCSN Sampling)

```
Given:
  - Trained score network sθ
  - Noise schedule [σ1 > σ2 > ... > σL]
  - Step size α (could vary with i)

x ← sample from N(0, I)  # or uniform, or any random init

for i in 1..L:
    for t in 1..T:  # T steps per noise scale
        z ← sample from N(0, I)
        # Langevin update:
        x ← x + α * sθ(x, σi) + sqrt(2 * α) * z
    # reduce noise level to next in schedule

return x
```

> **Note**: In practice, $\alpha$ can be tuned or chosen by certain heuristics. Some methods treat the sampling as a continuous-time SDE.

---

## 5. Comparison with DDPM

**DDPM (Denoising Diffusion Probabilistic Models)** is another popular framework for diffusion-based generative modeling. Key similarities and differences:

- **Similarities**:
  1. Both add noise to data and learn to reverse that process.
  2. Both rely on a network to predict how to “denoise” a corrupted sample.
  3. Both can produce high-quality samples.

- **Differences**:
  1. **Score-based (NCSN)** explicitly learns $\nabla \log p_\sigma(\mathbf{x})$; it’s a continuous-time or multi-noise-level approach that focuses on the **score** directly.
  2. **DDPM** uses a Markov chain forward process with a fixed schedule of noise addition and a backward pass trained with a variational bound or an equivalent MSE on the noise.  
  3. **NCSN** sampling is often done via **Langevin dynamics** or SDE sampling; **DDPM** sampling is done by the Markov chain defined in the reverse diffusion process.

### Pseudocode (DDPM Training vs. Score-Based)

| **DDPM Training**                                | **NCSN Training**                                     |
|--------------------------------------------------|-------------------------------------------------------|
| sample x from data                               | sample x from data                                   |
| sample t from {1, ..., T} uniformly              | sample σ from [σ_min, σ_max]                         |
| construct q(x_t|x) (Gaussian)                    | x_tilde = x + ε, ε ~ N(0, σ^2 I)                     |
| network predicts noise (or x_0)                  | network predicts score = (x - x_tilde)/σ^2           |
| loss = MSE(noise_pred, noise)                    | loss = MSE(score_pred, score_target)                 |

Despite different viewpoints, the final goal is to **learn how to invert the noising process** and generate from the data distribution.

---

## 6. Practical Computer Vision Example

Let’s do a **toy** training + sampling example in Python-like pseudo-code. We’ll demonstrate how you might train a small score network on a simple image dataset (e.g., MNIST). We’ll focus on the essential bits rather than a fully optimized script.

### 6.1. Setup

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ----- Hyperparameters -----
batch_size = 64
lr = 1e-4
num_epochs = 5
sigma_min = 0.01
sigma_max = 1.0
num_noise_levels = 5  # for example
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

### 6.2. Define a Simple Score Network

A very simplistic CNN that takes in (noisy_image, sigma) and outputs the predicted score. We’ll just treat sigma as an additional channel or embed it somehow.

```python
class ScoreNet(nn.Module):
    def __init__(self):
        super(ScoreNet, self).__init__()
        # A minimal CNN architecture (for demonstration only)
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1)
        )
        # A small MLP to transform sigma to a scale/bias or embedding
        self.sigma_embedding = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x_noisy, sigma):
        # x_noisy shape: (batch, 1, H, W)
        # sigma shape: (batch, )
        # we can add a trivial approach: just add sigma as an extra channel or scale
        sigma = sigma.view(-1, 1)  # shape (batch,1)
        embedded = self.sigma_embedding(sigma)  # shape (batch,1)
        # Reshape for broadcast
        embedded = embedded.view(-1, 1, 1, 1)
        
        # For demonstration, multiply the input by embedded just to incorporate sigma
        x_scaled = x_noisy * (1 + embedded)
        
        return self.net(x_scaled)
```

In real code, people often adopt more sophisticated ways to condition on $\sigma$. The key is that the model can see both the noisy image and the noise level.

### 6.3. Data Loader

```python
transform = transforms.Compose([
    transforms.ToTensor(), 
    # optional: transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='.', download=True, train=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
```

### 6.4. Training Loop

```python
score_model = ScoreNet().to(device)
optimizer = optim.Adam(score_model.parameters(), lr=lr)

def get_loss(x_noisy, x, sigma):
    # Score target = (x - x_noisy)/sigma^2
    # x, x_noisy: (batch, 1, H, W)
    # sigma: (batch,)
    sigma_2 = sigma**2
    # Broadcast shapes
    sigma_2 = sigma_2.view(-1,1,1,1)
    
    score_target = (x - x_noisy) / sigma_2
    score_pred = score_model(x_noisy, sigma)
    loss = torch.mean((score_pred - score_target)**2)
    return loss

for epoch in range(num_epochs):
    for x, _ in train_loader:
        x = x.to(device)
        
        # sample a sigma for the entire batch, or per-sample:
        sigma_vals = torch.rand(x.size(0), device=device) * (sigma_max - sigma_min) + sigma_min
        
        # add Gaussian noise
        noise = torch.randn_like(x) * sigma_vals.view(-1,1,1,1)
        x_noisy = x + noise
        
        # compute loss
        loss = get_loss(x_noisy, x, sigma_vals)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch} - Loss {loss.item():.4f}")
```

- In practice, you might sample multiple $\sigma$ values per mini-batch, or implement continuous training strategies.  
- The above is a simplistic demonstration.

### 6.5. Sampling (Annealed Langevin)

After training:

```python
import math

def sample_from_model(score_model, shape=(1, 28, 28), 
                      sigma_min=0.01, sigma_max=1.0, 
                      steps=5, step_size=0.0001):
    """
    shape: the shape of the final image we want
    steps: how many noise scales we step through
    step_size: step size for Langevin update
    """
    score_model.eval()
    with torch.no_grad():
        x = torch.randn((1, *shape), device=device)  # start from noise
        sigmas = torch.linspace(sigma_max, sigma_min, steps).to(device)
        
        for sigma in sigmas:
            # Possibly do multiple Langevin steps per sigma:
            for t in range(3):  # e.g. 3 internal steps
                grad = score_model(x, sigma.unsqueeze(0))
                # Langevin update
                x = x + step_size * grad
                # Add standard Gaussian noise scaled by sqrt(2 * step_size)
                z = torch.randn_like(x)
                x = x + math.sqrt(2 * step_size) * z
        
        return x
```

You can now call:

```python
generated_sample = sample_from_model(score_model, shape=(1,28,28))
# visualize the output, e.g. via matplotlib:
import matplotlib.pyplot as plt

plt.imshow(generated_sample.cpu().squeeze().numpy(), cmap='gray')
plt.title("Sample from NCSN")
plt.show()
```

---

Below is a concise **advantage vs. disadvantage** comparison between **Noise Conditional Score Networks (NCSN)** and **Denoising Diffusion Probabilistic Models (DDPM)**. Both are types of diffusion-like generative models but differ in how they parameterize and learn the “reverse” (denoising) process.

---

## NCSN/DDPM Pros and Cons:
### NCSN (Noise Conditional Score Networks)

### Advantages
1. **Direct Score Learning**  
   - NCSN directly estimates the *score* $\nabla_x \log p_\sigma(x)$ for multiple noise levels.  
   - This yields a continuous (or semi-continuous) representation of how to denoise data at different noise scales.

2. **Strong Theoretical Foundation in Score Matching**  
   - NCSN is grounded in the concept of **score matching** (Hyvärinen, 2005), with extensions to *denoising score matching*.  
   - The math for gradient-based sampling (e.g., **Annealed Langevin Dynamics** or **SDE-based** sampling) is quite elegant.

3. **Flexibility with Noise Schedules**  
   - In principle, one can adapt the **noise schedule** (ranging from $\sigma_\text{max}$ down to $\sigma_\text{min}$) to suit different data types or complexity levels.

4. **Connections to Denoising Diffusion and SDEs**  
   - NCSN paved the way for more general **Stochastic Differential Equation (SDE)** frameworks (e.g., Score-Based Generative Modeling via SDEs).  
   - These approaches unify various noise processes (e.g., VE/VP SDEs) under one umbrella, often with excellent empirical performance.

### Disadvantages
1. **Potential Training Instability**  
   - Directly learning the score can be numerically sensitive, especially if the network *overestimates* or *underestimates* gradients at particular noise scales.  
   - NCSNv1 sometimes suffered from artifacts without careful training tricks; later versions (NCSNv2, etc.) introduced improvements.

2. **Sampling Speed**  
   - **Annealed Langevin Dynamics** or SDE-based sampling often requires **many iterations** to produce high-quality samples (although alternative samplers exist).  
   - This can be **slower** compared to some other generative models unless carefully optimized or used with accelerated samplers.

3. **Hyperparameter Sensitivity**  
   - The choice of noise schedule, step size in Langevin updates, and weighting across noise levels can be *tricky* to tune.  
   - Poor choices can lead to slow convergence or poor sample quality.

4. **Less Common than DDPM in Some Frameworks**  
   - DDPM has become quite popular in mainstream applications (e.g., large text-to-image systems), so codebases and community support might be stronger there.

---

## DDPM (Denoising Diffusion Probabilistic Models)

### Advantages
1. **Stable & Well-Studied Training**  
   - DDPM uses a *simple denoising objective* — typically predicting the added noise $\epsilon$ (or predicting $\mathbf{x}_0$) in a discretized forward diffusion process.  
   - This approach is known to be **empirically stable** and relatively straightforward to implement.

2. **High-Quality Results & Widespread Adoption**  
   - DDPMs have led to many state-of-the-art image (and other modality) generation systems (e.g., Stable Diffusion variants, latent diffusion, etc.).  
   - Large-scale industrial efforts have focused on optimizing, scaling, and refining the DDPM pipeline.

3. **Discrete Timesteps = Easier Implementation**  
   - The forward noising is broken into a finite number of steps $1,2,\ldots,T$.  
   - Each step is a standard Gaussian transition with a closed-form variance schedule.  
   - The backward (reverse) process has a well-defined Markov chain structure.

4. **Flexible Samplers**  
   - Many specialized samplers (ancestral sampling, DDIM, DPM++, etc.) speed up or refine the sampling process.  
   - This fosters a large ecosystem of improved, faster sampling algorithms.

### Disadvantages
1. **Discrete, Fixed Noise Schedule**  
   - Typically, we define a discrete schedule $\beta_1, \beta_2, \ldots, \beta_T$.  
   - If you want to adapt noise scales on the fly, you need more complex scheduling methods or continuous-time extensions.

2. **Sampling Speed**  
   - Like NCSN, naive DDPM sampling can be *slow* if you must run through all $T$ steps.  
   - Optimized samplers exist but still can be more computationally heavy than, say, a single forward pass in a GAN or an autoregressive model.

3. **Hyperparameter Tuning**  
   - While training is stable, the choice of $\beta_t$ schedules, number of diffusion steps $T$, and the weighting in the loss can impact final quality and sampling speed.

4. **Less Direct Theoretical Link to Score Matching**  
   - DDPM training can be seen as *equivalent* to certain forms of denoising score matching, but it’s not as explicitly framed in that manner.  
   - Some prefer the direct interpretation of “learning the score” in NCSN.

---

## When to Choose Which?

- **If you’re comfortable with continuous noise processes** (or you want to unify everything under an SDE perspective), or you like the *direct* score-matching formulation, **NCSN** (score-based) might be more appealing.  
- **If you prefer a discrete step approach** with a more widely adopted codebase, more off-the-shelf tools, and strong community support, **DDPM** might be more convenient.

In practice, **both methods** can achieve excellent results. They share many similarities (iterative denoising, Gaussian noise corruption) and differ primarily in how the network is parameterized and trained (i.e., predicting the score function vs. predicting the noise or the clean image at each diffusion step).  

Ultimately, which approach you pick can depend on:
- Familiarity and community adoption  
- Desired sampler flexibility (continuous vs. discrete)  
- Implementation details and existing frameworks  
- Empirical performance on your specific dataset or domain.

## Putting It All Together

1. **Train** the score network on your dataset, at multiple noise levels.  
2. **Sample** from the trained model by running annealed Langevin (or SDE-based sampling).  
3. Optionally, refine hyperparameters like the step sizes, the number of noise levels, etc., to improve the sample quality.

---

## Final Summary

- **NCSN** (Noise Conditional Score Network) learns $\nabla_x \log p_\sigma(\mathbf{x})$.  
- Denoising Score Matching is the practical objective: predict $(\mathbf{x}-\mathbf{x}^\sim)/\sigma^2$.  
- **Training** involves MSE between network’s score predictions and ground-truth gradient from Gaussian corruption.  
- **Sampling** uses **Langevin dynamics** (or an SDE) to iteratively refine noise into a coherent sample.  
- **Comparison with DDPM**: Both rely on iterative denoising but have different parameterizations and training objectives (noise prediction vs. score prediction).  

This framework has proven effective for high-quality image (and other modality) generation. You now have a conceptual foundation, plus a reference implementation outline for training and sampling with NCSN. Have fun experimenting!
