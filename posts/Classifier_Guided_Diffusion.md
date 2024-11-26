[![Home](https://img.shields.io/badge/Home-Click%20Here-blue?style=flat&logo=homeadvisor&logoColor=white)](../)

## Classifier-Guided Diffusion: An In-Depth Exploration:

Classifier-guided diffusion is a powerful technique in generative modeling that leverages an external classifier to steer the generation process toward desired attributes or classes. This method enhances the quality and controllability of generated data, such as images, by integrating class-specific information during the diffusion process.

In this comprehensive guide, we'll delve into:

1. **Intuitive Understanding of Classifier-Guided Diffusion**
2. **Why Use the Gradient of the Log Probability of the Classifier?**
3. **Mathematical Formulation**
4. **Pseudo-Code for Training and Inference**
5. **Example Code Implementation**

---

## 1. Intuitive Understanding of Classifier-Guided Diffusion

### **Diffusion Models Overview**

Diffusion models are a class of generative models that create data by iteratively denoising a sample, starting from pure noise and gradually refining it to produce coherent outputs (e.g., images). The process involves a **forward diffusion** phase, where data is progressively corrupted with noise, and a **reverse diffusion** phase, where the model learns to reconstruct the original data from the noisy versions.

### **Guiding the Diffusion Process**

While diffusion models can generate diverse and high-quality samples, controlling the generation process to produce specific attributes (like generating images of a particular class) can be challenging. This is where **classifier-guided diffusion** comes into play.

By integrating a separate classifier into the diffusion process, we can **steer** the generation toward desired classes or attributes. The classifier provides **gradient information** that nudges the diffusion model in the direction of the target class, enhancing both the fidelity and relevance of the generated data.

---

## 2. Why Use the Gradient of the Log Probability of the Classifier?

The core idea behind classifier-guided diffusion is to **modify the reverse diffusion process** using gradients derived from a classifier. Here's why and how:

### **Maximizing Class Probability**

Suppose we have a pre-trained classifier $p_{\textrm{cls}}(y \mid x)$ that assigns a probability to a class $y$ given an input $x$. During generation, we aim to generate samples $x$ that not only resemble the training data but also belong to a specific class $y$.

To achieve this, we want to **maximize the probability** $p_{\textrm{cls}}(y \mid x)$ of the desired class. Taking the gradient of the log probability $\nabla_x \log p_{\textrm{cls}}(y \mid x)$ gives us the direction in the data space that increases the likelihood of class $y$.

### **Steering the Generation Process**

By adding this gradient to the reverse diffusion step, we **guide** the diffusion model to produce samples that are more likely to be classified as $y$. Intuitively, the gradient tells the model how to adjust the current sample $x_t$ to better align with the target class $y$.

### **Intuitive Analogy**

Imagine navigating a landscape where each point represents a potential image. The classifier's gradient points "uphill" towards regions where images are more likely to belong to class $y$. By following this gradient during the generation process, the model is directed towards areas of higher class probability, resulting in more accurate and relevant samples.

---

## 3. Mathematical Formulation

### **Notation**

- $x_0$: Original data sample (e.g., an image).
- $x_t$: Noisy version of $x_0$ at time step $t$.
- $\epsilon_\theta(x_t, t)$: Neural network predicting noise at step $t$.
- $p_\theta(x_{t-1} \mid x_t)$: Reverse diffusion step from $t$ to $t-1$.
- $p_{\text{cls}}(y \mid x_t)$: Pre-trained classifier's probability for class $y$ given $x_t$.
- $s$: Guidance scale factor.

### **Reverse Diffusion with Classifier Guidance**

The standard reverse diffusion step updates $x_{t-1}$ based on $x_t$ and the predicted noise:

$$ x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z $$

where $\alpha_t$, $\beta_t$, $\bar{\alpha}_t$, and $\sigma_t$ are predefined noise schedule parameters, and $z$ is Gaussian noise.

With classifier guidance, we modify the prediction by adding the gradient of the log probability:


$$ \epsilon_\theta(x_t, t) - s \cdot \sigma_t \nabla_{x_t} \log p_{\text{cls}}(y \mid x_t) $$

Thus, the guided reverse diffusion step becomes:

$$ x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \left[ \epsilon_\theta(x_t, t) - s \cdot \sigma_t \nabla_{x_t} \log p_{\text{cls}}(y \mid x_t) \right] \right) + \sigma_t z $$

### **Intuition Behind the Formula**

- **Noise Prediction Adjustment:** The term $s \cdot \sigma_t \nabla_{x_t} \log p_{\text{cls}}(y \mid x_t)$ adjusts the noise prediction to favor the target class.
  
- **Guidance Scale $s$:** Controls the strength of the guidance. A higher $s$ makes the model adhere more strictly to the class $y$, potentially at the cost of diversity.

---

## 4. Pseudo-Code for Training and Inference

### **Training Phase**

Training in classifier-guided diffusion involves two main components:

1. **Diffusion Model ($\epsilon_\theta$) Training:** Trains the diffusion model to predict noise.
2. **Classifier ($p_{\text{cls}}$) Training:** Trains a classifier to predict class probabilities from noisy data.

#### **Pseudo-Code: Diffusion Model Training**

```python
for batch in data_loader:
    x_0, y = batch  # x_0: original data, y: class labels
    t = sample_random_t()  # Sample a random time step
    noise = sample_noise()
    x_t = forward_diffusion(x_0, t, noise)  # Add noise to x_0 to get x_t
    noise_pred = diffusion_model(x_t, t)  # Predict noise
    loss = mse(noise_pred, noise)  # Mean squared error loss
    loss.backward()
    optimizer.step()
```

#### **Pseudo-Code: Classifier Training**

```python
for batch in data_loader:
    x_0, y = batch
    t = sample_random_t()
    noise = sample_noise()
    x_t = forward_diffusion(x_0, t, noise)  # Add noise to x_0 to get x_t
    logits = classifier(x_t, t)  # Predict class logits
    loss = cross_entropy_loss(logits, y)
    loss.backward()
    optimizer.step()
```

### **Inference Phase**

During inference, the classifier guides the reverse diffusion process to generate samples conditioned on a target class $y$.

#### **Pseudo-Code: Classifier-Guided Inference**

```python
x_T = sample_noise()  # Start from pure noise
for t in reversed(range(1, T+1)):
    epsilon = diffusion_model(x_t, t)
    grad_log_p = classifier_gradient(x_t, y, t)
    epsilon_guided = epsilon - s * sigma_t * grad_log_p
    x_{t-1} = reverse_diffusion_step(x_t, epsilon_guided, t)
    # Optionally add noise if t > 1
return x_0
```

**Functions Explained:**

- `forward_diffusion(x_0, t, noise)`: Adds noise to $x_0$ at time $t$.
- `reverse_diffusion_step(x_t, epsilon, t)`: Computes $x_{t-1}$ from $x_t$ and $\epsilon$.
- `classifier_gradient(x_t, y, t)`: Computes $\nabla_{x_t} \log p_{\text{cls}}(y \mid x_t)$.

---

## 5. Example Code Implementation

Below is a simplified implementation of classifier-guided diffusion using PyTorch. This example includes:

- A basic diffusion model.
- A simple classifier.
- Training loops for both models.
- Inference with classifier guidance.

**Note:** This code is for educational purposes and omits many optimizations and complexities of real-world implementations.

### **Dependencies**

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
```

### **Define the Diffusion Model**

A simple UNet-like architecture is commonly used, but for simplicity, we'll use a basic neural network.

```python
class DiffusionModel(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=512, output_dim=784):
        super(DiffusionModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),  # +1 for time step
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x, t):
        # x: [batch_size, input_dim]
        # t: [batch_size, 1]
        return self.net(torch.cat([x, t], dim=1))
```

### **Define the Classifier**

A simple classifier that takes noisy data and time step as input.

```python
class Classifier(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=512, num_classes=10):
        super(Classifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),  # +1 for time step
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x, t):
        # x: [batch_size, input_dim]
        # t: [batch_size, 1]
        return self.net(torch.cat([x, t], dim=1))
```

### **Forward Diffusion Process**

Adds noise to the data at a given time step.

```python
def forward_diffusion(x_0, t, noise, alpha_bar):
    return torch.sqrt(alpha_bar[t]) * x_0 + torch.sqrt(1 - alpha_bar[t]) * noise
```

### **Reverse Diffusion Step**

Computes $x_{t-1}$ from $x_t$ and guided noise prediction.

```python
def reverse_diffusion_step(x_t, epsilon, t, alpha, beta, alpha_bar, sigma):
    coeff1 = 1 / torch.sqrt(alpha[t])
    coeff2 = beta[t] / torch.sqrt(1 - alpha_bar[t])
    x_prev = coeff1 * (x_t - coeff2 * epsilon) + sigma[t] * torch.randn_like(x_t)
    return x_prev
```

### **Compute Classifier Gradient**

Computes the gradient of the log probability of the target class with respect to $x_t$.

```python
def classifier_gradient(classifier, x_t, y, t):
    x_t = x_t.detach().requires_grad_(True)
    logits = classifier(x_t, t)
    log_prob = F.log_softmax(logits, dim=1)
    target_log_prob = log_prob[torch.arange(len(y)), y]
    target_log_prob.sum().backward()
    return x_t.grad
```

### **Training Loops**

#### **Prepare Data**

Using MNIST for simplicity.

```python
# Hyperparameters
batch_size = 128
learning_rate = 1e-3
num_epochs = 10
T = 1000  # Number of diffusion steps

# Prepare DataLoader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # Flatten images
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
```

#### **Initialize Models and Optimizers**

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

diffusion_model = DiffusionModel().to(device)
classifier = Classifier().to(device)

optimizer_diffusion = optim.Adam(diffusion_model.parameters(), lr=learning_rate)
optimizer_classifier = optim.Adam(classifier.parameters(), lr=learning_rate)
```

#### **Define Noise Schedule**

Using linear noise schedule for simplicity.

```python
beta = torch.linspace(1e-4, 0.02, T).to(device)  # Beta schedule
alpha = 1.0 - beta
alpha_bar = torch.cumprod(alpha, dim=0)
sigma = torch.sqrt(beta)
```

#### **Train the Diffusion Model**

```python
def train_diffusion_model():
    diffusion_model.train()
    for epoch in range(num_epochs):
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            batch_size_current = x.size(0)
            t = torch.randint(0, T, (batch_size_current,), device=device).long()
            noise = torch.randn_like(x)
            x_t = forward_diffusion(x, t, noise, alpha_bar)
            t_input = (t / T).unsqueeze(1).float()  # Normalize t to [0,1]
            epsilon_pred = diffusion_model(x_t, t_input)
            loss = F.mse_loss(epsilon_pred, noise)
            optimizer_diffusion.zero_grad()
            loss.backward()
            optimizer_diffusion.step()
        
        print(f'Diffusion Model - Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')
```

#### **Train the Classifier**

```python
def train_classifier():
    classifier.train()
    for epoch in range(num_epochs):
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            batch_size_current = x.size(0)
            t = torch.randint(0, T, (batch_size_current,), device=device).long()
            noise = torch.randn_like(x)
            x_t = forward_diffusion(x, t, noise, alpha_bar)
            t_input = (t / T).unsqueeze(1).float()  # Normalize t to [0,1]
            logits = classifier(x_t, t_input)
            loss = F.cross_entropy(logits, y)
            optimizer_classifier.zero_grad()
            loss.backward()
            optimizer_classifier.step()
        
        print(f'Classifier - Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')
```

### **Training Execution**

```python
# Train both models
print("Training Diffusion Model...")
train_diffusion_model()

print("Training Classifier...")
train_classifier()
```

### **Inference with Classifier Guidance**

```python
def generate_sample(target_class, guidance_scale=1.0):
    diffusion_model.eval()
    classifier.eval()
    
    with torch.no_grad():
        x_t = torch.randn(1, 784).to(device)  # Start from noise
        for t in reversed(range(T)):
            t_tensor = torch.tensor([t / T]).to(device).unsqueeze(1).float()
            # Predict noise
            epsilon = diffusion_model(x_t, t_tensor)
            
            # Compute classifier gradient
            x_t.requires_grad = True
            logits = classifier(x_t, t_tensor)
            log_prob = F.log_softmax(logits, dim=1)
            target_log_prob = log_prob[:, target_class].sum()
            target_log_prob.backward()
            grad = x_t.grad.detach()
            x_t = x_t - guidance_scale * sigma[t] * grad  # Adjust epsilon
            x_t = reverse_diffusion_step(x_t, epsilon, t, alpha, beta, alpha_bar, sigma)
            x_t = torch.clamp(x_t, -1.0, 1.0)  # Optional: Clamp to valid range
            
    return x_t
```

### **Generate and Visualize a Sample**

```python
import matplotlib.pyplot as plt

def visualize_sample(x):
    x = x.cpu().numpy().reshape(28, 28)
    plt.imshow(x, cmap='gray')
    plt.axis('off')
    plt.show()

# Example: Generate a sample of digit '3'
generated_x = generate_sample(target_class=3, guidance_scale=5.0)
visualize_sample(generated_x)
```

---

## **Explanation of the Example Code**

1. **Model Definitions:**
   - **DiffusionModel:** Predicts the noise component in the data at a given time step.
   - **Classifier:** Predicts class probabilities from noisy data at a given time step.

2. **Training Process:**
   - **Diffusion Model Training:** Trains to minimize the mean squared error between the predicted noise and the actual noise added during the forward diffusion.
   - **Classifier Training:** Trains to classify the noisy data at various time steps into the correct classes.

3. **Inference with Classifier Guidance:**
   - Starts with pure noise.
   - Iteratively applies the reverse diffusion step, adjusting the noise prediction using the gradient from the classifier to steer the generation toward the target class.
   - The `guidance_scale` controls the strength of the guidance; higher values enforce stronger adherence to the target class.

4. **Visualization:**
   - After generation, the sample is reshaped and visualized as an image. For instance, generating a digit '3' from the MNIST dataset.

---

## References

1. **Denoising Diffusion Probabilistic Models**
   - **Authors:** Jonathan Ho, Ajay Jain, Pieter Abbeel
   - **Link:** [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
   
2. **Classifier Guidance for Diffusion Models**
   - **Authors:** Prafulla Dhariwal, Alexander Nichol
   - **Link:** [GitHub Repository](https://github.com/openai/guided-diffusion)
   
3. **Improved Techniques for Training Score-based Generative Models**
   - **Authors:** Yang Song, Stefano Ermon
   - **Link:** [arXiv:2006.09011](https://arxiv.org/abs/2006.09011)
   
4. **Diffusion Models Beat GANs on Image Synthesis**
   - **Authors:** Prafulla Dhariwal, Alexander Nichol
   - **Link:** [arXiv:2105.05233](https://arxiv.org/abs/2105.05233)
   
5. **Guided Diffusion Models**
   - **Authors:** OpenAI
   - **Link:** [OpenAI Blog](https://openai.com/blog/guided-diffusion)
   
6. **Score-Based Generative Modeling through Stochastic Differential Equations**
   - **Authors:** Yang Song, Stefano Ermon
   - **Link:** [arXiv:2011.13456](https://arxiv.org/abs/2011.13456)
   
7. **Stable Diffusion: High-Resolution Image Synthesis with Latent Diffusion Models**
   - **Authors:** Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer
   - **Link:** [arXiv:2112.10752](https://arxiv.org/abs/2112.10752)

## **Conclusion**

Classifier-guided diffusion enhances generative models by incorporating external classifiers to steer the generation process toward desired attributes or classes. By leveraging the gradient of the classifier's log probability, the diffusion model can produce more controlled and accurate outputs. This technique has been instrumental in advancing the quality and applicability of generative models in various domains, including image synthesis, text generation, and beyond.

Understanding both the theoretical underpinnings and practical implementation details is crucial for effectively utilizing classifier-guided diffusion in your projects. The provided pseudo-code and example implementation offer a foundational starting point for experimentation and further development.

If you're venturing into building or refining generative models, exploring classifier-guided diffusion can significantly enhance the controllability and relevance of your generated data.
