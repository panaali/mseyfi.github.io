# GAN Tutorial: Connecting BCE Loss to Minimax Game and Understanding Non-Saturating Loss

## 1. Introduction to GANs

Generative Adversarial Networks (GANs) are composed of two neural networks:

* A **Generator (G)**: learns to generate fake samples $G(z)$ from random noise $z \sim p(z)$
* A **Discriminator (D)**: learns to classify samples as real (from data) or fake (from the generator)

These networks are trained in a two-player minimax game.

---

## 2. Binary Cross-Entropy Loss for Discriminator and Generator

### Binary Cross-Entropy (BCE)

Given a prediction $\hat{y} \in (0, 1)$ and a true label $y \in \{0, 1\}$, the BCE loss is:
$\text{BCE}(y, \hat{y}) = -y \log(\hat{y}) - (1 - y) \log(1 - \hat{y})$

### BCE Loss for Discriminator (D)

D is a binary classifier:

* For real data $x \sim p_{\text{data}}(x)$, label is 1
* For fake data $G(z)$, label is 0

So discriminator loss becomes:

$$
\mathcal{L}_D = \mathbb{E}_{x \sim p_{\text{data}}}[-\log D(x)] + \mathbb{E}_{z \sim p(z)}[-\log(1 - D(G(z)))]
$$

### BCE Loss for Generator (G)

Original generator loss (from minimax formulation) $\rightarrow$ maximize the loss below:

$$
\mathcal{L}_G^{\text{original}} = \mathbb{E}_{z \sim p(z)}[-\log(1 - D(G(z)))]
$$

But maximizing this loss means maximizing the cross entropy. That means make $D(G(z))\rigtharrow 1$ which is desired. 
But the problem is if during the training the output loss becomes $0$ then the gradient flow stops and we cannot go beyond $0$. Therefore, we should think about a new way of representing this. 

---

## 3. Minimax GAN Objective

The original GAN paper (Goodfellow et al. 2014) defines the objective as:

$$
\min_G \max_D \; \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p(z)}[\log (1 - D(G(z)))]
$$

* $D$ maximizes this expression (equivalent to minimizing BCE loss for real/fake classification)
* $G$ minimizes the same expression (equivalent to minimizing BCE loss with label 0 for fake)

Thus, BCE losses for $D$ and $G$ naturally merge into this min-max framework.

---

## 4. Saturation Problem and Non-Saturating Generator Loss

### Saturation

If $D(G(z)) \to 0$, then $\log(1 - D(G(z))) \to 0$ and gradient $\nabla_G \mathcal{L}_G \to 0$

This means:

> The generator receives no learning signal when it's weak (bad at fooling D). This is called **loss saturation**.

### Non-Saturating Loss (Practical Generator Loss)

Instead of minimizing:
$\mathbb{E}_{z} [\log(1 - D(G(z)))]$
We minimize:
$\mathcal{L}_G^{\text{non-saturating}} = -\mathbb{E}_{z} [\log D(G(z))]$

This avoids saturation and gives large gradients when $D(G(z)) \approx 0$.

---

## 5. PyTorch Code: Training a GAN on MNIST

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Generator Network
class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, img_dim),
            nn.Tanh(),  # Output in [-1, 1]
        )

    def forward(self, z):
        return self.model(z)

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(img_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)

# Setup
z_dim = 100
img_dim = 28 * 28
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

G = Generator(z_dim, img_dim).to(device)
D = Discriminator(img_dim).to(device)

opt_G = optim.Adam(G.parameters(), lr=2e-4)
opt_D = optim.Adam(D.parameters(), lr=2e-4)
criterion = nn.BCELoss()

# Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),  # Scale to [-1, 1]
])
dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(20):
    for real_imgs, _ in dataloader:
        real_imgs = real_imgs.view(-1, img_dim).to(device)
        batch_size = real_imgs.size(0)

        # Labels
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Train Discriminator
        z = torch.randn(batch_size, z_dim).to(device)
        fake_imgs = G(z)
        real_preds = D(real_imgs)
        fake_preds = D(fake_imgs.detach())  # detach() avoids computing gradients for G

        d_loss = criterion(real_preds, real_labels) + criterion(fake_preds, fake_labels)

        opt_D.zero_grad()
        d_loss.backward()
        opt_D.step()

        # Train Generator (non-saturating loss)
        # Note: we do NOT use torch.no_grad() here
        # Because we want to compute gradients w.r.t. G
        # Although D's weights won't be updated, we need gradients to flow through D(G(z))
        fake_preds = D(fake_imgs)
        g_loss = criterion(fake_preds, real_labels)  # Pretend fakes are real

        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()

    print(f"Epoch {epoch+1}: D_loss={d_loss.item():.4f}, G_loss={g_loss.item():.4f}")
```

---

## 6. Conclusion

* GANs are trained with BCE loss under a min-max framework.
* The discriminator uses standard BCE classification.
* The generator originally minimized $\log(1 - D(G(z)))$, but this saturates.
* The non-saturating loss $-\log D(G(z))$ is used in practice to ensure strong gradients.
* We do **not** use `torch.no_grad()` in the generator step because we need gradients to flow from `D(G(z))` back to `G`. However, gradients with respect to D’s parameters are still computed — but unused — causing some **waste of memory**. A more efficient alternative is to **freeze D's parameters** using `requires_grad_(False)` during generator update to save memory.

This tutorial provides both the mathematical reasoning and practical code to understand and train GANs effectively.
