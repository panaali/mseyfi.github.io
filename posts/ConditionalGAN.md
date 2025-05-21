[![Home](https://img.shields.io/badge/Home-Click%20Here-blue?style=flat&logo=homeadvisor&logoColor=white)](../)

## [![CV](https://img.shields.io/badge/CV-Selected_Topics_in_Computer_Vision-green?style=for-the-badge&logo=github)](../main_page/CV)

This is a conditional GAN Training loop, which is exactly the same as DCGANs, with one minor change. The input to both Generator and Discriminator is concatenated (via embedding layers and maybe some conv layers) to the original inputs we had before. So labels are concatenated to noise for the Generator and are concatenated to the input image for the Discriminator. To match the size we can use some intermediate layers. 

**Note that the input images are aligned with the input labels, so the Discriminator and Generator implicitly learn their dependence to the input images.**

### How a conditional GAN (cGAN) “ties” an image to its label

**Key point:** during training the **pair** *(image `x`, label `y`)* is what is judged as real or fake.
If the image and the label don’t “match”, the discriminator immediately wins, so gradients push both networks toward *“image ↔ label consistency”*.

---

## 1  Conditional adversarial objective

$$
\min_{G}\,\max_{D}\;
\mathcal{L}(G,D)=
\mathbb{E}_{x,y\sim p_{\text{data}}}\bigl[\log D(x,y)\bigr] \;+\;
\mathbb{E}_{z\sim p(z),\,y\sim p_{\text{data}}}\bigl[\log(1-D(G(z,y),y))\bigr]
$$

* **Real term** `(x,y)` sampled from the dataset → target **1**
* **Fake term** `(G(z,y),y)` → target **0**

Because `y` is fed into **both** networks:

* **Discriminator** learns a similarity score *f(x,y)* that is **high** when the visual evidence in `x` agrees with the semantics of `y`.
* **Generator** receives gradients through the **fake term**; if it outputs a digit *5* while `y=3`, `D` easily rejects it (`f` low), so the gradient nudges `G` to make its output look more “3-like”.

At equilibrium $p_{G}(x\,|\,y)=p_{\text{data}}(x\,|\,y)$.

---

## 2  Where the label enters

| Network                    | Injection method                                                                                         | What happens during back-prop?                                                                                                                           |
| -------------------------- | -------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Generator $G(z,y)$**     | concatenate `y`-embedding to `z`, or use Conditional BatchNorm, FiLM, AdaIN, etc.                        | Gradients flow into the **embedding vector for `y`** and all layers that see it, allowing each class to carve out its own region of the output manifold. |
| **Discriminator $D(x,y)$** | *Concatenation* (`x‖y`) or *projection* trick: $D(x,y)=\sigma\big(f(x)+\langle\phi(y),h(x)\rangle\big)$. | If the image content contradicts the label, the inner product term becomes negative, lowering $D$’s score and producing a strong gradient signal.        |

### Embedding layer vs. Linear layer (why we embed `y`)

* **`nn.Embedding(C,d)`**: a *lookup* table $W\in\mathbb{R}^{C\times d}$.
  Input is an **integer id**; output is the corresponding row. Only that row’s parameters receive gradients.
* **`nn.Linear(C,d)`**: a dense affine map that expects a **real vector** (usually one-hot if used for labels); *every* weight updates each step—wasteful for small $C$.

The embedding therefore gives each class its own learnable “style vector” with minimal compute.

---

## 3  Why the link cannot be ignored

*Suppose `G` tries to ignore `y`.*
Then $G(z,y)\approx G(z)$. The discriminator still sees the label, so it quickly discovers a pattern:

> “Whenever the label is 3, the strokes look like random digits, not like a 3.”
> `D`’s accuracy shoots to 1, its gradients ≠ 0, and `G` is forced (by the adversarial loss) to make images that reduce this discrepancy.
> Conversely, if `D` decides to ignore `y`, its real/fake accuracy drops because real and fake *pairs* become indistinguishable—`D` is pushed to use the label.

Thus the *zero-sum* game drives both players to exploit every predictive bit in `y`, resulting in label-conditioned synthesis.

---

## 4  Practical signs the conditioning is working

* **Per-class FID / accuracy** drops: generated “3”s are evaluated only against real “3”s.
* **Interference test**: feed the same latent `z` with all labels 0-9. A well-trained cGAN will morph the same base strokes into each digit while preserving style.
* **Latent traversal** within a fixed label changes pose/thickness but not class, showing disentanglement.

---

### TL;DR

The discriminator judges *pairs* $(x,y)$; the generator can fool it **only** by emitting images whose class evidence truly matches `y`. Gradients flowing through the shared label embeddings make this association explicit and trainable.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Hyperparameters
batch_size = 64
num_epochs = 20
learning_rate = 0.0002
latent_dim = 100
num_classes = 10
img_size = 28
channels = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# MNIST dataset
train_dataset = datasets.MNIST(
    root='./data', train=True, transform=transform, download=True
)

train_loader = DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True
)

# Generator model 
class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, momentum=0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim + num_classes, 128, normalize=False),  # [B, 110] -> [B, 128]
            *block(128, 256),                                        # [B, 128] -> [B, 256]
            *block(256, 512),                                        # [B, 256] -> [B, 512]
            *block(512, 1024),                                       # [B, 512] -> [B, 1024]
            nn.Linear(1024, channels * img_size * img_size),         # [B, 1024] -> [B, 784]
            nn.Tanh()                                                # Scale to [-1, 1]
        )

    def forward(self, noise, labels):
        # noise: [B, 100], labels: [B]
        gen_input = torch.cat((noise, self.label_emb(labels)), -1)  # [B, 110]
        img = self.model(gen_input)                                 # [B, 784]
        img = img.view(img.size(0), channels, img_size, img_size)   # [B, 1, 28, 28]
        return img

# Discriminator model
class Discriminator(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.model = nn.Sequential(
            nn.Linear(channels * img_size * img_size + num_classes, 512),  # [B, 794] -> [B, 512]
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),                                           # [B, 512] -> [B, 256]
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),                                             # [B, 256] -> [B, 1]
            nn.Sigmoid(),
        )

    def forward(self, img, labels):
        img_flat = img.view(img.size(0), -1)                              # [B, 1, 28, 28] -> [B, 784]
        d_in = torch.cat((img_flat, self.label_emb(labels)), -1)         # [B, 784] + [B, 10] = [B, 794]
        validity = self.model(d_in)                                      # [B, 1]
        return validity

# Initialize models 
generator = Generator(latent_dim, num_classes).to(device)
discriminator = Discriminator(latent_dim, num_classes).to(device)

# Loss function
adversarial_loss = nn.BCELoss()

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# Training
for epoch in range(num_epochs):
    for i, (imgs, labels) in enumerate(train_loader):
        batch_size = imgs.size(0)
        real_imgs = imgs.to(device)                    # [B, 1, 28, 28]
        labels = labels.to(device)                     # [B]

        # Real and fake labels
        valid = torch.ones(batch_size, 1, device=device)  # [B, 1]
        fake = torch.zeros(batch_size, 1, device=device)  # [B, 1]

        # Train Generator
        optimizer_G.zero_grad()
        z = torch.randn(batch_size, latent_dim, device=device)       # [B, 100]
        gen_labels = torch.randint(0, num_classes, (batch_size,), device=device)  # [B]
        gen_imgs = generator(z, gen_labels)                          # [B, 1, 28, 28]
        validity = discriminator(gen_imgs, gen_labels)               # [B, 1]
        g_loss = adversarial_loss(validity, valid)
        g_loss.backward()
        optimizer_G.step()

        # Train Discriminator
        optimizer_D.zero_grad()
        real_validity = discriminator(real_imgs, labels)             # [B, 1]
        d_real_loss = adversarial_loss(real_validity, valid)
        fake_validity = discriminator(gen_imgs.detach(), gen_labels)  # [B, 1]
        d_fake_loss = adversarial_loss(fake_validity, fake)
        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        if i % 400 == 0:
            print(f"Epoch [{epoch}/{num_epochs}] Batch {i}/{len(train_loader)} Loss D: {d_loss.item():.4f}, loss G: {g_loss.item():.4f}")
```
print("Training finished.")
**Explanation:**

- **Generator (`Generator` class):**
  - Takes a noise vector (`z`) and a label (`labels`) as input.
  - Embeds the label and concatenates it with the noise vector.
  - Passes the concatenated vector through several fully connected layers.
  - Outputs a generated image corresponding to the input label.

- **Discriminator (`Discriminator` class):**
  - Takes an image (`img`) and a label (`labels`) as input.
  - Embeds the label and concatenates it with the flattened image.
  - Passes the concatenated vector through several fully connected layers.
  - Outputs a scalar value representing the validity of the input.

- **Training Loop:**
  - **Generator Training:**
    - Generates fake images conditioned on random labels.
    - Aims to fool the discriminator into classifying fake images as real.
  - **Discriminator Training:**
    - Evaluates real images with true labels and fake images with generated labels.
    - Aims to correctly classify real and fake images.
  - **Loss Functions:**
    - Uses Binary Cross Entropy Loss (`nn.BCELoss()`) for both generator and discriminator.
  - **Optimization:**
    - Updates the generator and discriminator parameters using Adam optimizer.

**Notes:**

- **Label Embedding:**
  - The labels are embedded into a continuous vector space.
  - This embedding is learned during training and helps the generator and discriminator to condition on labels.

