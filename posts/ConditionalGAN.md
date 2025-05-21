[![Home](https://img.shields.io/badge/Home-Click%20Here-blue?style=flat&logo=homeadvisor&logoColor=white)](../)

## [![CV](https://img.shields.io/badge/CV-Selected_Topics_in_Computer_Vision-green?style=for-the-badge&logo=github)](../main_page/CV)
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

