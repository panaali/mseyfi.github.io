Below is a Python implementation of a Conditional Generative Adversarial Network (cGAN) using PyTorch. This example uses the MNIST dataset to generate images conditioned on class labels (digits from 0 to 9).

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
    def __init__(self):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, momentum=0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim + num_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, channels * img_size * img_size),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate noise and label embedding
        gen_input = torch.cat((noise, self.label_emb(labels)), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), channels, img_size, img_size)
        return img

# Discriminator model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.model = nn.Sequential(
            nn.Linear(channels * img_size * img_size + num_classes, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img, labels):
        # Flatten image
        img_flat = img.view(img.size(0), -1)
        # Concatenate image and label embedding
        d_in = torch.cat((img_flat, self.label_emb(labels)), -1)
        validity = self.model(d_in)
        return validity

# Initialize models
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Loss function
adversarial_loss = nn.BCELoss()

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# Training
for epoch in range(num_epochs):
    for i, (imgs, labels) in enumerate(train_loader):

        batch_size = imgs.size(0)
        real_imgs = imgs.to(device)
        labels = labels.to(device)

        # Adversarial ground truths
        valid = torch.ones(batch_size, 1, device=device)
        fake = torch.zeros(batch_size, 1, device=device)

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = torch.randn(batch_size, latent_dim, device=device)
        gen_labels = torch.randint(0, num_classes, (batch_size,), device=device)

        # Generate images
        gen_imgs = generator(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_imgs, gen_labels)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Real images
        real_validity = discriminator(real_imgs, labels)
        d_real_loss = adversarial_loss(real_validity, valid)

        # Fake images
        fake_validity = discriminator(gen_imgs.detach(), gen_labels)
        d_fake_loss = adversarial_loss(fake_validity, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        # Print training stats
        if i % 400 == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {i}/{len(train_loader)} \
Loss D: {d_loss.item():.4f}, loss G: {g_loss.item():.4f}"
            )

    # (Optional) Save generated images to monitor training progress
    # You can use torchvision.utils.save_image here

print("Training finished.")
```

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

- **Data Normalization:**
  - Images are normalized to the range \([-1, 1]\) using `transforms.Normalize([0.5], [0.5])`.
  - The generator uses a `Tanh` activation function in the output layer to match this range.

- **Device Configuration:**
  - The code automatically uses GPU if available.
  - Ensures faster training when a compatible GPU is present.

- **Monitoring Training:**
  - Training progress is printed every 400 batches.
  - You can modify this interval or add code to save generated images for better monitoring.

**Optional Enhancements:**

- **Image Saving:**
  - Use `torchvision.utils.save_image` to save generated images at each epoch.
  - Helps in visualizing the generator's performance over time.

- **Model Checkpointing:**
  - Save the model parameters using `torch.save` to resume training later.
  - Useful for long training sessions.

- **Hyperparameter Tuning:**
  - Experiment with different learning rates, batch sizes, and network architectures.
  - Adjust the number of epochs based on convergence.

**Dependencies:**

Make sure you have the following packages installed:

```bash
pip install torch torchvision
```

