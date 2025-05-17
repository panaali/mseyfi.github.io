[![Home](https://img.shields.io/badge/Home-Click%20Here-blue?style=flat&logo=homeadvisor&logoColor=white)](../)

## [![CV](https://img.shields.io/badge/CV-Selected_Topics_in_Computer_Vision-green?style=for-the-badge&logo=github)](../main_page/CV)

**Sub-pixel convolution** (also known as **pixel shuffle**) is a technique primarily used for **image super-resolution** and other upsampling tasks in deep learning. Instead of upsampling via interpolation or transposed convolution, it learns to generate a **high-resolution image** from a low-resolution feature map by **reorganizing the channels**.

---

## 🔧 Core Idea:

Given a low-resolution feature map with shape:
**(B, C × r², H, W)**

It rearranges it to shape:
**(B, C, H × r, W × r)**

Here:

* `B` = batch size
* `C` = number of channels in output
* `H`, `W` = spatial size of the feature map
* `r` = upscaling factor (e.g., 2, 3, 4)

---

## 🧠 Why Use It?

* Avoids checkerboard artifacts common in transposed convolutions.
* Reduces computational cost: instead of working in high-res space, the convolution is done in low-res and then upscaled.

---

## 🧮 Mathematical Overview

Suppose you want to upscale by a factor `r`. Instead of upsampling directly, you:

1. **Use a convolution layer** that outputs `C × r²` channels.
2. **Rearrange (reshape + transpose)** the channels into spatial dimensions.

### Example:

For an upscaling factor `r = 2`:

* Input: tensor with shape `(B, 4, H, W)`
* Pixel shuffle output: `(B, 1, 2H, 2W)`

Why 4? Because `r² = 2² = 4`

---

## 📦 PyTorch Code Example

```python
import torch
import torch.nn as nn

# Simulate input
input = torch.randn(1, 4, 5, 5)  # B=1, C=4, H=5, W=5

# Pixel shuffle
pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
output = pixel_shuffle(input)  # output shape will be (1, 1, 10, 10)
```

---

## 🧱 Block Diagram (Conceptual)

```
Input Feature Map: (B, C*r^2, H, W)
     ↓
Convolution (learns features for upsampling)
     ↓
PixelShuffle (rearranges channels into space)
     ↓
Output: (B, C, H*r, W*r)
```

---

## 🔁 Used In:

* ESPCN (Efficient Sub-Pixel Convolutional Neural Network)
* Real-Time Single Image Super-Resolution (Shi et al., 2016) [\[paper\]](https://arxiv.org/abs/1609.05158)

---

## 🆚 Compared to Transposed Convolution

| Aspect      | Sub-pixel Convolution         | Transposed Convolution              |
| ----------- | ----------------------------- | ----------------------------------- |
| Artifacts   | Less prone to checkerboard    | Can suffer from checkerboard        |
| Speed       | Faster (low-res domain)       | Slower (operates in upsampled size) |
| Flexibility | Needs careful channel shaping | More flexible                       |

---

