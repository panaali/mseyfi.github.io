[![Home](https://img.shields.io/badge/Home-Click%20Here-blue?style=flat&logo=homeadvisor&logoColor=white)](../)

## [![CV](https://img.shields.io/badge/CV-Selected_Topics_in_Computer_Vision-green?style=for-the-badge&logo=github)](../main_page/CV)

# 📘 Full Tutorial: High-Pass and Low-Pass Filtering with Laplacian, Gaussian Kernels, and Image Sharpening

---

## 🔷 1. Introduction

This tutorial walks through:

* High-pass and low-pass filtering
* How Laplacian detects curvature and edge polarity
* How Gaussian blurs smooth images
* How to build high-pass filters from low-pass ones
* **How to sharpen images using Laplacian or Gaussian blur**

---

## 🔷 2. Kernels Overview

### 🟩 Low-Pass Filter (Gaussian Kernel)

$$
K_{\text{gauss}} = \frac{1}{16}
\begin{bmatrix}
1 & 2 & 1 \\
2 & 4 & 2 \\
1 & 2 & 1
\end{bmatrix}
$$

* Smooths the image
* Preserves low frequencies
* Kernel sum = **1**

---

### 🟥 High-Pass Filter (Laplacian Kernel)

$$
K_{\text{laplace}} =
\begin{bmatrix}
-1 & -1 & -1 \\
-1 & 8 & -1 \\
-1 & -1 & -1
\end{bmatrix}
$$

* Enhances edges and fine texture
* Kernel sum = **0**
* Sensitive to **curvature** and **polarity**

---

## 🔶 3. Laplacian as a Curvature Detector

The Laplacian operator:

$$
\Delta I(x, y) = \frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2}
$$

Measures **local curvature**:

| Region        | Laplacian Output |
| ------------- | ---------------- |
| Flat          | 0                |
| Bright center | Positive         |
| Dark center   | Negative         |

Used for **edge detection** and **edge polarity**.

---

## 🔷 4. Bright vs. Dark Edge Behavior

| Type        | Center   | Neighbors | Output      |
| ----------- | -------- | --------- | ----------- |
| Bright blob | Brighter | Darker    | 🔺 Positive |
| Dark pit    | Darker   | Brighter  | 🔻 Negative |

To visualize both types, take `abs(laplacian)` or encode sign with color.

---

## 🔶 5. Blurring with Gaussian (LPF)

Gaussian smoothing:

* Reduces high-frequency noise
* Creates a **blurred version** of the image
* Used before downsampling or as part of denoising

OpenCV:

```python
blurred = cv2.GaussianBlur(image, (5, 5), sigmaX=1.0)
```

---

## 🔷 6. High-Pass from Blurring

You can generate a high-pass image by:

$$
\text{HPF} = \text{Original} - \text{Blurred}
$$

This removes the low-frequency parts (smooth regions) and retains high-frequency details (edges).

---

## ✅ 7. Image Sharpening Techniques

### 🔹 1. **Sharpening via Laplacian (Unsharp Masking)**

$$
\text{Sharpened} = \text{Original} - \lambda \cdot \text{Laplacian}
$$

* Subtract Laplacian → enhances edges
* $\lambda$ controls intensity (e.g., 1.0)

```python
lap = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
sharpened = cv2.addWeighted(img, 1.0, lap, -1.0, 0)
```

---

### 🔹 2. **Sharpening via Gaussian Blur (Unsharp Masking)**

$$
\text{HPF} = \text{Original} - \text{Blurred} \\
\text{Sharpened} = \text{Original} + \lambda \cdot \text{HPF}
$$

This gives:

$$
\text{Sharpened} = (1 + \lambda) \cdot \text{Original} - \lambda \cdot \text{Blurred}
$$

```python
blur = cv2.GaussianBlur(img, (5, 5), 1.0)
hpf = cv2.subtract(img, blur)
sharpened = cv2.addWeighted(img, 1.0, hpf, 1.0, 0)
```

---

## 🔷 8. Python Implementation

```python
import cv2
import numpy as np

# Load image
img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Gaussian Blur (LPF)
blur = cv2.GaussianBlur(img, (5, 5), 1.0)

# Laplacian (HPF)
lap = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
lap_abs = np.uint8(np.clip(np.abs(lap), 0, 255))

# HPF from blur
hpf_from_blur = cv2.subtract(img, blur)

# Sharpen using Laplacian
sharpen_lap = cv2.addWeighted(img, 1.0, lap, -1.0, 0)
sharpen_lap = np.clip(sharpen_lap, 0, 255).astype(np.uint8)

# Sharpen using blur subtraction
sharpen_blur = cv2.addWeighted(img, 1.0, hpf_from_blur, 1.0, 0)

# Save outputs
cv2.imwrite('blur.jpg', blur)
cv2.imwrite('laplacian.jpg', lap_abs)
cv2.imwrite('hpf_from_blur.jpg', hpf_from_blur)
cv2.imwrite('sharpen_lap.jpg', sharpen_lap)
cv2.imwrite('sharpen_blur.jpg', sharpen_blur)
```

---

## ✅ 9. Summary Table

| Method              | Kernel Sum | Enhances   | Removes    | Notes                   |
| ------------------- | ---------- | ---------- | ---------- | ----------------------- |
| Gaussian LPF        | 1          | Smoothness | Detail     | All-positive weights    |
| Laplacian HPF       | 0          | Edges      | Flat areas | Signed curvature        |
| HPF via Blur        | 0          | Edges      | Blur       | Easy and fast           |
| Sharpen (Laplacian) | —          | Contrast   | None       | Subtracts curvature     |
| Sharpen (Gaussian)  | —          | Contrast   | Blur       | Unsharp masking variant |

---

# 🧠 Tutorial: Laplacian Pyramid Blending — Full Theory, Intuition, and Code

---

## 🔷 Overview

Laplacian pyramid blending is a **multi-scale image fusion technique** designed to seamlessly combine two images. Instead of directly blending pixel values, it fuses **image structures at different frequency bands**, which avoids visible seams and preserves both low-frequency background and high-frequency details.

---

## 🔶 Problem: Why Naive Blending Fails

Given:

* Image A
* Image B
* Mask M (defines blending region: 1 = A, 0 = B)

### ❌ Naive blending:

```
blend = M * A + (1 - M) * B
```

* Produces sharp seams if M is binary.
* Fails at preserving texture, edges.

---

## ✅ Solution: Multi-scale Blending with Laplacian Pyramids

### 🎯 Core Idea:

Blend **low frequencies** over **broad regions** and **high frequencies** only **near transitions**. This creates a **smooth blend** in global color and sharpness.

---

## 🔷 Background: Gaussian and Laplacian Pyramids

### 🟩 Gaussian Pyramid (G):

* Sequence of **blurred and downsampled** images.
* Each level is a **lower resolution** approximation.
* Used to extract low-frequency (smooth) content.

### 🟥 Laplacian Pyramid (L):

* Stores **difference between levels** in Gaussian pyramid:

$L_i = G_i - \text{Upsample}(G_{i+1})$

* Captures **details (high-frequency residuals)**
* Final level $L_N = G_N$: lowest-res base image

---

## 🔷 Algorithm Steps

### Input:

* Image A, Image B (same size)
* Mask M (values $\in [0, 1]$)
* Pyramid depth $N$

### Step-by-step:

1. **Build Gaussian pyramids**:

   * $G_A, G_B, G_M$
2. **Build Laplacian pyramids**:

   * $L_A, L_B$
3. **Blend pyramids**:

   * For each level $i$:
     $L_{blend}^i = G_M^i \cdot L_A^i + (1 - G_M^i) \cdot L_B^i$
4. **Collapse blended Laplacian pyramid** to reconstruct final image.

---

## 🔶 Why Multi-scale Works (Intuition)

| Frequency | What is blended  | Result                  |
| --------- | ---------------- | ----------------------- |
| Low freq  | Global structure | Smooth transitions      |
| High freq | Edges, texture   | Sharp seams are avoided |

### ⚠ Problem with hard masks:

Without pyramid mask smoothing, high-res levels will sharply cut edges. That leads to ghosting or seams.

### ✅ Pyramid mask = frequency-aware blending:

Blending at each scale with a smoothed mask ensures transitions are aligned to scale.

---

## 🎨 Analogies

* **Audio Equalizer**: Each Laplacian level = frequency band
* **Painter's Brush**:

  * Broad brush for backgrounds (low-freq)
  * Fine brush for detail (high-freq)
* **Focal blending**: coarse vision blends wide zones, fine vision blends contours

---

## 🧮 Mathematical Formulation

Let:

* $G_A^i$: Gaussian pyramid of A
* $G_B^i$: Gaussian pyramid of B
* $G_M^i$: Gaussian pyramid of mask M
* $L_A^i, L_B^i$: Laplacian pyramids
* $L_{blend}^i$: blended pyramid

Then:

$$
L_A^i = G_A^i - \text{Upsample}(G_A^{i+1}) \quad \text{(same for B)}
$$

$$
L_{blend}^i = G_M^i \cdot L_A^i + (1 - G_M^i) \cdot L_B^i
$$

$$
\text{Reconstruct: } R^{i} = L_{blend}^{i} + \text{Upsample}(R^{i+1})
$$

---

## 🔁 Flowchart

```
Image A         Image B         Mask
   |               |              |
   v               v              v
GaussianPyrA   GaussianPyrB   GaussianPyrMask
   |               |              |
   v               v              v
LaplacianPyrA  LaplacianPyrB      |
   \_________   ________/         |
             \ /                  |
         Blend at each level <----
                |
         Collapse pyramid
                |
             Output
```

---

## 💻 Code (Python + OpenCV)

```python
import cv2
import numpy as np

def build_gaussian_pyramid(img, levels):
    gp = [img.copy()]
    for _ in range(levels):
        img = cv2.pyrDown(img)
        gp.append(img)
    return gp

def build_laplacian_pyramid(gp):
    lp = []
    for i in range(len(gp) - 1):
        up = cv2.pyrUp(gp[i + 1], dstsize=(gp[i].shape[1], gp[i].shape[0]))
        lp.append(cv2.subtract(gp[i], up))
    lp.append(gp[-1])
    return lp

def blend_pyramids(lpA, lpB, gpM):
    return [gm * la + (1 - gm) * lb for la, lb, gm in zip(lpA, lpB, gpM)]

def reconstruct_from_laplacian(lp):
    img = lp[-1]
    for i in range(len(lp) - 2, -1, -1):
        img = cv2.pyrUp(img, dstsize=(lp[i].shape[1], lp[i].shape[0]))
        img = cv2.add(img, lp[i])
    return img

# Load and normalize
A = cv2.imread('A.jpg').astype(np.float32) / 255
B = cv2.imread('B.jpg').astype(np.float32) / 255
M = cv2.imread('mask.jpg', 0).astype(np.float32) / 255
M = cv2.merge([M, M, M])

levels = 5

gpA = build_gaussian_pyramid(A, levels)
gpB = build_gaussian_pyramid(B, levels)
gpM = build_gaussian_pyramid(M, levels)

lpA = build_laplacian_pyramid(gpA)
lpB = build_laplacian_pyramid(gpB)

blended_lp = blend_pyramids(lpA, lpB, gpM)
blended_img = reconstruct_from_laplacian(blended_lp)

cv2.imwrite('blended.jpg', (np.clip(blended_img, 0, 1) * 255).astype(np.uint8))
```

---

## ✅ Conclusion

Laplacian pyramid blending is a powerful method for **seamless image fusion**. By decomposing and recomposing images across multiple resolutions, it allows fine control of transition zones and detail preservation — solving the fundamental challenge of edge artifacts in image compositing.


