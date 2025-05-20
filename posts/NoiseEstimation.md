
# 🧠 Noise Estimation in a Single Image – Tutorial

---

## 🎯 Goal

> Estimate the **noise variance** or **noise standard deviation (σ)** from an image $g(x, y) = f(x, y) + \eta(x, y)$, where $f$ is the clean image and $\eta$ is additive noise.

We often assume **zero-mean white Gaussian noise**, i.e., $\eta \sim \mathcal{N}(0, \sigma^2)$.

---

## 📚 Common Methods

### 🟡 1. **High-Pass Filtering Method** (Residual-Based)

**Assumption**: Noise is high-frequency, image content is mostly low-frequency.

### 📌 Steps:

1. Apply a **Gaussian blur** to remove low-frequency content.
2. Subtract blurred image from original to isolate **residual (mostly noise)**.
3. Compute the **standard deviation** of this residual.

### 🧪 Code (OpenCV/Numpy):

```python
import cv2
import numpy as np

def estimate_noise_highpass(img):
    img = img.astype(np.float32)
    blurred = cv2.GaussianBlur(img, (7, 7), 1.5)
    residual = img - blurred
    return np.std(residual)

img = cv2.imread('image.png', 0)  # grayscale
sigma_est = estimate_noise_highpass(img)
print(f"Estimated noise std: {sigma_est:.2f}")
```

---

### 🟡 2. **MAD on Wavelet Detail Coefficients**

**MAD**: Median Absolute Deviation

### 📌 Steps:

1. Apply a **wavelet decomposition** (e.g., using Haar or Daubechies).
2. Use **high-frequency subbands** (e.g., horizontal, vertical, diagonal).
3. Estimate σ from detail coefficients:

$$
\sigma \approx \frac{\text{median}(|c|)}{0.6745}
$$

This formula works for Gaussian noise due to properties of MAD.

### 🧪 Code (PyWavelets):

```python
import pywt
import numpy as np
import cv2

def estimate_noise_wavelet(img):
    coeffs = pywt.dwt2(img, 'db1')
    _, (cH, cV, cD) = coeffs
    detail_coeffs = np.concatenate([cH.ravel(), cV.ravel(), cD.ravel()])
    sigma = np.median(np.abs(detail_coeffs)) / 0.6745
    return sigma

img = cv2.imread('image.png', 0)
sigma_wavelet = estimate_noise_wavelet(img.astype(np.float32))
print(f"Estimated noise σ (wavelet): {sigma_wavelet:.2f}")
```

---

### 🟡 3. **Flat Patch Analysis**

Assume **flat regions** (low gradient) contain mostly noise.

### 📌 Steps:

1. Divide image into blocks (e.g., 8×8)
2. Compute local variance for each patch
3. Keep patches with **low edge activity** (low gradient or entropy)
4. Estimate noise as **mean or median variance** of flat patches

This method is robust to structured content.

---

### 🟡 4. **Blind Estimators (No Training)**

#### 📘 Method by Immerkaer (1996) – uses Laplacian

$$
\sigma \approx \sqrt{\frac{\pi}{2}} \cdot \frac{1}{6(N-2)(M-2)} \sum_{x=2}^{N-1} \sum_{y=2}^{M-1} |(g * \nabla^2)(x, y)|
$$

Where $\nabla^2$ is the Laplacian kernel:

$$
\begin{bmatrix}
0 & 1 & 0 \\
1 & -4 & 1 \\
0 & 1 & 0
\end{bmatrix}
$$

---

## 🔬 When to Use What?

| Method              | Strengths               | Weaknesses             |
| ------------------- | ----------------------- | ---------------------- |
| High-pass residual  | Fast, simple            | Sensitive to textures  |
| Wavelet MAD         | Robust, well-founded    | Needs wavelet lib      |
| Flat patch variance | Good for natural images | Needs threshold tuning |
| Laplacian estimator | Fully blind, elegant    | Assumes Gaussian noise |

---

## ✅ Summary

* **If you want a quick estimate**: use high-pass filtering or wavelet-MAD.
* **For robust methods**: wavelet-based MAD is commonly used in denoising papers.
* Noise estimation is essential for **adaptive filters** (Wiener, BM3D, DnCNN).

---
