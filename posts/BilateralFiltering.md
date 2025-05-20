[![Home](https://img.shields.io/badge/Home-Click%20Here-blue?style=flat&logo=homeadvisor&logoColor=white)](../)

## [![CV](https://img.shields.io/badge/CP-Selected_Topics_in_Computational_Photography-green?style=for-the-badge&logo=github)](../main_page/CP)

# 🧠 **Bilateral Filtering in Image Processing – Tutorial**

---

## 🟡 1. Motivation

In image denoising, we often want to remove **noise** but preserve **edges**. Classical filters like Gaussian blur are fast but **blur across edges**, causing loss of detail.

**Bilateral filtering** solves this by combining:

* **Spatial closeness** (like Gaussian blur)
* **Intensity similarity** (to avoid averaging across edges)

---

## 🟩 2. Intuition

Imagine averaging only those pixels that:

1. Are **close in distance**
2. Have **similar brightness/color**

So, a pixel is influenced only by **neighboring pixels that look similar**.

> It’s like a **smart blur** that doesn’t mix objects of different intensities.

---

## 🔢 3. Mathematical Formulation

Let $I(x)$ be the intensity at pixel $x$. Then the bilateral filter output at $x$ is:

$$
I_{\text{out}}(x) = \frac{1}{W(x)} \sum_{x_i \in \Omega} I(x_i) \cdot f_s(\|x - x_i\|) \cdot f_r(|I(x) - I(x_i)|)
$$

Where:

* $\Omega$ = local neighborhood of $x$
* $f_s$ = **spatial Gaussian** → penalizes far-away pixels
* $f_r$ = **range Gaussian** → penalizes intensity differences
* $W(x)$ = normalization factor:

  $$
  W(x) = \sum_{x_i \in \Omega} f_s(\|x - x_i\|) \cdot f_r(|I(x) - I(x_i)|)
  $$

---

## 🧮 4. Kernel Details

* $f_s(d) = \exp\left(-\frac{d^2}{2\sigma_s^2}\right)$
* $f_r(\Delta I) = \exp\left(-\frac{(\Delta I)^2}{2\sigma_r^2}\right)$

| Term       | Meaning                                                              |
| ---------- | -------------------------------------------------------------------- |
| $\sigma_s$ | Controls spatial influence radius                                    |
| $\sigma_r$ | Controls range sensitivity (how different intensities are tolerated) |

---

## 📊 5. Visual Flowchart

```plaintext
For each pixel x:
    └─ For each neighbor x_i in window:
         ├─ Compute spatial weight: f_s(||x - x_i||)
         ├─ Compute range weight: f_r(|I(x) - I(x_i)|)
         ├─ Multiply both weights and image value: weight * I(x_i)
    └─ Normalize by total weight sum
```

---

## 🧪 6. Python Code Example (Simplified)

```python
import cv2
import numpy as np

img = cv2.imread('image.jpg')
filtered = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
```

| Param                     | Meaning                                |
| ------------------------- | -------------------------------------- |
| `d`                       | Diameter of neighborhood               |
| `sigmaColor` = $\sigma_r$ | How much color difference is tolerated |
| `sigmaSpace` = $\sigma_s$ | How much spatial distance is tolerated |

---

## 🧠 7. Comparison to Non-Local Means (NLM)

| Feature                           | Bilateral Filter               | Non-Local Means (NLM)        |
| --------------------------------- | ------------------------------ | ---------------------------- |
| **Neighborhood**                  | Local (fixed radius)           | Non-local (can span image)   |
| **Weight based on**               | Spatial + Intensity difference | Patch similarity             |
| **Edge-preserving?**              | Yes                            | Yes                          |
| **Texture handling**              | OK                             | Better (uses structure)      |
| **Computation**                   | Fast                           | Slow (can be accelerated)    |
| **Noise removal quality**         | Moderate                       | High                         |
| **Sensitive to parameter tuning** | Moderate                       | High                         |
| **Mathematical base**             | Point-wise similarity          | Patch-wise similarity        |
| **Implemented in OpenCV**         | `cv2.bilateralFilter()`        | `cv2.fastNlMeansDenoising()` |

---

## 🧩 8. Intuition: Bilateral vs NLM

**Bilateral**:

> "I’ll average with nearby pixels that are close in value."

**Non-Local Means**:

> "I’ll average with any patch in the image that *looks like me*, regardless of where it is."

---

## 🧬 9. Non-Local Means (NLM) Brief Overview

Let $P(x)$ be a patch around pixel $x$. NLM is:

$$
I_{\text{out}}(x) = \frac{1}{Z(x)} \sum_{x_i} I(x_i) \cdot \exp\left(-\frac{\|P(x) - P(x_i)\|^2}{h^2}\right)
$$

Where $h$ controls filtering strength.

---

## 🧪 NLM Python Code

```python
dst = cv2.fastNlMeansDenoisingColored(img, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)
```

---

## 🧠 10. When to Use What?

| Scenario                                      | Use                                     |
| --------------------------------------------- | --------------------------------------- |
| Fast edge-preserving smoothing                | **Bilateral**                           |
| High-quality denoising of textures (but slow) | **NLM**                                 |
| Real-time system                              | **Bilateral**                           |
| Medical, satellite, or low-light images       | **NLM** (better structure preservation) |

---

## ✅ Summary

* **Bilateral Filter**: Edge-preserving blur using spatial + intensity proximity.
* **NLM**: Powerful denoiser using patch-based similarity, non-local in nature.
* Bilateral = simple, fast, local
* NLM = accurate, slow, global

Would you like me to walk through an end-to-end image example in NumPy, or illustrate both filters side-by-side?
