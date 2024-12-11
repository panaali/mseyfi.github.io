[![Home](https://img.shields.io/badge/Home-Click%20Here-blue?style=flat&logo=homeadvisor&logoColor=white)](../)

## [![CV](https://img.shields.io/badge/CV-Selected_Topics_in_Generative_AI-green?style=for-the-badge&logo=github)](../main_page/GenAI)

Below is a comprehensive tutorial on the Non-Local Means (NLM) algorithm, a highly-regarded method for image denoising. By the end of this tutorial, you will understand the theory behind Non-Local Means, why it is effective, how it is formulated mathematically, and how to implement it in code.

---

## Introduction

Image denoising is a fundamental problem in image processing and computer vision. Given a noisy image, the goal is to recover a clean version that preserves important details and structures. Traditional denoising methods often rely on local information, for example:

- **Linear Filtering (e.g., Gaussian blur):** Averages pixel values within a local neighborhood. While simple, it tends to smooth out edges and important details along with the noise.
- **Non-Linear Methods (e.g., Median Filter):** Looks at local intensity distributions to remove impulse noise. While preserving edges better than linear smoothing, it may still not make full use of the image’s global repetitive structures.

**Non-Local Means (NLM)** introduces a paradigm shift. Instead of considering only local neighborhoods, NLM looks for similar patterns (or patches) throughout the entire image. The key insight is that natural images often contain repetitive structures (e.g., texture patterns or small objects) at different locations. By leveraging these similar structures no matter where they occur, NLM can produce very effective denoising results.

---

## The Intuition Behind Non-Local Means

Consider a pixel $$ i $$ in the noisy image. Traditional smoothing filters estimate the denoised intensity at $$ i $$ by averaging pixels *near* $$ i $$. Non-Local Means, however, says: “To estimate the denoised intensity of pixel $$ i $$, we can average not just the pixels near $$ i $$ but all pixels $$ j $$ in the entire image that have a similar local neighborhood (patch) to that of $$ i $$.”

In other words, NLM searches for patches similar to the patch around the current pixel of interest. If many patches across the image look similar, averaging them produces a cleaner estimate of the underlying structure, while random noise tends to be averaged out.

---

## Mathematical Formulation

### Notation

- Let the noisy image be $$ v = \{ v(x) : x \in \Omega \} $$ where $$ x $$ indexes pixels in the 2D domain $$\Omega$$.
- We want to estimate a denoised image $$ u = \{ u(x) : x \in \Omega \} $$.

For each pixel $$ x $$, consider a patch $$ N_x $$ centered around $$ x $$. A patch is typically a small square window around the pixel, for example $$ 7 \times 7 $$ or $$ 5 \times 5 $$ pixels.

### Weighting Scheme

The non-local means estimate of $$ u(x) $$ is given by a weighted average of all pixels $$ v(y) $$ in the image:

$$
u(x) = \sum_{y \in \Omega} w(x,y) v(y),
$$

where $$ w(x,y) $$ are weights that depend on the similarity of patches $$ N_x $$ and $$ N_y $$.

### Similarity Measure

To determine how similar two patches $$ N_x $$ and $$ N_y $$ are, we compute a Gaussian-weighted Euclidean distance between them:

$$
\| v(N_x) - v(N_y) \|_2^2 = \sum_{\xi \in N_x} G(\xi) [v(x+\xi)-v(y+\xi)]^2
$$

Here, $$ G(\xi) $$ is often a Gaussian kernel that gives higher weight to pixels near the center of the patch, making the similarity measure more robust to outliers or patch boundaries.

### Exponential Weighting

The weights $$ w(x,y) $$ are then defined as:

$$
w(x,y) = \frac{1}{Z(x)} \exp \left(-\frac{\| v(N_x)-v(N_y) \|_2^2}{h^2}\right),
$$

where $$ h $$ is a filtering parameter that controls the decay of the weights with respect to the patch distance. $$ Z(x) $$ is a normalization constant ensuring that $$\sum_{y} w(x,y) = 1$$:

$$
Z(x) = \sum_{y \in \Omega} \exp \left(-\frac{\| v(N_x)-v(N_y) \|_2^2}{h^2}\right).
$$

### Key Parameters

- **Patch size:** Size of the neighborhood patches (e.g., $$7 \times 7$$).
- **Search window size:** Size of the region around $$ x $$ (or possibly the entire image) over which we look for similar patches.
- **Parameter $$ h $$:** Controls how quickly the influence of dissimilar patches falls off. Larger $$ h $$ includes more patches but potentially blur; smaller $$ h $$ is more selective but may lead to less smoothing.
- **Optional pre-filtering (e.g., Gaussian smoothing):** Sometimes done to improve robustness.

---

## Why Non-Local Means Is Effective

1. **Exploits Redundancy:** Natural images have many repetitive structures. Even if noise is random, similar patches across the image can help isolate the underlying structure.
2. **Preserves Edges and Details:** By averaging only among patches that are very similar, NLM avoids mixing unrelated structures. This leads to better detail preservation than classic local filters.
3. **Handles Various Noise Types:** Although often discussed for additive Gaussian noise, the NLM concept can be generalized to other noise models as well.

---

## Complexity Considerations

A naive implementation of NLM is computationally expensive because for each pixel $$ x $$, you might compare its patch $$ N_x $$ with every other patch in the image. If the image has $$ N $$ pixels and each patch comparison takes $$ P $$ operations, a naive approach leads to $$ O(N^2 P) $$ complexity, which can be prohibitively large.

### Speed-Up Techniques

- **Limited Search Window:** Instead of searching the entire image for similar patches, limit the search to a neighborhood around $$ x $$.
- **Pre-Classification of Patches / KD-Trees / Hashing:** Group similar patches to reduce the complexity of comparisons.
- **Integral Images or Fast Convolutions:** Precompute certain statistics to speed up patch comparisons.
- **GPU/Parallel Processing:** The NLM algorithm is highly parallelizable.

---

## Step-by-Step NLM Procedure

1. **Input:** A noisy image $$ v $$.
2. **For each pixel $$ x $$ in $$ v $$:**
   - Extract the patch $$ N_x $$ around $$ x $$.
   - For every candidate pixel $$ y $$ in the search window:
     - Extract patch $$ N_y $$.
     - Compute the weighted Euclidean distance between $$ N_x $$ and $$ N_y $$.
   - Convert distances to weights using the exponential function and parameter $$ h $$.
   - Normalize weights so they sum to 1.
   - Compute the final estimate $$ u(x) $$ as the weighted sum of pixels $$ v(y) $$.
3. **Output:** The denoised image $$ u $$.

---

## Example Implementation in Python

Below is a simple (but not highly optimized) implementation of the Non-Local Means algorithm in Python using NumPy. For demonstration purposes, we will:

- Generate a synthetic noisy image.
- Apply NLM denoising.
- Compare results.

**Requirements:**
- Python 3.x
- NumPy
- Matplotlib (for visualization)

**Note:** This implementation may be slow for large images due to its naive structure. Consider using a smaller image or optimize further as discussed.

```python
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage.util import random_noise

def gaussian_kernel(size, sigma=1):
    """Create a 2D Gaussian kernel."""
    ax = np.linspace(-(size//2), size//2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2*sigma**2))
    return kernel / np.sum(kernel)

def non_local_means(noisy, patch_size=7, search_size=21, h=0.1):
    """
    Non-Local Means denoising.
    Parameters:
        noisy (2D array): Noisy input image (grayscale).
        patch_size (int): Patch size (odd number).
        search_size (int): Search window size around each pixel (odd number).
        h (float): Filtering parameter controlling decay of weights.
    Returns:
        denoised (2D array): Denoised image.
    """
    # Pad image to handle boundaries
    pad_size = search_size // 2
    patch_radius = patch_size // 2
    padded = np.pad(noisy, pad_size + patch_radius, mode='reflect')

    # Gaussian weighting for patch comparison (makes center pixels more important)
    gaussian_weights = gaussian_kernel(patch_size, sigma=patch_size/3)

    denoised = np.zeros_like(noisy)
    rows, cols = noisy.shape

    for i in range(rows):
        for j in range(cols):
            # Coordinates in padded image
            i_p = i + pad_size + patch_radius
            j_p = j + pad_size + patch_radius

            # Extract reference patch
            ref_patch = padded[i_p - patch_radius : i_p + patch_radius + 1,
                               j_p - patch_radius : j_p + patch_radius + 1]

            # Search window coordinates
            i_min = i_p - pad_size
            i_max = i_p + pad_size + 1
            j_min = j_p - pad_size
            j_max = j_p + pad_size + 1

            # Initialize weights
            weights = np.zeros((search_size, search_size))

            # Compute weights
            for di in range(search_size):
                for dj in range(search_size):
                    i_s = i_min + di
                    j_s = j_min + dj

                    # Extract candidate patch
                    cand_patch = padded[i_s - patch_radius : i_s + patch_radius + 1,
                                        j_s - patch_radius : j_s + patch_radius + 1]
                    
                    # Squared difference weighted by Gaussian
                    dist2 = gaussian_weights * (ref_patch - cand_patch)**2
                    dist2_sum = np.sum(dist2)
                    w = np.exp(-dist2_sum / (h**2))
                    weights[di, dj] = w

            # Normalize weights
            weights /= np.sum(weights)

            # Weighted sum of pixels in search window
            search_region = padded[i_min : i_max, j_min : j_max]
            denoised[i, j] = np.sum(weights * search_region)

    return denoised

# Example usage:
if __name__ == "__main__":
    # Load a sample image
    image = img_as_float(data.camera())  # grayscale image

    # Add Gaussian noise
    noisy_img = random_noise(image, var=0.01)

    # Apply NLM denoising
    denoised_img = non_local_means(noisy_img, patch_size=7, search_size=21, h=0.1)

    # Display results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title("Original")
    axes[0].axis('off')

    axes[1].imshow(noisy_img, cmap='gray')
    axes[1].set_title("Noisy")
    axes[1].axis('off')

    axes[2].imshow(denoised_img, cmap='gray')
    axes[2].set_title("NLM Denoised")
    axes[2].axis('off')

    plt.show()
```

---

## Observations and Variations

- **Comparison to BM3D or Wavelet Denoising:** NLM was a groundbreaking technique that inspired more sophisticated approaches (like BM3D). However, NLM often performs very well compared to simpler methods.
- **Color Images:** NLM can be extended to color images by comparing patches across multiple channels, or by operating on a luminance-chrominance representation.
- **Anisotropic NLM:** Variations exist that adapt the search window shape or give different weighting schemas to improve results.
- **Fast Approximation:** Using integral images or hierarchical search can speed up NLM significantly for practical use.

---

## Conclusion

Non-Local Means is a powerful denoising algorithm leveraging the idea that natural images contain repetitive structures. By comparing patches globally and averaging only among similar patches, NLM achieves effective noise reduction while preserving edges and fine details better than conventional local smoothing filters. 

Understanding the mathematics, the algorithmic workflow, and the complexity considerations lays a foundation for applying, optimizing, or extending the NLM approach in your image processing tasks.
