[![Home](https://img.shields.io/badge/Home-Click%20Here-blue?style=flat&logo=homeadvisor&logoColor=white)](../)

## [![CV](https://img.shields.io/badge/CV-Selected_Topics_in_Computational_Photography-green?style=for-the-badge&logo=github)](../main_page/CP)

# Motion Estimation and Compensation in Video Coding: A Comprehensive Guide

## Introduction

**Motion estimation** and **motion compensation** are critical components in video compression algorithms. They exploit temporal redundancies between consecutive frames in a video sequence to reduce the amount of data required for efficient storage and transmission. By predicting the motion of objects from one frame to another, we can represent a video more compactly without significantly compromising visual quality.

This guide provides an in-depth exploration of motion estimation and compensation, covering:

- Mathematical foundations
- Detailed algorithms with step-by-step explanations
- Code implementations with inline comments
- Flowcharts illustrating the algorithms
- Application of motion estimation in video coding

---

## Table of Contents

1. [Mathematical Background](#1-mathematical-background)
2. [Motion Estimation Algorithms](#2-motion-estimation-algorithms)
   - [2.1. Exhaustive Search (Full Search) Algorithm](#21-exhaustive-search-full-search-algorithm)
   - [2.2. Three-Step Search Algorithm](#22-three-step-search-algorithm)
   - [2.3. Diamond Search Algorithm](#23-diamond-search-algorithm)
   - [2.4. Hierarchical (Multi-Resolution) Search Algorithm](#24-hierarchical-multi-resolution-search-algorithm)
3. [Implementation of Algorithms](#3-implementation-of-algorithms)
4. [Flowcharts of Algorithms](#4-flowcharts-of-algorithms)
5. [Motion Estimation in Video Coding](#5-motion-estimation-in-video-coding)
   - [5.1. Steps in Video Encoding with Motion Estimation](#51-steps-in-video-encoding-with-motion-estimation)
6. [Conclusion](#6-conclusion)
7. [References](#7-references)

---

## 1. Mathematical Background

### Block-Based Motion Estimation

In video sequences, motion estimation is typically performed on a block-by-block basis. An image frame is divided into non-overlapping blocks of pixels (e.g., 16x16, 8x8). The goal is to find, for each block in the current frame, the best matching block within a search window in a reference frame (usually the previous frame).

**Motion Vector (MV):** The displacement between the position of the block in the current frame and the position of the best matching block in the reference frame.

### Error Metrics

To determine the best match, an error metric quantifies the similarity between blocks. Commonly used error metrics include:

- **Sum of Absolute Differences (SAD):**

  $$
  \text{SAD} = \sum_{i=1}^{N} \sum_{j=1}^{N} |C(i, j) - R(i + \delta x, j + \delta y)|
  $$

- **Mean Squared Error (MSE):**

  $$
  \text{MSE} = \frac{1}{N^2} \sum_{i=1}^{N} \sum_{j=1}^{N} [C(i, j) - R(i + \delta x, j + \delta y)]^2
  $$

- **Peak Signal-to-Noise Ratio (PSNR):**

  $$
  \text{PSNR} = 10 \log_{10}\left( \frac{L^2}{\text{MSE}} \right)
  $$

  Where $$L$$ is the maximum possible pixel value.

**Variables:**

- $$C(i, j)$$: Pixel value at position $$(i, j)$$ in the current block.
- $$R(i + \delta x, j + \delta y)$$: Pixel value at the corresponding position in the reference block shifted by motion vector $$(\delta x, \delta y)$$.

---

## 2. Motion Estimation Algorithms

Motion estimation algorithms aim to efficiently find the best matching block in the reference frame. We'll discuss several algorithms:

1. Exhaustive Search (Full Search)
2. Three-Step Search
3. Diamond Search
4. Hierarchical (Multi-Resolution) Search

### 2.1. Exhaustive Search (Full Search) Algorithm

**Description:**

- The most straightforward method.
- Searches every possible position within the search window.
- Guarantees the optimal match but is computationally expensive.

**Algorithm Steps:**

1. For each block in the current frame:
   - Initialize minimum error to infinity.
   - For each candidate block within the search window in the reference frame:
     - Compute the error metric (e.g., SAD).
     - If the computed error is less than the minimum error:
       - Update the minimum error.
       - Store the motion vector corresponding to the candidate block.

### 2.2. Three-Step Search Algorithm

**Description:**

- Reduces computational complexity compared to exhaustive search.
- Uses a coarse-to-fine search strategy.

**Algorithm Steps:**

1. **Initialization:**
   - Set the initial step size $$s = 4$$ (or $$s = \text{Maximum Search Range} / 2$$).
   - Start at the center of the search window.

2. **First Step:**
   - Evaluate the nine points around the center (including the center) with displacements $$\{(-s, -s), (-s, 0), (-s, s), (0, -s), (0, 0), (0, s), (s, -s), (s, 0), (s, s)\}$$.
   - Find the point with the minimum error.

3. **Second Step:**
   - Reduce the step size $$s = s / 2$$.
   - Center the search around the best point from the previous step.
   - Evaluate the nine points as before.

4. **Third Step:**
   - Repeat the process until $$s = 1$$.

### 2.3. Diamond Search Algorithm

**Description:**

- Further reduces computations compared to three-step search.
- Uses a diamond-shaped pattern for searching.

**Algorithm Steps:**

1. **Large Diamond Search Pattern (LDSP):**
   - Use a diamond-shaped pattern to search around the current point.

2. **Search Process:**
   - Start at the center.
   - Apply LDSP.
   - If the minimum error point is at the center:
     - Switch to Small Diamond Search Pattern (SDSP).
   - Else:
     - Move the center to the minimum error point and repeat LDSP.

3. **Termination:**
   - When SDSP is applied, and the minimum error point is at the center, the search ends.

### 2.4. Hierarchical (Multi-Resolution) Search Algorithm

**Description:**

- Uses multiple resolutions of the frames (pyramids).
- Coarse motion is estimated at lower resolutions.
- Refinement is done at higher resolutions.

**Algorithm Steps:**

1. **Construct Image Pyramids:**
   - Create a Gaussian pyramid for both current and reference frames.

2. **Coarse Level Search:**
   - At the lowest resolution, perform motion estimation using a simple algorithm.

3. **Refinement:**
   - Use the estimated motion vectors as a starting point for the next higher resolution.
   - Refine the motion vectors at each level.

---

## 3. Implementation of Algorithms

Let's implement each algorithm using Python and NumPy. We will use grayscale images for simplicity.

### Common Setup

```python
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color

# Load frames
frame1 = io.imread('frame1.png')  # Reference frame
frame2 = io.imread('frame2.png')  # Current frame

# Convert to grayscale
frame1_gray = color.rgb2gray(frame1)
frame2_gray = color.rgb2gray(frame2)

# Ensure frames are the same size
assert frame1_gray.shape == frame2_gray.shape, "Frames must be the same size"

# Parameters
block_size = 16
search_range = 7  # Maximum displacement in pixels

height, width = frame1_gray.shape

# Initialize motion vector field
mv_field = np.zeros((height // block_size, width // block_size, 2), dtype=np.int)
```

### 3.1. Exhaustive Search Implementation

```python
# Exhaustive Search Motion Estimation
for i in range(0, height - block_size + 1, block_size):
    for j in range(0, width - block_size + 1, block_size):
        # Current block
        current_block = frame2_gray[i:i+block_size, j:j+block_size]

        min_mse = float('inf')
        best_mv = (0, 0)

        # Search window boundaries in the reference frame
        i_min = max(i - search_range, 0)
        i_max = min(i + search_range, height - block_size)
        j_min = max(j - search_range, 0)
        j_max = min(j + search_range, width - block_size)

        # Exhaustive search over the search window
        for m in range(i_min, i_max + 1):
            for n in range(j_min, j_max + 1):
                reference_block = frame1_gray[m:m+block_size, n:n+block_size]
                mse = np.mean((current_block - reference_block) ** 2)

                if mse < min_mse:
                    min_mse = mse
                    best_mv = (m - i, n - j)

        mv_field[i // block_size, j // block_size] = best_mv
```

**Explanation:**

- For each block in the current frame, we search all possible positions within the search window in the reference frame.
- We compute the Mean Squared Error (MSE) between the current block and each candidate block.
- We select the motion vector corresponding to the candidate block with the minimum MSE.

### 3.2. Three-Step Search Implementation

```python
# Three-Step Search Motion Estimation
for i in range(0, height - block_size + 1, block_size):
    for j in range(0, width - block_size + 1, block_size):
        # Current block
        current_block = frame2_gray[i:i+block_size, j:j+block_size]

        center_m, center_n = i, j
        step_size = max(2 ** (int(np.ceil(np.log2(search_range))) - 1), 1)

        while step_size >= 1:
            min_mse = float('inf')
            best_mv = (0, 0)

            # Search points
            for m in range(-step_size, step_size + 1, step_size):
                for n in range(-step_size, step_size + 1, step_size):
                    ref_m = center_m + m
                    ref_n = center_n + n

                    if (0 <= ref_m <= height - block_size) and (0 <= ref_n <= width - block_size):
                        reference_block = frame1_gray[ref_m:ref_m+block_size, ref_n:ref_n+block_size]
                        mse = np.mean((current_block - reference_block) ** 2)

                        if mse < min_mse:
                            min_mse = mse
                            best_mv = (ref_m - i, ref_n - j)
                            best_m, best_n = ref_m, ref_n

            center_m, center_n = best_m, best_n
            step_size = step_size // 2

        mv_field[i // block_size, j // block_size] = (center_m - i, center_n - j)
```

**Explanation:**

- Starts with a large step size and progressively reduces it.
- At each step, searches the 9 points around the current center.
- Updates the center to the point with the minimum MSE.
- Repeats until the step size becomes zero.

### 3.3. Diamond Search Implementation

```python
# Diamond Search Motion Estimation
LDSP = [(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)]
SDSP = [(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)]

for i in range(0, height - block_size + 1, block_size):
    for j in range(0, width - block_size + 1, block_size):
        # Current block
        current_block = frame2_gray[i:i+block_size, j:j+block_size]

        center_m, center_n = i, j
        pattern = LDSP
        search = True

        while search:
            min_mse = float('inf')
            best_mv = (0, 0)
            for (dm, dn) in pattern:
                ref_m = center_m + dm
                ref_n = center_n + dn

                if (0 <= ref_m <= height - block_size) and (0 <= ref_n <= width - block_size):
                    reference_block = frame1_gray[ref_m:ref_m+block_size, ref_n:ref_n+block_size]
                    mse = np.mean((current_block - reference_block) ** 2)

                    if mse < min_mse:
                        min_mse = mse
                        best_mv = (ref_m - i, ref_n - j)
                        best_m, best_n = ref_m, ref_n

            if (best_m == center_m) and (best_n == center_n):
                if pattern == LDSP:
                    pattern = SDSP
                else:
                    search = False
            else:
                center_m, center_n = best_m, best_n

        mv_field[i // block_size, j // block_size] = (center_m - i, center_n - j)
```

**Explanation:**

- Uses two patterns: Large Diamond Search Pattern (LDSP) and Small Diamond Search Pattern (SDSP).
- Begins with LDSP; if the best match is at the center, switches to SDSP.
- Continues until the center point is the best match in SDSP.

### 3.4. Hierarchical Search Implementation

```python
from skimage.transform import pyramid_gaussian

# Create Gaussian pyramids
pyramid_levels = 3
pyramid_frame1 = tuple(pyramid_gaussian(frame1_gray, max_layer=pyramid_levels - 1, downscale=2))
pyramid_frame2 = tuple(pyramid_gaussian(frame2_gray, max_layer=pyramid_levels - 1, downscale=2))

# Initialize motion vectors at the coarsest level
mv_field_pyramid = [None] * pyramid_levels
for level in reversed(range(pyramid_levels)):
    frame1_level = pyramid_frame1[level]
    frame2_level = pyramid_frame2[level]

    scale = 2 ** level
    height_level, width_level = frame1_level.shape
    mv_field_level = np.zeros((height_level // block_size, width_level // block_size, 2), dtype=np.int)

    for i in range(0, height_level - block_size + 1, block_size):
        for j in range(0, width_level - block_size + 1, block_size):
            # Current block
            current_block = frame2_level[i:i+block_size, j:j+block_size]

            if level == pyramid_levels - 1:
                # Coarsest level: use exhaustive search
                min_mse = float('inf')
                best_mv = (0, 0)

                # Search window (small at coarsest level)
                search_range_level = 1
                i_min = max(i - search_range_level, 0)
                i_max = min(i + search_range_level, height_level - block_size)
                j_min = max(j - search_range_level, 0)
                j_max = min(j + search_range_level, width_level - block_size)

                for m in range(i_min, i_max + 1):
                    for n in range(j_min, j_max + 1):
                        reference_block = frame1_level[m:m+block_size, n:n+block_size]
                        mse = np.mean((current_block - reference_block) ** 2)

                        if mse < min_mse:
                            min_mse = mse
                            best_mv = (m - i, n - j)

                mv_field_level[i // block_size, j // block_size] = best_mv
            else:
                # Higher levels: refine motion vectors
                prev_mv = mv_field_pyramid[level + 1][i // (2 * block_size), j // (2 * block_size)] * 2
                center_m = i + prev_mv[0]
                center_n = j + prev_mv[1]

                min_mse = float('inf')
                best_mv = (0, 0)

                search_range_level = 1
                for dm in range(-search_range_level, search_range_level + 1):
                    for dn in range(-search_range_level, search_range_level + 1):
                        ref_m = center_m + dm
                        ref_n = center_n + dn

                        if (0 <= ref_m <= height_level - block_size) and (0 <= ref_n <= width_level - block_size):
                            reference_block = frame1_level[ref_m:ref_m+block_size, ref_n:ref_n+block_size]
                            mse = np.mean((current_block - reference_block) ** 2)

                            if mse < min_mse:
                                min_mse = mse
                                best_mv = (ref_m - i, ref_n - j)

                mv_field_level[i // block_size, j // block_size] = best_mv

    mv_field_pyramid[level] = mv_field_level

# The final motion vector field is at level 0
mv_field = mv_field_pyramid[0]
```

**Explanation:**

- Constructs Gaussian pyramids of the frames.
- Performs motion estimation starting from the coarsest level.
- At each finer level, refines the motion vectors obtained from the previous level.
- Motion vectors are scaled appropriately when moving between levels.

---

## 4. Flowcharts of Algorithms

### 4.1. Exhaustive Search Flowchart

```
Start
 |
For each block in current frame
 |
For each candidate position in search window
 |
Compute error metric (e.g., SAD)
 |
If error < min_error
 |
Update min_error and best_mv
 |
Store best_mv for the block
 |
End
```

### 4.2. Three-Step Search Flowchart

```
Start
 |
For each block in current frame
 |
Initialize step_size = max_search_range / 2
 |
While step_size >= 1
 |
Evaluate 9 points around center
 |
Find point with min_error
 |
Update center to best point
 |
step_size = step_size / 2
 |
Store best_mv for the block
 |
End
```

### 4.3. Diamond Search Flowchart

```
Start
 |
For each block in current frame
 |
Initialize center position
 |
Set pattern = LDSP
 |
While searching
 |
Apply pattern around center
 |
Find point with min_error
 |
If center is best point
 |
If pattern == LDSP
 |
Switch to SDSP
 |
Else
 |
Search = False
 |
Else
 |
Update center to best point
 |
Store best_mv for the block
 |
End
```

### 4.4. Hierarchical Search Flowchart

```
Start
 |
Construct image pyramids
 |
For level from coarsest to finest
 |
If coarsest level
 |
Perform motion estimation (e.g., exhaustive)
 |
Else
 |
Use previous level's mv as starting point
 |
Refine motion vectors
 |
Store mv for the level
 |
End
```

---

## 5. Motion Estimation in Video Coding

Motion estimation is integral to video coding standards like MPEG, H.264/AVC, HEVC, and beyond. It enables efficient inter-frame compression by predicting frames based on motion information from reference frames.

### 5.1. Steps in Video Encoding with Motion Estimation

1. **Frame Partitioning:**
   - Divide the current frame into macroblocks (e.g., 16x16 pixels).

2. **Motion Estimation:**
   - For each macroblock, perform motion estimation to find the best matching block in the reference frame.
   - Determine the motion vector.

3. **Motion Compensation:**
   - Use the motion vector to predict the macroblock from the reference frame.
   - Compute the residual (difference) between the actual macroblock and the predicted macroblock.

4. **Transform and Quantization:**
   - Apply a transform (e.g., DCT) to the residual.
   - Quantize the transform coefficients to reduce precision and achieve compression.

5. **Entropy Coding:**
   - Encode the quantized coefficients and motion vectors using entropy coding techniques (e.g., Huffman coding, arithmetic coding).

6. **Reconstruction for Reference Frames:**
   - In the decoder, reconstruct the macroblock by inverse quantization and inverse transform.
   - Add the reconstructed residual to the predicted macroblock.
   - Store the reconstructed frame for future predictions.

7. **Output Bitstream:**
   - Combine encoded motion vectors and residual data into a compressed bitstream.

**Flowchart of Video Encoding with Motion Estimation:**

```
Input Frame
    |
[Frame Partitioning]
    |
Divide into Macroblocks
    |
For each Macroblock:
    |
[Motion Estimation]
    |
Determine Motion Vector
    |
[Motion Compensation]
    |
Predict Macroblock
    |
[Compute Residual]
    |
Residual = Actual - Predicted
    |
[Transform (e.g., DCT)]
    |
[Quantization]
    |
[Entropy Coding]
    |
Encode Motion Vector and Quantized Coefficients
    |
[Reconstruction]
    |
Inverse Quantization and Transform
    |
Add Predicted Macroblock
    |
Store Reconstructed Frame
    |
[Output Bitstream]
```

---

## 6. Conclusion

Motion estimation and compensation are fundamental techniques in video compression, enabling efficient encoding by leveraging temporal redundancies. Various algorithms offer trade-offs between computational complexity and estimation accuracy. Understanding these algorithms and their implementations is crucial for developing efficient video coding systems and improving existing standards.

---

## 7. References

1. **Richardson, Iain E. G.** *H.264 and MPEG-4 Video Compression: Video Coding for Next-generation Multimedia*. John Wiley & Sons, 2004.

2. **Sullivan, Gary J., et al.** "Overview of the High Efficiency Video Coding (HEVC) standard." *IEEE Transactions on Circuits and Systems for Video Technology* 22.12 (2012): 1649-1668.

3. **K. R. Rao, D. N. Kim, J. J. Hwang.** *Video Coding Standards: AVS China, H.264/MPEG-4 Part 10, HEVC, VP6, DIRAC and VC-1*. Springer, 2014.

4. **Tourapis, Alexis Michael.** "Enhanced predictive zonal search for single and multiple frame motion estimation." *Visual Communications and Image Processing 2002*. Vol. 4671. International Society for Optics and Photonics, 2002.

---

**Note:** The code examples provided are for educational purposes and may not be optimized for performance. In practice, optimized algorithms and hardware accelerations are used to achieve real-time video encoding.
