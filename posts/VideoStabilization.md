[![Home](https://img.shields.io/badge/Home-Click%20Here-blue?style=flat&logo=homeadvisor&logoColor=white)](../)

## [![CV](https://img.shields.io/badge/CP-Selected_Topics_in_Computational_Photography-green?style=for-the-badge&logo=github)](../main_page/CP)

Below is a comprehensive, step-by-step tutorial on image (video) stabilization. The tutorial will cover the conceptual underpinnings and the mathematical formulations often used in state-of-the-art stabilizing pipelines. We will discuss various motion models, explain how to extract and match features, how to identify and separate camera motion from subject (object) motion, and finally how to warp frames to achieve a stabilized sequence. 

Because you requested a deeper, manual discussion (without relying on OpenCV or high-level libraries), this tutorial will focus on the principles, the math, and illustrative pseudo-code. Please note that you can implement these ideas in your own codebase in any language of your choice (C++, Python, MATLAB, etc.), so long as you handle the corresponding linear algebra operations and optimization routines.

---

## Table of Contents

1. **Introduction and Overview**
2. **Motion Models and Transformation Matrices**
3. **Feature Extraction Approaches**
4. **Feature Matching Methods**
5. **Estimating Camera Motion vs. Subject Motion**
6. **Computing the Transformation (Homography)**
7. **Stabilization Pipeline and Smoothing**
8. **Warping and Rendering the Final Stabilized Frame**
9. **Flowchart of the Stabilization Pipeline**
10. **Pseudo-code**

---

## 1. Introduction and Overview

Image (or video) stabilization seeks to remove unwanted motions (often from camera shake or jitter) in consecutive frames of a video. The principal steps are:

1. **Detect and extract features** (or measure motion in some way).
2. **Match features** across consecutive frames (or measure pixel shifts using optical flow).
3. **Estimate a transformation** (e.g., translation, affine, homography) that explains the dominant camera motion.
4. **Smooth or filter** the transformation across the video frames.
5. **Warp the frames** according to the smoothed transformations to produce the stabilized output.

Stabilization is vital in many scenarios—handheld videography, drone footage, surveillance video, etc.

---

## 2. Motion Models and Transformation Matrices

### 2.1 Types of Motion Models

1. **2D Translation**  
   $$
   \begin{pmatrix}
   x' \\
   y'
   \end{pmatrix}
   =
   \begin{pmatrix}
   x + t_x \\
   y + t_y
   \end{pmatrix}
   $$
   This model only allows shifting in x and y directions.

2. **Euclidean (Rigid) Transformation** (Rotation + Translation)  
   $$
   \begin{pmatrix}
   x' \\
   y'
   \end{pmatrix}
   =
   \begin{pmatrix}
   \cos\theta & -\sin\theta \\
   \sin\theta & \cos\theta
   \end{pmatrix}
   \begin{pmatrix}
   x \\
   y
   \end{pmatrix}
   +
   \begin{pmatrix}
   t_x \\
   t_y
   \end{pmatrix}
   $$

3. **Similarity Transformation** (Scale + Rotation + Translation)  
   $$
   \begin{pmatrix}
   x' \\
   y'
   \end{pmatrix}
   =
   s
   \begin{pmatrix}
   \cos\theta & -\sin\theta \\
   \sin\theta & \cos\theta
   \end{pmatrix}
   \begin{pmatrix}
   x \\
   y
   \end{pmatrix}
   +
   \begin{pmatrix}
   t_x \\
   t_y
   \end{pmatrix}
   $$

4. **Affine Transformation** (Scale + Rotation + Shear + Translation)  
   $$
   \begin{pmatrix}
   x' \\
   y'
   \end{pmatrix}
   =
   \begin{pmatrix}
   a_{11} & a_{12} \\
   a_{21} & a_{22}
   \end{pmatrix}
   \begin{pmatrix}
   x \\
   y
   \end{pmatrix}
   +
   \begin{pmatrix}
   t_x \\
   t_y
   \end{pmatrix}
   $$

5. **Homography** (Projective Transformation)  
   $$
   \begin{pmatrix}
   x' \\
   y' \\
   1
   \end{pmatrix}
   =
   \begin{pmatrix}
   h_{11} & h_{12} & h_{13} \\
   h_{21} & h_{22} & h_{23} \\
   h_{31} & h_{32} & h_{33}
   \end{pmatrix}
   \begin{pmatrix}
   x \\
   y \\
   1
   \end{pmatrix}
   $$
   After applying the matrix, you normalize by dividing through by the third component, i.e.,  
   $$
   x' = \frac{h_{11}x + h_{12}y + h_{13}}{h_{31}x + h_{32}y + h_{33}}, 
   \quad
   y' = \frac{h_{21}x + h_{22}y + h_{23}}{h_{31}x + h_{32}y + h_{33}}.
   $$

For most handheld camera scenarios (non-parallax or small depth changes), an affine or homography-based model often suffices. A homography is the most general 2D projective transformation.  

---

## 3. Feature Extraction Approaches

Stabilization typically relies on detecting and tracking consistent features across frames. Feature extraction can be broadly split into:

1. **Sparse Feature Detectors and Descriptors**  
   - **SIFT** (Scale-Invariant Feature Transform)  
   - **SURF** (Speeded Up Robust Features)  
   - **ORB** (Oriented FAST and Rotated BRIEF)  
   - Others: Harris corners, FAST corners, etc.

2. **Dense Optical Flow**  
   - **Lucas-Kanade** method  
   - **Farneback** method  
   - **Gunnar Farneback** approach for dense flow

### 3.1 SIFT and SURF in a Nutshell

**SIFT** involves:
1. Building Gaussian pyramid and Difference-of-Gaussian (DoG) pyramids.  
2. Detecting scale-space extrema (keypoints).  
3. Determining keypoint orientation from gradient histogram.  
4. Computing local descriptor (gradient orientation distribution around the keypoint).  

**SURF** is similar in spirit but uses box filters and integral images for faster approximation of Gaussian derivatives.

Mathematically, if $I(x, y) $ is your image, SIFT looks for extrema of:
$$
D(x, y, \sigma) = (G(x, y, k\sigma) - G(x, y, \sigma)) * I(x, y),
$$
where $G(\cdot, \sigma) $ is a Gaussian function with standard deviation \(\sigma$, and \(k$ is a constant multiplicative factor between successive scales. Points which are local maxima in the DoG space are used as candidate keypoints.

### 3.2 Optical Flow in a Nutshell

Optical flow methods attempt to find the apparent motion of pixels from frame \(t$ to frame \(t+1$. The **Lucas-Kanade** approach solves for small motion under the brightness constancy assumption:
$$
I(x, y, t) \approx I(x + \delta x, y + \delta y, t+1),
$$
leading to the well-known optical flow constraint:
$$
\frac{\partial I}{\partial x} \, \delta x + \frac{\partial I}{\partial y} \, \delta y + \frac{\partial I}{\partial t} = 0.
$$
This is solved in neighborhoods to yield \((\delta x, \delta y)$ for each feature point.

---

## 4. Feature Matching Methods

After extracting features (keypoints + descriptors) from consecutive frames:

1. **Nearest Neighbor Matching**  
   - For each feature in Frame \(k$, find the closest descriptor in Frame \(k+1$.  
   - Usually done in descriptor space (e.g., Euclidean distance for SIFT or SURF descriptors).

2. **Ratio Test (Lowe’s test)**  
   - Compare the closest match distance \(d_1$ to the second-closest match distance \(d_2$.  
   - Accept the match only if \(\frac{d_1}{d_2} < \text{some threshold}$ (e.g., 0.7 or 0.8).

3. **Optical Flow**  
   - Track corners using a local window-based approach (Lucas-Kanade).  
   - Minimizes intensity differences between frames for small neighborhoods.  

---

## 5. Estimating Camera Motion vs. Subject Motion

In real footage, not all motion is from the camera; some objects or subjects move within the scene. We want to separate the **dominant motion** (camera-induced) from local motions (moving objects). Common strategies:

1. **Robust Estimation / RANSAC**  
   - Even if some features belong to moving objects, a RANSAC-based approach will try random subsets of matched points to fit a transformation model (e.g., homography).  
   - The model with the largest consensus (inliers) is deemed to represent the dominant (camera) motion. Outliers correspond to independently moving objects.

2. **Region-based**  
   - Segment the image into multiple regions. The largest region’s motion is often the camera motion.  

3. **Motion segmentation**  
   - If necessary, advanced methods track multiple motion layers. The largest layer typically corresponds to the background or camera motion.  

---

## 6. Computing the Transformation (Homography)

### 6.1 Mathematical Formulation

Assuming we have matched pairs \(\{(x_i, y_i)\}$ in frame \(k$ and \(\{(x'_i, y'_i)\}$ in frame \(k+1$. We want to solve for the homography matrix \(H$:
$$
\begin{pmatrix}x'_i \\ y'_i \\ 1\end{pmatrix}
\sim
H
\begin{pmatrix}x_i \\ y_i \\ 1\end{pmatrix}
=
\begin{pmatrix}
h_{11} & h_{12} & h_{13} \\
h_{21} & h_{22} & h_{23} \\
h_{31} & h_{32} & h_{33}
\end{pmatrix}
\begin{pmatrix}
x_i \\
y_i \\
1
\end{pmatrix}.
$$

This gives two linear equations per point:
$$
x'_i = \frac{h_{11}x_i + h_{12}y_i + h_{13}}{h_{31}x_i + h_{32}y_i + h_{33}}, \quad
y'_i = \frac{h_{21}x_i + h_{22}y_i + h_{23}}{h_{31}x_i + h_{32}y_i + h_{33}}.
$$

Typically, you rewrite in a linear least squares form (or direct linear transform – DLT):
$$
\begin{pmatrix}
x_i & y_i & 1 & 0 & 0 & 0 & -x'_i x_i & -x'_i y_i & -x'_i \\
0 & 0 & 0 & x_i & y_i & 1 & -y'_i x_i & -y'_i y_i & -y'_i
\end{pmatrix}
\begin{pmatrix}
h_{11} \\ h_{12} \\ h_{13} \\ h_{21} \\ h_{22} \\ h_{23} \\ h_{31} \\ h_{32} \\ h_{33}
\end{pmatrix}
= 0.
$$

Stacking enough points (at least 4 correspondences for a total of 8 equations, if not degenerate) yields a system \(A \vec{h} = 0$. You can solve it by SVD (Singular Value Decomposition), picking the \(\vec{h}$ that corresponds to the singular vector of the smallest singular value. Then you reshape \(\vec{h}$ into the 3×3 matrix \(H$.  

**Use RANSAC** to robustly estimate \(H$. In each iteration:
1. Pick a minimal subset of matched points (4 for a homography).  
2. Compute \(H$.  
3. Count the number of inliers (matches that agree with this \(H$ within a threshold).  
4. Keep the best \(H$ that yields the most inliers.  

---

## 7. Stabilization Pipeline and Smoothing

Once we have a sequence of transformations \(\{H_1, H_2, \ldots, H_N\}$ relating consecutive frames, we want to produce a stable sequence. Simply applying each raw transformation might still produce jitter. Therefore we **smooth** or **filter** these transformations over time.

1. **Cumulative Transforms**  
   - We can transform each frame relative to the first frame by composing transformations:  
     $$
     C_k = H_k \, H_{k-1} \, \ldots \, H_2 \, H_1
     $$  
     So \(C_k$ is the transform from frame 1 to frame k.  

2. **Filter the Cumulative Transform**  
   - Let \(\hat{C}_k$ be the smoothed version of \(C_k$. You can use:  
     - **Moving average** filter,  
     - **Kalman filter**,  
     - **Gaussian smoothing** in the parameter space (translation, rotation, scale).  

3. **Stabilized Transformation for Frame \(k$**  
   - The final transform that you apply to frame \(k$ might be:  
     $$
     S_k = \hat{C}_k \, C_k^{-1}.
     $$  
     This ensures the frame is transformed from its original motion to the smoothed motion.  

---

## 8. Warping and Rendering the Final Stabilized Frame

Given your final transformation (let’s call it \(T_k$ for each frame \(k$), to warp the frame you perform:

$$
\begin{pmatrix}
x' \\
y' \\
1
\end{pmatrix}
=
T_k
\begin{pmatrix}
x \\
y \\
1
\end{pmatrix}.
$$

For each pixel \((x', y')$ in the output (stabilized) frame, you sample the corresponding color from \((x, y)$ in the original frame using bilinear interpolation or another interpolation method. This process is also called **inverse mapping**:  
1. For each pixel in the output image, apply \(T_k^{-1}$ to find the corresponding source position in the original image.  
2. Interpolate that color.  

---

## 9. Flowchart of the Stabilization Pipeline

Below is a high-level flowchart capturing the main steps:

```
+------------------------------------------+
|            VIDEO INPUT FRAMES           |
+------------------------------------------+
                  |  
                  v  
+------------------------------------------+
|          FEATURE EXTRACTION             |
| (e.g. SIFT, SURF, or Optical Flow)      |
+------------------------------------------+
                  |
                  v
+------------------------------------------+
|         FEATURE MATCHING /              |
|    MOTION ESTIMATION (RANSAC)           |
|      => Homography or Affine            |
+------------------------------------------+
                  |  
                  v  
+------------------------------------------+
|    ACCUMULATE TRANSFORMS ACROSS         |
|           FRAMES (C_k)                  |
+------------------------------------------+
                  |
                  v
+------------------------------------------+
|  SMOOTH / FILTER THE TRANSFORM SEQUENCE |
|   (e.g., moving average, Kalman filter) |
+------------------------------------------+
                  |
                  v
+------------------------------------------+
|    COMPUTE STABILIZED TRANSFORM S_k     |
+------------------------------------------+
                  |
                  v
+------------------------------------------+
|         WARP FRAMES USING S_k           |
+------------------------------------------+
                  |
                  v
+------------------------------------------+
|         OUTPUT STABILIZED VIDEO         |
+------------------------------------------+
```

---

## 10. Pseudo-code

### 10.1 Overall Stabilization Routine

Below is a simplified pseudo-code, avoiding use of OpenCV. We assume you have your own linear algebra operations (like SVD, matrix multiply, etc.) available.

```plaintext
function stabilize_video(frames):
    # frames is a list/array of images (frame_1, frame_2, ..., frame_N)
    
    # 1. Feature extraction for the first frame
    keypoints_prev, descriptors_prev = extract_features(frames[1])

    # Initialize an array to store pairwise transforms
    transforms = array of size (N-1)  # each transform is a 3x3 matrix if using homography

    # 2. Loop through frames 2..N
    for k in 2..N:
        # Extract features from current frame
        keypoints_curr, descriptors_curr = extract_features(frames[k])
        
        # Match features
        matches = match_features(descriptors_prev, descriptors_curr)
        
        # Filter matches via RANSAC to find transformation
        transforms[k-1] = estimate_transform_RANSAC(keypoints_prev, keypoints_curr, matches)
        
        # Prepare for next iteration
        keypoints_prev = keypoints_curr
        descriptors_prev = descriptors_curr
    
    # 3. Cumulative transforms
    cumulative_transforms = array of size (N)  # each is a 3x3 matrix
    cumulative_transforms[1] = Identity(3x3) 
    for k in 2..N:
        cumulative_transforms[k] = cumulative_transforms[k-1] * transforms[k-1]  

    # 4. Smooth cumulative transforms
    smoothed_cumulative = smooth_transforms(cumulative_transforms)

    # 5. Compute stabilized transforms
    stabilized_transforms = array of size (N)
    for k in 1..N:
        # We want S_k = smoothed_cumulative[k] * inverse(cumulative_transforms[k])
        stabilized_transforms[k] = smoothed_cumulative[k] * inverse(cumulative_transforms[k])

    # 6. Warp frames according to stabilized transforms
    stabilized_frames = array of size(N)
    for k in 1..N:
        stabilized_frames[k] = warp_image(frames[k], stabilized_transforms[k])

    return stabilized_frames
```

### 10.2 Feature Extraction (e.g., SIFT-like steps)

```plaintext
function extract_features(image):
    # Pseudo-code for a SIFT-like pipeline
    # 1) Build Gaussian pyramid
    gaussians = build_gaussian_pyramid(image)
    
    # 2) Build Difference-of-Gaussian pyramid
    dog = build_DoG_pyramid(gaussians)
    
    # 3) Detect local extrema in scale-space
    keypoints = detect_DoG_extrema(dog)
    
    # 4) Assign orientation and build descriptors
    descriptors = []
    for kp in keypoints:
        orientation = compute_orientation(kp, image)
        descriptor = compute_descriptor(kp, orientation, image)
        descriptors.push(descriptor)
    
    return keypoints, descriptors
```

### 10.3 RANSAC-based Homography Estimation

```plaintext
function estimate_transform_RANSAC(keypoints1, keypoints2, matches):
    best_inlier_count = 0
    best_H = Identity(3x3)
    
    for iteration in 1..RANSAC_MAX_ITERATIONS:
        # 1) Randomly select 4 matches
        sample = random_subset(matches, 4)
        
        # 2) Compute candidate H from these 4 points
        H_candidate = compute_homography(sample)
        
        # 3) Count inliers among all matches
        inliers = 0
        for m in matches:
            p1 = keypoints1[m.query_index]
            p2 = keypoints2[m.train_index]
            
            # Project p1 by H_candidate
            p1_h = [p1.x, p1.y, 1]^T
            p2_est = H_candidate * p1_h
            p2_est = p2_est / p2_est[2]  # normalize
            
            # Check distance to actual p2
            if distance(p2_est, p2) < ransac_threshold:
                inliers += 1
        
        # 4) Update best if current inliers is better
        if inliers > best_inlier_count:
            best_inlier_count = inliers
            best_H = H_candidate
    
    return best_H
```

### 10.4 Smoothing Transforms

```plaintext
function smooth_transforms(cumulative_transforms):
    # naive example: moving average in parameter space

    # Convert each 3x3 matrix into a parameter vector (e.g. translation, rotation, scale)
    parameter_vectors = []
    for C in cumulative_transforms:
        parameters = matrix_to_parameters(C)  
        parameter_vectors.push(parameters)
    
    # Apply a 1D moving average or some filter on each parameter dimension
    smoothed_parameter_vectors = moving_average_filter(parameter_vectors)
    
    # Convert parameter vectors back to matrices
    smoothed_matrices = []
    for spv in smoothed_parameter_vectors:
        smoothed_matrices.push(parameters_to_matrix(spv))
    
    return smoothed_matrices
```

### 10.5 Warping the Image

```plaintext
function warp_image(image, transform):
    # transform is a 3x3 matrix
    # create an output image of the same size
    out_image = zeros_like(image)
    
    # for each pixel (x_out, y_out) in out_image
    for y_out in range(0, out_image.height):
        for x_out in range(0, out_image.width):
            # Construct homogeneous coords
            p_out = [x_out, y_out, 1]^T
            
            # Map back to original image
            p_in = inverse(transform) * p_out
            px_in = p_in[0] / p_in[2]
            py_in = p_in[1] / p_in[2]
            
            # If (px_in, py_in) is within bounds of 'image', sample it
            if (0 <= px_in < image.width) and (0 <= py_in < image.height):
                out_image[y_out, x_out] = bilinear_sample(image, px_in, py_in)
    
    return out_image
```

---

## Conclusion

By understanding the concepts above—feature extraction, feature matching, homography (or affine) estimation, RANSAC filtering, smoothing transforms, and warping—you can implement a robust video stabilization pipeline from scratch (i.e., without high-level libraries like OpenCV). 

Key points to remember:
1. **Dominant motion** is extracted by robust estimations (e.g., RANSAC).  
2. **Subject motion** is largely treated as outliers in that estimation step.  
3. **Smoothing** the transformations in time is the crucial step that actually removes jitter.  
4. **Warping** final frames based on the smoothed transformations produces stabilized results.

---

### Further Extensions

- **Kalman Filters** or **Particle Filters** for more sophisticated smoothing.  
- **3D camera motion** if you have additional sensor data (IMU, etc.).  
- **Adaptive warping** if large parallax is involved (layer-based or advanced 3D mapping).  

The above tutorial should provide a foundation to build and deepen your understanding of video stabilization techniques. Once you master the math and the logic here, you can replace any step with more optimized or advanced methods (e.g., advanced descriptors, GPU-accelerated optical flow, etc.) or incorporate high-level libraries for convenience.
