[![Home](https://img.shields.io/badge/Home-Click%20Here-blue?style=flat&logo=homeadvisor&logoColor=white)](../)
## [![ML](https://img.shields.io/badge/ML-Selected_Topics_in_Machine_Learning-green?style=for-the-badge&logo=github)](../ML)

# RANSAC
Below is a comprehensive tutorial on the RANSAC (Random Sample Consensus) algorithm. We will cover the intuition behind it, the mathematical underpinnings, reasons why we need it, and provide a complete working example in Python at the end. By the end, you should have a solid understanding of RANSAC and how to apply it to a simple parameter estimation problem, such as fitting a linear model to data with outliers.

---

## Introduction

When working with real-world data, it is quite common to encounter *outliers* — data points that do not follow the general trend or underlying model you assume. Standard parameter estimation methods (e.g., least-squares fitting) can be severely skewed by these outliers. To counter this problem, we need a method that is *robust* against such aberrations. This is where RANSAC comes into play.

**RANSAC (Random Sample Consensus)** is an iterative method for robust parameter estimation. It attempts to find a model that best fits the *inlier* data points while minimizing the effect of *outlier* points.

---

## Why Do We Need RANSAC?

1. **Outlier Resistance:** Traditional curve or line fitting methods (like ordinary least squares) minimize the squared error over all points. A few extreme outliers can significantly distort the results, causing the model to not represent the main bulk of the data well. RANSAC, however, explicitly tries to separate inliers from outliers and is not fooled by a large fraction of outliers.

2. **Model Estimation in Noisy Environments:** For tasks such as:
   - Fitting a line or polynomial to data with spurious measurements,
   - Estimating a homography or fundamental matrix in computer vision from sets of matched features (many of which may be incorrect),
   
   RANSAC provides a strong, empirically proven tool to achieve robust estimates.

3. **Simplicity and Generality:** The idea behind RANSAC is relatively simple and can be applied to various parametric models. It does not rely on distributional assumptions about the errors and can be adapted to different types of problems.

---

## Intuition Behind RANSAC

The basic intuition is as follows:

1. **Minimal Subset for Model Fitting:** Suppose you want to fit a line to 2D data points $$(x_i, y_i)$$. A line model $$y = mx + c$$ can be uniquely determined by just two distinct points. If you pick two points at random, you can compute a line that passes through them.

2. **Check How Many Points Agree:** Once you have this candidate model (e.g., the line estimated from those two randomly chosen points), you count how many data points lie close enough to this line. These “close enough” points are considered *inliers*. Points that lie too far are considered *outliers*.

3. **Repeat and Find the Best:** By repeatedly sampling random subsets of points and computing a model, you generate many candidate models. Each candidate model will have a certain number of inliers. The model with the maximum number of inliers is deemed the “best” model. After identifying the best model, you can even refine it by re-fitting the model to all the inliers to get a more accurate final estimate.

---

## Mathematical Formulation

Let’s consider the problem of fitting a linear model $$y = m x + c$$ to a set of $$N$$ points $$\{(x_i, y_i)\}_{i=1}^{N}$$.

**Without RANSAC (Ordinary Least Squares):**

We typically find $$m$$ and $$c$$ by minimizing the sum of squared residuals:

$$
\min_{m,c} \sum_{i=1}^{N} (y_i - (m x_i + c))^2.
$$

If there are a large number of outliers, this optimization may yield poor estimates for $$m$$ and $$c$$.

**With RANSAC:**

1. **Model Parameterization:**
   The model parameters $$\theta$$ for a line can be represented as $$\theta = (m, c)$$.

2. **Minimal Subset Sampling:**
   For a line, the minimal subset $$S$$ could be just 2 distinct points. From points $$(x_i, y_i)$$ and $$(x_j, y_j)$$, we can solve:
   $$
   m = \frac{y_j - y_i}{x_j - x_i}, \quad c = y_i - m x_i.
   $$

3. **Distance Threshold (Consensus Criterion):**
   We define a threshold $$\epsilon$$ that determines what we call an inlier. For each data point $$(x_k, y_k)$$, compute the residual:
   $$
   r_k = |y_k - (m x_k + c)|.
   $$
   If $$r_k \le \epsilon$$, then the point is considered an inlier for that particular model.

4. **Inlier Count:**
   Count the number of inliers $$I(\theta)$$ for the model $$\theta$$. The quality of the model is measured by how many inliers it has.

5. **Repetition:**
   Repeat the above sampling many times. Each time:
   - Randomly select a minimal subset of data (in the line example, two points).
   - Estimate the model parameters from that subset.
   - Compute the number of inliers.
   Keep track of the model that yields the maximum number of inliers.

6. **Refinement:**
   Once you identify the best model $$\theta^*$$, you can optionally re-fit the model to all the inliers of $$\theta^*$$ using a more robust estimation (like a standard least-squares fit only on the inliers).

**Key RANSAC parameters:**
- **Number of iterations $$T$$:** How many random subsets to sample.
- **Inlier threshold $$\epsilon$$:** Determines how close a point must be to the model to count as an inlier.
- **Minimal number of points $$n$$:** The smallest number of points required to estimate the model parameters (2 for a line, 3 for a plane, etc.).

A common guideline for selecting $$T$$ is based on the probability of success. If $$w$$ is the probability that any chosen data point is an inlier, then the probability that a chosen subset of $$n$$ points consists entirely of inliers is $$w^n$$. We want to choose $$T$$ such that the probability of never picking an all-inlier subset is small (e.g., $$0.01$$). Thus:
$$
1 - (w^n)^T = \delta \implies T = \frac{\log \delta}{\log(1 - w^n)},
$$
where $$\delta$$ is the desired probability of failure. This gives a theoretical grounding for choosing the number of iterations.

---

## Pseudocode

Here is a high-level pseudocode for RANSAC applied to line fitting:

```
Input: data points { (x_i, y_i) }, number of iterations T, threshold ε

best_model = None
best_inlier_count = 0

for t in 1 to T:
    # 1. Randomly select a minimal subset of data points
    choose two distinct points at random from the dataset
    # 2. Compute model parameters (m, c) from these two points
    fit_line(points[subset]) -> (m, c)

    # 3. Determine inliers
    inliers = []
    for each point (x_i, y_i):
        residual = |y_i - (m*x_i + c)|
        if residual <= ε:
            inliers.append((x_i, y_i))

    # 4. Check if we have a better model
    if count(inliers) > best_inlier_count:
        best_inlier_count = count(inliers)
        best_model = (m, c)

# Optional: Refit best_model using all inliers for a refined estimate

return best_model
```

---

## Example Code in Python

In this example, we will:

1. Generate synthetic data for a line $$y = 2x + 1$$.
2. Add noise and outliers.
3. Run RANSAC to estimate the line parameters.
4. Compare the RANSAC estimate to the ordinary least squares solution.

**Requirements:**  
- Python 3.x
- NumPy for numerical computations
- Matplotlib for visualization (optional)

```python
import numpy as np
import matplotlib.pyplot as plt

def generate_data(num_points=100, m=2.0, c=1.0, noise_std=0.1, outlier_ratio=0.3, outlier_range=10):
    np.random.seed(42)
    X = np.linspace(-5, 5, num_points)
    # True line: y = m*x + c
    Y = m * X + c
    # Add Gaussian noise
    Y += np.random.normal(0, noise_std, size=num_points)
    # Add outliers
    num_outliers = int(num_points * outlier_ratio)
    outlier_indices = np.random.choice(num_points, size=num_outliers, replace=False)
    Y[outlier_indices] += np.random.uniform(-outlier_range, outlier_range, size=num_outliers)
    return X, Y

def fit_line_from_points(x1, y1, x2, y2):
    # Compute slope m and intercept c for line passing through points (x1,y1) and (x2,y2)
    m = (y2 - y1) / (x2 - x1)
    c = y1 - m*x1
    return m, c

def ransac_line_fit(X, Y, num_iterations=1000, threshold=0.5):
    best_model = None
    best_inlier_count = 0

    data = np.column_stack((X, Y))
    n = data.shape[0]

    for _ in range(num_iterations):
        # Randomly sample two distinct points
        idx = np.random.choice(n, 2, replace=False)
        x1, y1 = data[idx[0]]
        x2, y2 = data[idx[1]]
        
        if np.isclose(x1, x2):
            # If the chosen points have almost the same x, skip to avoid division by zero
            continue

        # Estimate model from these two points
        m, c = fit_line_from_points(x1, y1, x2, y2)

        # Count inliers
        residuals = np.abs(Y - (m*X + c))
        inliers = residuals <= threshold
        inlier_count = np.sum(inliers)

        # Update best model if needed
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_model = (m, c)

    # Optional refinement: Refit the model using all inliers of the best model
    if best_model is not None:
        m_best, c_best = best_model
        residuals = np.abs(Y - (m_best*X + c_best))
        inliers = residuals <= threshold
        if np.sum(inliers) > 2:
            X_in = X[inliers]
            Y_in = Y[inliers]
            # Perform a simple least squares fit on inliers
            A = np.vstack([X_in, np.ones_like(X_in)]).T
            m_refined, c_refined = np.linalg.lstsq(A, Y_in, rcond=None)[0]
            best_model = (m_refined, c_refined)

    return best_model

# Generate synthetic data
X, Y = generate_data()

# Fit with ordinary least squares (for comparison)
A = np.vstack([X, np.ones_like(X)]).T
m_ols, c_ols = np.linalg.lstsq(A, Y, rcond=None)[0]

# Fit using RANSAC
m_ransac, c_ransac = ransac_line_fit(X, Y, num_iterations=2000, threshold=0.5)

print("Ordinary Least Squares estimate:")
print(f"m = {m_ols:.2f}, c = {c_ols:.2f}")

print("RANSAC estimate:")
print(f"m = {m_ransac:.2f}, c = {c_ransac:.2f}")

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(X, Y, label='Data Points', alpha=0.7)
plt.plot(X, m_ols*X + c_ols, 'r-', label='OLS Fit', linewidth=2)
plt.plot(X, m_ransac*X + c_ransac, 'g-', label='RANSAC Fit', linewidth=2)
plt.legend()
plt.title('RANSAC vs OLS')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()
```

---

## Interpretations and Observations

- **Without RANSAC (OLS):** The resulting line is often influenced by outliers. If a significant fraction of points are far off the true line, the OLS solution can deviate substantially from the underlying model.

- **With RANSAC:** By sampling pairs of points at random and checking the consensus (inliers), the RANSAC approach is likely to find a pair that truly represents the underlying line. Consequently, the final RANSAC solution will often be much closer to the true model parameters, assuming a sufficient number of iterations and appropriate thresholding.

- **Choosing Parameters:**  
  The performance of RANSAC depends on the chosen parameters:
  - **Number of iterations (T):** More iterations increase the probability of finding a good model, but also increase computation time.
  - **Threshold (ε):** Determines what counts as an inlier. Too large, and everything might become an inlier; too small, and even good points might be deemed outliers.
  - **Minimum subset size (n):** Must be chosen according to the model. For a line, n=2. For a homography in computer vision, n=4 corresponding point pairs, etc.

---

## Conclusion

RANSAC is a powerful, general-purpose algorithm for robust parameter estimation in the presence of a high fraction of outliers. By relying on random sampling of minimal subsets and consensus checking, it can find the model that best fits the majority of the data points, leaving outliers out of consideration. Understanding the RANSAC framework is invaluable for tasks in computer vision, robotics, and many other fields dealing with noisy, real-world datasets.

You now have the intuition, the mathematics, and a working code example to start using RANSAC for your own problems.
