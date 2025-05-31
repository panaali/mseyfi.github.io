**Support Vector Machines (SVMs) — Full Tutorial**

---

### **1. SVM Analogy: Medical School Admission**

Imagine you're trying to decide whether students get into medical school based on two features:

* **GPA (Grade Point Average)**
* **MCAT score (Medical College Admission Test)**

Plotting these as points in 2D space (GPA on x-axis, MCAT on y-axis), suppose accepted students are mostly in the upper-right (high GPA, high MCAT), while rejected students are in the lower-left.

We aim to find a **decision boundary** (a line) that separates the two classes.

**Definitions:**

* **Hyperplane**: In 2D, this is a line. In general, it's an (n-1)-dimensional flat subspace separating two classes.
* **Margin**: The distance between the hyperplane and the closest points from each class.
* **Support Vectors**: The data points that lie exactly on the margin boundaries. These are the critical points that "support" the hyperplane.
* **Optimal Hyperplane**: The one that **maximizes the margin** between classes.

---
![svm](../images/svm.png)

*Fig.~1 Support Vector Machines, Hard Marging SVM (left) vs Soft Margin SVM (right)*

### **2. Hard Margin SVM (Linearly Separable Case)**

We assume the data is **perfectly linearly separable**.

#### **Goal:**

Find a hyperplane $w^T x + b = 0$ that:

* Separates the data.
* Maximizes the margin (i.e., minimizes $\|w\|^2$).

#### **Formulation:**

Let $(x_i, y_i)$ be the training data with $y_i \in \{-1, +1\}$

**Constraints:**

$$
 y_i(w^T x_i + b) \geq 1 \quad \forall i
$$

**Optimization Problem:**

$$
\min_{w, b} \frac{1}{2} \|w\|^2 \quad \text{subject to} \quad y_i(w^T x_i + b) \geq 1
$$

This is a **convex quadratic optimization problem with linear constraints**. We solve it using **quadratic programming**, not gradient descent.

#### ❗ Why Not Gradient Descent?

Hard-margin SVM uses exact inequality constraints: $y_i(w^T x_i + b) \geq 1$. These are not amenable to standard gradient descent:

* We'd have to use **penalty methods** to turn constraints into penalties, effectively creating a soft-margin SVM.
* The true hard-margin problem is typically solved via **quadratic programming solvers**, which handle constraints directly.

Thus, **gradient descent is not used** for training hard-margin SVMs in practice.

#### **Lagrangian Dual Derivation**

We derive the dual form by introducing **Lagrange multipliers** $\alpha_i \geq 0$ for the constraints.

**Primal Lagrangian:**

$$
\mathcal{L}(w, b, \alpha) = \frac{1}{2} \|w\|^2 - \sum_i \alpha_i [y_i(w^T x_i + b) - 1]
$$

To find the saddle point, we take partial derivatives and set them to zero:

**1. Derivative with respect to $w$:**

$$
\frac{\partial \mathcal{L}}{\partial w} = w - \sum_i \alpha_i y_i x_i = 0 \Rightarrow w = \sum_i \alpha_i y_i x_i
$$

**2. Derivative with respect to $b$:**

$$
\frac{\partial \mathcal{L}}{\partial b} = -\sum_i \alpha_i y_i = 0 \Rightarrow \sum_i \alpha_i y_i = 0
$$

**Plug back into the Lagrangian:**

$$
\mathcal{L}(w, b, \alpha) = \frac{1}{2} \left\|\sum_i \alpha_i y_i x_i \right\|^2 - \sum_i \alpha_i \left[y_i\left(\sum_j \alpha_j y_j x_j^T x_i + b\right) - 1\right]
$$

Simplify:

* First term becomes: $\frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j x_i^T x_j$
* Second term: $-\sum_i \alpha_i y_i \sum_j \alpha_j y_j x_j^T x_i = -\sum_{i,j} \alpha_i \alpha_j y_i y_j x_i^T x_j$
* Third term: $-\sum_i \alpha_i y_i b = 0$ by constraint
* Constant: $+\sum_i \alpha_i$

So the dual becomes:

$$
\max_{\alpha} \sum_i \alpha_i - \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j x_i^T x_j
$$

Subject to:

$$
\alpha_i \geq 0, \quad \sum_i \alpha_i y_i = 0
$$

#### **Why Do We Maximize the Dual? (Intuition)**
Perfect — let’s address this precisely.

---

### ❓ Your question:

> Why is minimizing the **primal problem with constraints** equivalent to solving the **saddle point problem**:

$$
\min_{\mathbf{w}, b} \max_{\boldsymbol{\alpha} \geq 0} \mathcal{L}(\mathbf{w}, b, \boldsymbol{\alpha})
$$

Why does this min-max give the **same result** as solving the original constrained optimization?

---

## ✅ Answer: **This is guaranteed by the Karush-Kuhn-Tucker (KKT) conditions and strong duality** — but let’s derive it intuitively and mathematically.

---

## 🧱 Step 1: Original Constrained Problem (Primal Form)

We want to solve:

$$
\begin{aligned}
\min_{\mathbf{w}, b} \quad & \frac{1}{2} \|\mathbf{w}\|^2 \\
\text{s.t.} \quad & y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1 \quad \forall i
\end{aligned}
$$

These are **inequality constraints**, so we introduce **Lagrange multipliers** $\alpha_i \geq 0$.

---

## 🧠 Step 2: Define the Lagrangian Function

We build:

$$
\mathcal{L}(\mathbf{w}, b, \boldsymbol{\alpha}) = \frac{1}{2} \|\mathbf{w}\|^2 - \sum_{i=1}^N \alpha_i \left[ y_i (\mathbf{w}^\top \mathbf{x}_i + b) - 1 \right]
$$

This combines:

* The **objective** $\frac{1}{2} \|\mathbf{w}\|^2$
* The **penalties** for constraint violation (through $\alpha_i$)

---

We now look for the **saddle point** of $\mathcal{L}$:

$$
\min_{\mathbf{w}, b} \max_{\boldsymbol{\alpha} \geq 0} \mathcal{L}(\mathbf{w}, b, \boldsymbol{\alpha})
$$

This saddle point represents:

* Minimizing over $(\mathbf{w}, b)$: trying to **reduce the objective**
* Maximizing over $\boldsymbol{\alpha} \geq 0$: trying to **enforce the constraints**

Here’s the deep **intuition**:

* If a constraint is **violated**, the corresponding $\alpha_i$ will grow and push the optimizer to satisfy the constraint.
* If a constraint is **satisfied**, the corresponding $\alpha_i$ will go to 0 (by complementary slackness).

This min-max structure **automatically balances** satisfying constraints and optimizing the objective.

This dual view also gives us the flexibility to introduce kernels, because it only depends on **dot products** $x_i^T x_j$.

#### **Pseudocode: Hard Margin SVM Training **

$$
\begin{aligned}
&\textbf{Input:} \ \text{Training data } \{(x_i, y_i)\}, \ y_i \in \{-1, +1\} \\
&\textbf{Output:} \ \text{Weight vector } w \text{ and bias } b \\
&1. \ \text{Formulate primal: minimize } \frac{1}{2} \|w\|^2 \\
&\quad \text{subject to } y_i(w^T x_i + b) \geq 1 \ \forall i \\
&2. \ \text{Form Lagrangian and derive dual:} \\
&\quad L(\alpha) = \sum_i \alpha_i - \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j x_i^T x_j \\
&\quad \text{subject to } \sum_i \alpha_i y_i = 0, \ \alpha_i \geq 0 \\
&3. \ \text{Solve using a QP solver to get } \alpha_i \\
&4. \ \text{Compute } w = \sum_i \alpha_i y_i x_i \\
&\quad \text{Select any support vector } x_k, \text{ then compute } b = y_k - w^T x_k \\
&5. \ \text{Return } w, b
\end{aligned}
$$

---
