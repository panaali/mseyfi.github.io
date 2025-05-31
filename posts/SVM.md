## Support Vector Machines (SVMs)

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
![svm](../images/SVM.png)

*Fig.~1 Support Vector Machines, Hard Marging SVM (left) vs Soft Margin SVM (right) where some outliers are allowed with a penalty*

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

#### Why is minimizing the **primal problem with constraints** equivalent to solving the **saddle point problem**:

$$
\min_{\mathbf{w}, b} \max_{\boldsymbol{\alpha} \geq 0} \mathcal{L}(\mathbf{w}, b, \boldsymbol{\alpha})
$$

Why does this min-max give the **same result** as solving the original constrained optimization?

---

**This is guaranteed by the Karush-Kuhn-Tucker (KKT) conditions and strong duality** — but let’s derive it intuitively and mathematically.

##  Original Constrained Problem (Primal Form)

We want to solve:

$$
\begin{aligned}
\min_{\mathbf{w}, b} \quad & \frac{1}{2} \|\mathbf{w}\|^2 \\
\text{s.t.} \quad & y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1 \quad \forall i
\end{aligned}
$$

These are **inequality constraints**, so we introduce **Lagrange multipliers** $\alpha_i \geq 0$.

---

##  Define the Lagrangian Function

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

### **3. Soft Margin SVM (Handling Non-Separable Data)**

In real-world problems (like the med school example), perfect separation is rare. For instance, a student with high GPA and low MCAT might still be accepted, creating an **outlier**.

To handle such cases, we allow some violations of the margin using **slack variables** $\xi_i \geq 0$.

#### **Soft Margin SVM Primal Formulation:**

We modify the constraints:

$$
 y_i(w^T x_i + b) \geq 1 - \xi_i
$$

Objective becomes:

$$
\min_{w, b, \xi} \quad \frac{1}{2} \|w\|^2 + C \sum_i \xi_i
$$

Where:

* $\frac{1}{2} \|w\|^2$: encourages a large margin
* $\sum \xi_i$: penalizes margin violations
* $C$: trade-off parameter between margin size and violations

---

### **Hinge Loss Reformulation**

The problem can also be expressed using **hinge loss**:

$$
L(w, b) = \frac{1}{2} \|w\|^2 + C \sum_i \max(0, 1 - y_i(w^T x_i + b))
$$

* If $y_i(w^T x_i + b) \geq 1$, loss is zero (correctly classified with margin)
* If $y_i(w^T x_i + b) < 1$, we incur linear loss

This form is subdifferentiable and suitable for **gradient-based optimization**.

---

### **Training Soft Margin SVM via Gradient Descent**

Let $z_i = y_i(w^T x_i + b)$.
Gradient of the hinge loss is:
$$
\nabla_w L = w - C \sum_{i: z_i < 1} y_i x_i
$$
$$
\nabla_b L = -C \sum_{i: z_i < 1} y_i
$$
#### **Pseudocode: Soft Margin SVM (Gradient Descent)**

$$
\begin{aligned}
&\textbf{Input:} \ \text{Training data } \{(x_i, y_i)\}, \text{learning rate } \eta, \text{regularization } C, \text{epochs } T \\
&\textbf{Output:} \ w, b \\
&1. \ \text{Initialize } w = 0, \ b = 0 \\
&2. \ \text{For } t = 1 \text{ to } T:\\
&\quad \text{For each } (x_i, y_i):\\
&\quad\quad \text{Compute margin: } m = y_i(w^T x_i + b) \\
&\quad\quad \text{If } m \geq 1:\\
&\quad\quad\quad w \leftarrow w - \eta w \\
&\quad\quad\quad b \leftarrow b \\
&\quad\quad \text{Else:}\\
&\quad\quad\quad w \leftarrow w - \eta (w - C y_i x_i) \\
&\quad\quad\quad b \leftarrow b + \eta C y_i \\
&3. \ \text{Return } w, b
\end{aligned}
$$

---

### **How the Dual Helps in High Dimensions**

The dual formulation of SVM is particularly powerful in high-dimensional spaces.

#### **Why?**

* In the dual, the data appears **only in terms of dot products**: $x_i^T x_j$.
* These dot products can be computed **without explicitly constructing high-dimensional features** via the **kernel trick**.
* This allows us to work in **infinite-dimensional** feature spaces (like with the RBF kernel) while computing everything efficiently in the original space.

#### **Intuition:**

Even if the data is not linearly separable in its original space, it may be separable in a higher-dimensional space. Instead of transforming all data points manually, we **implicitly map them using a kernel**, which computes dot products in that higher-dimensional space **without ever forming the mapped vectors**.

This makes the dual SVM especially suited for high-dimensional or nonlinearly separable problems.

#### **Dual Optimization vs Support Vectors**

* The dual optimization problem introduces one Lagrange multiplier $\alpha_i$ **for every training point**.
* So during training, the dual is optimized over **all data points**.
* However, **after training**, most $\alpha_i$ are zero.
* Only the **support vectors** — points that lie exactly on the margin or violate it — have non-zero $\alpha_i$.

**Therefore:**

* **Dual training** involves all data.
* **Inference (prediction)** only uses support vectors:

$$
f(x) = \sum_{i \in \text{SV}} \alpha_i y_i K(x_i, x) + b
$$

This sparsity is a key reason why SVMs are efficient at prediction time.

---

