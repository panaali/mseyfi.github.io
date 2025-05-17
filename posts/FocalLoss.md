[![Home](https://img.shields.io/badge/Home-Click%20Here-blue?style=flat&logo=homeadvisor&logoColor=white)](../)
## [![ML](https://img.shields.io/badge/ML-Selected_Topics_in_Machine_Learning-green?style=for-the-badge&logo=github)](../main_pages/ML)

**Focal Loss** is a modified version of the standard cross-entropy loss, designed to address the **class imbalance problem**, especially in tasks like **object detection** (e.g. RetinaNet) or **extremely imbalanced binary classification**.

---

## 🔷 1. **Motivation**

In many tasks:

* Easy negatives dominate the training.
* Standard binary cross-entropy does **not differentiate** between hard and easy examples.
* So we want a loss that **downweights easy examples** and **focuses on hard ones**.

---

## 🔶 2. **Binary Cross-Entropy (Review)**

For binary classification, with predicted probability $\hat{p} \in (0,1)$, true label $y \in \{0,1\}$:

$$
\mathcal{L}_{\text{CE}} = -[y \log(\hat{p}) + (1 - y)\log(1 - \hat{p})]
$$

---

## 🔷 3. **Focal Loss (Binary Case)**

Focal Loss adds a **modulating factor** to the CE loss:

$$
\mathcal{L}_{\text{focal}} = - \alpha (1 - \hat{p})^\gamma \log(\hat{p}) \quad \text{if } y = 1
$$

$$
\mathcal{L}_{\text{focal}} = - (1 - \alpha) \hat{p}^\gamma \log(1 - \hat{p}) \quad \text{if } y = 0
$$

Or unified as:

$$
\mathcal{L}_{\text{focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t)
$$

Where:

* $$p_t = \begin{cases}
  \hat{p} & \text{if } y = 1 \\
  1 - \hat{p} & \text{if } y = 0
  \end{cases}$$
* $$\alpha_t = \begin{cases}
  \alpha & \text{if } y = 1 \\
  1 - \alpha & \text{if } y = 0
  \end{cases}$$

---

### 🔹 Parameters

| Parameter           | Meaning                                                    |
| ------------------- | ---------------------------------------------------------- |
| $\gamma \in [0, 5]$ | Focusing parameter. Higher γ focuses more on hard examples |
| $\alpha \in (0, 1)$ | Class weighting. Helps balance positive/negative classes   |

---

### 🔹 Behavior

* If $p_t$ is **close to 1** (correct confident prediction):
  $(1 - p_t)^\gamma \approx 0$ → loss ≈ 0
* If $p_t$ is **close to 0** (incorrect prediction):
  $(1 - p_t)^\gamma \approx 1$ → full loss applied

So **easy examples are downweighted**, **hard examples are focused on**.

---

## 🔶 4. **Focal Loss in PyTorch**

```python
import torch
import torch.nn.functional as F

def focal_loss(logits, targets, alpha=0.25, gamma=2.0, reduction='mean'):
    """
    logits: Tensor of raw predictions (before sigmoid), shape (N,)
    targets: Tensor of binary labels (0 or 1), shape (N,)
    """
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')  # shape (N,)
    probs = torch.sigmoid(logits)
    p_t = probs * targets + (1 - probs) * (1 - targets)  # same as p_t

    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)

    focal_term = (1 - p_t) ** gamma
    loss = alpha_t * focal_term * bce_loss

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss  # no reduction
```

---

## 🔷 5. **Comparison to Cross-Entropy**

| Property                  | Cross-Entropy | Focal Loss       |
| ------------------------- | ------------- | ---------------- |
| Focuses on hard examples? | ❌             | ✅                |
| Handles class imbalance?  | ❌             | ✅ via $\alpha$   |
| Used in RetinaNet?        | ❌             | ✅                |
| Extra parameters?         | ❌             | $\gamma, \alpha$ |

---

## 🔶 6. **Multiclass Focal Loss**

For multiclass classification with softmax over $K$ classes:

$$
\mathcal{L}_{\text{focal}} = - \sum_{c=1}^{K} \alpha_c (1 - p_c)^\gamma y_c \log(p_c)
$$

Where:

* $y_c$ is 1 only for the ground-truth class
* $p_c$ is the predicted softmax probability for class $c$
* $\alpha_c$ is class weighting

---

## ✅ Summary

| Term               | Meaning                              |
| ------------------ | ------------------------------------ |
| $(1 - p_t)^\gamma$ | Downweights easy examples            |
| $\alpha_t$         | Adjusts for class imbalance          |
| γ = 0              | Becomes normal cross-entropy         |
| Common usage       | RetinaNet, imbalanced classification |

---



In **focal loss** and **cross-entropy**, if **positives are the minority**, you should **give more weight to positives** to compensate for imbalance.

So:

* **You should set $\alpha = 0.75$** (not 0.25) if positives are rare.
* In the focal loss:

  * $\alpha_t = 0.75$ for **positives (y = 1)**
  * $\alpha_t = 0.25$ for **negatives (y = 0)**

This is **exactly the same principle** as class-weighted cross-entropy.

---

## 🔍 Where the confusion happened

In your earlier example:

> "If $\alpha = 0.25$", then:
>
> * $\alpha_t = 0.25$ for **positive**
> * $\alpha_t = 0.75$ for **negative**

That means: you’re assigning **less weight to positives**, which is appropriate **only if positives are abundant** (i.e. majority), which is **not** typical.

---

## 🧠 Proper Setting of $\alpha$

Let’s get it straight:

| Scenario                           | Class           | Weight in Focal Loss      |
| ---------------------------------- | --------------- | ------------------------- |
| **Positives are rare** (e.g. 1:10) | Positives (y=1) | $\alpha = 0.75$ or higher |
|                                    | Negatives (y=0) | $1 - \alpha = 0.25$       |
| **Negatives are rare**             | Positives (y=1) | $\alpha = 0.25$           |
|                                    | Negatives (y=0) | $1 - \alpha = 0.75$       |

So: **Higher alpha means "give more weight to class 1 (positives)."**

---

## ✅ Consistency with Weighted Cross-Entropy

In standard **weighted binary cross-entropy**:

$$
\text{Loss} = - w_1 y \log(\hat{p}) - w_0 (1 - y)\log(1 - \hat{p})
$$

To handle imbalance:

* Set $w_1 > w_0$ when **class 1 (positive)** is underrepresented
* This is equivalent to choosing $\alpha > 0.5$ in focal loss

---

## 🔁 Why Then Did RetinaNet Use $\alpha = 0.25$?

Good catch: RetinaNet (Lin et al., 2017) uses:

* $\alpha = 0.25$
* Because **positives are extremely rare** (\~1:100) in dense object detection
* But in their definition:

  * They assign \*\*$\alpha = 0.25$ to **positives**
  * Not because it's optimal, but because the large **focusing parameter** $\gamma = 2$ already downweights easy negatives harshly
  * So they experimentally found **$\alpha = 0.25$** was enough

But in **general use**, especially in class-imbalanced problems, it's safer to follow:

> **Set $\alpha$ higher for the underrepresented class.**

---

## ✅ Final Rule of Thumb

> If **positives are rare**, use **$\alpha > 0.5$** (e.g., 0.75 or 0.9) to give them more weight.

> If **negatives are rare**, use **$\alpha < 0.5$** (e.g., 0.25).

---

