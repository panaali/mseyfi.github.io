Focal Loss is a modified version of the standard cross-entropy loss, designed to address the **class imbalance problem**, especially in tasks like **object detection** (e.g. RetinaNet) or **extremely imbalanced binary classification**.

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

* $p_t = \begin{cases}
  \hat{p} & \text{if } y = 1 \\
  1 - \hat{p} & \text{if } y = 0
  \end{cases}$
* $\alpha_t = \begin{cases}
  \alpha & \text{if } y = 1 \\
  1 - \alpha & \text{if } y = 0
  \end{cases}$

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
