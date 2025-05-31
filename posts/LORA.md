# LoRA (Low-Rank Adaptation) for Multi-Head Attention: Complete Tutorial

## 1. Introduction to LoRA

**LoRA (Low-Rank Adaptation)** is a parameter-efficient fine-tuning technique for large-scale pre-trained models. It allows us to adapt a model by introducing low-rank trainable matrices into certain parts of the network while keeping the original pre-trained weights frozen.

## 2. Core Idea

In standard training, a weight matrix $W \in \mathbb{R}^{d \times k}$ is updated directly. LoRA freezes $W$ and instead introduces a trainable low-rank perturbation:

$W' = W + \Delta W \quad \text{where} \quad \Delta W = A B \quad A \in \mathbb{R}^{d \times r}, B \in \mathbb{R}^{r \times k}, r \ll \min(d, k)$

This allows only $A$ and $B$ to be trained.

### Why Efficient?

* The full matrix $W$ has $d \times k$ parameters.
* LoRA adds $d \times r + r \times k = r (d + k)$ trainable parameters.
* For square matrices where $d = k$, LoRA has $2 \times r \times d$ parameters.

This is a massive reduction when $r \ll d$.

### Why Can We Approximate a Matrix with a Low-Rank One?

Most real-world weight matrices contain significant redundancy. The effective information often lies in a low-dimensional subspace. Through **Singular Value Decomposition (SVD)**:

$W = U \Sigma V^\top$

We can keep only the top $r$ singular values and associated vectors:

$W \approx U_r \Sigma_r V_r^\top = A B$

LoRA mimics this with trainable $A$ and $B$, assuming that the model only needs a low-dimensional update for fine-tuning. This insight drastically reduces the number of trainable parameters without a significant drop in performance.

## 3. Where LoRA is Applied in Attention

In Transformers, attention is computed as:

$\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{Q K^\top}{\sqrt{d_k}} \right)V$

Where:

* $Q = X W^Q$, $K = X W^K$, $V = X W^V$

LoRA is usually applied to:

* $W^Q$, $W^V$
* Optionally $W^K$

**Why Q and V?**

* $W^Q$: Affects how each token queries others — essential for attention pattern.
* $W^V$: Affects the actual values propagated through attention.

**Why not always K?**

Let’s extend the classroom analogy:

* Imagine you're in a math classroom that was pre-trained on general math topics (arithmetic, algebra, calculus).
* Now you want to fine-tune the class for **geometry**.

Here’s what happens:

* **Queries (Q)** are how students ask questions. With LoRA, we teach them to ask better geometry questions.
* **Values (V)** are the answers students give. With LoRA, we make sure the answers contain geometry-relevant information.

But what about **Keys (K)**?

* Keys are like the **name tags** or **subject expertise labels** students wear, indicating what they know.

Your concern is valid: if no student is labeled as a "geometry expert" (i.e., there is no pre-existing K for geometry), how can questions about geometry be matched to the right answers?

Here’s the insight:

* In pretraining, the model may not have explicit geometry tokens, but many tokens might partially relate to geometry (e.g., "shape," "angle," "proof").
* LoRA **adjusts Queries (Q)** to align geometry questions toward **existing Keys (K)** that are semantically close.
* It also **adjusts Values (V)** to ensure those matched tokens return geometry-relevant content.

By doing this, we exploit the latent expressiveness of the pretrained model — even if no student had a "geometry" badge before, many had enough math foundation to help.

Updating K can cause instability:

* A token formerly recognized as an algebra expert might suddenly appear to specialize in geometry — but without learning how to actually answer geometry questions (unless V is also changed).
* This disrupts the learned alignment between Q and K and often hurts performance.

So instead of relabeling students (K), we teach students to direct their questions more effectively (Q) and provide new, relevant answers (V) using the same matching system.

However, what if we **do update both K and V**?

If we carefully adapt **both**:

* **K (Keys)** to label tokens as geometry-relevant entities, and
* **V (Values)** to ensure those tokens now provide appropriate geometry answers,

then we can potentially avoid destabilization.

But this approach requires **tight coupling between K and V updates**:

* Any shift in K must be mirrored by a shift in V to preserve semantic alignment.
* If K is changed but V is not updated accordingly, attention scores may be redirected to the wrong tokens, leading to degraded performance.

So, yes—it is **possible** to update both K and V safely. But:

* It demands **more parameters** and
* **More careful training** (e.g., warm-up, learning rate tuning)

In low-rank settings like LoRA, Q and V often give the best return on investment with the least risk.

Empirically, this is enough to shift the model toward the new domain without corrupting its internal structure.
**Not applied to:**

* $QK^\top$, the attention score matrix, which is a result of computation.
* $QK^\top V$, the attention output, which is not a parameter but an intermediate result.

## 4. Rank of Matrix in LoRA

The **rank** of a matrix is the number of linearly independent columns (or rows). In LoRA, we approximate a high-rank matrix by multiplying two low-rank matrices:

$\Delta W = A B \quad \text{rank}(\Delta W) \le r$

This reduces both memory and compute. The success of this approximation depends on the low intrinsic dimensionality of useful updates.

## 5. LoRA-Enhanced Attention: How It Works

Given a frozen weight $W^Q$, we replace it with:

$W^Q_{\text{LoRA}} = W^Q + \alpha A_Q B_Q$

During training, only $A_Q, B_Q$ are updated. $W^Q$ remains frozen.

This is similarly done for $W^V$, optionally $W^K$.

## 6. LoRA Multi-Head Attention Code (No LLM Libraries)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=4, alpha=1.0, dropout=0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha

        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.weight.requires_grad = False

        self.A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.B = nn.Parameter(torch.randn(out_features, r) * 0.01)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        delta_w = torch.matmul(self.B, self.A)
        adapted_weight = self.weight + self.alpha * delta_w
        return F.linear(self.dropout(x), adapted_weight)

class MultiHeadAttentionWithLoRA(nn.Module):
    def __init__(self, embed_dim, num_heads, r=4):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = LoRALinear(embed_dim, embed_dim, r=r)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = LoRALinear(embed_dim, embed_dim, r=r)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, T, C = x.shape

        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_probs, V)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(attn_output)

# Example
x = torch.randn(2, 16, 64)
mha_lora = MultiHeadAttentionWithLoRA(embed_dim=64, num_heads=4, r=8)
out = mha_lora(x)
print(out.shape)  # (2, 16, 64)
```

## 7. Summary Table

| Component         | LoRA Applied? | Explanation                               |
| ----------------- | ------------- | ----------------------------------------- |
| $W^Q, W^V$        | ✅             | Most commonly adapted weights             |
| $W^K$             | Optional      | Sometimes skipped to save compute         |
| $QK^\top$         | ❌             | No learnable parameters                   |
| $QK^\top V$       | ❌             | Not a target of LoRA                      |
| Output proj $W^O$ | Optional      | Sometimes adapted in large-scale settings |
