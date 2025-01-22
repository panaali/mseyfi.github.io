# Axial Attention Simplified

Below is illustrative pseudo-code comparing **traditional (vanilla) 2D self-attention** vs. **axial attention** for a batch of 2D feature maps (e.g., an image). We’ll assume:

- **Input** $\mathbf{X}$ has shape $(B, H, W, d)$  
  - $B$: Batch size  
  - $H$: Height  
  - $W$: Width  
  - $d$: Feature (embedding) dimension per spatial location

We will show:

1. **Vanilla 2D self-attention** – Flatten the 2D grid $(H \times W)$ into one dimension of length $N = H \times W$, and then compute full self-attention.  
2. **Axial attention** – Factor the attention into a “row” step (attending across width $W$) and then a “column” step (attending across height $H$).

---

## 1. Vanilla 2D Self-Attention

### Key Idea

1. Reshape the $(H, W)$ grid into one sequence of length $N = H \times W$.  
2. Compute standard (multi-head) self-attention over those $N$ tokens.  
3. Reshape back to $(H, W)$.

### Pseudo-Code

```python
##################################################
# VANILLA 2D SELF-ATTENTION
##################################################

# X.shape = (B, H, W, d)
# Flatten 2D grid into a single dimension
N = H * W
X_flat = reshape(X, (B, N, d))  # Now each item is a "token"

# Create learnable projection matrices Wq, Wk, Wv (d x d) for queries, keys, values
# (For multi-head attention, you'd split d into multiple heads, but here is a single-head illustration)

Q = X_flat @ Wq   # (B, N, d)
K = X_flat @ Wk   # (B, N, d)
V = X_flat @ Wv   # (B, N, d)

# Compute attention scores: QK^T / sqrt(d)
scores = Q @ transpose(K, (0, 2, 1))  # (B, N, N)
scores = scores / sqrt(d)

# Softmax over the last dimension
weights = softmax(scores, dim=-1)  # (B, N, N)

# Multiply by V to get the output
out = weights @ V  # (B, N, d)

# Reshape the output back to (B, H, W, d)
X_out = reshape(out, (B, H, W, d))
```

- **Memory/Compute Complexity** roughly $O(N^2 \cdot d)$ = $O((HW)^2 \cdot d)$.  
- For large $H,W$, this becomes very expensive.

---

## 2. Axial Attention

### Key Idea

1. Perform attention **across rows** (the width dimension, $W$) for each of the $H$ rows, independently.  
2. Then perform attention **across columns** (the height dimension, $H$) for each of the $W$ columns, independently.  
3. Each step is effectively 1D self-attention, so the cost scales like $O(H \cdot W^2 + W \cdot H^2)$ instead of $O(H^2 W^2)$.

### Step-by-Step Diagram

1. **Row Attention**  
   - Treat each row of length $W$ as a 1D sequence.  
   - Compute attention over that sequence, for all $H$ rows.

2. **Column Attention**  
   - Next, treat each column of length $H$ as a 1D sequence.  
   - Compute attention over that sequence, for all $W$ columns.

### Pseudo-Code

```python
##################################################
# AXIAL ATTENTION
##################################################

# X.shape = (B, H, W, d)

# 1. ROW ATTENTION: Attend across the width dimension (W)

row_out = zeros_like(X)  # (B, H, W, d)

for i in range(H):
    # Extract the i-th row from the batch
    # shape of X_i => (B, W, d)
    X_i = X[:, i, :, :]  
    
    # Project to Q, K, V
    Q_i = X_i @ Wq_row    # (B, W, d)
    K_i = X_i @ Wk_row    # (B, W, d)
    V_i = X_i @ Wv_row    # (B, W, d)

    # Compute scores = Q_i K_i^T / sqrt(d)
    scores_i = Q_i @ transpose(K_i, (0, 2, 1))  # (B, W, W)
    scores_i = scores_i / sqrt(d)

    # Softmax
    weights_i = softmax(scores_i, dim=-1)       # (B, W, W)

    # Output
    out_i = weights_i @ V_i                     # (B, W, d)

    # Place it back in the row_out
    row_out[:, i, :, :] = out_i

# 2. COLUMN ATTENTION: Now attend across the height dimension (H)

col_out = zeros_like(row_out)  # (B, H, W, d)

for j in range(W):
    # Extract the j-th column from row_out
    # shape of row_j => (B, H, d)
    row_j = row_out[:, :, j, :]

    # Project to Q, K, V
    Q_j = row_j @ Wq_col   # (B, H, d)
    K_j = row_j @ Wk_col   # (B, H, d)
    V_j = row_j @ Wv_col   # (B, H, d)

    # Compute scores = Q_j K_j^T / sqrt(d)
    scores_j = Q_j @ transpose(K_j, (0, 2, 1))  # (B, H, H)
    scores_j = scores_j / sqrt(d)

    # Softmax
    weights_j = softmax(scores_j, dim=-1)       # (B, H, H)

    # Output
    out_j = weights_j @ V_j                     # (B, H, d)

    # Place it back into col_out
    col_out[:, :, j, :] = out_j

# col_out is now the final output after axial attention
X_out = col_out
```

- **Memory/Compute Complexity** is closer to $O(H \times W^2 + W \times H^2)$, which is much less than $O(H^2 W^2)$ when $H$ and $W$ are large.

---

## 3. Shape Clarifications

- **Initial Input**: $\mathbf{X} \in \mathbb{R}^{B \times H \times W \times d}$.  
- **Flattened for Vanilla**: $\mathbf{X}_{\text{flat}} \in \mathbb{R}^{B \times (HW) \times d}$.  
- **Row Attention**:  
  - Process each row: $\mathbf{X}_i \in \mathbb{R}^{B \times W \times d}$.  
  - Attention scores: $\mathbf{scores}_i \in \mathbb{R}^{B \times W \times W}$.  
  - Row output: $\mathbf{row\_out}_i \in \mathbb{R}^{B \times W \times d}$.  
- **Column Attention**:  
  - Process each column: $\mathbf{row\_j} \in \mathbb{R}^{B \times H \times d}$.  
  - Attention scores: $\mathbf{scores}_j \in \mathbb{R}^{B \times H \times H}$.  
  - Column output: $\mathbf{col\_out}_j \in \mathbb{R}^{B \times H \times d}$.  

---

## 4. Why Axial Attention Can Be Better

- **Reduced Complexity**  
  - Vanilla: $O((HW)^2 \cdot d)$.  
  - Axial: $O(H \cdot W^2 \cdot d + W \cdot H^2 \cdot d)$.  
  - For large $H$ and $W$, this is a major reduction.

- **Scalability**  
  - Can be extended to 3D (e.g., height $\times$ width $\times$ depth) by applying attention along each axis in turn.

- **Maintains Global Context**  
  - Even though we split across rows then columns, the final representation still mixes information globally.

---

### Final Note

In practice, **multi-head attention** is commonly used (where $d$ is split into multiple heads), but the overall pattern remains the same. Axial attention is a powerful technique to make self-attention feasible for large 2D (or higher-dimensional) data without the prohibitive $(HW)^2$ cost of naive full self-attention.
