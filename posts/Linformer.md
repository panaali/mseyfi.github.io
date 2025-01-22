# Linformer
Below is an overview of **Linformer**, why we need it, and **side-by-side pseudo-code** comparing **traditional self-attention** to **Linformer** self-attention. We’ll keep the example to **single-head** attention for clarity, but in practice you would typically use **multi-head** attention (with separate projections for each head).

---

## 1. What Is Linformer?

**Linformer** (Wang et al., 2020) is a method to reduce the $\mathcal{O}(N^2)$ complexity of self-attention (where $N$ is the sequence length) to approximately $\mathcal{O}(N)$ (or more precisely $\mathcal{O} (N \times r)$ ), by projecting the attention’s **key** and **value** representations into a **lower-dimensional space** of size $r \ll N$.  

### The Core Idea

- In standard self-attention, we compute:
  
  $\text{Attention}(Q, K, V) = \text{softmax} \Bigl(\frac{QK^\top}{\sqrt{d}}\Bigr)V$
  
  - $Q, K, V \in \mathbb{R}^{B \times N \times d}$
  - This yields an attention matrix of shape $\mathbb{R}^{B \times N \times N}$ (when computing $QK^\top$).

- **Linformer** approximates this large $(N \times N)$ attention matrix by reducing $K$ and $V$ from sequence length $N$ down to a smaller dimension $r$.  
  - Introduce a learnable projection $E \in \mathbb{R}^{N \times r}$.  
  - Project $K, V$ along the sequence dimension, obtaining $K' \in \mathbb{R}^{B \times r \times d}$ and $V' \in \mathbb{R}^{B \times r \times d}$.  
  - Then compute attention as:
    
    $\text{Attention}(Q, K', V') = \text{softmax}\Bigl(\frac{Q K'^\top}{\sqrt{d}}\Bigr) V'$

  - Now the attention matrix is $\mathbb{R}^{B \times N \times r}$ instead of $\mathbb{R}^{B \times N \times N}$.

**Result**: The complexity becomes $\mathcal{O}(N \times r \times d)$ instead of $\mathcal{O}(N^2 \times d)$. For $r \ll N$, this is a major saving in both memory and computation.

---

## 2. Why Do We Use Linformer?

1. **Reduced Memory/Compute**  
   - Standard self-attention’s $\mathcal{O}(N^2)$ memory usage can be prohibitive for large $N$.  
   - Linformer compresses the sequence dimension from $N$ to $r$, reducing complexity to $\mathcal{O}(N \times r)$.

2. **Scalability to Long Sequences**  
   - Tasks like long-document modeling, large-context language modeling, or processing high-resolution images in “patch” form often involve large $N$.  
   - Linformer can handle much longer sequences than vanilla self-attention without exhausting GPU memory.

3. **Low-Rank Assumption**  
   - Linformer is motivated by the idea that the attention matrix often lies in a low-rank subspace for many practical tasks, so a smaller projection $r$ can capture most of the important structure.

---

## 3. Pseudo-Code: Traditional vs. Linformer

Below, we compare:

- **Traditional Self-Attention**  
- **Linformer Self-Attention**  

We'll assume a single head for simplicity, with inputs:

- $\mathbf{X} \in \mathbb{R}^{B \times N \times d}$  
  - $B$: Batch size  
  - $N$: Sequence length  
  - $d$: Embedding (hidden) dimension

We have trainable weight matrices $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{d \times d}$. In Linformer, we additionally learn a projection $\mathbf{E}\in \mathbb{R}^{N \times r}$ (and often a separate $\mathbf{E}_K, \mathbf{E}_V$ for keys and values, but we’ll keep one $\mathbf{E}$ for simplicity).

---

### 3.1 Traditional Self-Attention (Vanilla)

```python
##################################################
# VANILLA SELF-ATTENTION (SINGLE-HEAD)
##################################################
# X.shape = (B, N, d)

# 1. Compute Q, K, V
#    Each is (B, N, d)
Q = X @ W_Q   # shape: (B, N, d)
K = X @ W_K   # shape: (B, N, d)
V = X @ W_V   # shape: (B, N, d)

# 2. Compute attention scores: QK^T / sqrt(d)
#    => (B, N, N)
scores = Q @ transpose(K, (0, 2, 1))  # shape: (B, N, N)
scores = scores / sqrt(d)

# 3. Softmax over the last dimension
weights = softmax(scores, dim=-1)     # shape: (B, N, N)

# 4. Multiply by V => (B, N, d)
out = weights @ V

# final output: (B, N, d)
X_out = out
```

- **Memory/Compute Complexity**: $\mathcal{O}(N^2 \times d)$.  
- The bottleneck is the $QK^\top$ operation, which yields an $(N \times N)$ attention matrix.

---

### 3.2 Linformer Self-Attention

```python
##################################################
# LINFORMER SELF-ATTENTION (SINGLE-HEAD)
##################################################
# X.shape = (B, N, d)

# 1. Compute Q, K, V (same as vanilla)
Q = X @ W_Q   # (B, N, d)
K = X @ W_K   # (B, N, d)
V = X @ W_V   # (B, N, d)

# 2. Project K and V along the sequence dimension
#    E.shape = (N, r)  (learnable)
#    K_prime.shape = (B, r, d)
#    V_prime.shape = (B, r, d)

# We want to reduce the "N" dimension of K,V to "r"
# Pseudo-code can be done with an einsum or loop, but conceptually:
K_prime = batch_matmul_seq_dim(K, E)  # shape: (B, r, d)
V_prime = batch_matmul_seq_dim(V, E)  # shape: (B, r, d)

# 3. Compute attention with smaller dimension
#    => Q.shape:       (B, N, d)
#       K_prime.shape: (B, r, d)
#    => QK'^T => (B, N, r)
scores = Q @ transpose(K_prime, (0, 2, 1))  # (B, N, r)
scores = scores / sqrt(d)

# 4. Softmax over the last dimension (r)
weights = softmax(scores, dim=-1)   # (B, N, r)

# 5. Multiply by V_prime => (B, N, d)
out = weights @ V_prime   # (B, N, d)

# final output: (B, N, d)
X_out = out
```

Where `batch_matmul_seq_dim(K, E)` roughly means:

```python
# Conceptual: for each example in the batch,
# we are applying a matrix multiply along the "N" dimension:
# (B, N, d) x (N, r) => (B, r, d)

# A simple way to express this in pseudo-code:
K_prime = zeros((B, r, d))
for b in range(B):
    # K[b].shape = (N, d)
    # E.shape =   (N, r)
    # => K_prime[b].shape = (r, d)
    K_prime[b] = transpose(K[b], (1, 0)) @ E   # or use an einsum
    # Then transpose back if needed
```

*(In an actual implementation with a deep learning framework, you might use an “einsum” notation or built-in multi-dimensional matmul to handle this elegantly.)*

- **Memory/Compute Complexity**:  
  - The main attention step $Q(K')^\top$ is $\mathcal{O}(N \times r \times d)$.  
  - The linear projections for $K, V$ are $\mathcal{O}(N \times d \times r)$.  
  - Overall, $\mathcal{O}(N \times r \times d)$ if $r \ll N$ is much smaller than $\mathcal{O}(N^2 \times d)$.

---

## 4. Shape Clarifications

- **Vanilla Self-Attention**  
  - **Input**: $\mathbf{X}\in \mathbb{R}^{B \times N \times d}$  
  - **Q, K, V**: $\in \mathbb{R}^{B \times N \times d}$  
  - **Scores**: $\in \mathbb{R}^{B \times N \times N}$  
  - **Output**: $\in \mathbb{R}^{B \times N \times d}$  

- **Linformer**  
  - **Input**: $\mathbf{X}\in \mathbb{R}^{B \times N \times d}$  
  - **Q, K, V**: $\in \mathbb{R}^{B \times N \times d}$  
  - **Projection $\mathbf{E}$**: $\in \mathbb{R}^{N \times r}$  
  - **$K'$, $V'$**: $\in \mathbb{R}^{B \times r \times d}$  (reduced along the $N$ dimension)  
  - **Scores**: $\in \mathbb{R}^{B \times N \times r}$  
  - **Output**: $\in \mathbb{R}^{B \times N \times d}$  

---

## 5. Summary

1. **Traditional Self-Attention**:  
   - Complexity: $\mathcal{O}(N^2 \times d)$.  
   - Creates a full $(N \times N)$ attention matrix.

2. **Linformer**:  
   - Complexity: $\mathcal{O}(N \times r \times d)$.  
   - Projects $K$ and $V$ to $(r \ll N)$ along the sequence dimension, drastically reducing both memory and compute overhead.

**Why Linformer?**  
- It allows Transformers to handle much **longer sequences** without blowing up compute/memory usage, by leveraging an assumption that the $(N \times N)$ attention matrix can be well-approximated in a lower-rank subspace.

## More Intuition

In **Linformer**, we typically project only the Key ($K$) and Value ($V$) matrices (along the sequence length) into a lower-dimensional space, while leaving the Query ($Q$) matrix at its original size. The main motivation is to reduce the **$\mathbf{N \times N}$** complexity in the attention computation without losing too much information about each query token. Below are some intuitive reasons why $Q$ is not compressed in Linformer:

---

## 1. How Linformer Reduces Complexity

Recall that in standard self-attention, we compute:

$\text{Attention}(Q, K, V) 
= \underbrace{\text{softmax}\Bigl(\frac{Q \,K^\top}{\sqrt{d}}\Bigr)}_{\displaystyle \text{size }N \times N} \; V,$

where $Q, K, V \in \mathbb{R}^{B \times N \times d}$ and $N$ is the sequence length. This yields an $\mathcal{O}(N^2)$ memory and compute cost (because of the $N\times N$ matrix).

- **Linformer** introduces a learnable projection $\mathbf{E}\in \mathbb{R}^{N \times r}$ (with $r \ll N$) that compresses $K$ and $V$ from shape $(N, d)$ to $(r, d)$ along the sequence dimension:

  $K' = K \times E,\quad V' = V \times E \quad (\text{schematically}),$
  
  so that $K' \in \mathbb{R}^{B \times r \times d}$ and $V' \in \mathbb{R}^{B \times r \times d}$.

- Then the new attention step becomes:

  $\text{Attention}(Q, K', V') = \text{softmax}\Bigl(\frac{Q \,K'^\top}{\sqrt{d}}\Bigr) \; V',$

  which yields an $N \times r$ attention matrix instead of $N \times N$, cutting complexity down to $\mathcal{O}(N \times r)$.

---

## 2. Why $Q$ is Kept at Full Dimension

1. **We Still Need an Output Token for Each Position ($N$)**  
   - The output of the attention mechanism is still $(B, N, d)$, meaning we produce a new embedding for each of the $N$ tokens.  
   - The Query dimension (the “rows” of the attention matrix) corresponds directly to the number of tokens ($N$) for which we want to compute an updated representation.  
   - If we also projected $Q$ down to size $r$, we would effectively reduce the number of output tokens from $N$ to $r$, which is **not** what we want—each position in the input still needs its own separate query and output.

2. **The Big Cost is in $\mathbf{QK^\top}$, Not in $Q$ Alone**  
   - Standard self-attention’s main memory/compute bottleneck arises from forming the $(N \times N)$ attention matrix $QK^\top$.  
   - By projecting $K$ down to shape $(r, d)$, that matrix shrinks to $(N \times r)$, which is the key to reducing complexity.  
   - Keeping $Q$ at $(N, d)$ ensures we still get an output for each of the $N$ tokens.  
   - If we also compressed $Q$, we would get $(r \times r)$ attention, which would no longer yield per-token outputs of size $N$.

3. **Low-Rank Assumption Typically Applies to $K$ and $V$**  
   - Linformer’s theoretical argument (and empirical justification) is that the attention matrix itself (related to $K$ and $V$) often lies in a low-rank subspace.  
   - $Q$ represents the queries from each token, which we generally want to keep at full dimension so that each position can preserve its unique “question” about the sequence.  
   - Keys and Values often exhibit redundancy across the sequence, making them more amenable to low-rank projection.

4. **Preserving Per-Token “Querying” Power**  
   - Each token’s query vector $\mathbf{q}$ is what “selects” or “extracts” relevant context from the rest of the sequence. Compressing $Q$ might lose some fidelity in how each token can attend.  
   - Keeping $Q$ uncompressed ensures each token still has the full embedding dimension to encode what it wants from the rest of the sequence, while the cost-saving measure is applied to $K$ and $V$.

---

## 3. Intuitive Summary

- **Linformer** only compresses the **sequence dimension** in $K$ and $V$.  
- **$Q$ remains at size $(N, d)$** so you still have a query vector for each of the $N$ tokens, maintaining a distinct output for every token.  
- This approach **drastically reduces** the memory and compute cost by shrinking the $\mathbf{N \times N}$ attention matrix to $\mathbf{N \times r}$ (with $r \ll N$), but still allows each token to “ask” a full-dimensional question.

Hence, the reason we **don’t** change $Q$ in Linformer is mainly to:

1. Avoid losing the ability to produce $\mathbf{N}$ distinct outputs, one for each token.  
2. Focus the low-rank compression where it matters most for computational savings—on the $\mathbf{N}$ dimension of $K$ and $V$.
