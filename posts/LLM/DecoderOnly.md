
## A Deep Dive into Decoder-Only Transformers

*Last Updated: June 8, 2025*

### 1. The Task and Core Concept: Autoregressive Generation

The core task of a decoder-only model is **autoregressive text generation**. The model learns to predict the very next word given a sequence of preceding words. It generates text one token at a time, feeding its own previous output back in as input to predict the next token. This simple, self-supervised objective, when applied at a massive scale, enables the model to learn grammar, facts, reasoning abilities, and style.

---
### 2. Architecture: The Power of Masking

A decoder-only model is a stack of decoder layers from the original Transformer architecture. Its critical component is **Causal Self-Attention** (or Masked Self-Attention), which ensures that the prediction for a token at position `i` can only depend on the known outputs at positions less than `i`.

This is achieved by adding a mask to the attention score matrix before the softmax operation, effectively zeroing out the probabilities for all future tokens.

#### The Mathematics of Causal Self-Attention

We modify the standard attention formula by adding a mask matrix $M$:

$$M_{ij} = \begin{cases} 0 & \text{if } i \geq j \\ -\infty & \text{if } i < j \end{cases}$$

$$\text{CausalAttention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

#### Code Example: Generating the Causal Attention Mask
Here is a practical PyTorch snippet demonstrating how this mask is generated on the fly.

```python
import torch

# Define the sequence length for a given batch
sequence_length = 4

# Create the mask that prevents attending to future tokens
# `torch.triu` creates an upper triangular matrix. diagonal=1 ensures the diagonal is 0.
causal_mask_base = torch.triu(torch.ones(sequence_length, sequence_length), diagonal=1).bool()
# tensor([[False,  True,  True,  True],
#         [False, False,  True,  True],
#         [False, False, False,  True],
#         [False, False, False, False]])

# The final mask uses a large negative number for masked positions and 0.0 for unmasked.
final_mask = torch.zeros(sequence_length, sequence_length)
final_mask.masked_fill_(causal_mask_base, float('-inf'))
# tensor([[0., -inf, -inf, -inf],
#         [0.,   0., -inf, -inf],
#         [0.,   0.,   0., -inf],
#         [0.,   0.,   0.,   0.]])

# This final_mask is then added to the attention scores before the softmax.
```

---
### 3. Positional Embeddings: Encoding Order
The Transformer architecture is permutation-invariant. We must explicitly inject positional information. The final input embedding for each token is the sum of its token embedding and its positional embedding.

#### 3.1 Absolute Positional Embeddings (e.g., GPT-3)
* **Mechanism:** A unique vector is learned for each absolute position in the sequence (1st, 2nd, 3rd, etc.) from an embedding matrix of size `[max_sequence_length, hidden_size]`.
* **Limitation:** This method has a hard maximum sequence length and may not generalize well beyond it.

#### 3.2 Deep Dive: Rotary Positional Embeddings (RoPE)
Modern state-of-the-art models like Llama use a more advanced method called Rotary Positional Embeddings (RoPE).

* **Intuition:** Instead of *adding* position information, RoPE *rotates* our Query and Key vectors based on their position. The key insight is that the dot product between two vectors is related to the cosine of the angle between them. If we rotate our Query vector by an angle `m` and our Key vector by an angle `n`, the dot product between the new vectors will depend on the *relative angle* `m-n`. By making the rotation angle a function of a token's position, the attention score naturally becomes dependent on the relative positions of the tokens, not their absolute ones.

* **Mathematical Formulation:**
    1.  First, we view our `d`-dimensional embedding vectors (Queries and Keys) as a sequence of `d/2` two-dimensional vectors. Let's take one such pair $(x_{2i}, x_{2i+1})$ from a vector $x$ at position $m$.
    2.  We define a rotation angle $\theta_{m,i} = m \cdot 10000^{-2i/d}$. The angle depends on both the token's position `m` and the feature index `i`. Different dimensions are rotated at different frequencies.
    3.  We apply a 2D rotation matrix to this pair:
        $$R_m = \begin{pmatrix} \cos(\theta_{m,i}) & -\sin(\theta_{m,i}) \\ \sin(\theta_{m,i}) & \cos(\theta_{m,i}) \end{pmatrix}$$
        The positionally-encoded feature pair is then computed as $R_m \begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}$.
    4.  This is done for both the Query vector $q_m$ at position `m` and the Key vector $k_n$ at position `n`. The crucial property is that the dot product of the rotated vectors, $\langle R_m q_m, R_n k_n \rangle$, can be shown through trigonometric identities to be a function of their original dot product and their relative position, $m-n$.

* **Benefits:** This elegant method provides robust relative position information and has shown excellent performance when generalizing to sequence lengths far beyond what the model saw during training.

---
### 4. Training: The Next-Token Prediction Objective

#### 4.1 Tokenization and Input-Output Pairs
Using a sub-word tokenizer like BPE, the training data is created by simply shifting the tokenized text sequence by one position.

* **Input (`X`):** `[The, quick, brown, fox, jumps]`
* **Label (`Y`):** `[quick, brown, fox, jumps, .]`

#### 4.2 Loss Function: Cross-Entropy
The model is trained to minimize the **Cross-Entropy Loss** between its predicted probability distribution for the next token and the one-hot encoded vector of the true next token. For a single sequence $X = (x_1, ..., x_S)$, the loss is the average negative log-likelihood of the true next token at each position:

$$L_{CE}(\theta) = - \frac{1}{S} \sum_{t=1}^{S} \log P(x_t | x_{<t}; \theta)$$

#### 4.3 Evaluation Metric: Perplexity (and its relation to Loss)
While we use Cross-Entropy as the *loss function to be minimized during training*, we often use **Perplexity (PPL)** as the primary *metric for evaluating and comparing* language models.

* **Relationship:** Perplexity is the direct exponential of the cross-entropy loss.

    $$\text{Perplexity} = e^{L_{CE}} = \exp\left(- \frac{1}{S} \sum_{t=1}^{S} \log P(x_t | x_{<t}; \theta)\right)$$

* **Intuition: Why "Perplexity"?** It is a measure of how "perplexed" or "surprised" the model is by the test data. A lower perplexity indicates that the model is less surprised, meaning it consistently assigns a higher probability to the correct next token. A perplexity of **100** can be interpreted as the model being, on average, as uncertain about the next word as if it were choosing uniformly from 100 different words. A perfect model would have a perplexity of 1.
* **In Summary:** We train the model by minimizing cross-entropy loss. We evaluate and compare models by reporting their perplexity. **Lower is better for both.**

---
### 5. Prominent Architectures: GPT and Llama Families
#### The GPT (Generative Pre-trained Transformer) Family
Developed by OpenAI, this family pioneered the scaling of decoder-only models, trained on datasets like BookCorpus and WebText, culminating in GPT-3's demonstration of **in-context learning**.

#### The Llama (Large Language Model Meta AI) Family
Developed by Meta AI, the Llama family focused on training efficiency, demonstrating that smaller models trained on vast amounts of data (1-2 Trillion tokens from sources like C4) could outperform larger models. Their open release spurred massive innovation, and Llama 2 incorporated RLHF for safety.

#### Key Differences: Llama vs. GPT
| Feature | GPT Family (e.g., GPT-3) | Llama Family (e.g., Llama 2) |
| :--- | :--- | :--- |
| **Access** | Closed-source, accessible via API. | Open-source (weights available), allowing local deployment and research. |
| **Key Innovation Focus** | Scaling parameters to achieve emergent abilities (in-context learning). | Training efficiency (more tokens for smaller models), open access, and RLHF for safety. |
| **Architectural Tweaks** | Standard Transformer decoder. | **Pre-normalization (RMSNorm):** Normalizes inputs to layers for better training stability. |
| **Activation Function** | ReLU | **SwiGLU:** A variant of the Gated Linear Unit, which has shown better performance than ReLU. |
| **Positional Embeddings** | Learned, absolute positional embeddings. | **Rotary Positional Embeddings (RoPE):** Encodes relative position, improving long-sequence performance. |

---
### 6. Fine-Tuning and Inference
* **Fine-Tuning:** A "base model" is further trained on a curated dataset of `(instruction, desired_output)` pairs to align its behavior.
* **Inference:** Generating text is an iterative process. The strategy we use to *select* a token from the model's probability distribution is critical.

#### 6.1 A Deep Dive into Sampling Strategies

1.  **Greedy Search (Deterministic):** Always select the single token with the highest probability (`argmax`). This is fast but highly repetitive and rarely used for creative generation.

2.  **Temperature Sampling:** Rescales the logits before the softmax function using a `temperature` parameter ($T$).
    * $T > 1$ increases randomness and creativity.
    * $T < 1$ increases determinism, making the output more focused.

3.  **Top-k Sampling:** Restricts the choice to a "shortlist" of the `k` most probable tokens, then samples from that smaller set. This avoids picking absurdly unlikely tokens from the distribution's tail.

4.  **Top-p (Nucleus) Sampling:** A more adaptive approach and the most common sophisticated method. It creates a shortlist of variable size by selecting the smallest set of tokens whose cumulative probability exceeds a threshold `p` (e.g., 0.95). If the model is certain, the set is small; if uncertain, the set is larger, allowing for more creativity.

5.  **Beam Search:** Used to find a high-probability *sequence* rather than just the next word. It keeps track of the `k` (the "beam width") most probable partial sequences at each step and expands them. It is more computationally intensive but often produces more coherent and optimal outputs for tasks like translation or summarization.

---
### 7. From-Scratch Implementation of a Decoder-Only Model
This implementation uses the computationally efficient, fused-layer approach for multi-head attention.

```python
import torch
import torch.nn as nn
from torch.nn import functional as F

# --- Hyperparameters for our Toy Model ---
batch_size = 32      # How many independent sequences will we process in parallel?
block_size = 128     # What is the maximum context length for predictions?
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 256         # Embedding dimension
n_head = 4           # Number of attention heads
n_layer = 1          # Number of decoder layers (for simplicity)
dropout = 0.2
eval_interval = 500
max_steps = 5001

# --- 1. Data Preparation and Tokenizer ---
# For this toy example, we'll use a simple character-level tokenizer.
try:
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
except FileNotFoundError:
    print("Warning: 'input.txt' not found. Using dummy text for demonstration.")
    text = "hello world, this is a demonstration of a toy gpt model for a top-tier graduate school class."

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    # Generate random starting points for each sequence in the batch
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # Input sequences (the context)
    x = torch.stack([data[i:i+block_size] for i in ix])
    # Target sequences (the next character to predict)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# --- 2. Model Components (Optimized Implementation) ---

class MultiHeadAttention(nn.Module):
    """ The efficient, fused implementation of Multi-Head Causal Self-Attention """
    def __init__(self, n_embd, num_heads):
        super().__init__()
        self.num_heads = num_heads
        assert n_embd % num_heads == 0
        self.head_size = n_embd // num_heads

        # A single, fused linear layer for Q, K, V projections
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # The final output projection layer
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask buffer
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))
                                        .view(1, 1, block_size, block_size))

    def forward(self, x):
        # Input x shape: [B, T, C] (Batch, Time/seq_len, Channels/n_embd)
        B, T, C = x.shape

        # 1. Fused Projection & Splitting
        qkv = self.c_attn(x) # Shape: [B, T, 3 * C]
        q, k, v = qkv.split(self.n_embd, dim=2) # Each is [B, T, C]

        # 2. Reshape for Multi-Head computation
        # (B, T, C) -> (B, T, num_heads, head_size) -> (B, num_heads, T, head_size)
        q = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_size).transpose(1, 2)

        # 3. Batched Scaled Dot-Product Attention
        # (B, num_heads, T, head_size) @ (B, num_heads, head_size, T) -> (B, num_heads, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / self.head_size**0.5)
        att = att.masked_fill(self.tril[:,:,:T,:T] == 0, float('-inf')) # Apply causal mask
        att = F.softmax(att, dim=-1)
        
        # (B, num_heads, T, T) @ (B, num_heads, T, head_size) -> (B, num_heads, T, head_size)
        y = att @ v

        # 4. Concatenate and Project back
        y = y.transpose(1, 2).contiguous().view(B, T, C) # Shape: [B, T, C]
        return self.resid_dropout(self.c_proj(y))

class FeedForward(nn.Module):
    """ A simple linear layer followed by a non-linearity """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)

class DecoderBlock(nn.Module):
    """ A single Transformer block: communication followed by computation """
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, n_head)
        self.ffwd = FeedForward(n_embd)
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)

    def forward(self, x):
        # Pre-normalization and residual connections
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# --- 3. Full Model ---
class SimpleDecoderOnlyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[DecoderBlock(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # Final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size) # Language model head

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Get token and position embeddings
        # idx shape: [B, T]
        tok_emb = self.token_embedding_table(idx) # Shape: [B, T, C]
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # Shape: [T, C]
        x = tok_emb + pos_emb # Shape: [B, T, C]
        
        # Pass through decoder blocks
        x = self.blocks(x) # Shape: [B, T, C]
        x = self.ln_f(x) # Shape: [B, T, C]
        
        # Final projection to vocabulary size
        logits = self.lm_head(x) # Shape: [B, T, vocab_size]
        
        # Calculate loss if targets are provided
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_for_loss = logits.view(B*T, C)
            targets_for_loss = targets.view(B*T)
            loss = F.cross_entropy(logits_for_loss, targets_for_loss)
            
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop context to the last block_size tokens to respect positional embedding limits
            idx_cond = idx[:, -block_size:]
            # Get the predictions
            logits, loss = self(idx_cond)
            # Focus only on the logit for the last time step
            logits = logits[:, -1, :] # Becomes (B, C)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

# --- 4. Training Loop ---
model = SimpleDecoderOnlyModel()
m = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("Starting training...")
for steps in range(max_steps):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if steps % eval_interval == 0:
        print(f"Step {steps}, Training Loss: {loss.item():.4f}")

# --- 5. Generation from the model ---
print("\n--- Generating Text from Trained Model ---")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_tokens = m.generate(context, max_new_tokens=200)[0].tolist()
print(decode(generated_tokens))
```
