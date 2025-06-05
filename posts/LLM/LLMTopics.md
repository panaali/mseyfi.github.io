# Big Topics in LLMs

1- what is Tokenizer, what are different tokenizers?

2- Embedding 

3- How de-tokenizatin

4-Transformer architechtures:
  - Encoder only models (BERT) sequence-to-sequence, (input output same size) or use case: sentiment, classification
  - Encoder-decoder models sequence-to-sequence with different length (BArt, T5)
  - Decoder only architechtures (Jurassic, Lama, GPT, BLUE)


- text you enter to the model is prompt
- output of the model is completion

- Prompt ---> inference ----> completion
- context window: full amount of text available to memory

5- In context Learning and zero/one/few shot inference

6- greedy(select the word according to the softmax highest probability) vs random sampling text generation (select a text randomly based on the softmax probability- if a word has 2% chance it can still be selected however in the greedy case there is no way a word with less maximum softmax probability be chosen)

7- top-k and top-p sampling and temperature

8- Encoder only models (BERT, ROBERTA) Autoencoding models ,  
  - masked languege modeling (MLM)
  - mask a word and predict a word (self supervisedly)
  - use bidirectional context, meaning both the past and the future words can take part to predicting the missing word.  - 
  - used for Named Entity Recognition)
  - Sentiment analysis
  - token/sentence classification
    -BERT and ROBERTA

9- Decoder only (GPT, Lamma) or autoregressive models (Causal Language modeling ) --> Text prediction from previsous tokens. 
The model has no knowledge of the end of the sentence.Unidirectional context, very good in zero shot inference

10- Encoder-Decoder models(T5, Bart) summarization, translation, question answering   
  - pretrain using span corruption
  - Sentinel tokens
# Computational Challenges of LLMs
BFLOAT16 and FP16
Model Sharding FSDP
Chinchilla Rule 
## Finetuning

ICL (in-contect learning zeor/one/few shot learning doesnt work for smaller models and may take a lot of space in the context window)

1- Instruction fine-tuning 
  - works on pairs of prompt-completion.
  - For a single task fine-tuning we need around 1K examples.
  - it's a full fine-tuning mechanism that updates all the model weights
  - Finetuning on a single task may cause catastrophic forgetting
  - If model performance drop on other tasks are important, you need to finetune over multiple tasks
    - Often 50K to 100K examples across multiple taks are needed.
  - **PEFT** or parameter efficient fine-tuning only adapt task-specific weights.
  - One important technique to prevent catasrophic forgetting is regularization on gradients norms. Using this technique, we prevent big updates on the backpropagation.
  - ## FLAN Finetuned LAnguage NET.
2- Metrics for evaluating LLMS
  - ## Definitions:
    -  **Unigram** is a single word  
    -  **Bigram** is a two words
    -  **n-gram** is a group of n *consecutive* words
  
  - Rouge: Used for text summarization
    - compares a summary to one or multiple other reference summaries. 
    
    EX:

    Reference: It is cold outside
    
    Generated output: it is very cols outside 
    
    $$
    Rouge-1_{\text{recall}} = \frac{\text{unigram matches}}{\text{unnigrams in reference}}
    $$

    
    $$
    Rouge-1_{\text{precision}} = \frac{\text{unigram matches}}{\text{unnigrams in output}}
    $$

    $$
    Rouge-1_{\text{F1}} = 2.\frac{\text{precision}\times \text{recall}}{\text{precision} + \text{recall}}
    $$
    
  - Issues: ordering of words are not taken into consideration therefore Rouge_1 can have a high number but they results may subjectively be poor.
  - We can use N-grams instead of word matching for Rouge
  - **LCS**: Longest common sequence
  - BLEU: Used for translation task
    - Compares translation to human-generated translation   

  # RLHF
-  Reward
-  Rollout
-  state and context
-  scaling instruction-finetuned language models
-  Convert ranking into pariwise training data for the reward model
- Reorder the prompts so that the prefered option comes first 

 ## Why Convert ranking into pariwise training data for the reward model

  Learning from full rankings (e.g., A > B > C > D) is more **complex and less stable** than learning from pairwise preferences due to several intertwined mathematical, computational, and statistical reasons. Here’s a breakdown:

---

### 📌 1. **Combinatorial Explosion of Permutations**

* A full ranking is a **permutation** over $n$ items.
* The number of possible permutations grows **factorially**: $n!$
* Modeling or assigning probabilities over all possible permutations (as in the **Plackett-Luce model**) becomes intractable for larger $n$.

#### 🔢 Example:

* For 4 items: 24 possible rankings.
* For 10 items: 3,628,800 possible rankings.

This makes the **output space huge**, and **gradient updates** less localized.

---

### 📌 2. **Label Ambiguity and Noise Amplification**

* Human rankings are **subjective and noisy**, especially as the number of choices increases.
* Annotators are less consistent with full rankings:

  * A > B might be confidently judged.
  * But deciding whether C > D is often weakly supported.

#### ✅ Pairwise comparisons focus only on two items at a time:

* Less cognitive load
* Easier and more accurate judgment
* Reduced label noise

---

### 📌 3. **Loss Function Instability**

* To learn from full rankings, we often use **listwise losses** such as:

  * ListMLE
  * Plackett-Luce
  * ListNet

These losses involve **normalizations over permutations** and are:

* **Non-convex**
* **Sensitive to small changes** in ranking order
* Difficult to optimize with SGD
* Require special sorting operations that **break gradient flow**

---

### 📌 4. **Difficulty in Generalizing**

* Ranking models often need to **predict the entire order** of a new list, not just compare two items.
* This task is **overconstrained**:

  * A small mistake (e.g., swapping B and C) drastically affects listwise metrics like NDCG, MAP.
  * Pairwise models are more forgiving and often generalize better under human-labeled data.

---

### 📌 5. **Sparse Learning Signal**

* A full ranking provides **only one ground truth permutation**.
* But from the same data, you can extract $\binom{n}{2}$ pairwise preferences.
* Pairwise learning gives **many training samples per annotation**, which helps generalization and convergence.

---

### 🔁 Comparison Table

| Aspect              | Full Ranking (Listwise)                 | Pairwise                     |
| ------------------- | --------------------------------------- | ---------------------------- |
| Complexity          | $O(n!)$                                 | $O(n^2)$                     |
| Labeling difficulty | High (subjective, error-prone)          | Low (binary comparison)      |
| Learning signal     | Sparse (1 permutation)                  | Dense (many comparisons)     |
| Optimization        | Complex loss, non-convex                | Simple sigmoid or hinge loss |
| Gradient stability  | Poor (discontinuous at sort boundaries) | Smooth (score differences)   |
| Scalability         | Poor for large $n$                      | Good                         |

---

### 🧠 Intuition

Training with full rankings is like trying to learn the rules of chess by memorizing full games. Training with pairwise comparisons is like learning which moves are better than others—simpler and more generalizable.



 ## reorder prompts so the preferred response comes first

In **RLHF reward model training**, we **reorder prompts so the preferred response comes first** (e.g., change (bad, good) → (good, bad)) before feeding them into the model. Here's **why**:



#### ✅ 1. **Standardization for Loss Computation**

Most reward models use a **pairwise loss** like:

$$
\mathcal{L}_{\text{RM}} = -\log \sigma(r(x_{\text{preferred}}) - r(x_{\text{dispreferred}}))
$$

To compute this, you need to know **which input is preferred**. By always putting the preferred item first, the model:

* Always computes $r_1 - r_2$ (where $r_1 = r(x_{\text{preferred}})$)
* Uses the same loss formula every time
* Avoids needing if-else logic or label-dependent signs

This is critical for clean and **vectorized implementation**.

---

#### ✅ 2. **Avoids Label Ambiguity**

If you feed unordered pairs with an extra label (`label = 0` means "first is better", `label = 1` means "second is better"), then:

* Your implementation must **flip** the sign of the reward difference based on the label
* Easy to introduce bugs like sign errors or flipped labels

Reordering the preferred item first removes this ambiguity.

---

#### ✅ 3. **Eases Batch Processing**

If all examples are ordered (preferred, dispreferred), then:

* You can construct two input batches:

  * `x_winners = [x1, x2, x3, ...]`
  * `x_losers  = [y1, y2, y3, ...]`
* Compute scores:

  $$
  r_{\text{win}} = r(x_winners),\quad r_{\text{lose}} = r(x_losers)
  $$
* Vectorized loss:

  $$
  \mathcal{L} = -\log \sigma(r_{\text{win}} - r_{\text{lose}})
  $$

---

#### ✅ 4. **Simplifies Human Feedback Integration**

When collecting rankings or binary preferences from humans, you only need to mark one response as preferred. Reordering ensures that training data **always places the good one first**, allowing clean downstream usage.

---

### 🧠 Summary

| Why Reorder?  | Benefit                                              |
| ------------- | ---------------------------------------------------- |
| Standard loss | Consistent $\log \sigma(r_1 - r_2)$ without flipping |
| Avoids bugs   | No need to handle label-dependent signs              |
| Vectorization | Enables batch training with simple subtraction       |
| Clean design  | Better integration with data pipelines and logging   |

---

Absolutely. Let’s now **rewrite everything using BERT** as the backbone of the reward model, and **track all tensor shapes step-by-step**, from input construction to loss computation.

---

# 🎯 Reward Model Using BERT – Full Tutorial with Tensor Sizes

---

## 1. 📥 **Inputs**

We start with:

* A prompt $x$
* Two completions $y^+$ (preferred), $y^-$ (dispreferred)

### Example:

```text
Prompt:     "Tell me a joke."
Preferred:  "Why don’t scientists trust atoms? Because they make up everything."
Dispreferred: "I'm not sure."
```

### Input Construction:

We concatenate prompt and completion with a separator:

```text
[CLS] Tell me a joke. [SEP] Why don’t scientists trust atoms? Because they make up everything. [SEP]
```

### Tokenization:

Assume BERT-base:

* Vocabulary size: $V = 30522$
* Hidden size: $H = 768$
* Max input length: $L = 512$

Let’s assume:

* Prompt + preferred response = 40 tokens
* Prompt + dispreferred response = 20 tokens

So:

```python
input_ids_preferred:    [batch_size, 40]
input_ids_dispreferred: [batch_size, 20]
```

After padding to max length in batch (say, 40), both inputs become:

```
input_ids: [batch_size, seq_len] = [B, 40]
attention_mask: [B, 40]
token_type_ids: [B, 40]
```

---

## 2. 🧠 **Model Architecture**

### We use:

* **BERT-base** (12 layers, 768 hidden dim, 12 heads)
* **Reward head**: 1-layer MLP on final token’s hidden state (typically `[CLS]` token)

---

### Forward Pass: Preferred Input

```python
outputs = BERT(input_ids, attention_mask, token_type_ids)
```

#### BERT Output:

```python
last_hidden_state: [B, L, H] = [B, 40, 768]
```

#### Extract \[CLS] token:

```python
cls_embedding: [B, H] = last_hidden_state[:, 0, :]  # [B, 768]
```

#### Reward head (linear projection):

```python
reward: [B, 1] = Linear(768 → 1)(cls_embedding)
```

---

### Repeat for dispreferred input:

* Compute $r^+$ and $r^-$ → both shapes: `[B, 1]`
* Squeeze to shape `[B]` for loss computation.

---

## 3. 📉 Loss Function

We use the **pairwise logistic loss**:

$$
\mathcal{L} = -\frac{1}{B} \sum_{i=1}^B \log \sigma(r^+_i - r^-_i)
$$

```python
r_diff = r_preferred - r_dispreferred       # shape: [B]
loss = -torch.log(torch.sigmoid(r_diff))    # shape: [B]
loss = loss.mean()                          # scalar
```

This encourages $r^+ > r^-$.

---

## 4. 🔄 Training Procedure

### Step-by-step:

1. Collect human preferences: tuples of (prompt, preferred, dispreferred)
2. Tokenize both pairs using BERT tokenizer
3. Pad to same length
4. Encode both via BERT → get CLS embeddings → project to scalar
5. Compute difference of scores
6. Apply sigmoid loss
7. Backpropagate and update parameters

> You can optionally **freeze BERT** and train only the reward head if data is limited.

---

## 5. 🧪 Evaluation Metrics

* Accuracy: % where $r^+ > r^-$
* AUC: if you treat preferences as binary classification
* Spearman/Pearson correlation (if comparing to real-valued ratings)

---

## 🧱 Summary of Tensors

| Component           | Tensor Shape     | Notes                            |
| ------------------- | ---------------- | -------------------------------- |
| `input_ids`         | `[B, L]`         | tokenized prompt + response      |
| `attention_mask`    | `[B, L]`         | 1 for real tokens, 0 for padding |
| `token_type_ids`    | `[B, L]`         | 0 for prompt, 1 for completion   |
| `last_hidden_state` | `[B, L, 768]`    | BERT output for each token       |
| `cls_embedding`     | `[B, 768]`       | first token representation       |
| `reward`            | `[B, 1]` → `[B]` | scalar reward per input          |
| `loss`              | `scalar`         | pairwise logistic loss           |

---

## 📦 Code Skeleton (PyTorch)

```python
class RewardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.reward_head = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_embed = outputs.last_hidden_state[:, 0, :]  # [B, 768]
        reward = self.reward_head(cls_embed).squeeze(-1)  # [B]
        return reward
```

---

## 📌 Notes:

* For long completions, use **RoBERTa** or **Longformer** (BERT is capped at 512 tokens).
* For image-text or multimodal reward models (e.g., in Flamingo or GPT-4V), visual tokens are fused at the input stage.

---

Would you like a full training script including data collation and batching?

