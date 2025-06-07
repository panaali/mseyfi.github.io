## A Deep Dive into BERT: The Complete Tutorial

*Last Updated: June 6, 2025*

### 1\. The Task: True Language Understanding

The core task of any foundational language model is not just to process text, but to *understand* it. For years, the dominant paradigm was unidirectional. Models like LSTMs and even the original GPT processed text sequentially, from left-to-right.

Consider this sentence:

> "The board of directors approved the new **board** for the game."

A left-to-right model, when it reaches the first "board," has no information about the word "game" which appears later. Its understanding is limited. The goal, therefore, is to create a model that can build a rich, contextual representation of each word by looking at the *entire sentence simultaneously*—both left and right. This is the task of creating a **deeply bidirectional language representation**.

### 2\. The Core Architecture: The Transformer Encoder

BERT is fundamentally a stack of Transformer Encoder layers. By eschewing the decoder, BERT can process an entire input sequence simultaneously, which is the key to its bidirectionality.

Each encoder layer is composed of two primary sub-layers: a Multi-Head Self-Attention mechanism and a Position-wise Feed-Forward Network.

#### 2.1 The Mathematics of Self-Attention

This mechanism allows the model to weigh the importance of other words when encoding a specific word. Let's formalize this.

Given an input sequence of token embeddings $X = (x\_1, x\_2, ..., x\_n)$, we first project them into three distinct spaces using learned weight matrices ($W^Q, W^K, W^V$) to create Query, Key, and Value matrices:

$$Q = XW^Q$$$$K = XW^K$$$$V = XW^V$$

The **Scaled Dot-Product Attention** is then calculated as:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Let's dissect this formula:

1.  **$QK^T$**: The dot product between the Query and Key matrices computes a similarity score, or an "attention score," between every pair of tokens in the sequence.
2.  **$\\frac{...}{\\sqrt{d\_k}}$**: The scores are scaled by the square root of the dimension of the key vectors ($d\_k$). This is a crucial normalization step that prevents the dot product values from growing too large, ensuring stable gradients.
3.  **softmax(...)**: The softmax function is applied row-wise to the scaled scores, converting them into a probability distribution of "attention weights."
4.  **...V**: Finally, we compute a weighted sum by multiplying the attention weights with the Value matrix. The result is a new sequence of embeddings where each token's representation is a rich, contextual blend of all other tokens in the sequence.

**Multi-Head Attention** performs this entire process multiple times in parallel with different learned projection matrices, allowing the model to capture different types of relationships simultaneously.

### 3\. BERT Architecture: An Overview

It is important to understand the concrete parameters of the models that were released.

| Parameter | BERT-Base | BERT-Large |
| :--- | :--- | :--- |
| **Encoder Layers (L)** | 12 | 24 |
| **Hidden Size (H)** | 768 | 1024 |
| **Attention Heads (A)** | 12 | 16 |
| **Total Parameters** | 110 Million | 340 Million |

  * **Embedding Size:** The hidden size (H) is synonymous with the embedding size. Each token is represented by a 768-dimensional vector in BERT-Base.
  * **Sequence Length:** BERT was pre-trained with a maximum sequence length of **512 tokens**.
  * **Batch Size:** For fine-tuning, smaller batch sizes like 16 or 32 are typically recommended.

#### How BERT Handles Sequences Longer Than 512 Tokens

The fixed 512-token limit presents a challenge for long documents. Standard strategies include:

1.  **Truncation:** Simply truncate the document to the first 512 tokens. This is easy but suffers from information loss.
2.  **Chunking and Pooling:** A more robust approach involves sliding a window of 512 tokens across the document. Each chunk is passed through BERT independently, and the resulting output vectors are aggregated (e.g., via `max` or `mean` pooling) to form a single representation for the entire document.

### 4\. A Deeper Look at the WordPiece Tokenizer

BERT uses a sub-word tokenization strategy called **WordPiece**.

  * **Vocabulary Creation:** The process starts with every individual character in the corpus and iteratively merges the most frequent adjacent pairs to form new sub-word units until a desired vocabulary size is reached (e.g., 30,000).
  * **Handling Out-of-Vocabulary (OOV) Words:** This is its primary strength. A traditional tokenizer would map an unknown word to a single `[UNK]` token. WordPiece effectively **eliminates the OOV problem** by greedily breaking down any unknown word into the longest possible known sub-words. For example, `epistemology` might become `epis`, `##tem`, `##ology`. The `##` indicates a sub-word that is part of a larger word.

### 5\. BERT's Input Representation

The final input vector for each token is the sum of three embeddings:

  * **Token Embeddings:** Learned vectors for each sub-word in the WordPiece vocabulary.
  * **Segment Embeddings:** A learned vector (`E_A` or `E_B`) indicating membership in the first or second sentence.
  * **Position Embeddings:** A learned vector for each position (0-511) to provide the model with a sense of order.
  * **Attention Mask:** A binary tensor that tells the attention mechanism to ignore `[PAD]` tokens.
  * **Special Tokens:**
      * **`[CLS]`:** Placed at the beginning of every sequence. Its final hidden state serves as the aggregate sequence representation for classification tasks.
      * **`[SEP]`:** Used to separate sentences.

### 6\. BERT's Pre-training Strategies

BERT is pre-trained on two unsupervised tasks **simultaneously**.

#### 6.1 Task 1: Masked Language Model (MLM)

This task enables deep bidirectional training. The model predicts randomly masked words in the input, forcing it to use both left and right context.

#### 6.2 Task 2: Next Sentence Prediction (NSP)

This task trains the model to understand logical relationships between sentences by predicting if sentence B is the actual sentence that follows sentence A.

#### 6.3 The Combined Loss Function (Mathematical Formulation)

The total loss for pre-training, parameterized by $\\theta$, is the sum of the MLM and NSP losses.

$$L(\theta) = L_{MLM}(\theta) + L_{NSP}(\theta)$$

  * **MLM Loss:** The average **Negative Log-Likelihood** (or Cross-Entropy Loss) over the masked positions. Let $M$ be the set of indices of the masked tokens.

    $$L_{MLM}(\theta) = - \frac{1}{|M|} \sum_{i \in M} \log p_i(x_i)$$

    where $p\_i(x\_i)$ is the model's predicted probability for the correct token $x\_i$ at masked position $i$.

  * **NSP Loss:** The Negative Log-Likelihood for the binary classification task on the `[CLS]` token's output.

    $$L_{NSP}(\theta) = - \log p_{cls}(y)$$

    where $y$ is the true label (`IsNext` or `NotNext`).

### 7\. Fine-Tuning BERT for Downstream Tasks

Fine-tuning adapts the pre-trained BERT model for a specific, supervised task by adding a small, task-specific layer.

#### 7.1 Text Classification (e.g., Sentiment Analysis)

  * **Methodology:** Add a single linear classification layer. The input to this layer is the final hidden state of the **`[CLS]`** token. Fine-tune by minimizing cross-entropy loss.

#### 7.2 Question Answering (Extractive QA)

  * **Task:** Given a question and a passage, find the span of text in the passage that answers the question.
  * **Methodology:** Add two specialized heads that take the final hidden state of *every token* in the passage as input: one to predict the probability of each token being the **start** of the answer, and another to predict the probability of it being the **end**.

#### 7.3 Named Entity Recognition (NER)

  * **Task:** Classify each token in a sentence into categories like Person, Organization, Location, etc.
  * **Methodology:** Add a classification layer that takes the final hidden state of **every token** and outputs a probability distribution over the NER labels for each token.

### 8\. Implementation in Practice

#### 8.1 Pseudocode for Pre-training

```
# Conceptual Pseudocode for BERT Pre-training
for each batch in large_unlabeled_corpus:
    # 1. Prepare batch and apply MLM masking
    input_tokens, mlm_labels, is_next_label = prepare_batch(batch)
    
    # 2. Construct BERT input tensors
    input_ids, segment_ids, attention_mask = construct_bert_input(input_tokens)
    
    # 3. Forward Pass
    sequence_output, cls_output = BERT(input_ids, segment_ids, attention_mask)
    
    # 4. Calculate Losses
    mlm_loss = CrossEntropyLoss(sequence_output, mlm_labels) # Over masked positions
    nsp_loss = CrossEntropyLoss(cls_output, is_next_label)   # On [CLS] output
    
    # 5. Backpropagation
    total_loss = mlm_loss + nsp_loss
    total_loss.backward()
    optimizer.step()
```

#### 8.2 Fine-Tuning Training and Validation Loop (PyTorch)

```python
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

# --- 1. Dataset Class ---
class IMDbDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# --- 2. Tokenization and DataLoader Setup ---
# Assume train_texts, train_labels, etc. are loaded
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

train_dataset = IMDbDataset(train_encodings, train_labels)
val_dataset = IMDbDataset(val_encodings, val_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# --- 3. Model, Optimizer, and Scheduler Setup ---
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * num_epochs)

# --- 4. Training Loop ---
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        # Tensors in `batch`:
        # 'input_ids':      [batch_size, sequence_length], e.g., [16, 128]
        # 'attention_mask': [batch_size, sequence_length], e.g., [16, 128]
        # 'labels':         [batch_size],                  e.g., [16]
        optimizer.zero_grad()
        outputs = model(input_ids=batch['input_ids'].to(device),
                        attention_mask=batch['attention_mask'].to(device),
                        labels=batch['labels'].to(device))
        
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

    # --- 5. Validation Loop ---
    model.eval()
    for batch in val_loader:
        with torch.no_grad():
            outputs = model(input_ids=batch['input_ids'].to(device),
                            attention_mask=batch['attention_mask'].to(device),
                            labels=batch['labels'].to(device))
            
        # Logits have shape: [batch_size, num_labels], e.g., [16, 2]
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        # ... (accuracy calculation logic) ...
```

### 9\. De-Tokenization

De-tokenization is the process of converting the model's output token IDs back into human-readable text. The tokenizer object provides this functionality, typically via a `decode` method. This method takes a sequence of token IDs, looks them up in its vocabulary, and intelligently stitches sub-words (those starting with `##`) back into complete words.
