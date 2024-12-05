[![Home](https://img.shields.io/badge/Home-Click%20Here-blue?style=flat&logo=homeadvisor&logoColor=white)](../)

# Understanding DETR: Object Detection with Transformers
#### Good video [Here](https://www.youtube.com/watch?v=T35ba_VXkMY&ab_channel=YannicKilcher)
**Table of Contents**

1. [Introduction](#introduction)
2. [Why Transformers for Object Detection?](#why-transformers-for-object-detection)
3. [DETR Architecture Overview](#detr-architecture-overview)
4. [Encoder Detailed Explanation](#encoder-detailed-explanation)
5. [Decoder Detailed Explanation](#decoder-detailed-explanation)
6. [Remarks on the Tensor Size and architecture wrap up](Remarks-on-the-Tensor-Size-and-architecture-wrap-up)
7. [Loss Functions and Bipartite Matching](#loss-functions-and-bipartite-matching)
8. [Conclusion](#conclusion)

---

## Introduction

The **Detection Transformer (DETR)** is a novel approach to object detection that leverages Transformers, which were originally designed for sequence-to-sequence tasks like machine translation. Introduced by Carion et al. in 2020, DETR simplifies the object detection pipeline by eliminating the need for hand-crafted components like anchor generation and non-maximum suppression (NMS).

---
![DetPipeline](../images/DetrPipeline.png)

## Why Transformers for Object Detection?

Traditional object detection models rely on convolutional neural networks (CNNs) with added complexities like region proposal networks, anchor boxes, and NMS. Transformers offer a simpler and more unified architecture by modeling object detection as a direct set prediction problem.

**Reasons for Using Transformers:**

- **Global Context Modeling:** Transformers can capture long-range dependencies, making them suitable for understanding global context in images.
- **Simplified Pipeline:** Eliminates the need for NMS and anchor boxes, reducing hyperparameters.
- **Set Prediction:** Treats object detection as a set prediction problem, which aligns well with the permutation-invariant nature of Transformers.

---

## DETR Architecture Overview

DETR consists of three main components:

1. **Backbone CNN:** Extracts feature maps from the input image.
2. **Transformer Encoder:** Processes the feature maps to capture global context.
3. **Transformer Decoder:** Generates object predictions using learned object queries.

Below is a high-level diagram of the DETR architecture:

```
Input Image --> Backbone CNN --> Transformer Encoder --> Transformer Decoder --> Predictions
```

---

![DetPipeline](../images/Detr.png)

## Encoder Detailed Explanation

### Role of the Encoder

The encoder processes the feature map from the backbone and outputs a sequence of context-rich feature representations. It models the relationships between all positions in the feature map, capturing global information.

### Mathematical Formulation

Let $X \in \mathbb{R}^{H \times W \times C}$ be the feature map from the backbone, where:

- $H$, $W$: Height and width of the feature map.
- $C$: Number of channels.

We flatten $X$ to $\mathbf{x} \in \mathbb{R}^{N \times C}$, where $N = H \times W$.

**Positional Encoding:**

Since Transformers lack inherent positional awareness, we add positional encodings $\mathbf{p}$ to $\mathbf{x}$:

$$\mathbf{z}_0 = \mathbf{x} + \mathbf{p}$$

**Encoder Layers:**

Each encoder layer consists of:

1. **Multi-Head Self-Attention (MHSA):**

   $\text{MHSA}(\mathbf{z}_{l-1}) = \textrm{Softmax}\left( \frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} \right)\mathbf{V}$

   Where:

   - $\mathbf{Q}, \mathbf{K}, \mathbf{V}$: Queries, keys, and values computed from $\mathbf{z}_{l-1}$.
   - $d_k$: Dimensionality of keys.


2. **Feed-Forward Network (FFN):**

   $$\text{FFN}(\mathbf{z}) = \textrm{ReLU}(\mathbf{z}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2$$

**Layer Normalization and Residual Connections:**

Each sub-layer is wrapped with residual connections and layer normalization:

$$
\mathbf{z}'_l = \textrm{LayerNorm}\left(\mathbf{z}_{l-1} + \textrm{MHSA}(\mathbf{z}_{l-1})\right)
$$

$$
\mathbf{z}_l = \textrm{LayerNorm}(\mathbf{z}'_l + \textrm{FFN}(\mathbf{z}'_l))
$$

### Intuition Behind the Encoder

The encoder allows each position in the feature map to attend to every other position, capturing global relationships. This is crucial for understanding complex scenes where objects might interact.

![DetTrans](../images/DetTrans.png)


### Code Snippet

```python
import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=6):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_encoder = PositionalEncoding(d_model)
    
    def forward(self, src):
        # src shape: (N, C) where N = H * W
        src = self.pos_encoder(src)  # Add positional encoding
        memory = self.encoder(src)   # Output shape: (N, C)
        return memory

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Implementation of positional encoding
        # ...
    
    def forward(self, x):
        # x shape: (N, C)
        # ...
        return x + self.pe[:x.size(0), :]
```

**Tensor Sizes:**

- Input `src`: $N \times C$
- Output `memory`: $N \times C$

---

## Decoder Detailed Explanation

### Role of the Decoder

The decoder generates object predictions by querying the encoded image features. It uses a fixed set of learned object queries to produce a set of predictions in parallel.

### Mathematical Formulation

**Object Queries:**

Let $\mathbf{q} \in \mathbb{R}^{M \times C}$ be the learned object queries, where $M$ is the number of object queries (e.g., 100).

**Decoder Layers:**

Each decoder layer consists of:

1. **Masked Multi-Head Self-Attention (MMHSA) on Object Queries:**

$$
   \mathbf{q}'_l = \text{LayerNorm}(\mathbf{q}_{l-1} + \text{MMHSA}(\mathbf{q}_{l-1}))
$$

2. **Multi-Head Cross-Attention (MHCA) between Object Queries and Encoder Output:**

   $\mathbf{q}^{\prime\prime}_l = \text{LayerNorm}(\mathbf{q}'_l + \text{MHCA}(\mathbf{q}'_l, \mathbf{z}))$

   Where $\mathbf{z}$ is the encoder output.

3. **Feed-Forward Network (FFN):**

   $\mathbf{q}_l = \text{LayerNorm}(\mathbf{q}^{\prime\prime}_l + \text{FFN}(\mathbf{q}^{\prime\prime}_l))$

### Intuition Behind the Decoder

- **Object Queries:** Act as slots that the model fills with detected objects. They attend to relevant parts of the encoder output to gather object-specific information.
- **Cross-Attention:** Enables object queries to focus on different parts of the image, effectively learning where objects are located.

### Code Snippet

```python
class TransformerDecoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=6, num_queries=100):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.query_pos = nn.Parameter(torch.randn(num_queries, d_model))  # Object queries
    
    def forward(self, tgt, memory):
        # tgt shape: (num_queries, C)
        # memory shape: (N, C)
        tgt2 = self.decoder(tgt, memory)  # Output shape: (num_queries, C)
        return tgt2
```

**Tensor Sizes:**

- Input `tgt`: $M \times C$ (object queries)
- Input `memory`: $N \times C$ (encoder output)
- Output `tgt2`: $M \times C$ (decoder output)

### Detailed Workflow of the Decoder

1. **Initialization:**

   - Start with learned object queries $\mathbf{q}_0$.

2. **Self-Attention on Object Queries:**

   - Each object query attends to other queries to capture dependencies among predicted objects.
   - Helps in modeling mutual exclusivity and interactions.

3. **Cross-Attention between Queries and Encoder Output:**

   - Object queries attend to the encoder output to gather relevant image features.
   - Cross-attention weights determine which parts of the image each query focuses on.

4. **Feed-Forward Network:**

   - Applies non-linear transformations to enhance feature representations.

5. **Output Layers:**

   - Apply linear layers to predict bounding boxes and class labels for each query.

### Mathematical Expressions

**Cross-Attention Computation:**

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}\left( \frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} \right)\mathbf{V}
$$

- $\mathbf{Q}$: Projected object queries.
- $\mathbf{K}, \mathbf{V}$: Projected encoder outputs.

---
## Remarks on the Tensor Size and architecture wrap up:

- The cross-attention in the DETR decoder is computed between a set of **N_query**(here **N_query = M**) queries and **N_enc** (here **N_enc = N**) encoder output tokens. Therefore, the cross-attention weight matrix has dimensions **(N_query × N_enc)**.  
- The encoder outputs a sequence of shape **(N_enc, D)**, where **N_enc** is typically the flattened spatial dimension of the image features and **D** is the model dimension.  
- The decoder queries consist of **N_query** learned embeddings, each of dimension **D**.  
- For cross-attention, the query (Q) matrix is of shape **(N_query, D)**, and the key (K) and value (V) matrices derived from the encoder output are of shape **(N_enc, D)**.

**Detailed Explanation:**

1. **Encoder Output Size:**  
   DETR first processes an input image through a backbone CNN (e.g., ResNet-50), resulting in a feature map of size `(C, H', W')`. These features are then flattened and projected into a `D`-dimensional space. Let’s define:
   - `H'` and `W'` are the spatial dimensions of the downsampled feature map after the backbone and possibly additional projections.
   - The total number of spatial positions is `N_enc = H' * W'`.
   - Each position is represented by a `D`-dimensional vector after linear projection.

   Thus, after positional encoding and passing through the Transformer encoder (with multiple layers), the **encoder output** is a sequence of shape:

   $$
   \text{Encoder Output} \in \mathbb{R}^{N_{\text{enc}} \times D}, \quad \text{where } N_{\text{enc}} = H'W'
   $$

3. **Decoder Queries:**  
   The DETR decoder uses a fixed set of **N_query** learned object queries. Each query is a `D`-dimensional embedding. These queries are independent of the input image content; they are trained to attend to different parts of the encoder output to detect objects. So the query set is:

   $$
   \text{Decoder Queries} \in \mathbb{R}^{N_{\text{query}} \times D}, \quad \text{where } N_{\text{query}} \text{ is often set to 100}
   $$

   Each decoder layer uses these queries to attend to the encoder outputs and produce refined queries (representations) that eventually lead to object predictions.

4. **Cross-Attention in the Decoder:**
   In a standard Transformer cross-attention module, you have three key matrices: Q (query), K (key), and V (value). For cross-attention in the DETR decoder:
   - **Q (Query):** Derived from the decoder queries. After a linear projection, these will still have dimension `(N_query, D)`.
   - **K (Key) and V (Value):** Derived from the encoder output. After linear projections, these remain `(N_enc, D)`.

   Concretely:
   - Q is obtained by applying a linear transformation to the decoder queries:  

     $$
     Q \in \mathbb{R}^{N_{\text{query}} \times D}
     $$
   
   - K and V are obtained by applying linear transformations to the encoder output:  
   
     $$
     K \in \mathbb{R}^{N_{\text{enc}} \times D}, \quad V \in \mathbb{R}^{N_{\text{enc}} \times D}
     $$

5. **Cross-Attention Computation:**
   The attention weights are computed as:

   $$
   \text{Attention} = \text{softmax}\left(\frac{QK^T}{\sqrt{D}}\right) V
   $$

   Here, `QK^T` results in a `(N_query, N_enc)` matrix:

   - `Q` has shape `(N_query, D)`.
   - `K^T` has shape `(D, N_enc)`.

   Multiplying `Q (N_query, D)` by `K^T (D, N_enc)` gives:

   $$
   QK^T \in \mathbb{R}^{N_{\text{query}} \times N_{\text{enc}}}
   $$

   This is the **cross-attention weight matrix**, which, after the softmax operation, is used to combine the values:
   
   $$
   \text{Attention Output} = \text{softmax}(QK^T) V \in \mathbb{R}^{N_{\text{query}} \times D}
   $$

**Summary of Dimensions:**
- **Encoder output:** `(N_enc, D)` where `N_enc = H' * W'`.
- **Decoder queries:** `(N_query, D)` (e.g., `N_query = 100`).
- **Cross-attention Q:** `(N_query, D)`
- **Cross-attention K and V:** `(N_enc, D)`
- **Cross-attention matrix (QK^T):** `(N_query, N_enc)`

This is the general scheme for the DETR architecture’s cross-attention dimensions.

## Loss Functions and Bipartite Matching

**Overview:**  
DETR's loss function and training procedure differ from traditional object detectors because it treats object detection as a direct set prediction problem. This approach removes the need for non-maximum suppression (NMS) and anchor generation. Instead, DETR predicts a fixed-size set of object "queries" and then finds a one-to-one matching between these predictions and the ground truth objects using the Hungarian algorithm. The loss is then computed based on this optimal matching.

**Key Steps in DETR’s Loss Computation:**

1. **Fixed-Size Predictions:**
   DETR outputs a fixed number of predictions (for example, 100 predictions per image). Each prediction consists of:
   - A class probability distribution (including the "no object" or background class).
   - A predicted bounding box (parameterized by its center coordinates, width, and height, or sometimes normalized coordinates).

   Let’s denote:
   - The set of predictions as $$\{\hat{y}_i\}_{i=1}^{N}$$, where $$N$$ is a fixed number like 100.
   - Each $$\hat{y}_i$$ includes $$\hat{p}_i(c)$$ for each class $$c$$ and a predicted bounding box $$\hat{b}_i$$.

2. **Constituting Ground Truth:**
   The ground truth for an image typically consists of:
   - A set of $$M$$ ground truth objects $$\{y_j\}_{j=1}^{M}$$ where each $$y_j$$ includes a ground-truth class label $$c_j$$ and a ground-truth bounding box $$b_j$$.

   DETR assumes $$M \leq N$$. If there are fewer ground-truth objects than the number of predictions, the unmatched predictions should ideally represent the “no object” class.

3. **Optimal Bipartite Matching with the Hungarian Algorithm:**
   One critical innovation in DETR is how it determines which predicted query corresponds to which ground truth object. This is done via a one-to-one matching using the Hungarian (a.k.a. Kuhn-Munkres) algorithm.

   **Forming the Cost Matrix:**
   - First, DETR computes a cost for matching each predicted box $$\hat{y}_i$$ with each ground truth object $$y_j$$.
   - The cost includes both classification and localization terms:
     1. **Classification Cost:** This is typically the negative log-likelihood of the ground truth class under the predicted class distribution:

        $$
        C_{\text{class}}(y_j, \hat{y}_i) = -\log \hat{p}_i(c_j)
        $$
        
     3. **Bounding Box Localization Cost:**  
        DETR uses a combination of:
        - **L1 loss** on bounding box coordinates: $$\| b_j - \hat{b}_i \|_1$$
        - **Generalized IoU (GIoU)** loss on bounding boxes:

          $$
          C_{\text{box}}(y_j, \hat{y}_i) = \lambda_{\text{L1}}\| b_j - \hat{b}_i \|_1 + \lambda_{\text{GIoU}}(1 - \text{GIoU}(b_j, \hat{b}_i))
          $$

   Thus, the total cost for matching ground-truth object $$j$$ and predicted object $$i$$ could be:

   $$
   C(y_j, \hat{y}_i) = C_{\text{class}}(y_j, \hat{y}_i) + C_{\text{box}}(y_j, \hat{y}_i)
   $$

   **Hungarian Matching:**
   - After forming the $$M \times N$$ cost matrix (with $$M \leq N$$), the Hungarian algorithm is applied to find the global one-to-one matching between predictions and ground truth that yields the minimal total cost.
   - This results in a permutation $$\sigma$$ of $$\{1,\ldots,N\}$$ (or a partial mapping if $$N>M$$) such that $$\sigma(j)$$ is the index of the prediction matched to the ground-truth object $$j$$.

5. **Computing the Loss After Matching:**
   Once the optimal matching is established, the loss is computed by summing over the matched pairs and also taking into account the unmatched predictions:

   **Matched Predictions:**
   For each matched pair $$(j, \sigma(j))$$:
   - **Classification Loss:** A cross-entropy loss for the matched prediction’s class distribution against the ground truth class. This encourages the matched prediction to classify correctly.
   - **Box Regression Loss:** A combination of L1 loss and GIoU loss between the matched predicted box and the ground truth box.

   **Unmatched Predictions:**
   For the predictions that are not matched to any ground truth object, the model expects them to predict the "no object" (or background) class. Thus, those predictions incur a classification loss pushing them towards predicting "no object."

   Formally, the final loss $$\mathcal{L}$$ is:

   $$
   \mathcal{L} = \sum_{j=1}^{M} [\mathcal{L}_{\text{class}}(y_j, \hat{y}_{\sigma(j)}) + \lambda_{\text{box}}\mathcal{L}_{\text{box}}(y_j, \hat{y}_{\sigma(j)})] + \sum_{i \notin \{\sigma(1), \ldots, \sigma(M)\}} \mathcal{L}_{\text{class}_\text{noobj}}(\hat{y}_{i})
   $$

   Where:
   - $$\mathcal{L}_{\text{class}}$$ is the cross-entropy loss for the correct class.
   - $$\mathcal{L}_{\text{box}}$$ includes both L1 and GIoU losses.
   - $$\mathcal{L}_{\text{class}_\text{noobj}}$$ is the loss that encourages predictions that aren't matched to a real object to predict the "no object" class.

**Summary:**
- **Ground Truth Constitution:** The ground truth is simply the set of annotated bounding boxes and their classes for each image.
- **Matching:** DETR uses the Hungarian algorithm to find a unique, one-to-one assignment between the predicted set of objects (queries) and the ground-truth objects.
- **Loss Function:**  
  - A combined classification and box regression loss is computed only for matched predictions.
  - Unmatched predictions are penalized if they fail to predict the "no object" class.
  - The Hungarian matching ensures a fair and stable assignment, enabling an end-to-end set prediction that does not require post-processing like NMS.
## Conclusion

DETR revolutionizes object detection by framing it as a direct set prediction problem using Transformers. The architecture simplifies the detection pipeline, removes the need for heuristic components like NMS, and provides a unified end-to-end trainable model.

**Key Takeaways:**

- **Transformer Encoder:** Captures global context in images.
- **Transformer Decoder:** Uses object queries to predict objects in parallel.
- **Set-Based Loss with Bipartite Matching:** Ensures unique assignment of predictions to ground truth objects.

By leveraging the strengths of Transformers, DETR opens new avenues for research and applications in object detection and beyond.
