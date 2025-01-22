# Efficient ML Techniques in Transformers
Vision Transformers (ViTs) have become a popular choice for image recognition and related tasks, but they can be computationally expensive and memory-heavy. Below is a list of common (and often complementary) techniques to optimize Transformers—including ViTs—for more efficient training and inference. Alongside each category, I’ve mentioned some influential or representative papers.

---

## 1. Efficient Attention Mechanisms

**Key Idea:** Replace the standard O(N²) self-attention (where N is the number of tokens) with more efficient variants, typically by imposing low-rank structure, using kernel approximations, or random feature mappings.

- **Linformer**  
  \- *Paper:* [Linformer: Self-Attention with Linear Complexity](https://arxiv.org/abs/2006.04768) (Wang et al., 2020)  
  \- *Idea:* Projects the sequence length dimension to a lower dimension, reducing the complexity from O(N²) to O(N).

- **Performer**  
  \- *Paper:* [Rethinking Attention with Performers](https://arxiv.org/abs/2009.14794) (Choromanski et al., 2021)  
  \- *Idea:* Uses random feature maps to approximate the softmax attention, enabling linear-time attention.

- **Reformer**  
  \- *Paper:* [Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451) (Kitaev et al., 2020)  
  \- *Idea:* Uses locality-sensitive hashing (LSH) to reduce the complexity of attention and reversible residual layers to save memory.

- **Big Bird**  
  \- *Paper:* [Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) (Zaheer et al., 2020)  
  \- *Idea:* Combines random, global, and local attentions to handle very long sequences efficiently.

---

## 2. Model Compression Techniques

### 2.1 Pruning

**Key Idea:** Remove weights or tokens deemed unimportant. 

- **Structured Pruning (e.g., heads, entire tokens):**  
  Removing entire attention heads or tokens (in the case of Vision Transformers) that contribute less to the final prediction.
  
- **Movement Pruning:**  
  \- *Paper:* [Movement Pruning: Adaptive Sparsity by Fine-Tuning](https://arxiv.org/abs/2005.07683) (Sanh et al., 2020)  
  \- *Idea:* Learns which weights to remove during fine-tuning, guided by the movement of weights during training.

- **Token Pruning / Early Exiting for ViT:**  
  Prune unimportant tokens dynamically or terminate computation early if predictions are sufficiently confident.  
  \- Example approach: [Dynamic Token Pruning for Vision Transformers](https://arxiv.org/abs/2106.16231).

### 2.2 Quantization

**Key Idea:** Use fewer bits (e.g., 8-bit or even lower precision) to represent weights and/or activations without significantly degrading accuracy.

- *Seminal Early Work:* [Model Compression via Distillation and Quantization](https://arxiv.org/abs/1506.02626) (Hinton et al., 2015) introduced the broader framework of compression.  
- *Applied to Transformers:* [Quantizing Deep Convolutional + Transformer Models](https://arxiv.org/abs/1910.06188), among others.

### 2.3 Low-Rank Factorization

**Key Idea:** Approximate large weight matrices with products of lower-rank matrices.

- *Representative Work:* [Tensorizing Neural Networks](https://arxiv.org/abs/1509.06569) shows how to reshape weights into tensors and factor them efficiently.  
- *Vision Transformer context:* Factoring projection matrices or MLP weights in Transformers can reduce parameters and computation.

### 2.4 Knowledge Distillation

**Key Idea:** Train a smaller “student” model (or same-sized model but more efficient architecture) to match outputs of a larger “teacher” model.

- **DeiT**  
  \- *Paper:* [Training Data-Efficient Image Transformers & Distillation through Attention](https://arxiv.org/abs/2012.12877) (Touvron et al., 2021)  
  \- *Idea:* Shows that Transformers can be trained effectively with fewer data if distilled from a CNN or a larger Transformer.

- **TinyBERT, MobileBERT, etc.** (more general for NLP, but the idea is the same)  
  \- *Papers:* [TinyBERT](https://arxiv.org/abs/1909.10351), [MobileBERT](https://arxiv.org/abs/2004.02984).

---

## 3. Parameter Efficient Fine-Tuning

**Key Idea:** Instead of fine-tuning all parameters, only update (or add) a small subset of the parameters. This is especially relevant when you deal with large-scale ViTs.

- **Adapters / LoRA (Low-Rank Adaptation):**  
  \- *Paper:* [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (Hu et al., 2021)  
  \- *Idea:* Insert small trainable low-rank matrices into Transformer layers to handle new tasks, reducing the overhead.

- **Prefix Tuning / Prompt Tuning** (originating from NLP, can also be adapted for ViTs).

---

## 4. Mixture-of-Experts (MoE)

**Key Idea:** Scale model capacity by having multiple “expert” layers, but activate only a subset for each input, reducing total computation.

- *Representative Work:* [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) (Fedus et al., 2021)  
- Though MoE approaches are more common in large-scale language models, they can theoretically be applied to ViTs as well.

---

## 5. Architectural Re-design

Sometimes, simply rethinking the architecture yields a more efficient design:

- **Hybrid CNN-Transformer Architectures:** Use convolution in early stages for low-level feature extraction, then apply Transformers on higher-level tokens (reducing total sequence length).  
  \- *Paper Example:* [LocalViT: Bringing Locality to Vision Transformers](https://arxiv.org/abs/2104.05707)

- **Pyramid Transformers / Swin Transformers:**  
  \- *Paper:* [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) (Liu et al., 2021)  
  \- *Idea:* Reduce sequence length by using patch hierarchies and windowed self-attention.

- **Patch Merging / Pooling:** Combine patches progressively so that later layers have fewer tokens.

---

## 6. Dynamic Computation / Early Exiting

**Key Idea:** Not all inputs require the same amount of computation. For some inputs, we can exit early or skip certain layers/tokens.

- *Representative Idea (in NLP):* [LayerDrop: Trading Accuracy for Efficiency by Dropping Layers in Pre-trained Models](https://arxiv.org/abs/1909.11556)  
- *Applied to Vision:* Dynamic token pruning / partial inference once the model is sufficiently confident.

---

## Summary of Techniques & Seminal References

1. **Efficient Attention**  
   - *Linformer* (Wang et al., 2020)  
   - *Performer* (Choromanski et al., 2021)  
   - *Reformer* (Kitaev et al., 2020)  
   - *Big Bird* (Zaheer et al., 2020)

2. **Model Compression**  
   - **Pruning:** *Movement Pruning* (Sanh et al., 2020)  
   - **Quantization:** Early works by Hinton et al. and subsequent follow-ups  
   - **Low-Rank Factorization:** *Tensorizing Neural Networks* (Novikov et al., 2015)  
   - **Distillation:** *DeiT* (Touvron et al., 2021), *Hinton et al. (2015)*

3. **Parameter-Efficient Fine-Tuning**  
   - *LoRA* (Hu et al., 2021), *Adapter-BERT* approaches

4. **Mixture-of-Experts**  
   - *Switch Transformer* (Fedus et al., 2021)

5. **Architectural Tweaks**  
   - *Swin Transformer* (Liu et al., 2021)  
   - Hybrid CNN + Transformer  
   - Pyramid ViTs

6. **Dynamic Computation / Early Exiting**  
   - *LayerDrop* (Fan et al., 2019)  
   - Dynamic Token Pruning (various recent works)

---

### Final Note

In practice, combining multiple of these strategies often yields the best tradeoff between accuracy and efficiency. For Vision Transformers, a common recipe might be:

1. Use an efficient attention scheme (like local/windowed attention).  
2. Add architectural innovations (pyramidal design, patch merging).  
3. Apply knowledge distillation for further accuracy boosts with fewer parameters.  
4. Optionally prune or quantize the final model for edge or latency-sensitive deployments.

Each of these categories has a rich body of research. If you’re aiming to build an efficient ViT from scratch, you could start with a well-known efficient ViT backbone (e.g., Swin Transformer, MobileViT, etc.) and then apply compression or distillation on top.
