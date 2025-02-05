[![Home](https://img.shields.io/badge/Home-Click%20Here-blue?style=flat&logo=homeadvisor&logoColor=white)](../)


<br>
<br>


[![Linformer](https://img.shields.io/badge/Linformer-An_Efficient_Transformer-blue?style=for-the-badge&logo=github)](../posts/Linformer)
<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">

Overview of Linformer, why we need it, and side-by-side pseudo-code comparing traditional self-attention to Linformer self-attention. We’ll keep the example to single-head attention for clarity, but in practice you would typically use multi-head attention (with separate projections for each head).
_Last updated: {{ site.time | date: "%B %d, %Y" }}_
</div>


[![Efficiency](https://img.shields.io/badge/Efficient_Transformers-Efficient_Techniques_in_Transformers-blue?style=for-the-badge&logo=github)](../posts/EfficientTransformers)
<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">

Vision Transformers (ViTs) have become a popular choice for image recognition and related tasks, but they can be computationally expensive and memory-heavy. Below is a list of common (and often complementary) techniques to optimize Transformers—including ViTs—for more efficient training and inference. Alongside each category, I’ve mentioned some influential or representative papers.
_Last updated: {{ site.time | date: "%B %d, %Y" }}_
</div>


[![AxialAttention](https://img.shields.io/badge/Axial_Attention-Attentions across axes-blue?style=for-the-badge&logo=github)](../posts/AxialAttention)

<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
Key ideas:
  
1. Perform attention **across rows** (the width dimension, $W$) for each of the $H$ rows, independently.  
2. Then perform attention **across columns** (the height dimension, $H$) for each of the $W$ columns, independently.  
3. Each step is effectively 1D self-attention, so the cost scales like $O(H \cdot W^2 + W \cdot H^2)$ instead of $O(H^2 W^2)$.

_Last updated: {{ site.time | date: "%B %d, %Y" }}_
</div>


[![TrackFormer](https://img.shields.io/badge/TrackFormer-Multi_Object_Tracking_with_Transformer-blue?style=for-the-badge&logo=github)](../posts/TrackFormer)

<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
Traditional multi-object tracking (MOT) systems often follow a two-step pipeline:
  
Detect objects in each frame independently.
Associate detections across frames to form trajectories.
This separation can lead to suboptimal solutions since detection and association are treated as separate problems. TrackFormer merges these steps by extending a Transformer-based detection architecture (inspired by DETR) to simultaneously detect and track objects. It does this by introducing track queries that carry information about previously tracked objects forward in time, allowing the network to reason about detection and association in a unified end-to-end manner. <p></p>
_Last updated: {{ site.time | date: "%B %d, %Y" }}_
</div>

[![DETR](https://img.shields.io/badge/DETR-Detection_Transformer-blue?style=for-the-badge&logo=github)](../posts/DETR)

<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
The Detection Transformer (DETR) is a novel approach to object detection that leverages Transformers, which were originally designed for sequence-to-sequence tasks like machine translation. Introduced by Carion et al. in 2020, DETR simplifies the object detection pipeline by eliminating the need for hand-crafted components like anchor generation and non-maximum suppression (NMS).
 <p></p>
_Last updated: {{ site.time | date: "%B %d, %Y" }}_
</div>

## [![VIT](https://img.shields.io/badge/VIT-Vision_Transformers-blue?style=for-the-badge&logo=github)](../posts/VIT)
<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
Vision Transformers (ViTs) apply the Transformer architecture, originally designed for natural language processing (NLP), to computer vision tasks like image classification. ViTs treat an image as a sequence of patches (akin to words in a sentence) and process them using Transformer encoders. <p></p>
_Last updated: {{ site.time | date: "%B %d, %Y" }}_
</div>

## [![CGANS](https://img.shields.io/badge/CGANs-Conditional_GAN-blue?style=for-the-badge&logo=github)](../posts/ConditionalGan)
<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
Python implementation of a Conditional Generative Adversarial Network (cGAN) using PyTorch.
 <p></p>
_Last updated: {{ site.time | date: "%B %d, %Y" }}_
</div>

## [![Distillation](https://img.shields.io/badge/Distillation-grey?style=for-the-badge&logo=github)](../posts/Distillation)
<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
Generalization issue with Distillation
 <p></p>
_Last updated: {{ site.time | date: "%B %d, %Y" }}_
</div>


[![SSD](https://img.shields.io/badge/SSD-Single_Shot_Object_Detector-blue?style=for-the-badge&logo=github)](../posts/SSD)
<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">

Single shot object detector
_Last updated: {{ site.time | date: "%B %d, %Y" }}_
</div>


