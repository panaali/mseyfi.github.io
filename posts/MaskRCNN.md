# Mask R-CNN: Extended Technical Deep Dive Tutorial (Fully Corrected)

---

## 🎯 Tutorial Objectives

This tutorial is written to provide an extensive understanding of the Mask R-CNN architecture by dissecting every individual component involved in its pipeline. You will:

* See annotated PyTorch code for every network block (Backbone, FPN, RPN, RoIAlign, Detection & Mask Heads).
* Understand how each tensor transforms through the network, with precise shape annotations.
* Learn about anchor generation, matching, and the similarities/differences between SSD and Mask R-CNN.
* Deep dive into loss function math and logic, especially focusing on segmentation loss choices.
* Get visual and conceptual clarity about how the model is trained, evaluated, and deployed.

---

## 🧱 1. Backbone – ResNet50/101

The backbone of Mask R-CNN is a deep residual network, most commonly ResNet-50 or ResNet-101. It acts as a powerful feature extractor. ResNet is composed of stacked residual blocks that mitigate vanishing gradients in very deep networks.

We take intermediate outputs after certain layers to form a feature hierarchy:

* `C2` from `layer1` (stride = 4)
* `C3` from `layer2` (stride = 8)
* `C4` from `layer3` (stride = 16)
* `C5` from `layer4` (stride = 32)

These are fed into the FPN.

```python
from torchvision.models import resnet50
import torch.nn as nn

class ResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        net = resnet50(pretrained=True)
        self.stage1 = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)  # (B, 64, H/4, W/4)
        self.stage2 = net.layer1  # (B, 256, H/4, W/4)
        self.stage3 = net.layer2  # (B, 512, H/8, W/8)
        self.stage4 = net.layer3  # (B, 1024, H/16, W/16)
        self.stage5 = net.layer4  # (B, 2048, H/32, W/32)

    def forward(self, x):
        x = self.stage1(x)
        c2 = self.stage2(x)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)
        return c2, c3, c4, c5
```

These features preserve increasing semantic depth as resolution decreases. They are passed to the Feature Pyramid Network.

---

## 🧭 2. FPN – Feature Pyramid Network

FPN is used to construct a pyramid of features with strong semantics at all levels. It helps with detecting objects of different sizes.

### Key Properties:

* Top-down pathway with upsampling
* Lateral connections to maintain spatial structure
* Produces `P2`, `P3`, `P4`, and `P5`, each with 256 channels

```python
class FPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lat5 = nn.Conv2d(2048, 256, 1)
        self.lat4 = nn.Conv2d(1024, 256, 1)
        self.lat3 = nn.Conv2d(512, 256, 1)
        self.lat2 = nn.Conv2d(256, 256, 1)

        self.smooth4 = nn.Conv2d(256, 256, 3, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, 3, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, 3, padding=1)

    def forward(self, c2, c3, c4, c5):
        p5 = self.lat5(c5)
        p4 = self.lat4(c4) + nn.functional.interpolate(p5, scale_factor=2)
        p3 = self.lat3(c3) + nn.functional.interpolate(p4, scale_factor=2)
        p2 = self.lat2(c2) + nn.functional.interpolate(p3, scale_factor=2)
        return [p2, self.smooth3(p3), self.smooth4(p4), p5]
```

### Output Shapes:

Assuming input = (B, 3, 800, 800):

* P2 = (B, 256, 200, 200)
* P3 = (B, 256, 100, 100)
* P4 = (B, 256, 50, 50)
* P5 = (B, 256, 25, 25)

---

## 🛰️ 3. Region Proposal Network (RPN)

The RPN is a fully convolutional network that proposes candidate object bounding boxes.

### Architecture:

* Shared 3×3 conv → ReLU
* Branch 1: 1×1 conv to predict objectness scores
* Branch 2: 1×1 conv to predict bounding box regression deltas

```python
class RPNHead(nn.Module):
    def __init__(self, in_channels=256, num_anchors=9):
        super().__init__()
        self.shared = nn.Conv2d(in_channels, 256, 3, padding=1)
        self.cls_logits = nn.Conv2d(256, num_anchors * 2, 1)
        self.bbox_pred = nn.Conv2d(256, num_anchors * 4, 1)

    def forward(self, feats):
        cls_logits, bbox_preds = [], []
        for feat in feats:
            t = nn.functional.relu(self.shared(feat))
            cls_logits.append(self.cls_logits(t))
            bbox_preds.append(self.bbox_pred(t))
        return cls_logits, bbox_preds
```

### Anchor Matching and Labeling:

* Anchor boxes (default boxes) are created for each pixel location.
* Positive: IoU ≥ 0.7 with GT box
* Negative: IoU ≤ 0.3
* Ignore: Otherwise

### Similarity to SSD:

Both use anchors, but:

* SSD has multi-class classification per anchor
* RPN only classifies **object vs background**

### Loss:

```math
L_{\text{RPN}} = \frac{1}{N_{cls}} \sum \text{BCE}(p_i, p_i^*) + \lambda \frac{1}{N_{reg}} \sum p_i^* \text{SmoothL1}(t_i, t_i^*)
```

### Hard Negative Mining:

Selects highest-loss negatives to maintain ratio (e.g., 1:3 pos\:neg)

---

## 🧾 4. RoIAlign

RoIAlign improves over RoIPool by removing quantization. It precisely extracts fixed-size (7×7) feature maps from input regions.

### Key Idea:

* Divide RoI into bins (7×7)
* For each bin, sample at 4 positions
* Apply bilinear interpolation on feature map

Output shape: `(B, 256, 7, 7)` per RoI

---

## 🧠 5. Heads

### Box Head (Classification + Regression)

```python
class BoxHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256*7*7, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.cls = nn.Linear(1024, 81)
        self.bbox = nn.Linear(1024, 81 * 4)

    def forward(self, x):
        x = x.flatten(1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        return self.cls(x), self.bbox(x)
```

### Mask Head (Per-class Binary Masks)

```python
class MaskHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU()
        )
        self.upsample = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.mask = nn.Conv2d(256, 81, 1)  # K = 81

    def forward(self, x):
        x = self.convs(x)
        x = nn.functional.relu(self.upsample(x))
        return self.mask(x)  # (B, 81, 28, 28)
```

---

## 📉 6. Losses

### 6.1 Detection Classification:

```math
L_{cls} = - \sum_{i} y_i \log(p_i) \quad \text{(softmax cross-entropy)}
```

### 6.2 BBox Regression:

```math
L_{box} = \text{SmoothL1}(t_i, t_i^*)
```

### 6.3 Mask Segmentation:

* Only 1 out of K channels is trained per RoI (the GT class)
* So a softmax over channels is invalid

```math
L_{mask} = \frac{1}{m^2} \sum_{i,j} \text{BCE}(M_k[i,j], M_k^*[i,j])
```

### Why BCE, Not Softmax + CCE?

* CCE assumes a full probability distribution over classes per pixel
* But only one mask (class-k) is supervised per RoI
* BCE treats each mask channel independently

---

## ✅ Summary

* RPN has its own cls + reg heads like SSD
* Anchors matched using IoU, balanced with hard negative mining
* RoIAlign allows precise fixed-size feature extraction
* Only the GT class mask is supervised, so BCE is correct
* Softmax + CCE would require all K masks to compete, which is not the design here

---

Let me know when you want the training loop, dataloader example, or inference code.
