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

$$
L_{\text{RPN}} = \frac{1}{N_{cls}} \sum \text{BCE}(p_i, p_i^*) + \lambda \frac{1}{N_{reg}} \sum p_i^* \text{SmoothL1}(t_i, t_i^*)
$$

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


Let's walk through the **complete flow from RPN to the final detection and classification heads** in Mask R-CNN, step by step, with **mathematical formulations** and **label generation logic**.

---

# 📊 Flow: From RPN → RoIAlign → Detection Head

---

## 🛰️ 1. RPN (Region Proposal Network)

### 🔹 Input

* Multi-scale FPN feature maps: `[P2, P3, P4, P5]` (shape: B×256×H×W per level)

### 🔹 Anchor Generation

* At each location in each FPN level, generate `k` anchors.

  * E.g. 3 scales × 3 aspect ratios = `k=9`
* Total number of anchors: $A = \sum_l H_l \cdot W_l \cdot k$

### 🔹 Predictions

For each anchor $a_i$, the RPN outputs:

* Objectness score $p_i \in [0,1]$
* Box deltas
$$
t_i = (\hat{t}_{x_i}, \hat{t}_{y_i}, \hat{t}_{w_i}, \hat{t}_{h_i})
$$

### 🔹 Training Labels for RPN

Each anchor $a_i$ is:

* Assigned a **label** $p_i^* \in \{0, 1, -1\}$

  * 1 → positive (IoU ≥ 0.7 with a GT box)
  * 0 → negative (IoU ≤ 0.3 with all GT boxes)
  * -1 → ignore (between thresholds)
* Assigned a **GT box** $g_i = (x_{gt}, y_{gt}, w_{gt}, h_{gt})$
* Compute **regression target deltas**:

$$
t_{xi}^* = \frac{x_{gt} - x_{a}}{w_a},\quad
t_{yi}^* = \frac{y_{gt} - y_{a}}{h_a}. \quad
t_{wi}^* = \log\left(\frac{w_{gt}}{w_a}\right),\quad
t_{hi}^* = \log\left(\frac{h_{gt}}{h_a}\right)
$$

### 🔹 RPN Loss

Let $N_{cls}$ and $N_{reg}$ be the number of samples:

$$
L_{RPN} = \frac{1}{N_{cls}} \sum_i \text{BCE}(p_i, p_i^*) + \lambda \frac{1}{N_{reg}} \sum_i p_i^* \cdot \text{SmoothL1}(t_i, t_i^*)
$$

Only **positive anchors** contribute to the regression loss.

---

## 📦 2. RoI Generation (Region Proposals)

### After RPN:

* Apply predicted deltas to anchors to get proposal boxes:

$$
x_p = \hat{t}_x \cdot w_a + x_a,\quad
y_p = \hat{t}_y \cdot h_a + y_a,\quad
w_p = w_a \cdot e^{\hat{t}_w},\quad
h_p = h_a \cdot e^{\hat{t}_h}
$$

* Apply **NMS** (e.g. IoU threshold 0.7)
* Keep top-N proposals (e.g. 1000 during training, 300 during test)
* These boxes are the **RoIs**

---

## 🧾 3. RoIAlign

### For each RoI:

* Map the proposal box to its corresponding feature map level $P_l$

  * Usually done using:

    $$
    l = \lfloor 4 + \log_2(\sqrt{wh} / 224) \rfloor
    $$
* Crop the feature map using **bilinear interpolation** into shape (B, 256, 7, 7)
# Each ROI is pooled out of the feature maps and create a 256 x 7 x 7 feature map, we have 1000 ROIs per image. 
### Output:

Tensor of shape (N, 256, 7, 7) — N = total number of RoIs across all images in the batch 

---

## 🎯 4. Detection Head (Classification + BBox Regression)

### Inputs:

* RoI-aligned features: $R_i \in \mathbb{R}^{256 \times 7 \times 7}$

### Network:

* Two FC layers → feature vector $f_i \in \mathbb{R}^{1024}$
* Heads:

  * **Classification**: $\hat{p}_i \in \mathbb{R}^{K+1}$ → softmax over classes
  * **Regression**: $\hat{t}_i \in \mathbb{R}^{(K+1) \times 4}$ → per-class bbox deltas

---

## 🏷️ 5. Training Labels for Detection Head

For each RoI $r_i$:

### 1. Match to GT box using IoU

* If IoU ≥ 0.5 → **positive**

  * Assign class label $c_i \in [1, K]$
  * Assign matched GT box $g_i$
* If IoU < 0.5 → **background**

  * Assign label $c_i = 0$
  * No regression target

### 2. Regression Target Deltas

For positive RoIs $r_i = (x_r, y_r, w_r, h_r)$:

$$
t_{xi}^* = \frac{x_{gt} - x_r}{w_r},\quad
t_{yi}^* = \frac{y_{gt} - y_r}{h_r}, \quad
t_{wi}^* = \log\left(\frac{w_{gt}}{w_r}\right),\quad
t_{hi}^* = \log\left(\frac{h_{gt}}{h_r}\right)
$$
Note: These are class-specific — only the **GT class channel** is trained.

---

## 🧮 6. Detection Loss

Let:

* $\hat{p}_i$: predicted class scores
* $c_i$: GT class
* $\hat{t}_{c_i}$: predicted deltas for class $c_i$
* $t_i^*$: GT deltas for RoI $i$

Then the **total loss**:

$$
L = \frac{1}{N_{cls}} \sum_i \text{CE}(\hat{p}_i, c_i) +
    \frac{1}{N_{reg}} \sum_i \mathbb{1}_{[c_i > 0]} \cdot \text{SmoothL1}(\hat{t}_{c_i}, t_i^*)
$$

* Classification: Cross-Entropy over K+1 classes
* Regression: Smooth L1 for **positive** RoIs only

---

## ✅ Summary Diagram

```
Anchors → RPN (cls + bbox) → Deltas + Scores → Proposals (RoIs)
     → RoIAlign (7x7x256)
         → FC → Classification Head → Class probs
         → FC → BBox Head → K-class regression
```



## 📉 6. Losses

### 6.1 Detection Classification:
Excellent question. Let's clarify how **detection** is performed in Mask R-CNN and whether **anchors** are involved at that stage.

---

**The detection head does not use anchors.**
Anchors are only used in the **RPN** (Region Proposal Network). The detection head operates on **refined RoIs** (region proposals) generated from RPN outputs.



#### 1. **Anchors Are Used Only in RPN**

* Anchors are generated at each location of FPN feature maps.
* For each anchor:

  * RPN predicts: objectness score + bbox regression offsets.
* Top scoring boxes (after NMS) are selected as **region proposals (RoIs)**.

These RoIs are then passed to the next stage — **the detection head**.

---

#### 2. **Detection Head Receives Aligned RoIs**

* RoIs are **refined bounding boxes**, not anchor templates.
* They are extracted via **RoIAlign** into fixed-size features (e.g., 7×7×256).
* These features are passed to the detection head.

---

### 📦 Detection Head

#### Inputs:

* RoI-aligned features of shape `(B, 256, 7, 7)`

#### Architecture:

* 2 Fully Connected (FC) layers
* Output:

  * **Classification logits** over `K` object classes
  * **Class-specific bbox deltas** (81 × 4 values in COCO)

#### Post-processing:

* Apply bbox deltas to RoIs to get refined boxes.
* Run **softmax** over classification logits.
* Apply **NMS** per class to suppress redundant detections.

---


| Stage              | Uses Anchors? | Description                                                  |
| ------------------ | ------------- | ------------------------------------------------------------ |
| **RPN**            | ✅ Yes         | Anchors + regression → Proposals                             |
| **Detection Head** | ❌ No          | Operates on RPN outputs (RoIs), classifies and refines boxes |

So detection is **not anchor-based**, but rather **proposal-based**, which are refined anchor regressions.

Let me know if you'd like to insert this explanation into the tutorial file.

$$
L_{cls} = - \sum_{i} y_i \log(p_i) \quad \text{(softmax cross-entropy)}
$$

### 6.2 BBox Regression:

$$
L_{box} = \text{SmoothL1}(t_i, t_i^*)
$$

### 6.3 Mask Segmentation:

* Only 1 out of K channels is trained per RoI (the GT class)
* So a softmax over channels is invalid

$$
L_{mask} = \frac{1}{m^2} \sum_{i,j} \text{BCE}(M_k[i,j], M_k^*[i,j])
$$
### Why Mask R-CNN Uses BCE Instead of Softmax + Categorical Cross-Entropy (CCE)

Mask R-CNN uses **Binary Cross-Entropy (BCE)** for its mask head loss, not softmax with categorical cross-entropy (CCE). Here's why:

---

### 🧩 Architecture Choice: One Binary Mask per Class

* The **mask head** outputs `K` channels (e.g., `K=81` for COCO), each of size `28x28`.
* Each channel represents a **binary mask** for one class.
* During training, **only the channel corresponding to the ground-truth class** is supervised.

  * For example, if a given RoI is labeled as “person” (class 1), only the 2nd mask is trained using the binary mask for “person”.
  * The other 80 channels are ignored.
  * This is because in general ROIs might overlap and segmentation masks for different ROIS might overlap. So it is not a categorical segmentation it could be multi labels.
  * In general for each ROI we calculated the mask loss separately.

---

### 🚫 Why Softmax + CCE Would Be Invalid

Softmax + categorical cross-entropy assumes:

1. All output channels represent **mutually exclusive** class probabilities.
2. A softmax across channels produces a **normalized distribution** over all `K` classes at each pixel.
3. You supervise **every pixel** to predict **exactly one** of the `K` classes.

But in Mask R-CNN:

* You **do not train all `K` channels**. Only the one for the GT class is used.
* The other channels are not supervised, so the softmax is ill-defined.
* This violates the assumption required for softmax + CCE to work (i.e., supervision for all classes per pixel).

---

### ✅ Why BCE Works for Mask R-CNN

* BCE is applied **per-pixel**, **per-class**, **independently**.
* It models: *“Is this pixel foreground for this class?”* (binary yes/no).
* Since we train only the GT class channel, BCE perfectly fits this logic.
* It allows us to treat each mask as a separate binary segmentation task.

---

### 🚨 What Happens If You Use Softmax + CCE Anyway?

If you incorrectly apply softmax across channels:

* It would force the network to produce **a probability distribution over all classes** per pixel.
* Since only one class is being supervised (GT class), the model has **no ground truth** to supervise the other `K-1` classes.
* This leads to **unstable training**, **false gradients**, and degraded performance.

In short:
**Softmax + CCE is semantically wrong for per-RoI binary masks** trained only for GT class.
**BCE is correct** because it models the actual training behavior: *1 mask per GT class per RoI*.

---

To redesign Mask R-CNN to use **softmax over mask outputs** with **categorical cross-entropy (CCE)**, you must change **how masks are predicted and supervised**. This fundamentally alters the mask head architecture and training logic.

---

### ✅ Objective: Softmax Mask Prediction with CCE

We want each pixel in the predicted mask to output a **single class label** (as in semantic segmentation). This requires:

| Requirement                  | Current Mask R-CNN           | Modified Design (Softmax)      |
| ---------------------------- | ---------------------------- | ------------------------------ |
| Per-pixel prediction         | Binary (1 per class)         | Categorical (1-of-K)           |
| Channels in mask head output | `K` binary masks (B×K×28×28) | 1 categorical mask (B×K×28×28) |
| Supervised channels          | Only GT class mask           | All pixels with class label    |
| Loss                         | BCE on GT class              | CCE over softmax(K)            |

---

### 🔧 Design Changes Required

#### 1. **Mask Head Output Remains (B, K, 28, 28)**

No change to the output shape — we still predict `K` channels per RoI.

#### 2. **Supervision Must Change**

You must now provide a **categorical mask label** for each pixel in the RoI crop. That is, instead of training a binary mask only for the GT class, you now train a **full pixel-wise class map** with values in `[0, ..., K-1]`.

This is hard because:

* Each RoI corresponds to **only one object**, which is of **a single class**.
* There is no semantic reason for pixels within an RoI to have multiple class labels.

To make this work:

* You’d need to **merge overlapping masks** and assign **per-pixel class labels** (like semantic segmentation).
* This makes it **no longer an instance segmentation problem**, but **semantic segmentation**.

#### 3. **Loss Function**

Replace BCE with softmax + cross-entropy:

```python
# logits: (B, K, 28, 28)
# targets: (B, 28, 28) with values in [0, K-1]
loss = nn.CrossEntropyLoss()(logits, targets)
```

#### 4. **Mask Target Construction**

Instead of a binary mask per RoI:

* You would need to project instance masks into a shared canvas per RoI.
* And resolve overlaps into a single class label per pixel.

This contradicts the core *instance-level* logic of Mask R-CNN.

---

### 🚫 Why This is Usually Not Done

* RoIs are defined per instance, not for the whole image.
* There is **no natural multi-class pixel label** per RoI.
* The model should only answer: *“Where is the mask for this object?”*, not *“Which class does each pixel belong to?”* — that's a **semantic segmentation** task, not instance segmentation.

---

### ✅ Summary

To support softmax + CCE for segmentation masks in Mask R-CNN:

| Step | Change Required                                       |
| ---- | ----------------------------------------------------- |
| 1    | Mask targets must be class labels per pixel           |
| 2    | Mask loss becomes CCE instead of BCE                  |
| 3    | Every mask channel must be supervised jointly         |
| 4    | Overlapping instances must be resolved by pixel class |

But this **destroys instance separation**, which is the entire point of Mask R-CNN.
This change essentially turns the model into a **semantic segmentation network**, not an instance segmentation one.

---





## ✅ Summary

* RPN has its own cls + reg heads like SSD
* Anchors matched using IoU, balanced with hard negative mining
* RoIAlign allows precise fixed-size feature extraction
* Only the GT class mask is supervised, so BCE is correct
* Softmax + CCE would require all K masks to compete, which is not the design here

---

