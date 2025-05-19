Here’s a simplified set of **rules for broadcasting in Python (NumPy):**

## ✅ Broadcasting: 3 Golden Rules

### 🔹 Rule 1: Compare from **trailing dimensions**

Start from the **rightmost dimension**, and work left. Dimensions are compatible if:

* they are equal, or
* one of them is 1

### 🔹 Rule 2: **Missing dimensions are treated as 1**

If arrays have different numbers of dimensions, the smaller one is **left-padded** with 1s.

### 🔹 Rule 3: Expand size 1 to match

If a dimension is 1 in one array and some number $N$ in the other, the 1 is **broadcast (stretched)** to match $N$.

---

## 🧪 Example

```python
A = np.ones((3, 1))       # shape: (3, 1)
B = np.array([10, 20, 30])  # shape: (3,) → broadcast to (1, 3)
C = A * B  # result shape: (3, 3)
```

### ✅ Why?

* A: (3, 1)
* B: → reshaped to (1, 3)
* Result: dimensions (3, 3)

---
