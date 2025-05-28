## 📚 Ensemble Learning: Bagging, Boosting, and Bootstrap Sampling

### Part 1: What is Bootstrapping in Ensembling?

**Bootstrapping** is the process of creating multiple new datasets by sampling from the original dataset **with replacement**.

* From an original dataset of size $N$, generate $B$ new datasets of the same size.
* Each dataset may include **duplicates** and **omit** some original samples.
* This is essential for **Bagging (Bootstrap Aggregating)** and often used in **Random Forests**.

#### Why Bootstrap?

* To train multiple **diverse models**.
* To reduce **variance** through aggregation.
* To simulate different training distributions from a single dataset.

#### Example

Original dataset:

```
ID | Feature | Label
-------------------
x1 |   3     |  A
x2 |   4     |  B
x3 |   5     |  A
x4 |   2     |  A
x5 |   6     |  B
```

Bootstrap Sample 1:

```
x2, x4, x2, x3, x5  ← x2 appears twice, x1 is omitted
```

Bootstrap Sample 2:

```
x3, x1, x3, x5, x4  ← x3 appears twice
```

### How are Duplicated Samples Treated?

* Samples are treated **independently** — each occurrence counts as a separate data point.
* If a sample like $x_2$ appears twice in the same bootstrapped dataset:

  * The model sees it as two separate instances.
  * Its influence on the training is effectively **doubled**.

This has the effect of **weighting** some samples more heavily (depending on how many times they are drawn).

### Theoretical Insight

* On average, each bootstrap sample contains about **63.2%** of the unique data points from the original dataset.

  * The probability a specific sample is not picked: $(1 - 1/N)^N \approx e^{-1} \approx 0.368$
  * So the chance a sample is included at least once: $1 - e^{-1} \approx 63.2\%$

#### Why this matters:

* It provides **natural variation** across training datasets.
* We can use the omitted samples (the remaining \~36.8%) to evaluate each model — this is called **Out-of-Bag (OOB) error estimation**.
* This allows validation **without needing a separate validation set**.

---

### Part 2: What is Bagging (Bootstrap Aggregating)?

**Bagging** is an ensemble learning technique that combines the predictions of multiple base models, each trained on a different **bootstrap sample** of the data.

#### Goal:

* Reduce **variance** of a model (especially high-variance models like decision trees).
* Improve **generalization** by averaging or voting across multiple models.

### How Bagging Works

1. Given training dataset $D$ with $N$ samples.
2. Generate $B$ bootstrap samples: $D_1, D_2, \dots, D_B$.
3. Train a base model $M_b$ on each $D_b$.
4. Aggregate the predictions from all models:

   * **Classification**: majority vote
   * **Regression**: average of outputs

### Pseudocode

```python
# Inputs: dataset D, number of models B, base_learner
models = []
for b in range(B):
    D_b = bootstrap_sample(D)  # sample with replacement
    model = base_learner.train(D_b)
    models.append(model)

def predict(x):
    preds = [m.predict(x) for m in models]
    return majority_vote(preds)  # or mean(preds) for regression
```

### Key Properties

* Works best with **unstable models** (like decision trees), where small data changes lead to different predictions.
* Each model sees a different subset of the data → ensemble becomes robust.
* **Out-of-Bag (OOB) error**: For each sample, average the predictions of models that didn’t train on it.

### Example

Say we create 5 bootstraps from the dataset of 100 samples, and train 5 decision trees. Then:

* Each tree sees ≈ 63 samples (due to bootstrapping).
* During inference, we get 5 predictions.
* If it's a classification task → use majority vote.
* If it's regression → take average of 5 outputs.


### Part 3: Mathematical Insights on Bagging

#### 1. Why Does Ensembling Reduce Variance?

Let $f_1(x), f_2(x), \ldots, f_B(x)$ be predictions of $B$ base learners trained on different bootstrapped datasets.

Define the ensemble prediction as:

$$
\bar{f}(x) = \frac{1}{B} \sum_{b=1}^{B} f_b(x)
$$

If each $f_b(x)$ has variance $\sigma^2$ and the models are **independent**, then:

$$
\text{Var}(\bar{f}(x)) = \frac{\sigma^2}{B}
$$

So as $B \to \infty$, the variance of the ensemble prediction approaches 0. Even when models are **correlated**, the ensemble still reduces variance:

$$
\text{Var}(\bar{f}(x)) = \frac{1}{B^2} \sum_{i=1}^{B} \sum_{j=1}^{B} \text{Cov}(f_i, f_j)
$$

This is why Bagging is effective for high-variance models like decision trees — averaging multiple overfit models leads to a smoother, more stable prediction.

---

#### 2. Why is the Probability of a Sample Not Picked ≈ 36.8%?

Let’s say you have $N$ samples, and you draw $N$ times **with replacement** to create a bootstrap sample.

For any specific sample $x_i$:

* The chance it’s **not picked** in one draw = $1 - \frac{1}{N}$
* The chance it’s **not picked** in $N$ draws =

$$
\left(1 - \frac{1}{N}\right)^N \approx e^{-1} \approx 0.368
$$

This uses the limit identity:

$$
\lim_{N \to \infty} \left(1 - \frac{1}{N}\right)^N = \frac{1}{e}
$$

So, about **36.8%** of the original dataset will **not** appear in a given bootstrap sample. This is why on average, **63.2%** of the data is included **at least once**.

This insight is fundamental to understanding how Bagging builds diversity, and how **OOB samples** can serve as an internal validation mechanism.

---

### Part 3B: Variance of the Ensemble with Correlated Predictors

Assume:

* Each model $f_i(x)$ has variance $\sigma^2$
* The average pairwise correlation between predictions is $\rho$

Then:

* $\text{Cov}(f_i, f_j) = \rho \sigma^2$ for $i \ne j$
* $\text{Var}(f_i) = \sigma^2$

Variance of the ensemble prediction:

$$
\text{Var}(\bar{f}(x)) = \text{Var}\left(\frac{1}{B} \sum_{i=1}^B f_i(x)\right) = \frac{1}{B^2} \sum_{i=1}^B \sum_{j=1}^B \text{Cov}(f_i, f_j)
$$

Separate diagonal and off-diagonal terms:

$$
= \frac{1}{B^2} \left( B \cdot \sigma^2 + B(B - 1) \cdot \rho \sigma^2 \right)
= \frac{\sigma^2}{B} (1 + (B - 1) \rho)
$$

**Interpretation:**

* If $\rho = 0$: $\text{Var}(\bar{f}) = \frac{\sigma^2}{B}$ → maximum variance reduction.
* If $\rho = 1$: $\text{Var}(\bar{f}) = \sigma^2$ → no reduction; models are redundant.

This shows that **lower correlation** between predictors increases the benefit of ensembling — motivating methods like Random Forests to inject feature randomness.

