# The Perceptron — Theory to Implementation

---

## 1. Biological Inspiration

The perceptron (Rosenblatt, 1958) is loosely modeled on a biological neuron. A neuron receives electrical signals through dendrites, accumulates them in the cell body, and fires an output signal down the axon only when the accumulated signal exceeds a threshold.

The perceptron is the mathematical abstraction of that idea:

- **Dendrites** → input features `x₁, x₂, ..., xₙ`
- **Synaptic strength** → weights `w₁, w₂, ..., wₙ`
- **Cell body accumulation** → weighted sum `z = Σ wᵢxᵢ + b`
- **Firing threshold** → step activation function
- **Axon output** → prediction `ŷ ∈ {0, 1}`

---

## 2. Mathematical Model

### 2.1 Inputs and Weights

A perceptron with `n` inputs is defined by:

- An input vector **x** = `[x₁, x₂, ..., xₙ]`
- A weight vector **w** = `[w₁, w₂, ..., wₙ]` (one weight per input)
- A scalar bias `b`

Each input `xᵢ` is a feature of a data point (e.g., for a 2-input AND gate, `x₁` and `x₂` are the two bits).

### 2.2 The Weighted Sum (Pre-activation)

The perceptron first computes the **pre-activation** `z`, also called the net input or logit:

```
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
  = w · x + b          (dot product notation)
```

The bias `b` is a learned offset that shifts the decision boundary away from the origin. Without it, the decision boundary is always forced to pass through the origin, which severely limits what the model can represent.

You can equivalently think of the bias as an extra weight `w₀` attached to a constant input `x₀ = 1`. This is a common notational simplification in textbooks.

### 2.3 The Step Activation Function

The pre-activation `z` is passed through a **Heaviside step function** to produce a binary output:

```
ŷ = step(z) = 1  if z ≥ 0
              0  otherwise
```

Geometrically, `z = 0` defines a hyperplane in n-dimensional input space. The step function maps everything on one side to 1 and everything on the other side to 0. This hyperplane **is** the decision boundary.

For 2 inputs (n=2), this is a straight line separating the 2D plane into two half-planes.

---

## 3. The Perceptron Learning Rule

This is the core algorithm. Given a labeled training example `(x, y)` where `y ∈ {0, 1}`:

### 3.1 Error Signal

```
error = y - ŷ
```

There are only three possible values:
- `error = 0` → prediction was correct, do nothing
- `error = +1` → predicted 0, actual was 1 (false negative) → push weights up
- `error = -1` → predicted 1, actual was 0 (false positive) → push weights down

### 3.2 Weight Update

```
wᵢ ← wᵢ + η × error × xᵢ      for each i
```

Where `η` (eta) is the **learning rate**, a small positive scalar (e.g., 0.1).

**Why this makes sense:**
- If `error = 0`, no update happens.
- If `error = +1` (we should have fired but didn't): we increase weights proportionally to the input. Features that were large get a bigger bump, making us more likely to fire on similar inputs next time.
- If `error = -1` (we fired when we shouldn't have): we decrease weights proportionally to the input.
- The `xᵢ` term means **irrelevant features (xᵢ = 0) don't get their weights changed**. Only features that actually contributed to the wrong prediction are corrected.

### 3.3 Bias Update

```
b ← b + η × error
```

The bias update is just the weight update without the `xᵢ` factor, because the bias corresponds to a constant input of 1.

### 3.4 Learning Rate `η`

- Too large: unstable, overshoots the correct boundary and oscillates
- Too small: converges correctly but very slowly
- Typical values: 0.01 to 0.5

---

## 4. Linear Separability

The perceptron can **only learn linearly separable problems** — problems where a straight line (hyperplane) can divide the positive class from the negative class.

### 4.1 AND Gate (linearly separable ✓)

| x₁ | x₂ | y |
|----|----|----|
| 0  | 0  | 0  |
| 0  | 1  | 0  |
| 1  | 0  | 0  |
| 1  | 1  | 1  |

A line can separate (1,1) from the rest.

### 4.2 OR Gate (linearly separable ✓)

| x₁ | x₂ | y |
|----|----|----|
| 0  | 0  | 0  |
| 0  | 1  | 1  |
| 1  | 0  | 1  |
| 1  | 1  | 1  |

A line can separate (0,0) from the rest.

### 4.3 XOR Gate (NOT linearly separable ✗)

| x₁ | x₂ | y |
|----|----|----|
| 0  | 0  | 0  |
| 0  | 1  | 1  |
| 1  | 0  | 1  |
| 1  | 1  | 0  |

No single straight line can separate (0,1) and (1,0) from (0,0) and (1,1). This is the fundamental limitation of the single-layer perceptron, famously described by Minsky & Papert (1969). Solving XOR requires a multi-layer network (Phase 4 of the roadmap).

---

## 5. The Perceptron Convergence Theorem

If the training data is linearly separable, the perceptron learning rule is **guaranteed to converge** to a perfect classifier in a finite number of steps.

Key conditions:
1. The data must be linearly separable.
2. The learning rate must be positive.

If the data is not linearly separable, the algorithm will cycle indefinitely — it never settles. This is why we'll later switch to sigmoid + MSE loss (Phase 2), which converges even for non-separable data (to the best linear approximation).

---

## 6. Implementation Walkthrough

The implementation lives at `src/main/scala/com/rohin/ann/Perceptron.scala`.

### 6.1 Design Decisions

**Private constructor, companion object as the public API.** The `case class` has a `private` constructor so the only way to build a `Perceptron` is through the companion:

```scala
case class Perceptron private (
    weights: RealVector,
    bias: Double,
    learningRate: Double
)
```

All operations — `predict`, `trainOne`, `train`, `updateAllWeights`, `updateBias` — live in the companion object as **extension methods**. This keeps the data type lean (pure data) while still allowing natural method-call syntax (`perceptron.predict(input)`). Tests import everything with `import Perceptron.*`.

**`learningRate` on the model.** It is part of the model's identity, not a transient training parameter. `copy(...)` carries it through every update automatically.

**Factory helpers on the companion.** `vec(xs: Double*)` and `zeroVec(size: Int)` wrap `ArrayRealVector` so neither tests nor callers ever need to import commons-math3 directly.

### 6.2 Smart constructors — `create`

Two overloads, one with a sensible default learning rate:

```scala
def create(bias: Double, weights: RealVector): Perceptron          // lr = 0.001
def create(bias: Double, weights: RealVector, learningRate: Double) // explicit lr
```

Typical usage in tests (with `import Perceptron.*`):

```scala
val p = create(bias = 0, weights = vec(1, 1), learningRate = 0.1)
val q = create(bias = -1, weights = zeroVec(2))   // uses default lr
```

### 6.3 `step` — Activation Function

```scala
def step(z: Double): Int =
  if (z >= 0) 1 else 0
```

Public on the companion — it is a pure math function, independently testable and reusable in future activation comparisons. `z = 0` maps to 1: the boundary hyperplane belongs to the positive class (Heaviside convention).

### 6.4 `predict` — Forward Pass

```scala
extension (perceptron: Perceptron)
  def predict(input: RealVector): Int =
    step(perceptron.weights.dotProduct(input) + perceptron.bias)
```

`RealVector.dotProduct` from commons-math3 computes the weighted sum natively and throws `DimensionMismatchException` automatically on a size mismatch — no manual validation needed.

### 6.5 `updateAllWeights` — Weight Update Step

```scala
extension (perceptron: Perceptron)
  def updateAllWeights(step: Double, ys: RealVector): Perceptron =
    perceptron.copy(
      weights = perceptron.weights.combine(1, step, ys)
    )
```

`RealVector.combine(a1, a2, v)` computes `a1·this + a2·v`. Called as `combine(1, updateMagnitude, input)` this gives:

```
weights_new = 1·weights + (η·error)·input
            = weights + η·error·input
```

This is the perceptron weight update rule expressed as a single native vector operation.

### 6.6 `updateBias` — Bias Update Step

```scala
extension (perceptron: Perceptron)
  def updateBias(updateMagnitude: Double): Perceptron =
    perceptron.copy(bias = perceptron.bias + updateMagnitude)
```

Decomposed into its own named method so `trainOne` reads as a sequence of focused steps rather than one big expression.

### 6.7 `trainOne` — One Example Update

```scala
extension (perceptron: Perceptron)
  def trainOne(input: RealVector, actual: Int): Perceptron =
    val predicted = perceptron.predict(input)
    val error = actual - predicted

    if error == 0 then perceptron
    else
      val updateMagnitude = perceptron.learningRate * error
      perceptron
        .updateAllWeights(updateMagnitude, input)
        .updateBias(updateMagnitude)
```

Short-circuits on `error == 0` — returns `this` with no allocation when the prediction is already correct. Otherwise chains the two update steps. `updateMagnitude = η × error` is computed once and shared by both, which makes the relationship between weight and bias update explicit.

### 6.8 `train` — Full Training Run

```scala
extension (perceptron: Perceptron)
  def train(dataset: List[(RealVector, Int)], epochs: Int): Perceptron =
    @tailrec
    def loop(iterate: Int, p: Perceptron): Perceptron =
      if (iterate == 0) then p
      else loop(iterate - 1, dataset.foldLeft(p) { case (acc, (input, actual)) =>
        acc.trainOne(input, actual)
      })
    loop(epochs, perceptron)
```

`@tailrec` inner loop — safe for large epoch counts with no stack risk. Each iteration runs one full epoch (a `foldLeft` over the entire dataset), then recurses with `epochs - 1`. The caller controls exactly how many epochs to run:

```scala
val trained = initial.train(data, epochs = 10)
```

---

## 7. Current State

All previously known issues are resolved in the current implementation:

| Area | Resolution |
|---|---|
| Dimension validation | `RealVector.dotProduct` throws `DimensionMismatchException` automatically |
| Weight update math | `combine(1, η·error, input)` — single native vector op from commons-math3 |
| Allocation on correct prediction | `trainOne` returns `this` when `error == 0` |
| Training API | `train(dataset, epochs)` — explicit epoch count, tail-recursive, stack-safe |
| Code organisation | Data in `case class`, all behaviour in companion as extension methods |

---

## 8. What's Next — Phase 2

The natural next step is to replace the step activation with **sigmoid** and introduce a loss function:

- `sigmoid(z) = 1 / (1 + exp(-z))` — differentiable, output is a probability in (0, 1)
- Loss: Mean Squared Error `L = (1/n) Σ (y - ŷ)²`
- Gradient descent instead of the binary perceptron rule

This unlocks continuous outputs and the ability to make progress on non-linearly-separable data, setting up gradient flow for multi-layer networks in Phase 4.
