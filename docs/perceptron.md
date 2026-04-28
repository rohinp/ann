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

The implementation lives at `src/main/scala/com/rohin/ann/Percentron.scala`.

### 6.1 Design Decisions

**Immutable case class.** The `Perceptron` is a `final case class`. Training does not mutate state; `trainOne` returns a new `Perceptron`. This is idiomatic Scala and makes reasoning about the model trivial — you can hold onto any intermediate state, compose training steps with `foldLeft`, and test pure functions without setup/teardown.

```scala
final case class Perceptron(
    weights: Vector[Double],
    bias: Double,
    learningRate: Double
)
```

**`learningRate` on the model.** The learning rate is part of the model's identity, not a transient training parameter. This means you can create differently-configured perceptrons and compare them without passing `η` into every training call.

### 6.2 `dot` — The Weighted Sum

```scala
private def dot(a: Vector[Double], b: Vector[Double]): Double =
  a.zip(b).map { case (x, y) => x * y }.sum
```

Computes `Σ aᵢbᵢ`. Currently uses Scala's `Vector`, which allocates intermediate collections. This will be replaced with `RealVector.dotProduct` from commons-math3, which operates on raw arrays with no allocation overhead.

**Current bug:** `zip` silently truncates to the shorter vector if lengths differ. A mismatched input produces a wrong (but not erroring) result. Dimension validation must be added.

### 6.3 `step` — Activation Function

```scala
private def step(z: Double): Int =
  if z >= 0 then 1 else 0
```

`z = 0` is classified as 1. This is a valid convention (Heaviside step). It means the boundary hyperplane itself belongs to the positive class.

### 6.4 `predict` — Forward Pass

```scala
def predict(input: Vector[Double]): Int = {
  val z = dot(weights, input) + bias
  step(z)
}
```

The complete forward pass: compute pre-activation, apply activation. Clean and direct.

### 6.5 `trainOne` — One Weight Update

```scala
def trainOne(input: Vector[Double], actual: Int): Perceptron = {
  val prediction = predict(input)
  val error = actual - prediction

  val updatedWeights = weights.zip(input).map { case (w, x) =>
    w + learningRate * error * x
  }
  val updatedBias = bias + learningRate * error

  copy(weights = updatedWeights, bias = updatedBias)
}
```

Implements the perceptron learning rule exactly. Returns a new `Perceptron`. Notice `copy(...)` — it reuses `learningRate` unchanged, updating only `weights` and `bias`.

**Missing:** A `train` method that runs one full epoch over a dataset:

```scala
def train(data: Seq[(Vector[Double], Int)]): Perceptron =
  data.foldLeft(this) { case (p, (input, label)) => p.trainOne(input, label) }
```

This is the idiomatic way to fold an immutable model over a dataset in Scala.

---

## 7. Known Issues in the Current Implementation

| Issue | Location | Impact |
|---|---|---|
| No dimension validation | `dot`, `trainOne` | Silent wrong results on mismatched inputs |
| No `train` method | — | Caller must manually fold over dataset |
| Uses `Vector[Double]` not `RealVector` | `dot`, weights field | Missed commons-math3 integration, allocates intermediates |
| `dot` re-implements `RealVector.dotProduct` | `dot` | Unnecessary, remove once commons-math3 is used |
| Filename is `Percentron.scala` | file | Typo — class is `Perceptron`, file should match |

---

## 8. Next Steps (Planned Redesign)

Replace `Vector[Double]` with `org.apache.commons.math3.linear.ArrayRealVector`:

```scala
import org.apache.commons.math3.linear.{ArrayRealVector, RealVector}

final case class Perceptron(
    weights: RealVector,
    bias: Double,
    learningRate: Double
)
```

This gives:
- `weights.dotProduct(input)` — native, zero-allocation dot product
- `weights.add(delta)` — vector arithmetic without `zip/map`
- `weights.mapMultiply(scalar)` — scalar multiplication
- Dimension mismatch throws `DimensionMismatchException` automatically

The step function and learning rule logic remain the same — only the representation changes.
