# Two-Layer Neural Network — Theory to Implementation

---

## 1. Why One Layer Is Not Enough

The perceptron draws a single straight line (hyperplane) through the input space. Any problem whose positive and negative examples can be separated by such a line is *linearly separable*, and the perceptron is guaranteed to solve it.

XOR is not linearly separable:

```
  x₂
  1 │  ★ (0,1)         · (1,1)
    │
  0 │  · (0,0)         ★ (1,0)
    └───────────────────────── x₁

  ★ = output 1    · = output 0
```

The two positive examples sit on the diagonal and no straight line can separate them from the two negative examples on the other diagonal.

The fix: add a **hidden layer**. The hidden layer learns a new representation of the input — it warps the input space — and the output neuron draws its line in that transformed space. After the hidden layer learns the right transformation, XOR becomes linearly separable in the new coordinate system.

---

## 2. From Step to Sigmoid

The perceptron uses a step function: output is exactly 0 or 1, and the function has zero gradient everywhere (and is undefined at zero). This makes it impossible to compute how much a weight change affects the output — there is no gradient to follow.

**Sigmoid** replaces it:

```
σ(z) = 1 / (1 + exp(-z))
```

Properties:
- Output is in (0, 1) — a continuous probability-like value
- Differentiable everywhere
- Derivative has a closed form: `σ'(z) = σ(z) · (1 − σ(z))`
- Equivalently, if `a = σ(z)` is already computed: `σ'(z) = a · (1 − a)` — no re-computation of σ needed

The derivative is always positive (sigmoid is monotone), small near the extremes (where the neuron is saturated), and largest at z = 0. This "squashing" of gradients near saturation is the origin of the *vanishing gradient* problem in deeper networks, but for a two-layer net it is fine.

---

## 3. Network Architecture

The two-layer network has:

- An input layer (no learnable parameters — just the raw features **x**)
- A hidden layer of `h` neurons
- An output layer of 1 neuron

Parameters:

| Symbol | Type | Shape | Role |
|--------|------|-------|------|
| W₁ | RealMatrix | n × h | hidden layer weights |
| b₁ | RealVector | h | hidden layer biases |
| W₂ | RealVector | h | output layer weights |
| b₂ | Double | 1 | output bias |

Where n = number of inputs, h = number of hidden neurons.

```
  Input x          Hidden layer              Output
  ─────────        ────────────────────      ──────
  x₁ ──────┐       ╔══════╗
           ├──W₁──►║  σ   ║ → a₁[0] ──┐
  x₂ ──────┘  +b₁  ╠══════╣           ├──W₂──► σ(z₂) → ŷ
                    ║  σ   ║ → a₁[1] ──┘   +b₂
                    ╚══════╝
                    (h neurons)
```

For XOR: n = 2 inputs, h = 2 hidden neurons, 1 output.

---

## 4. Forward Pass

Given an input vector **x**:

**Step 1 — Hidden layer pre-activation:**
```
z₁ = W₁ · x + b₁        (matrix-vector multiply + bias, result is RealVector of length h)
```

**Step 2 — Hidden layer activation:**
```
a₁ = σ(z₁)              (applied element-wise, result is RealVector of length h)
```

**Step 3 — Output pre-activation:**
```
z₂ = W₂ · a₁ + b₂       (dot product + scalar bias, result is Double)
```

**Step 4 — Output activation:**
```
a₂ = σ(z₂)              (scalar, the network's prediction ŷ ∈ (0, 1))
```

The `ForwardCache` stores all four intermediate values because the backward pass needs them:

```scala
case class ForwardCache(z1: RealVector, a1: RealVector, z2: Double, a2: Double)
```

`a1` is needed twice during backprop: once to update W₂ (it was the input to the output neuron) and once to compute the hidden delta (its derivative factors into the gradient).

---

## 5. Backpropagation

Backprop is the chain rule applied layer by layer, going backward through the network.

**Loss function:** Mean Squared Error  
```
L = ½ (a₂ − y)²
```
(The ½ is a convention that cancels the 2 when differentiated.)

### 5.1 Output Layer Delta

The output delta `δ₂` measures how much the output neuron's pre-activation `z₂` needs to change to reduce the loss.

By the chain rule:
```
δ₂ = ∂L/∂z₂ = ∂L/∂a₂ · ∂a₂/∂z₂
             = (a₂ − y) · σ'(z₂)
             = (a₂ − y) · a₂(1 − a₂)
```

Three cases:
- `a₂ > y` → δ₂ is positive → reduce z₂ → reduce W₂ and b₂
- `a₂ < y` → δ₂ is negative → increase z₂ → increase W₂ and b₂
- `a₂ = y` → δ₂ = 0 → no change

### 5.2 Output Layer Weight Update

The gradient of L with respect to each weight in W₂:
```
∂L/∂W₂[j] = δ₂ · a₁[j]
```

The update rule (gradient descent):
```
W₂ ← W₂ − η · δ₂ · a₁
b₂ ← b₂ − η · δ₂
```

### 5.3 Hidden Layer Delta

To update W₁, we need to know how much each hidden neuron's pre-activation `z₁[j]` contributed to the error. The error is propagated back through W₂ and scaled by the hidden neuron's own derivative:

```
δ₁[j] = (W₂[j] · δ₂) · σ'(z₁[j])
       = (W₂[j] · δ₂) · a₁[j] · (1 − a₁[j])
```

In vector form:
```
δ₁ = (W₂ · δ₂) ⊙ a₁ ⊙ (1 − a₁)
```

Where `⊙` is element-wise multiply.

Intuition: a hidden neuron's error signal is the output error scaled by how much that neuron's weight connects to the output, further scaled by how sensitive the hidden neuron was (its sigmoid derivative). If the hidden neuron was saturated (`a₁ ≈ 0` or `≈ 1`), its derivative is near zero and it receives almost no error signal — this is the vanishing gradient in miniature.

### 5.4 Hidden Layer Weight Update

The gradient of L with respect to W₁[i][j] is `δ₁[j] · x[i]`.

In matrix form, this is the **outer product** `δ₁ ⊗ x`:

```
W₁ ← W₁ − η · (δ₁ ⊗ x)
b₁ ← b₁ − η · δ₁
```

The outer product gives a rank-1 matrix where entry (i, j) = δ₁[j] · x[i], which is exactly the gradient for weight W₁[i][j].

---

## 6. Implementation Walkthrough

The implementation lives in `src/main/scala/com/rohin/ann/twolayer/`.

### 6.1 Design Decisions

Same pattern as `Perceptron`: `case class` with `private` constructor, all behaviour as extension methods on the companion. Construction via:

```scala
TwoLayerNN.create(inputSize = 2, hiddenSize = 2, lr = 0.1)
```

Weights are initialised randomly in (0, 1) via `randomMatrix` and `randomVec`. Random initialisation breaks the *symmetry problem* — if all weights started at the same value, every hidden neuron would compute the same thing and learn identically, making the hidden layer useless.

### 6.2 `Activation` — Shared Functions

`sigmoid` and `sigmoidDerivativeFromActivation` live in the separate `Activation` object so they can be used by any future model without depending on `TwoLayerNN`.

**Scala/Java interop detail:** `RealVector.map` accepts a `UnivariateFunction` (Java interface). Passing a Scala lambda or a method reference (e.g., `z1.map(Activation.sigmoid)`) causes a compile error because the compiler cannot resolve which Java SAM type to target when there are multiple `map` overloads. The workaround:

```scala
val sigmoidFn = new UnivariateFunction {
  def value(z: Double): Double = Activation.sigmoid(z)
}
val a1 = z1.map(sigmoidFn)
```

### 6.3 `forward` — Forward Pass

```scala
extension (nn: TwoLayerNN)
  def forward(input: RealVector): ForwardCache =
    val z1 = nn.W1.operate(input).add(nn.b1)   // W₁ · x + b₁
    val a1 = z1.map(sigmoidFn)                  // σ applied element-wise
    val z2 = nn.W2.dotProduct(a1) + nn.b2       // W₂ · a₁ + b₂
    val a2 = Activation.sigmoid(z2)
    ForwardCache(z1, a1, z2, a2)
```

`RealMatrix.operate(v)` is the matrix-vector product `W₁ · x`. `RealVector.add` adds b₁. The result is cached rather than returned directly so backprop can access all intermediate values.

### 6.4 `computeOutputDelta` — Output Error

```scala
def computeOutputDelta(a2: Double, y: Int): Double =
  val error = a2 - y
  val grad  = a2 * (1 - a2)
  error * grad
```

This is `δ₂ = (a₂ − y) · a₂(1 − a₂)`. The derivative is computed from the activation (`a2`) rather than from `z2`, avoiding a redundant sigmoid call.

### 6.5 `computeHiddenDelta` — Propagated Error

```scala
def computeHiddenDelta(W2: RealVector, delta2: Double, a1: RealVector): RealVector =
  val propagated = W2.mapMultiply(delta2)       // W₂ · δ₂ (element-wise scale)
  val grad       = a1.map(a => a * (1 - a))     // σ'(z₁) = a₁ ⊙ (1 − a₁)
  propagated.ebeMultiply(grad)                  // element-wise multiply
```

`mapMultiply` scales the entire vector by a scalar. `ebeMultiply` is commons-math3's element-by-element multiply (⊙).

### 6.6 `updateOutputLayer` and `updateHiddenLayer`

```scala
extension (nn: TwoLayerNN)
  def updateOutputLayer(a1: RealVector, a2: Double, y: Int): TwoLayerNN =
    val delta2 = computeOutputDelta(a2, y)
    val newW2  = nn.W2.subtract(a1.mapMultiply(nn.learningRate).mapMultiply(delta2))
    val newB2  = nn.b2 - nn.learningRate * delta2
    nn.copy(W2 = newW2, b2 = newB2)

extension (nn: TwoLayerNN)
  def updateHiddenLayer(delta1: RealVector, input: RealVector): TwoLayerNN =
    val gradW1 = delta1.outerProduct(input)     // δ₁ ⊗ x  →  rank-1 matrix
    val newW1  = nn.W1.subtract(gradW1.scalarMultiply(nn.learningRate))
    val newB1  = nn.b1.subtract(delta1.mapMultiply(nn.learningRate))
    nn.copy(W1 = newW1, b1 = newB1)
```

`outerProduct` returns a `RealMatrix`. `scalarMultiply` scales it by `η`. `RealMatrix.subtract` subtracts element-wise. All operations are from commons-math3 — no manual loops.

### 6.7 `trainOne` — Full Update for One Example

```scala
extension (nn: TwoLayerNN)
  def trainOne(input: RealVector, y: Int): TwoLayerNN =
    val cache  = nn.forward(input)
    val delta2 = computeOutputDelta(cache.a2, y)
    val delta1 = computeHiddenDelta(nn.W2, delta2, cache.a1)
    nn
      .updateOutputLayer(a1 = cache.a1, a2 = cache.a2, y = y)
      .updateHiddenLayer(delta1 = delta1, input = input)
```

Note that `delta2` is computed here (not inside `updateOutputLayer`) because it is needed for both the output update *and* for computing `delta1`. Computing it once and sharing it avoids a redundant call.

`updateOutputLayer` is applied first, then `updateHiddenLayer`. Order matters here for correctness: `updateHiddenLayer` uses the original `nn.W2` (via `delta1` which was computed from `nn.W2`), not the updated one. Because `updateOutputLayer` returns a new `TwoLayerNN` via `copy`, the chained call `.updateHiddenLayer(delta1, input)` operates on the updated output weights — but `delta1` was already fixed from the original W₂, so the hidden update is correct.

### 6.8 `train` — Full Training Run

Same `@tailrec` foldLeft pattern as `Perceptron.train`:

```scala
extension (nn: TwoLayerNN)
  def train(dataset: List[(RealVector, Int)], epochs: Int): TwoLayerNN =
    @tailrec
    def loop(iterate: Int, accNN: TwoLayerNN): TwoLayerNN =
      if iterate == 0 then accNN
      else loop(iterate - 1, dataset.foldLeft(accNN)((acc, ex) => acc.trainOne(ex._1, ex._2)))
    loop(epochs, nn)
```

---

## 7. Current State

| Component | Status |
|---|---|
| Sigmoid activation + derivative | `Activation.scala` |
| Two-layer forward pass with `ForwardCache` | `TwoLayerNN.forward` |
| Output delta (MSE gradient through sigmoid) | `computeOutputDelta` |
| Hidden delta (backprop through W₂ + sigmoid') | `computeHiddenDelta` |
| Output layer update (W₂, b₂) | `updateOutputLayer` |
| Hidden layer update (W₁ via outer product, b₁) | `updateHiddenLayer` |
| XOR convergence test (10,000 epochs, ±0.1 tolerance) | `TwoLayerNNSuite` ✅ |

---

## 8. What's Next — Phase 5: Pluggable / Composable Neurons

`TwoLayerNN` is monolithic: the number of layers, the activation function, and the backprop logic are all hardwired. It solves XOR but it doesn't teach you *how* neurons compose.

The goal of Phase 5 is to replace this with explicit building blocks:

1. **`Neuron`** — one unit: holds its weights, bias, and an activation function. Can run forward and receive a delta during backward.
2. **`Layer`** — a `Seq[Neuron]` that runs forward as a unit (each neuron independently), collects its activations, and distributes deltas during backward.
3. **`Network`** — a `Seq[Layer]` that chains the forward pass left-to-right and the backward pass right-to-left.

The goal is to have the same XOR test pass, but now driven by a composed structure where the plumbing is explicit. Each neuron is individually addressable, each layer's forward/backward is its own concern, and the network is just the wiring.

This directly answers the question: *what actually happens when neurons are composed?*
