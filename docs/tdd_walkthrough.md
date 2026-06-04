# TDD Learning Walkthrough

This document is a guided tour through the test suites in the order they were written. Each suite introduced one concept. Follow the suites in order — run the tests, read the source file alongside each test, and let the failing tests drive the implementation.

---

## Quick Reference

| Order | Suite | Source | Concept introduced |
|-------|-------|--------|--------------------|
| 1 | `PerceptronSuite` | `Perceptron.scala` | Forward pass, step activation, perceptron learning rule |
| 2 | `PerceptronSuiteV1` | `PerceptronV1.scala` | Sigmoid on a single neuron, gradient direction, saturation |
| 3 | `ActivationSuite` | `Activation.scala` | Sigmoid and its derivative as standalone functions |
| 4 | `TwoLayerNNSuite` | `TwoLayerNN.scala` + `ForwardCache.scala` | Backpropagation, hidden layer, XOR |

Run the full sequence:
```bash
sbt test
```

Run one suite at a time:
```bash
sbt "testOnly com.rohin.ann.PerceptronSuite"
sbt "testOnly com.rohin.ann.PerceptronSuiteV1"
sbt "testOnly com.rohin.ann.ActivationSuite"
sbt "testOnly com.rohin.ann.twolayer.TwoLayerNNSuite"
```

---

## Suite 1 — `PerceptronSuite`

**Source:** `src/main/scala/com/rohin/ann/Perceptron.scala`  
**Tests:** `src/test/scala/com/rohin/ann/PerceptronSuite.scala`

**What this suite establishes:** The complete perceptron — forward pass, the three-valued error signal, weight and bias updates, full training on AND and OR, and a documented failure on XOR.

---

### Test 1 — `predict returns 1 when weighted sum is positive`

```scala
val perceptron = create(bias = 0, weights = vec(1, 1))
perceptron.predict(vec(1, 1))  // → 1
```

The core mechanic: `z = w·x + b`, then `step(z)`. With weights `[1, 1]` and input `[1, 1]`, `z = 2 > 0` so the output is 1.

**What you learn:** The forward pass is just a dot product plus a bias, passed through a threshold.

---

### Test 2 — `predict returns 1 for boundary case`

```scala
val perceptron = create(bias = 0, weights = vec(1, -1))
perceptron.predict(vec(1, 1))  // z = 1 - 1 = 0 → 1
```

`z = 0` maps to 1. This is not arbitrary — the Heaviside step convention places the boundary hyperplane in the positive class.

**What you learn:** The decision boundary (`z = 0`) belongs to class 1. Important to test this explicitly because any off-by-one in the `if z >= 0` check would flip the result.

---

### Test 3 — `weights are updated when prediction is wrong`

```scala
val perceptron = create(bias = 0, weights = vec(-1, -1), learningRate = 1.0)
perceptron.trainOne(vec(1, 1), actual = 1)
// → weights: [0, 0],  bias: 1.0
```

With weights `[-1, -1]` and input `[1, 1]`, `z = -2`, prediction = 0, actual = 1, so `error = 1`.

```
Δw = η × error × x = 1.0 × 1 × [1, 1] = [1, 1]
w_new = [-1, -1] + [1, 1] = [0, 0]

Δb = η × error = 1.0 × 1 = 1
b_new = 0 + 1 = 1
```

**What you learn:** Exact delta verification — the update rule is `w ← w + η·error·x`, `b ← b + η·error`. Using `learningRate = 1.0` makes the arithmetic readable without a calculator.

---

### Test 4 — `No update when prediction is correct`

```scala
val perceptron = create(bias = 0, weights = vec(1, 1), learningRate = 1.0)
perceptron.trainOne(vec(1, 1), actual = 1)
// → weights: [1, 1],  bias: 0.0   (unchanged)
```

`z = 2`, prediction = 1, actual = 1, `error = 0`. No update.

**What you learn:** `trainOne` short-circuits when `error == 0` and returns the same instance without allocation. Only wrong predictions change the weights.

---

### Test 5 — `perceptron learns AND gate`

```scala
val trained = initial.train(andData, epochs = 10)
// All four predictions correct after 10 epochs
```

Training converges in 10 epochs from zero weights. AND is linearly separable — one line can isolate `(1,1)` from the rest.

**What you learn:** The full training loop works. `train(dataset, epochs)` is a `@tailrec` foldLeft — each epoch passes all four examples through `trainOne` in sequence.

---

### Test 6 — `perceptron learns OR gate`

Same structure as AND, different data. OR is also linearly separable.

**What you learn:** The same learning rule generalises to a different linear boundary without any code change.

---

### Test 7 — `perceptron learns XOR gate`  *(asserts that it FAILS)*

```scala
val correct = trained.predict(vec(0, 0)) == 0 &&
              trained.predict(vec(0, 1)) == 1 &&
              trained.predict(vec(1, 0)) == 1 &&
              trained.predict(vec(1, 1)) == 0

assertEquals(correct, false)   // ← the test asserts failure
```

XOR is not linearly separable. No single straight line can separate `(0,1)` and `(1,0)` from `(0,0)` and `(1,1)`.

**What you learn:** The perceptron convergence theorem guarantees convergence only when the data is linearly separable. When it is not, the algorithm cycles forever. This test documents the limitation explicitly — the test name says "learns XOR" but the assertion says it does not. Reading the comment above the test explains why.

**This is the motivating question for everything that follows.**

---

## Suite 2 — `PerceptronSuiteV1`

**Source:** `src/main/scala/com/rohin/ann/PerceptronV1.scala`  
**Tests:** `src/test/scala/com/rohin/ann/PerceptronSuiteV1.scala`

**What this suite establishes:** Replacing the step function with sigmoid on a *single* neuron. This is the minimal step before multi-layer networks — verify that sigmoid-based gradient descent works at all.

`PerceptronV1` has the same structure as `Perceptron` but uses `predictProb` (returns a `Double` via sigmoid) instead of `predict` (returns an `Int` via step). The learning rule changes from `error × x` to `error × σ'(z) × x`.

---

### Test 1 — `weights move in correct direction with sigmoid learning`

```scala
val p = create(bias = 0, weights = vec(0, 0), learningRate = 1.0)
val updated = p.trainOne(vec(1, 1), y = 1)

assert(updated.weights.getEntry(0) > 0)   // direction only, not exact value
assert(updated.weights.getEntry(1) > 0)
assert(updated.bias > 0)
```

Starting at zero weights, `σ(0) = 0.5`. The network predicts 0.5 but the target is 1, so the error is positive and all weights should increase.

**What you learn:** With a continuous loss gradient, the test checks *direction* rather than exact values. This is intentional — the exact update involves `σ'(z) = σ(z)(1 − σ(z)) = 0.25`, which makes the math less transparent. Testing direction is sufficient to verify correctness and generalises better. Contrast with Suite 1 where exact deltas were tested because the arithmetic was simple integers.

**Why a new class instead of modifying `Perceptron`:** `PerceptronV1` is an intermediate learning step — a single neuron with sigmoid. It is not a replacement for `Perceptron`; both are kept as independent milestones.

---

### Test 2 — `updates are smaller when prediction is confident`

```scala
val p = create(bias = 0, weights = vec(5, 5), learningRate = 1.0)
val input = vec(1, 1)
val before = p.predictProb(input)     // → close to 1.0 (z = 10)
val updated = p.trainOne(input, y = 1)
val after = updated.predictProb(input)
assert(Math.abs(after - before) < 0.01)   // very small change
```

With `weights = [5, 5]` and `input = [1, 1]`, `z = 10` and `σ(10) ≈ 0.9999`. The network is already very confident and correct. The derivative `σ'(10) = σ(10)·(1−σ(10)) ≈ 0.0001` is tiny.

**What you learn:** Sigmoid's derivative is near zero when the neuron is saturated. The update is small precisely because the neuron is already confident. This is the "self-regulating" property of gradient descent — larger mistakes cause larger corrections. It is also the origin of the *vanishing gradient* problem in deeper networks.

---

## Suite 3 — `ActivationSuite`

**Source:** `src/main/scala/com/rohin/ann/Activation.scala`  
**Tests:** `src/test/scala/com/rohin/ann/ActivationSuite.scala`

**What this suite establishes:** Sigmoid and its derivative as pure, independently testable functions extracted from any neuron implementation.

The motivation for extracting `Activation`: `PerceptronV1` inlined sigmoid via `Activation.*`. `TwoLayerNN` needs the same functions. Extracting them with their own test suite means neither model needs to re-test the math — any model can depend on `Activation` knowing it is already verified.

---

### Test 1 — `sigmoid produces expected values`

```scala
assertEquals(sigmoid(0), 0.5)   // centre point
assert(sigmoid(10) > 0.99)      // large positive → near 1
assert(sigmoid(-10) < 0.01)     // large negative → near 0
```

Three anchor points that characterise the full curve: the exact centre value and the asymptotic behaviour at both extremes.

**What you learn:** Sigmoid maps all of ℝ to (0, 1). These three tests cover the three qualitatively different regions of the function.

---

### Test 2 — `sigmoid derivative is correct at 0`

```scala
val d = sigmoidDerivative(0)
assertEquals(d, 0.25)
```

`σ'(0) = σ(0) · (1 − σ(0)) = 0.5 × 0.5 = 0.25`

This is the maximum of the derivative — sigmoid is most sensitive at `z = 0`.

**What you learn:** The derivative formula `σ'(z) = σ(z)(1 − σ(z))` is verified at its peak. The test also confirms the specific implementation: `sigmoidDerivative` calls `sigmoid` internally (uses the `z` form), while `sigmoidDerivativeFromActivation(a)` computes `a * (1 - a)` from an already-activated value. Both are in `Activation.scala`.

---

## Suite 4 — `TwoLayerNNSuite`

**Source:** `src/main/scala/com/rohin/ann/twolayer/TwoLayerNN.scala`, `ForwardCache.scala`  
**Tests:** `src/test/scala/com/rohin/ann/twolayer/TwoLayerNNSuite.scala`

**What this suite establishes:** A two-layer network with backpropagation. Each test isolates one piece of the backward pass before the final integration test (XOR). The strategy is: test each component with known inputs and verify exact outputs, then verify the full system end-to-end.

---

### Test 1 — `two layer network produces output between 0 and 1`

```scala
val nn = TwoLayerNN.create(inputSize = 2, hiddenSize = 2)
val output = nn.forward(vec(1, 0)).a2
assert(output > 0.0 && output < 1.0)
```

Sanity check: random initialisation and a forward pass run without error, and the output is a valid probability.

**What you learn:** `forward` returns a `ForwardCache` containing all intermediate values (`z1`, `a1`, `z2`, `a2`). The cache is not an optimisation — it is required: backprop needs `a1` to compute both the output weight gradient and the hidden delta.

---

### Test 2 — `output delta is computed correctly`

```scala
val delta = computeOutputDelta(a2 = 0.5, y = 1)
assertEquals(delta, -0.125)
```

With `a2 = 0.5` and `y = 1`:
```
δ₂ = (a₂ − y) · a₂(1 − a₂)
   = (0.5 − 1) × 0.5 × 0.5
   = -0.5 × 0.25
   = -0.125
```

**What you learn:** `computeOutputDelta` is a pure function tested in isolation. Negative delta means the network underpredicted (target was 1, output was 0.5), so weights need to increase. The sign of the delta drives the direction of the update.

---

### Test 3 — `output weights move in correct direction`

```scala
val updated = nn.updateOutputLayer(a1 = vec(0.5, 0.5), a2 = 0.5, y = 1)
assert(updated.W2.getEntry(0) > nn.W2.getEntry(0))
assert(updated.W2.getEntry(1) > nn.W2.getEntry(1))
assert(updated.b2 > nn.b2)
```

Underprediction (target 1, output 0.5) → δ₂ is negative → `W2 − η·δ₂·a1` increases W2.

**What you learn:** The direction test pattern from Suite 2 reappears. When the math involves random initial weights, testing direction is both simpler and more robust than computing exact values. The sign logic is: negative delta → subtracting a negative → weights increase.

---

### Test 4 — `hidden delta is computed correctly`

```scala
val delta1 = computeHiddenDelta(W2 = vec(1, 1), delta2 = -0.125, a1 = vec(0.5, 0.5))
assertEquals(delta1.getEntry(0), -0.03125)
assertEquals(delta1.getEntry(1), -0.03125)
```

```
δ₁[j] = W₂[j] · δ₂ · a₁[j] · (1 − a₁[j])
       = 1 × (−0.125) × 0.5 × 0.5
       = −0.03125
```

**What you learn:** The hidden delta is the output error propagated back through W₂ and scaled by the hidden neuron's own derivative. This is the chain rule — the hidden neuron only receives the fraction of the error that passed through its connection to the output. Using controlled inputs (`W2 = [1,1]`, `a1 = [0.5, 0.5]`) makes the arithmetic exact and checkable by hand.

---

### Test 5 — `hidden layer weights update correctly`

```scala
val updated = nn.updateHiddenLayer(delta1 = vec(-0.03125, -0.03125), input = vec(1, 1))
assert(updated.W1.getEntry(0, 0) > nn.W1.getEntry(0, 0))
assert(updated.W1.getEntry(1, 1) > nn.W1.getEntry(1, 1))
assert(updated.b1.getEntry(0) > nn.b1.getEntry(0))
assert(updated.b1.getEntry(1) > nn.b1.getEntry(1))
```

Negative delta → subtracting a negative outer product → W1 entries increase.

**What you learn:** The W1 gradient is the outer product `δ₁ ⊗ x` (a matrix, not a vector). `delta1.outerProduct(input)` from commons-math3 produces this directly. Each entry `[i][j]` of the gradient matrix is `δ₁[j] × x[i]` — exactly how much weight `W1[i][j]` contributed to the error.

---

### Test 6 — `two layer network learns XOR`  *(integration test)*

```scala
val trained = nn.train(xorData, epochs = 10000)
assert(math.abs(trained.forward(vec(0, 0)).a2 - 0) <= 0.1)
assert(trained.forward(vec(0, 1)).a2 > 0.9)
assert(trained.forward(vec(1, 0)).a2 > 0.9)
assert(math.abs(trained.forward(vec(1, 1)).a2 - 0) <= 0.1)
```

All five previous tests were unit tests of individual components. This one ties everything together: 10,000 epochs of `train` → `trainOne` → `forward` + backprop for every example. The tolerance of ±0.1 reflects that this is a continuous sigmoid output, not a binary step — "close to 0" and "close to 1" is the correct formulation.

**What you learn:** The XOR problem that `PerceptronSuite` documented as unsolvable is now solved. The hidden layer learned to transform the input space into a representation where XOR becomes linearly separable, and the output neuron learned the boundary in that new space.

This is the payoff of the entire sequence.

---

## Phase 5 — pytorchlite

Phase 5 revisits everything you just learned, but rearranges the same math into reusable bricks. Instead of one hard-coded `TwoLayerNN`, the `pytorchlite` package introduces a `Dense` layer abstraction, a `Sequential` container, explicit gradient data structures, and a small logging DSL for debugging. The tests mirror that progression.

### Test 1 — `ActivationFnSuite` keeps nonlinear math isolated

```scala
val s = ActivationFn.Sigmoid
assertEquals(s.forward(0), 0.5)
assertEquals(s.derivativeFromActivation(0.5), 0.25)
```

**What you learn:** The activation is now an explicit strategy object. `ActivationFn` also exposes `forwardFn`, a tiny adapter so commons-math's `RealVector.map` can invoke the Scala-defined function. That fixes the eta-expansion problem noted back in `TwoLayerNN.forward`.

### Test 2 — `Dense` layer forward preserves shape contracts

```scala
val layer = Dense(inputSize = 2, outputSize = 3, act = ActivationFn.Sigmoid)
val (_, a) = layer.forward(vec(1, 0))
assertEquals(a.getDimension, 3)
```

**What you learn:** Each layer owns its weight matrix (`output × input`), bias vector, and activation. The constructor randomises both weights and biases in `(-0.5, 0.5)` via helpers in `package.scala`. `forward` now returns both `z` and `a` so the caller can cache them for backprop.

### Test 3 — `Loss.outputDelta` reuses the MSE + sigmoid gradient

```scala
val delta = Loss.outputDelta(output = vec(0.5), target = vec(1.0))
assertClose(delta.getEntry(0), -0.125)
```

**What you learn:** The output gradient stayed the same math `(a − y) · a · (1 − a)`, but it's now extracted into a dedicated helper so the sequential stack doesn't mix loss-specific formulas with layer orchestration.

### Test 4 — `Sequential.forwardPass` caches activations

```scala
val pass = net.forwardPass(vec(1, 0))
assertEquals(pass.activations.size, 3)
assertEquals(pass.zs.size, 2)
```

**What you learn:** `ForwardPass` is the new `ForwardCache`. It stores every `z` and `a`, including the input activation at index 0. This keeps the later gradient code identical regardless of depth.

### Test 5 — `Sequential.backward` + `Gradients` propagate deltas layer-by-layer

```scala
val pass = net.forwardPass(vec(1, 0))
val grads = net.backward(pass, vec(1))
assertEquals(grads.deltas.size, 2)
```

**What you learn:** Gradients flow from the output delta computed in Test 3, through each layer's transposed weight matrix, and are Hadamard-multiplied by the activation derivative for that layer. The result is a list of deltas aligned with the layer order.

### Test 6 — `applyGradients`, `trainOne`, and `train` mutate weights repeatedly

Each helper gets its own assertion that weights actually change:

```scala
val grads = net.backward(pass, vec(1))
val updated = net.applyGradients(pass, grads, learningRate = 0.1)
assertNotEquals(updated.layers.head.W, net.layers.head.W)
```

`trainOne` and `train` simply compose the helpers, but having discrete tests ensures regressions surface at the exact stage they originate. These specs also exercise the optional `ConsoleLogging.DebugConfig` so you can dump gradients mid-test when needed.

### Test 7 — `TrainingSuite` proves the composed stack solves XOR

```scala
val xor = List((vec(0, 0), vec(0)), ..., (vec(1, 1), vec(0)))
val trained = net.train(xor, epochs = 5000, learningRate = 0.1)
assert(trained.forward(vec(0, 0)) < 0.2)
...
```

**What you learn:** Even though the architecture is now generic, the same 2-2-1 topology, sigmoid activations, and SGD learning loop still crack XOR. That validates the refactor: you gained composability without sacrificing behaviour.

## What's Next

Phase 6 picks up from the new `Sequential` API and scales it horizontally: batched forward passes, vectorised gradient accumulation, and experiments with JVM performance characteristics. The goal is to keep treating the docs as a lab notebook—once batching lands, expect another round of tests + theory write-ups here.
