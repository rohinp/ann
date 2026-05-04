# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A ground-up implementation of neural networks on the JVM using Scala 3, built for learning. The goal is to understand and implement ML from first principles, then explore JVM performance and GPU acceleration (Project Babylon / HAT) in later phases.

Phases 1–4 are complete. Currently targeting **Phase 5 — Pluggable / Composable Neurons**. See `README.md` for the full multi-phase roadmap.

## Commands

```bash
sbt compile          # compile
sbt run              # run Main
sbt test             # run all tests
sbt "testOnly MySuite"   # run a single test suite by name
sbt scalafmtAll      # format all sources
```

## Architecture

All neural network code lives under `src/main/scala/com/rohin/ann/`.

### `Perceptron` (Phase 1)

`Perceptron` is a `case class` with a `private` constructor. All behaviour lives in its companion object as **extension methods** (`predict`, `trainOne`, `train`, `updateAllWeights`, `updateBias`). Construction goes through:

- `Perceptron.create(bias, weights)` — default learning rate 0.001
- `Perceptron.create(bias, weights, learningRate)` — explicit learning rate
- `Perceptron.vec(xs: Double*)` / `Perceptron.zeroVec(size)` — vector factory helpers so callers never import `ArrayRealVector` directly

`weights` is `RealVector` from commons-math3. The weight update uses `RealVector.combine(1, η*error, input)` which is `w + η·error·x` as a single native vector op. `trainOne` returns `this` unchanged when `error == 0`.

`train(dataset, epochs)` runs N epochs via a `@tailrec` inner loop. Training duration is always explicit — there is no "run until converged" API.

### `Activation` (Phase 2)

`Activation` is a plain `object` with pure math functions:

- `sigmoid(z)` — `1 / (1 + exp(-z))`, output in (0, 1)
- `sigmoidDerivative(z)` — `σ(z) · (1 − σ(z))`, computed from z
- `sigmoidDerivativeFromActivation(a)` — `a · (1 − a)`, computed from an already-activated value (avoids re-computing sigmoid during backprop)

These are defined independently of any model and are reused by `TwoLayerNN`.

**Scala/Java interop note:** Passing `Activation.sigmoid` as a lambda to `RealVector.map` fails at compile time because the Scala compiler cannot resolve the overload between `UnivariateFunction` and `DoubleUnaryOperator`. The workaround is to wrap the call in an explicit `new UnivariateFunction { def value(z) = Activation.sigmoid(z) }`. This is documented inline in `TwoLayerNN.forward`.

### `TwoLayerNN` and `ForwardCache` (Phase 4)

`TwoLayerNN` is a `case class` with a `private` constructor. All behaviour lives in its companion object as extension methods. Fields:

- `W1: RealMatrix` — hidden layer weights, shape `(inputSize × hiddenSize)`
- `b1: RealVector` — hidden layer biases, length `hiddenSize`
- `W2: RealVector` — output layer weights, length `hiddenSize`
- `b2: Double` — output bias
- `learningRate: Double`

Construction: `TwoLayerNN.create(inputSize, hiddenSize, lr)` — initialises all weights and biases randomly in (0, 1).

**`forward(input)`** runs the two-step forward pass and returns a `ForwardCache(z1, a1, z2, a2)`. Caching intermediate values is essential: backprop needs `a1` to update W₂ and `a1` again (for its derivative) to propagate the hidden delta.

**Backprop** is split into named helpers to keep the math readable:

- `computeOutputDelta(a2, y)` — `(a₂ − y) · a₂(1 − a₂)` (MSE loss gradient through sigmoid)
- `computeHiddenDelta(W2, delta2, a1)` — `(W₂ · δ₂) ⊙ a₁(1 − a₁)` (error propagated back through output weights, scaled by hidden sigmoid derivative)
- `updateOutputLayer(a1, a2, y)` — applies `δ₂` to W₂ and b₂
- `updateHiddenLayer(delta1, input)` — applies `δ₁` to W₁ (via outer product) and b₁

`trainOne` composes these: forward → compute deltas → update output → update hidden.

`train(dataset, epochs)` is the same `@tailrec` foldLeft pattern as `Perceptron.train`.

`Main.scala` is a placeholder.

## Testing

Tests are written TDD-style — the test is the specification. Tests live in `src/test/scala/com/rohin/ann/` and use [munit](https://scalameta.org/munit/). Every piece of production code must have test coverage.

Test suites use `import ModelName.*` to access extension methods and factory helpers without qualification.

**`PerceptronSuite`** tests:
- Correct prediction (forward pass)
- Boundary case (`z = 0` → positive class)
- Weight and bias update when prediction is wrong (verify exact delta)
- No update when prediction is already correct
- Dataset-level training: AND and OR converge; XOR is tested and documented as non-convergent

**`ActivationSuite`** tests:
- `sigmoid` output range and known values
- `sigmoidDerivative` at known points

**`TwoLayerNNSuite`** tests:
- Forward pass output is in (0, 1)
- `computeOutputDelta` exact value
- Output weights and bias move in the correct direction after an update
- `computeHiddenDelta` exact value
- Hidden weights and bias move in the correct direction after an update
- XOR convergence: train 10,000 epochs, assert all four predictions within 0.1 of ground truth

Run a single suite during development:
```bash
sbt "testOnly com.rohin.ann.PerceptronSuite"
sbt "testOnly com.rohin.ann.ActivationSuite"
sbt "testOnly com.rohin.ann.twolayer.TwoLayerNNSuite"
```

## Code Conventions

- Scala 3 syntax throughout (indentation-based, `then`/`end`, etc.)
- `.scalafmt.conf` dialect is `scala213` but the project uses Scala 3 — update to `runner.dialect = scala3` before relying on formatter output.
- Math operations use `org.apache.commons.math3.linear` (`ArrayRealVector`, `RealVector`) — do not re-implement what commons-math3 already provides.

## Documentation

Theory and implementation notes for each model live in `docs/`:

- `docs/tdd_walkthrough.md` — **start here** — guided tour through all four test suites in the order they were written; explains what each test teaches and how the suites connect
- `docs/perceptron.md` — math, learning rule, linear separability, implementation walkthrough (Phase 1)
- `docs/perceptron_diagram.md` — Mermaid diagrams: forward pass, training loop, decision boundary
- `docs/two_layer_nn.md` — sigmoid, backpropagation, XOR solution, implementation walkthrough (Phases 2–4)
