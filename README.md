# JVM Neural Network From Scratch 🚀

This is an experimental, ground-up implementation of neural networks on the JVM using Scala.

The goal is not just to *use* machine learning libraries, but to **understand and build the core mechanics from scratch**, and then progressively explore **performance optimization and GPU acceleration using emerging JVM technologies like Project Babylon and HAT**.

---

## 🎯 Objectives

* Build neural networks from first principles
* Understand the math behind learning algorithms
* Explore JVM as a viable platform for ML systems
* Experiment with modern JVM projects (Babylon, HAT)
* Document the journey as a reproducible learning path

---

## 🧠 Learning Philosophy

This project follows a **bottom-up approach**:

1. Start with simple, fully transparent implementations
2. Avoid abstractions until concepts are understood
3. Gradually introduce performance and system-level concerns
4. Treat each stage as both a learning milestone and a reusable module

---

## 🗺️ Roadmap

### Phase 1 — Perceptron (Foundations) ✅

**Goal:** Understand the core learning loop

* [x] Implement a single perceptron
* [x] Dot product via Apache Commons Math (`RealVector.dotProduct`) — deliberately chose library over manual implementation
* [x] Step activation function
* [x] Weight & bias updates (learning rule) — using `RealVector.combine` for the vector update
* [x] Train on simple datasets:

  * AND ✓ (verified in tests)
  * OR ✓ (verified in tests)
* [x] Observe failure on XOR — XOR test documents the limitation with an explanation

**Key Concepts:**

* Linear separability
* Weights & bias
* Error-driven learning

---

### Phase 2 — Activations & Continuous Output ✅

**Goal:** Move beyond binary threshold models

* [x] Replace step function with sigmoid — `Activation.sigmoid(z) = 1 / (1 + exp(-z))`
* [x] Implement continuous output — sigmoid maps to (0, 1), not {0, 1}
* [x] Loss gradient implicit in backprop — output delta `δ₂ = (a₂ − y) · a₂(1 − a₂)` encodes MSE gradient
* [ ] Explicit loss tracking across epochs (still to-do)

**Key Concepts:**

* Differentiability — step has no gradient; sigmoid does
* Probability-like outputs
* Loss minimization via gradient

---

### Phase 3 — Matrix Structure ✅

**Goal:** Shift weight representation to matrices

* [x] Vector representation — `RealVector` from Phase 1
* [x] Matrix structure — `RealMatrix` (`Array2DRowRealMatrix`) for hidden layer weights W₁
* [x] Outer product for gradient computation — `delta1.outerProduct(input)` gives the W₁ gradient

**Key Concepts:**

* Hidden layer weights as a matrix (rows = inputs, cols = hidden neurons)
* Matrix-vector multiply for forward pass: `W₁ · x + b₁`
* Outer product for weight updates in backprop

---

### Phase 4 — Two-Layer Neural Network ✅

**Goal:** Build a real neural network that solves XOR

* [x] Implement hidden layer (W₁ ∈ ℝ^{n×h}, b₁ ∈ ℝ^h)
* [x] Forward propagation with `ForwardCache` (stores z₁, a₁, z₂, a₂)
* [x] Backpropagation — manual chain rule, output delta and hidden delta
* [x] Train on XOR successfully — test passes at 10,000 epochs

**Key Concepts:**

* Chain rule — gradient flows backward through layers
* Non-linear decision boundary (hidden layer transforms the input space)
* `ForwardCache` — storing intermediate activations for use during the backward pass

---

### Phase 5 — Pluggable / Composable Neurons (Next)

**Goal:** Understand *how neurons compose* — replace the monolithic `TwoLayerNN` with explicit building blocks

The current `TwoLayerNN` is self-contained: it knows its own shape, forward pass, and backward pass hardwired together. The next step is to decompose it into first-class pieces so a network is assembled from parts:

* [ ] `Neuron` — a single unit with a configurable activation function
* [ ] `Layer` — an ordered collection of neurons that forward/backward as a unit
* [ ] `Network` — an ordered sequence of layers; forward pass composes left-to-right, backward pass composes right-to-left
* [ ] Verify: the composed network replicates XOR (same result, different structure)

**Key Concepts:**

* Separation of topology from learning rule
* Activation as a pluggable strategy (step, sigmoid, ReLU, …)
* Gradient flowing through a composed structure

---

### Phase 6 — Batch Processing & Matrix Ops

**Goal:** Move toward scalable computation

* [ ] Batch forward pass — multiple inputs at once via matrix multiply
* [ ] Vectorised gradient updates
* [ ] Compare performance: single-example loop vs batched matrix ops

**Key Concepts:**

* Batching as outer product over examples
* Computational efficiency — fewer JVM allocations per example
* Foundations for GPU offloading (Phase 7)

---

### Phase 7 — JVM Performance Exploration

**Goal:** Push JVM limits for ML workloads

* [ ] Benchmark implementations (JMH)
* [ ] Explore JVM memory behaviour under training loops
* [ ] Evaluate libraries (optional): Breeze, ND4J

**Key Concepts:**

* Memory layout (object headers, boxing, cache lines)
* CPU vs JVM overhead
* Trade-offs of abstraction

---

### Phase 8 — GPU & Modern JVM (Experimental)

**Goal:** Explore GPU acceleration on JVM

* [ ] Identify compute-heavy parts (matrix multiply, batch forward pass)
* [ ] Experiment with Project Babylon
* [ ] Explore HAT (Heterogeneous Accelerator Toolkit)
* [ ] Attempt GPU offloading of core operations

**Key Concepts:**

* Data parallelism
* GPU vs CPU workloads
* JVM-native acceleration

---

## 🛠️ Tech Stack

* Scala 3 (JVM)
* sbt
* Apache Commons Math 3 — vector/matrix operations (`RealVector`, `RealMatrix`)
* munit — test framework
* (Later) ND4J for batched matrix ops, Project Babylon / HAT for GPU

---

## 📂 Project Structure

```
src/main/scala/com/rohin/ann/
  Perceptron.scala          — Phase 1: single perceptron, step activation, learning rule
  Activation.scala          — Phase 2: sigmoid and its derivative (shared across models)
  twolayer/
    TwoLayerNN.scala        — Phase 4: two-layer network with backprop
    ForwardCache.scala      — Phase 4: intermediate activations stored for backward pass

src/test/scala/com/rohin/ann/
  PerceptronSuite.scala     — perceptron unit tests
  ActivationSuite.scala     — sigmoid / derivative tests
  twolayer/
    TwoLayerNNSuite.scala   — two-layer NN tests (incl. XOR convergence)

docs/
  tdd_walkthrough.md        — guided tour through all test suites in the order they were written
  perceptron.md             — theory + implementation walkthrough (Phase 1)
  perceptron_diagram.md     — Mermaid diagrams: forward pass, training loop, decision boundary
  two_layer_nn.md           — theory + implementation walkthrough (Phases 2–4)
```

---

## 📈 Progress Tracking

This repository will evolve step-by-step, with each phase:

* Implemented from scratch
* Documented with learnings
* Benchmarked where relevant

---

## ⚠️ Disclaimer

This is an experimental and educational project.

* Not optimized for production
* Focused on clarity over performance (initially)
* Later phases may use unstable or experimental JVM features

---

## 🤝 Contributions

This is primarily a personal learning project, but ideas, discussions, and critiques are welcome.

---

## 🔥 End Goal

By the end of this project:

* A working multi-layer neural network built from scratch
* Clear understanding of how ML systems work internally
* Exploration of JVM as a serious platform for AI workloads
* Early experimentation with next-gen JVM GPU capabilities

---

## 🧭 Why this project?

Because using frameworks is easy.

Understanding them deeply is not.
