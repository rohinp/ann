# Perceptron Diagrams

---

## 1. Structure — Forward Pass

```mermaid
flowchart LR
    subgraph Inputs
        x1(["x₁"])
        x2(["x₂"])
        xn(["xₙ"])
    end

    subgraph Summation
        S(["z = Σwᵢxᵢ + b"])
    end

    subgraph Activation
        A(["step(z)"])
    end

    x1 -->|"w₁"| S
    x2 -->|"w₂"| S
    xn -->|"wₙ"| S
    b(["bias b"]) --> S
    S --> A
    A --> out(["ŷ ∈ {0, 1}"])
```

**ASCII equivalent** (for environments that don't render Mermaid):

```
 x₁ ──(w₁)──┐
             │
 x₂ ──(w₂)──┤
             ├──► [ z = Σwᵢxᵢ + b ] ──► [ step(z) ] ──► ŷ
 xₙ ──(wₙ)──┤
             │
 b  ─────────┘
```

---

## 2. Training Loop — One Epoch

```mermaid
flowchart TD
    A([Start: initialise w, b randomly]) --> B

    B[/"Next example (x, y)"/] --> C
    C["Compute z = w · x + b"] --> D
    D["Apply activation: ŷ = step(z)"] --> E
    E["Compute error = y − ŷ"] --> F

    F{error = 0?}
    F -->|Yes — correct| G([No update needed])
    F -->|No — wrong| H

    H["Update weights:\nwᵢ ← wᵢ + η · error · xᵢ"] --> I
    I["Update bias:\nb ← b + η · error"] --> J

    G --> J
    J{More examples\nin epoch?}
    J -->|Yes| B
    J -->|No| K

    K{All predictions\ncorrect?}
    K -->|No — run another epoch| B
    K -->|Yes — converged| L([Done])
```

---

## 3. Weight Update — What Changes and Why

```mermaid
flowchart LR
    subgraph "Case: error = +1  (predicted 0, actual 1)"
        direction LR
        e1["wᵢ too small"] --> u1["wᵢ ← wᵢ + η·xᵢ"] --> r1["Next time more likely to fire"]
    end

    subgraph "Case: error = −1  (predicted 1, actual 0)"
        direction LR
        e2["wᵢ too large"] --> u2["wᵢ ← wᵢ − η·xᵢ"] --> r2["Next time less likely to fire"]
    end

    subgraph "Case: error = 0   (correct)"
        direction LR
        e3["Already right"] --> u3["No change"] --> r3["Weights unchanged"]
    end
```

---

## 4. Decision Boundary — 2D Geometric View

For a 2-input perceptron, the decision boundary is the line `w₁x₁ + w₂x₂ + b = 0`.  
Everything above the line → ŷ = 1. Everything below → ŷ = 0.

### AND Gate (linearly separable ✓)

```
  x₂
  1 │  · (0,1)      ✗       ★ (1,1)
    │
  0 │  ✗ (0,0)              · (1,0)  ✗
    └──────────────────────────────── x₁
         0                  1

  ✗ = class 0   ★ = class 1

  Decision boundary separates ★(1,1) from the three ✗ points:

  x₂
  1 │  ·            ╲       ★
    │                ╲
  0 │  ✗              ╲     ·
    └──────────────────╲──────── x₁
                        ╲
```

### XOR Gate (NOT linearly separable ✗)

```
  x₂
  1 │  ★ (0,1)              · (1,1)  ✗
    │
  0 │  · (0,0)  ✗           ★ (1,0)
    └──────────────────────────────── x₁

  No single straight line can separate the two ★ points from the two ✗ points.
  The positive class is diagonal — requires a hidden layer (Phase 4).
```

---

## 5. Immutable Training — Scala Model

Each call to `trainOne` produces a **new** `Perceptron` (or returns `this` unchanged when `error == 0`). One epoch is a `foldLeft` over the dataset. `train` runs N epochs via a tail-recursive loop:

```mermaid
flowchart LR
    P0(["Perceptron₀\n(initial)"]) -->|"epoch 1\nfoldLeft over dataset"| P1
    P1(["Perceptron₁"]) -->|"epoch 2\nfoldLeft over dataset"| P2
    P2(["Perceptron₂"]) -->|"..."| Pn(["PerceptronN\n(after N epochs)"])
```

Inside each epoch, one `foldLeft` step:

```mermaid
flowchart LR
    Pa(["Pᵢ"]) -->|"trainOne(x, y)"| Pb(["Pᵢ₊₁"])
```

In code (all methods are extension methods on the companion — import with `import Perceptron.*`):

```scala
// one example at a time
val updated = perceptron.trainOne(input, actual)

// one full epoch (foldLeft inside train)
val afterOneEpoch = perceptron.train(dataset, epochs = 1)

// N epochs, tail-recursive, stack-safe
val trained = initial.train(data, epochs = 10)
```

The epoch count is explicit — the caller decides how long to train. There is no "train until converged" API; call `train` with increasing epochs and observe predictions to determine when results are good enough.
