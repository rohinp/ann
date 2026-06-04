# Phase 5 — pytorchlite

`pytorchlite` is the first refactor of this project that treats layers, activations, and training utilities as composable parts. The target was to keep the math from `TwoLayerNN`, but expose it in the way PyTorch lets you stack layers and call `Sequential`.

---

## Design Goals

- Make activations pluggable instead of hardcoding sigmoid in the model
- Encapsulate the weight matrix + bias vector into a `Dense` layer
- Use an ordered container (`Sequential`) so arbitrary stacks are possible
- Surface explicit gradient objects to keep forward/backward readable
- Keep dependencies limited to commons-math3 and Scala stdlib

---

## Building Blocks

### ActivationFn

- Trait with `forward` and `derivativeFromActivation`
- `forwardFn: UnivariateFunction` adapter exists purely so `RealVector.map` will accept the function without SAM-conversion issues
- Ships with `ActivationFn.Sigmoid`; more activations can extend the trait

### Dense

- Holds a `RealMatrix` (`output × input`) and a bias `RealVector`
- Randomly initialises both via helpers in `package.scala` (`Random.nextDouble() - 0.5`)
- `forward` returns `(z, a)` so callers can cache linear and activated values
- `update` returns a copy with new weights/biases, keeping instances immutable

### ForwardPass & Gradients

- `ForwardPass` is the cache: `zs` and `activations`, with `activations.head` equal to the input vector
- `Gradients` wraps the list of deltas, aligned with the layer order (index `i` matches `layers(i)`)

### Loss

- `Loss.outputDelta` implements `(a − y) · a · (1 − a)` (MSE loss flowing through sigmoid)
- Returns a `RealVector` that seeds backprop

### Sequential

- Stores `List[Dense]`
- `forward` reduces the layers and returns a scalar for convenience when the last layer has a single output
- `forwardPass` accumulates caches for every layer
- `backward` walks the layers in reverse, multiplying each delta by the transposed weights of the next layer and the derivative of the current layer's activation
- `applyGradients` computes the outer product `delta ⊗ activation` for every layer and subtracts the scaled gradient/bias delta
- `trainOne` = forward + backward + apply
- `train` folds `trainOne` across a dataset for `epochs` iterations (tail-recursive loop)

### Logging Helpers

- `ConsoleLogging.DebugConfig` controls whether gradient math is printed
- `FunSuiteWithLogging` in tests provides a default config so you can flip verbosity per suite without wiring loggers through every method call

---

## Training Flow Recap

1. Build a network with as many `Dense` layers as you like:
   ```scala
   val net = Sequential(
     Dense(2, 2, ActivationFn.Sigmoid),
     Dense(2, 1, ActivationFn.Sigmoid)
   )
   ```
2. Call `forwardPass` to capture activations if you need manual inspection
3. Use `trainOne` inside a fold to apply SGD to each example, or call `train` with a dataset + epoch count
4. Tune `learningRate` per experiment; defaults live at the call site so every test is explicit

---

## Test Map

| Suite | Focus |
| --- | --- |
| `ActivationFnSuite` | Sigmoid math stays correct when factored out |
| `DenseSuite` | Layer forward pass keeps dimensions consistent |
| `LossSuite` | Output delta matches the MSE + sigmoid derivative |
| `SequentialSuite` | Forward cache, backward deltas, gradient steps, and training helpers all mutate the network as expected |
| `TrainingSuite` | End-to-end XOR training using the sequential stack |

Refer back to `docs/tdd_walkthrough.md` for the narrative behind each spec and how it connects to earlier phases.

---

## Next Steps

Phase 6 will build on this abstraction with batched matrix ops, which should slot naturally into `Sequential` once inputs, deltas, and gradients are treated as matrices instead of vectors. Until then, `pytorchlite` is the cleanest playground for experimenting with layer shapes, activations, and logging.
