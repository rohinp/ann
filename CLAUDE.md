# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A ground-up implementation of neural networks on the JVM using Scala 3, built for learning. The goal is to understand and implement ML from first principles, then explore JVM performance and GPU acceleration (Project Babylon / HAT) in later phases.

Currently in **Phase 1 (Perceptron)**. See `README.md` for the full multi-phase roadmap.

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

**`Perceptron`** is a `case class` with a `private` constructor. All behaviour lives in its companion object as **extension methods** (`predict`, `trainOne`, `train`, `updateAllWeights`, `updateBias`). Construction goes through:

- `Perceptron.create(bias, weights)` — default learning rate 0.001
- `Perceptron.create(bias, weights, learningRate)` — explicit learning rate
- `Perceptron.vec(xs: Double*)` / `Perceptron.zeroVec(size)` — vector factory helpers so callers never import `ArrayRealVector` directly

`weights` is `RealVector` from commons-math3. The weight update uses `RealVector.combine(1, η*error, input)` which is `w + η·error·x` as a single native vector op. `trainOne` returns `this` unchanged when `error == 0`.

`train(dataset, epochs)` runs N epochs via a `@tailrec` inner loop. Training duration is always explicit — there is no "run until converged" API.

`Main.scala` is a placeholder.

## Testing

Tests are written TDD-style — the test is the specification. Tests live in `src/test/scala/com/rohin/ann/` and use [munit](https://scalameta.org/munit/). Every piece of production code must have test coverage.

Test suites use `import Perceptron.*` to access extension methods and factory helpers without qualification.

Every model/component needs tests for:
- Correct prediction (forward pass)
- Boundary case (`z = 0` → positive class)
- Weight and bias update when prediction is wrong (verify exact delta)
- No update when prediction is already correct
- Dataset-level training: verify AND and OR converge; XOR test is commented out with an explanation of why it cannot converge

Run a single suite during development:
```bash
sbt "testOnly com.rohin.ann.PerceptronSuite"
```

## Code Conventions

- Scala 3 syntax throughout (indentation-based, `then`/`end`, etc.)
- `.scalafmt.conf` dialect is `scala213` but the project uses Scala 3 — update to `runner.dialect = scala3` before relying on formatter output.
- Math operations use `org.apache.commons.math3.linear` (`ArrayRealVector`, `RealVector`) — do not re-implement what commons-math3 already provides.

## Documentation

Theory and implementation notes for each model live in `docs/`. Start with `docs/perceptron.md` — it covers the math, learning rule, linear separability, and a walkthrough of the current implementation including known issues.
