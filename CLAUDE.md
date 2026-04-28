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

`Perceptron` (`com.rohin.ann.Perceptron`) is an **immutable case class** — training returns a new instance rather than mutating state. This is the deliberate design pattern for all model types in this project: pure functions, no side effects, fold over a dataset to produce a trained model.

The learning rule in `trainOne` follows the classic perceptron update:
- `w_new = w + lr * (actual - predicted) * x`
- Returns a `copy(...)` with updated weights and bias

`Main.scala` is currently a placeholder. As phases progress, it should become a training harness or demo entry point.

## Testing

Every piece of production code must have test coverage. Tests live in `src/test/scala/` and use [munit](https://scalameta.org/munit/).

Required coverage for each model/component:
- The happy path (correct prediction, correct weight update)
- Edge cases: zero weights, zero input, learning rate of 0, single-feature input
- Dataset-level training: train to convergence on AND and OR, verify the model correctly refuses XOR (never converges)
- Dimension mismatch must throw, not silently produce a wrong answer

Run a single suite during development:
```bash
sbt "testOnly com.rohin.ann.PerceptronSuite"
```

## Code Conventions

- Scala 3 syntax throughout (indentation-based, `then`/`end`, etc.)
- `.scalafmt.conf` dialect is `scala213` but the project uses Scala 3 — update to `runner.dialect = scala3` before relying on formatter output.
- Math operations use `org.apache.commons.math3.linear` (`ArrayRealVector`, `RealMatrix`) — do not re-implement what commons-math3 already provides.

## Documentation

Theory and implementation notes for each model live in `docs/`. Start with `docs/perceptron.md` — it covers the math, learning rule, linear separability, and a walkthrough of the current implementation including known issues.
