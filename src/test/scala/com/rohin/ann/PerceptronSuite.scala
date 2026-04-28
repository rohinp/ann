package com.rohin.ann

import munit.FunSuite
import org.apache.commons.math3.linear.{ArrayRealVector, RealVector}

class PerceptronSuite extends FunSuite:

  // ── helpers ────────────────────────────────────────────────────────────────

  val lr: LearningRate = LearningRate.of(0.1).getOrElse(sys.error("bad lr"))

  def vec(xs: Double*): RealVector = new ArrayRealVector(xs.toArray)

  // truth-table datasets
  val andData: Seq[(RealVector, Label)] = Seq(
    (vec(0.0, 0.0), Label.Zero),
    (vec(0.0, 1.0), Label.Zero),
    (vec(1.0, 0.0), Label.Zero),
    (vec(1.0, 1.0), Label.One)
  )

  val orData: Seq[(RealVector, Label)] = Seq(
    (vec(0.0, 0.0), Label.Zero),
    (vec(0.0, 1.0), Label.One),
    (vec(1.0, 0.0), Label.One),
    (vec(1.0, 1.0), Label.One)
  )

  val xorData: Seq[(RealVector, Label)] = Seq(
    (vec(0.0, 0.0), Label.Zero),
    (vec(0.0, 1.0), Label.One),
    (vec(1.0, 0.0), Label.One),
    (vec(1.0, 1.0), Label.Zero)
  )

  // ── LearningRate ───────────────────────────────────────────────────────────

  test("LearningRate.of accepts values in (0.0, 1.0]") {
    assert(LearningRate.of(0.01).isRight)
    assert(LearningRate.of(0.5).isRight)
    assert(LearningRate.of(1.0).isRight)
  }

  test("LearningRate.of rejects zero") {
    assert(LearningRate.of(0.0).isLeft)
  }

  test("LearningRate.of rejects negative values") {
    assert(LearningRate.of(-0.1).isLeft)
  }

  test("LearningRate.of rejects values greater than 1.0") {
    assert(LearningRate.of(1.1).isLeft)
  }

  test("LearningRate.value returns the underlying Double") {
    assertEquals(LearningRate.of(0.3).map(_.value), Right(0.3))
  }

  // ── Label ──────────────────────────────────────────────────────────────────

  test("Label.fromInt(0) returns Zero") {
    assertEquals(Label.fromInt(0), Right(Label.Zero))
  }

  test("Label.fromInt(1) returns One") {
    assertEquals(Label.fromInt(1), Right(Label.One))
  }

  test("Label.fromInt rejects any value other than 0 or 1") {
    assert(Label.fromInt(-1).isLeft)
    assert(Label.fromInt(2).isLeft)
  }

  test("Label.value returns the underlying Int") {
    assertEquals(Label.Zero.value, 0)
    assertEquals(Label.One.value, 1)
  }

  // ── Perceptron construction ────────────────────────────────────────────────

  test("zeroed creates a perceptron with all-zero weights and bias") {
    val p = Perceptron.zeroed(2, lr)
    assertEquals(p.weights.getEntry(0), 0.0)
    assertEquals(p.weights.getEntry(1), 0.0)
    assertEquals(p.bias, 0.0)
  }

  test("zeroed creates weights of the correct dimension") {
    assertEquals(Perceptron.zeroed(3, lr).weights.getDimension, 3)
    assertEquals(Perceptron.zeroed(5, lr).weights.getDimension, 5)
  }

  // ── predict ────────────────────────────────────────────────────────────────

  test("predict returns One when pre-activation is positive") {
    val p = Perceptron(vec(1.0, 1.0), bias = 0.0, learningRate = lr)
    assertEquals(p.predict(vec(1.0, 1.0)), Label.One)
  }

  test("predict returns Zero when pre-activation is negative") {
    val p = Perceptron(vec(1.0, 1.0), bias = -10.0, learningRate = lr)
    assertEquals(p.predict(vec(1.0, 1.0)), Label.Zero)
  }

  test("predict returns One when pre-activation is exactly zero (boundary is positive class)") {
    // w·x + b = 1*1 + (-1)*1 + 0 = 0
    val p = Perceptron(vec(1.0, -1.0), bias = 0.0, learningRate = lr)
    assertEquals(p.predict(vec(1.0, 1.0)), Label.One)
  }

  test("predict throws IllegalArgumentException on dimension mismatch") {
    val p = Perceptron.zeroed(2, lr)
    intercept[IllegalArgumentException] {
      p.predict(vec(1.0, 2.0, 3.0))
    }
  }

  // ── trainOne ───────────────────────────────────────────────────────────────

  test("trainOne returns the same reference when prediction is already correct (no-op)") {
    // w=(1,1), bias=0.5 → z=2.5 → One; actual=One → error=0
    val p = Perceptron(vec(1.0, 1.0), bias = 0.5, learningRate = lr)
    val result = p.trainOne(vec(1.0, 1.0), Label.One)
    assert(result eq p, "Expected same object reference — trainOne must not allocate on error=0")
  }

  test("trainOne increases weights on a false negative (predicted Zero, actual One)") {
    // bias=-1.0 forces z=-1.0 regardless of input, so initial prediction is always Zero
    val p = Perceptron(vec(0.0, 0.0), bias = -1.0, learningRate = lr)
    val trained = p.trainOne(vec(1.0, 1.0), Label.One)
    assert(trained.weights.getEntry(0) > p.weights.getEntry(0))
    assert(trained.weights.getEntry(1) > p.weights.getEntry(1))
  }

  test("trainOne decreases weights on a false positive (predicted One, actual Zero)") {
    val p = Perceptron(vec(1.0, 1.0), bias = 10.0, learningRate = lr)
    val trained = p.trainOne(vec(1.0, 1.0), Label.Zero)
    assert(trained.weights.getEntry(0) < p.weights.getEntry(0))
    assert(trained.weights.getEntry(1) < p.weights.getEntry(1))
  }

  test("trainOne applies the correct weight delta: Δwᵢ = η × error × xᵢ") {
    // bias=-1.0 ensures z<0 → predicts Zero, so error = 1 - 0 = 1
    val p = Perceptron(vec(0.0, 0.0), bias = -1.0, learningRate = lr)
    val trained = p.trainOne(vec(0.5, 0.8), Label.One)
    assertEqualsDouble(trained.weights.getEntry(0), 0.1 * 1 * 0.5, delta = 1e-10)
    assertEqualsDouble(trained.weights.getEntry(1), 0.1 * 1 * 0.8, delta = 1e-10)
  }

  test("trainOne applies the correct bias delta: Δb = η × error") {
    // bias=-1.0 ensures initial prediction is Zero, so error = 1
    val p = Perceptron(vec(0.0, 0.0), bias = -1.0, learningRate = lr)
    val trained = p.trainOne(vec(1.0, 1.0), Label.One)
    assertEqualsDouble(trained.bias, -1.0 + 0.1 * 1, delta = 1e-10)
  }

  test("trainOne does not update the weight for a zero-valued input feature") {
    // bias=-1.0 ensures initial prediction is Zero, so error = 1
    val p = Perceptron(vec(0.0, 0.0), bias = -1.0, learningRate = lr)
    val trained = p.trainOne(vec(0.0, 1.0), Label.One)
    assertEquals(trained.weights.getEntry(0), 0.0) // x=0 → no update regardless of error
    assert(trained.weights.getEntry(1) > 0.0)
  }

  test("trainOne preserves learningRate") {
    val p = Perceptron.zeroed(2, lr)
    assertEquals(p.trainOne(vec(1.0, 1.0), Label.One).learningRate, lr)
  }

  test("trainOne does not mutate the original Perceptron") {
    val p       = Perceptron.zeroed(2, lr)
    val wBefore = p.weights.getEntry(0)
    val bBefore = p.bias
    p.trainOne(vec(1.0, 1.0), Label.One)
    assertEquals(p.weights.getEntry(0), wBefore)
    assertEquals(p.bias, bBefore)
  }

  // ── train (epoch) ──────────────────────────────────────────────────────────

  test("train on an empty dataset returns the same Perceptron") {
    val p = Perceptron.zeroed(2, lr)
    assertEquals(p.train(Seq.empty), p)
  }

  test("train on a single example is equivalent to trainOne") {
    val p       = Perceptron.zeroed(2, lr)
    val input   = vec(1.0, 0.5)
    val fromTrain    = p.train(Seq((input, Label.One)))
    val fromTrainOne = p.trainOne(input, Label.One)
    assertEqualsDouble(fromTrain.weights.getEntry(0), fromTrainOne.weights.getEntry(0), delta = 1e-10)
    assertEqualsDouble(fromTrain.weights.getEntry(1), fromTrainOne.weights.getEntry(1), delta = 1e-10)
    assertEqualsDouble(fromTrain.bias, fromTrainOne.bias, delta = 1e-10)
  }

  // ── isConverged ────────────────────────────────────────────────────────────

  test("isConverged returns true when all predictions are correct") {
    // known analytical solution to AND: w=(1,1), b=-1.5
    val p = Perceptron(vec(1.0, 1.0), bias = -1.5, learningRate = lr)
    assert(p.isConverged(andData))
  }

  test("isConverged returns false when any prediction is wrong") {
    val p = Perceptron.zeroed(2, lr) // always predicts Zero → wrong on (1,1)→One
    assert(!p.isConverged(andData))
  }

  // ── trainUntilConverged ────────────────────────────────────────────────────

  test("trainUntilConverged solves AND gate") {
    val (trained, converged) = Perceptron.zeroed(2, lr).trainUntilConverged(andData)
    assert(converged)
    assert(trained.isConverged(andData))
  }

  test("trainUntilConverged solves OR gate") {
    val (trained, converged) = Perceptron.zeroed(2, lr).trainUntilConverged(orData)
    assert(converged)
    assert(trained.isConverged(orData))
  }

  test("trainUntilConverged does not converge on XOR (not linearly separable)") {
    val (_, converged) = Perceptron.zeroed(2, lr).trainUntilConverged(xorData, maxEpochs = 1000)
    assert(!converged)
  }

  test("trainUntilConverged respects maxEpochs and stops — does not run indefinitely") {
    val (_, converged) = Perceptron.zeroed(2, lr).trainUntilConverged(xorData, maxEpochs = 5)
    assert(!converged)
  }

  test("trainUntilConverged returns converged=true when already converged before training") {
    val p = Perceptron(vec(1.0, 1.0), bias = -1.5, learningRate = lr)
    val (_, converged) = p.trainUntilConverged(andData)
    assert(converged)
  }
