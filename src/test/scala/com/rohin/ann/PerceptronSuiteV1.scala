package com.rohin.ann

import munit.FunSuite
import org.apache.commons.math3.linear.{ArrayRealVector, RealVector}
import com.rohin.ann.PerceptronV1

class PerceptronSuiteV1 extends FunSuite {
  import PerceptronV1.*

  test("weights move in correct direction with sigmoid learning") {

    val p = create(bias = 0, weights = vec(0, 0), learningRate = 1.0)

    val updated = p.trainOne(input = vec(1, 1), y = 1)

    /*
    We are NOT asserting exact values.
    Importnt: Only direction of change
     */
    // weights should increase
    assert(updated.weights.getEntry(0) > 0)
    assert(updated.weights.getEntry(1) > 0)

    // bias should increase
    assert(updated.bias > 0)
  }

  test("updates are smaller when prediction is confident") {

    val p = create(bias = 0, weights = vec(5, 5), learningRate = 1.0)

    val input = vec(1, 1)

    val before = p.predictProb(input)

    val updated = p.trainOne(input, 1)

    val after = updated.predictProb(input)

    // change should be small because already near 1
    assert(Math.abs(after - before) < 0.01)
  }
}
