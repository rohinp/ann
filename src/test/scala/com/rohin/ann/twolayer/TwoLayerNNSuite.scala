package com.rohin.ann.twolayer

import munit.FunSuite
import org.apache.commons.math3.linear.RealVector
import com.rohin.ann.twolayer.TwoLayerNN

class TwoLayerNNSuite extends FunSuite {
  import TwoLayerNN.*

  test("two layer network produces output between 0 and 1") {
    val nn = TwoLayerNN.create(
      inputSize = 2,
      hiddenSize = 2
    )

    val output = nn.forward(vec(1, 0)).a2

    assert(output > 0.0 && output < 1.0)
  }

  test("output delta is computed correctly") {

    val a2 = 0.5
    val y = 1

    val delta = computeOutputDelta(a2, y)

    assertEquals(delta, -0.125)
  }

  test("output weights move in correct direction") {

    val nn = TwoLayerNN.create(
      inputSize = 2,
      hiddenSize = 2
    )

    val a1 = vec(0.5, 0.5)
    val a2 = 0.5
    val y = 1

    val updated = nn.updateOutputLayer(a1, a2, y)

    // weights should increase
    assert(updated.W2.getEntry(0) > nn.W2.getEntry(0))
    assert(updated.W2.getEntry(1) > nn.W2.getEntry(1))

    // bias should increase
    assert(updated.b2 > nn.b2)
  }

  test("hidden delta is computed correctly") {

    val W2 = vec(1, 1)
    val delta2 = -0.125
    val a1 = vec(0.5, 0.5)

    val delta1 = computeHiddenDelta(W2, delta2, a1)

    assertEquals(delta1.getEntry(0), -0.03125)
    assertEquals(delta1.getEntry(1), -0.03125)
  }

  test("hidden layer weights update correctly") {

    val nn = TwoLayerNN.create(
      inputSize = 2,
      hiddenSize = 2
    )

    val input = vec(1, 1)
    val delta1 = vec(-0.03125, -0.03125)

    val updated = nn.updateHiddenLayer(delta1, input)

    // weights should increase
    assert(updated.W1.getEntry(0, 0) > nn.W1.getEntry(0, 0))
    assert(updated.W1.getEntry(1, 1) > nn.W1.getEntry(1, 1))

    // bias should increase
    assert(updated.b1.getEntry(0) > nn.b1.getEntry(0))
    assert(updated.b1.getEntry(1) > nn.b1.getEntry(1))
  }

  test("two layer network learns XOR") {

    val data = List(
      (vec(0, 0), 0),
      (vec(0, 1), 1),
      (vec(1, 0), 1),
      (vec(1, 1), 0)
    )

    val nn = TwoLayerNN.create(inputSize = 2, hiddenSize = 2, lr = 0.5)

    val trained = nn.train(data, epochs = 10000)

    /*
    Asserting when we get very close to the actual prediction of values, not exact.
     */
    assert(math.abs(trained.forward(vec(0, 0)).a2 - 0) <= 0.1)
    assert(
      trained.forward(vec(0, 1)).a2 > 0.9 && trained.forward(vec(0, 1)).a2 < 1.0
    )
    assert(
      trained.forward(vec(1, 0)).a2 > 0.9 && trained.forward(vec(1, 0)).a2 < 1.0
    )
    assert(math.abs(trained.forward(vec(1, 1)).a2 - 0) <= 0.1)
  }
}
