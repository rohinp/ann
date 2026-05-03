package com.rohin.ann

import munit.FunSuite

class ActivationSuite extends FunSuite {
  import com.rohin.ann.Activation.*

  test("sigmoid produces expected values") {

    assertEquals(sigmoid(0), 0.5)
    assert(sigmoid(10) > 0.99)
    assert(sigmoid(-10) < 0.01)
  }

  test("sigmoid derivative is correct at 0") {
    val s = sigmoid(0) // = 0.5
    val d = sigmoidDerivative(0)

    assertEquals(d, 0.25)
  }
}
