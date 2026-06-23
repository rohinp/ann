package com.rohin.ann.pytorchlite

import munit.FunSuite

class ActivationFnSuite extends FunSuite {
  import com.rohin.ann.pytorchlite.dense.ActivationFn.*

  test("sigmoid activation works") {
    val s = Sigmoid
    assertEquals(s.forward(0), 0.5)
    assertEquals(s.derivativeFromActivation(0.5), 0.25)
  }

  test("relu activation works") {

    assertEquals(ReLU.forward(-5), 0.0)
    assertEquals(ReLU.forward(0), 0.0)
    assertEquals(ReLU.forward(5), 5.0)

  }

  test("relu derivative works") {

    assertEquals(
      ReLU.derivativeFromActivation(0),
      0.0
    )

    assertEquals(
      ReLU.derivativeFromActivation(5),
      1.0
    )
  }
}
