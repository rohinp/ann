package com.rohin.ann.pytorchlite

import munit.FunSuite

class ActivationFnSuite extends FunSuite {
  import com.rohin.ann.pytorchlite.ActivationFn

  test("sigmoid activation works") {
    val s = ActivationFn.Sigmoid
    assertEquals(s.forward(0), 0.5)
    assertEquals(s.derivativeFromActivation(0.5), 0.25)
  }
}
