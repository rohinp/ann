package com.rohin.ann.pytorchlite

import munit.FunSuite
import com.rohin.ann.pytorchlite.Dense
import com.rohin.ann.pytorchlite.ActivationFn.*
class SequentialSuite extends FunSuite {
  test("sequential network forward works") {

    val net = Sequential(
      Dense(2, 2, Sigmoid),
      Dense(2, 1, Sigmoid)
    )

    val output = net.forward(vec(1, 0))

    assert(output > 0.0 && output < 1.0)
  }

  test("forward pass stores activations and zs") {

    val net = Sequential(
      Dense(2, 2, Sigmoid),
      Dense(2, 1, Sigmoid)
    )

    val pass = net.forwardPass(vec(1, 0))

    assertEquals(pass.activations.size, 3)
    assertEquals(pass.zs.size, 2)
  }
}
