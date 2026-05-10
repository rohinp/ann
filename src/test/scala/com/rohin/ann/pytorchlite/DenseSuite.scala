package com.rohin.ann.pytorchlite

import munit.FunSuite
import com.rohin.ann.pytorchlite.Dense

class DenseSuite extends FunSuite {
  test("dense layer forward produces output of correct size") {

    val layer =
      Dense(inputSize = 2, outputSize = 3, act = ActivationFn.Sigmoid)

    val input = vec(1, 0)

    val output = layer.forward(input)._2

    assertEquals(output.getDimension, 3)
  }
}
