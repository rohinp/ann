package com.rohin.ann.pytorchlite

import com.rohin.ann.pytorchlite.loss.{LossFunction, MeanSquaredError}

class LossSuite extends FunSuitePlus {
  test("output delta computed correctly") {

    val loss:LossFunction = MeanSquaredError
    val output = vec(0.5)
    val target = vec(1.0)

    val delta = loss.outputDelta(output, target)

    assertClose(delta.getEntry(0), -0.125)
  }

}
