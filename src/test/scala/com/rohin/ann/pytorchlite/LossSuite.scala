package com.rohin.ann.pytorchlite

class LossSuite extends FunSuitePlus {
  test("output delta computed correctly") {

    val output = vec(0.5)
    val target = vec(1.0)

    val delta = Loss.outputDelta(output, target)

    assertClose(delta.getEntry(0), -0.125)
  }

}
