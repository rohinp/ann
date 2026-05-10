package com.rohin.ann.pytorchlite

import munit.FunSuite

trait FunSuitePlus extends FunSuite {
  def assertClose(
      expected: Double,
      obtained: Double,
      tolerance: Double = 0.001
  ): Unit = {
    assert(
      Math.abs(expected - obtained) < tolerance,
      s"$obtained was not close to $expected"
    )
  }

}
