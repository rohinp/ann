package com.rohin.ann.pytorchlite

import org.apache.commons.math3.linear.RealVector

object Loss {

  def outputDelta(
      output: RealVector,
      target: RealVector
  ): RealVector = {

    val result = output.copy()

    for (i <- 0 until output.getDimension) {
      val a = output.getEntry(i)
      val y = target.getEntry(i)

      result.setEntry(
        i,
        (a - y) * (a * (1 - a))
      )
    }

    result
  }
}
