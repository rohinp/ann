package com.rohin.ann.pytorchlite.loss

import org.apache.commons.math3.linear.RealVector

trait LossFunction {

  def outputDelta(
                   output: RealVector,
                   target: RealVector
                 ): RealVector

}