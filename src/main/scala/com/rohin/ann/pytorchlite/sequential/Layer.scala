package com.rohin.ann.pytorchlite.sequential

import com.rohin.ann.pytorchlite.dense.ActivationFn
import org.apache.commons.math3.linear.{RealMatrix, RealVector}

trait Layer {
  val W: RealMatrix
  val b: RealVector
  val activation: ActivationFn
  def forward(input: RealVector): (RealVector, RealVector)
  def update(w: RealMatrix, b: RealVector): Layer
}
