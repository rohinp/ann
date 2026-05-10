package com.rohin.ann.pytorchlite

import org.apache.commons.math3.analysis.UnivariateFunction

trait ActivationFn {
  def forward(x: Double): Double
  /*
    Special case because if a map
    ```z1.map(ActivationFn.forward)```
    is used scala compiler tries to eta-expantion but it is not a java function type.
    Compilation fails and it makes scala compiler go bonkers.
   */
  def forwardFn: UnivariateFunction = new UnivariateFunction {
    def value(z: Double): Double = forward(z)
  }

  def derivativeFromActivation(a: Double): Double
}
object ActivationFn {
  object Sigmoid extends ActivationFn {
    def forward(x: Double) = 1.0 / (1.0 + math.exp(-x))
    def derivativeFromActivation(a: Double) = a * (1 - a)
  }
}
