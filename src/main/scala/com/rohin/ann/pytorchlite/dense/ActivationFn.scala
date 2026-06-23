package com.rohin.ann.pytorchlite.dense

import org.apache.commons.math3.analysis.UnivariateFunction

trait ActivationFn {
  def forward(x: Double): Double
  /*
    Special case because if a map
    ```z1.map(ActivationFn.forward)```
    is used scala compiler tries to eta-expantion but it is not a java function type.
    Compilation fails and it makes scala compiler go bonkers.
   */
  def forwardFn: UnivariateFunction = (z: Double) => forward(z)

  def derivativeFromActivation(a: Double): Double
}
object ActivationFn {
  object Sigmoid extends ActivationFn {
    def forward(x: Double): Double = 1.0 / (1.0 + math.exp(-x))
    def derivativeFromActivation(a: Double): Double = a * (1 - a)

    override def toString: String = "σ(x)"
  }

  object ReLU extends ActivationFn {

    override def forward(x: Double): Double = Math.max(0,x)

    override def derivativeFromActivation(a: Double): Double = if a > 0 then 1 else 0

    override def toString: String = "ReLU(x)"
  }
}
