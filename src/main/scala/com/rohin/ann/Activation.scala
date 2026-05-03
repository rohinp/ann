package com.rohin.ann

object Activation {

  // 𝜎(𝑥)
  def sigmoid(z: Double): Double =
    1.0 / (1.0 + Math.exp(-z))

  // 𝜎(𝑥)⋅(1−𝜎(𝑥))
  def sigmoidDerivative(z: Double): Double =
    val s = sigmoid(z)
    s * (1 - s)

  def sigmoidDerivativeFromActivation(a: Double): Double =
    a * (1 - a)

}
