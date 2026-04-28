package com.rohin.ann

final case class Perceptron(
    weights: Vector[Double],
    bias: Double,
    learningRate: Double
) {

  private def dot(a: Vector[Double], b: Vector[Double]): Double =
    a.zip(b).map { case (x, y) => x * y }.sum

  private def step(z: Double): Int =
    if z >= 0 then 1 else 0

  def predict(input: Vector[Double]): Int = {
    val z = dot(weights, input) + bias
    step(z)
  }

  def trainOne(input: Vector[Double], actual: Int): Perceptron = {
    val prediction = predict(input)
    val error = actual - prediction

    val updatedWeights =
      weights.zip(input).map { case (w, x) =>
        w + learningRate * error * x
      }

    val updatedBias =
      bias + learningRate * error

    copy(weights = updatedWeights, bias = updatedBias)
  }
}
