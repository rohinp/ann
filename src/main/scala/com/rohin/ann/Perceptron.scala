package com.rohin.ann

import org.apache.commons.math3.linear.{ArrayRealVector, RealVector}
import scala.annotation.tailrec
import javax.print.DocFlavor.READER

case class Perceptron private (
    weights: RealVector,
    bias: Double,
    learningRate: Double
)

object Perceptron {

  def vec(xs: Double*): RealVector =
    new ArrayRealVector(xs.toArray)

  def zeroVec(size: Int): RealVector =
    new ArrayRealVector(size)

  // With default learning rate
  def create(bias: Double, weights: RealVector): Perceptron =
    new Perceptron(weights, bias, 0.001)

  def create(
      bias: Double,
      weights: RealVector,
      learningRate: Double
  ): Perceptron =
    new Perceptron(weights, bias, learningRate)

  def step(z: Double): Int =
    if (z >= 0) 1 else 0

  extension (perceptron: Perceptron)
    def predict(input: RealVector): Int =
      step(perceptron.weights.dotProduct(input) + perceptron.bias)
  extension (perceptron: Perceptron)
    def updateAllWeights(step: Double, ys: RealVector): Perceptron =
      // Create an initial safe copy
      val workingWeights = perceptron.weights.copy()

      // Perform fast, mutating updates during the training loop
      perceptron.copy(
        weights = workingWeights.combine(1, step, ys)
      )

  extension (perceptron: Perceptron)
    def updateBias(updateMagnitude: Double): Perceptron =
      perceptron.copy(
        bias = perceptron.bias + updateMagnitude
      )

  extension (perceptron: Perceptron)
    def trainOne(input: RealVector, actual: Int): Perceptron =
      val predicted = perceptron.predict(input)
      val error = actual - predicted

      if error == 0 then perceptron
      else
        val updateMagnitude = perceptron.learningRate * error
        perceptron
          .updateAllWeights(updateMagnitude, input)
          .updateBias(updateMagnitude)

  extension (perceptron: Perceptron)
    def train(dataset: List[(RealVector, Int)], epochs: Int): Perceptron =
      @tailrec
      def loop(iterate: Int, p: Perceptron): Perceptron =
        if (iterate == 0) then p
        else
          loop(
            iterate - 1,
            dataset.foldLeft(p) { case (acc, (input, actual)) =>
              acc.trainOne(input, actual)
            }
          )
      loop(epochs, perceptron)
}
