package com.rohin.ann

import org.apache.commons.math3.linear.{ArrayRealVector, RealVector}
import scala.annotation.tailrec
import Activation.*

case class PerceptronV1 private (
    weights: RealVector,
    bias: Double,
    learningRate: Double
)

object PerceptronV1 {

  def vec(xs: Double*): RealVector =
    new ArrayRealVector(xs.toArray)

  def zeroVec(size: Int): RealVector =
    new ArrayRealVector(size)

  // With default learning rate
  def create(bias: Double, weights: RealVector): PerceptronV1 =
    new PerceptronV1(weights, bias, 0.001)

  def create(
      bias: Double,
      weights: RealVector,
      learningRate: Double
  ): PerceptronV1 =
    new PerceptronV1(weights, bias, learningRate)

  def step(z: Double): Int =
    if (z >= 0) 1 else 0

  extension (perceptronV1: PerceptronV1)
    def predictProb(input: RealVector): Double =
      val z = perceptronV1.weights.dotProduct(input) + perceptronV1.bias
      sigmoid(z)

  // w = w + η * (y - a) * x
  extension (perceptronV1: PerceptronV1)
    def updateAllWeights(step: Double, input: RealVector): PerceptronV1 =
      // Create an initial safe copy
      val workingWeights = perceptronV1.weights.copy()

      // Perform fast, mutating updates during the training loop
      perceptronV1.copy(
        weights = workingWeights.combine(1, step, input)
      )

  // b = b + η * (y - a)
  extension (perceptronV1: PerceptronV1)
    def updateBias(updateMagnitude: Double): PerceptronV1 =
      perceptronV1.copy(
        bias = perceptronV1.bias + updateMagnitude
      )

  extension (perceptronV1: PerceptronV1)
    def trainOne(input: RealVector, y: Int): PerceptronV1 =
      val a = perceptronV1.predictProb(input) // sigmoid
      val error = y - a
      // η * (y - a)
      val grad = error * a * (1 - a)
      val updateMagnitude = perceptronV1.learningRate * grad
      perceptronV1
        .updateAllWeights(updateMagnitude, input)
        .updateBias(updateMagnitude)

  extension (perceptronV1: PerceptronV1)
    def train(dataset: List[(RealVector, Int)], epochs: Int): PerceptronV1 =
      @tailrec
      def loop(iterate: Int, p: PerceptronV1): PerceptronV1 =
        if (iterate == 0) then p
        else
          loop(
            iterate - 1,
            dataset.foldLeft(p) { case (acc, (input, actual)) =>
              acc.trainOne(input, actual)
            }
          )
      loop(epochs, perceptronV1)
}
