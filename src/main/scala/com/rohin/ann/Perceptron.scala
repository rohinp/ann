package com.rohin.ann

import org.apache.commons.math3.linear.{ArrayRealVector, RealVector}

final case class Perceptron(
    weights: RealVector,
    bias: Double,
    learningRate: LearningRate
):

  private def step(z: Double): Label =
    if z >= 0.0 then Label.One else Label.Zero

  def predict(input: RealVector): Label =
    require(
      input.getDimension == weights.getDimension,
      s"Input dimension ${input.getDimension} does not match weight dimension ${weights.getDimension}"
    )
    step(weights.dotProduct(input) + bias)

  def trainOne(input: RealVector, actual: Label): Perceptron =
    val error = actual.value - predict(input).value
    if error == 0 then this
    else
      val lr = learningRate.value
      copy(
        weights = weights.add(input.mapMultiply(lr * error)),
        bias    = bias + lr * error
      )

  def train(data: Seq[(RealVector, Label)]): Perceptron =
    data.foldLeft(this) { case (p, (input, label)) => p.trainOne(input, label) }

  def isConverged(data: Seq[(RealVector, Label)]): Boolean =
    data.forall { case (input, label) => predict(input) == label }

  def trainUntilConverged(
      data: Seq[(RealVector, Label)],
      maxEpochs: Int = 1000
  ): (Perceptron, Boolean) =
    @annotation.tailrec
    def loop(current: Perceptron, remaining: Int): (Perceptron, Boolean) =
      if current.isConverged(data) then (current, true)
      else if remaining == 0 then (current, false)
      else loop(current.train(data), remaining - 1)
    loop(this, maxEpochs)

object Perceptron:
  def zeroed(inputDimension: Int, learningRate: LearningRate): Perceptron =
    Perceptron(new ArrayRealVector(inputDimension), 0.0, learningRate)
