package com.rohin.ann.twolayer

import org.apache.commons.math3.linear.RealMatrix
import org.apache.commons.math3.linear.RealVector
import org.apache.commons.math3.linear.ArrayRealVector
import scala.util.Random
import org.apache.commons.math3.linear.Array2DRowRealMatrix
import org.apache.commons.math3.analysis.UnivariateFunction
import com.rohin.ann.Activation
import scala.annotation.tailrec

case class TwoLayerNN private (
    W1: RealMatrix, // hidden layer weights
    b1: RealVector,
    W2: RealVector, // output weights
    b2: Double,
    learningRate: Double
)

object TwoLayerNN {

  def vec(xs: Double*): RealVector =
    new ArrayRealVector(xs.toArray)

  def randomVec(size: Int): RealVector =
    new ArrayRealVector(Array.fill(size)(Random.between(0d, 1d)))

  def randomMatrix(rows: Int, cols: Int): RealMatrix = {
    val matrix = Array2DRowRealMatrix(rows, cols)
    for i <- (0 until rows) do matrix.setRowVector(i, randomVec(cols))
    matrix
  }

  def create(inputSize: Int, hiddenSize: Int, lr: Double = 1) = {
    val w1 = randomMatrix(inputSize, hiddenSize)
    val b1 = randomVec(hiddenSize)
    val w2 = randomVec(hiddenSize)
    val b2 = Random.between(0d, 1d)
    TwoLayerNN(W1 = w1, b1 = b1, W2 = w2, b2 = b2, lr)
  }

  extension (nn: TwoLayerNN)
    def forward(input: RealVector): ForwardCache = {
      /*Step 1:
        z₁ = W₁ · x + b₁
        a₁ = sigmoid(z₁)
       */
      val z1 = nn.W1.operate(input).add(nn.b1)
      /*
        Special case because if a map
        ```val a1 = z1.map(Activation.sigmoid)```
        is used scala compiler tries to convert it to a java function type.
        Compilation fails and it makes scala compiler go bonkers.
       */
      val sigmoidFn = new UnivariateFunction {
        def value(z: Double): Double = Activation.sigmoid(z)
      }
      val a1 = z1.map(sigmoidFn)
      /*Step 2:
        z₂ = W₂ · a₁ + b₂
        a₂ = sigmoid(z₂)
       */
      val z2 = nn.W2.dotProduct(a1) + nn.b2
      val a2 = Activation.sigmoid(z2)
      ForwardCache(
        z1 = z1,
        a1 = a1,
        z2 = z2,
        a2 = a2
      )
    }

  def computeOutputDelta(a2: Double, y: Int): Double = {
    /*
    (a₂ - y) → how wrong you are
    a₂(1-a₂) → how sensitive the neuron is
     */
    val error = a2 - y
    val grad = a2 * (1 - a2)
    error * grad
  }

  extension (nn: TwoLayerNN)
    def updateOutputLayer(
        a1: RealVector,
        a2: Double,
        y: Int
    ): TwoLayerNN = {
      val delta2 = computeOutputDelta(a2, y)
      // W2 = W2 - (η * δ₂ * a1)
      val newW2 =
        nn.W2.subtract(a1.mapMultiply(nn.learningRate).mapMultiply(delta2))
      // b₂ = b₂ - η * δ₂
      val newB2 = nn.b2 - nn.learningRate * delta2
      nn.copy(W2 = newW2, b2 = newB2)
    }

  def computeHiddenDelta(
      W2: RealVector,
      delta2: Double,
      a1: RealVector
  ): RealVector = {
    val propagated = W2.mapMultiply(delta2)
    val grad = a1.map(a => a * (1 - a))
    propagated.ebeMultiply(grad)
  }

  extension (nn: TwoLayerNN)
    def updateHiddenLayer(
        delta1: RealVector,
        input: RealVector
    ): TwoLayerNN = {
      // W1​=W1​−η⋅δ1​⋅xT
      val gradW1 = delta1.outerProduct(input)

      val newW1 = nn.W1.subtract(gradW1.scalarMultiply(nn.learningRate))
      val newB1 = nn.b1.subtract(delta1.mapMultiply(nn.learningRate))

      nn.copy(W1 = newW1, b1 = newB1)
    }
  extension (nn: TwoLayerNN)
    def trainOne(input: RealVector, y: Int): TwoLayerNN = {
      val cache = nn.forward(input)
      val delta2 = computeOutputDelta(cache.a2, y)
      val delta1 = computeHiddenDelta(nn.W2, delta2, cache.a1)

      nn
        .updateOutputLayer(a1 = cache.a1, a2 = cache.a2, y = y)
        .updateHiddenLayer(delta1 = delta1, input = input)
    }

  extension (nn: TwoLayerNN)
    def train(dataset: List[(RealVector, Int)], epochs: Int): TwoLayerNN = {
      @tailrec
      def loop(iterate: Int, accNN: TwoLayerNN): TwoLayerNN = {
        if iterate == 0 then accNN
        else
          loop(
            iterate - 1,
            dataset.foldLeft(accNN) { case (acc, (input, actual)) =>
              acc.trainOne(input, actual)
            }
          )
      }
      loop(epochs, nn)
    }
}
