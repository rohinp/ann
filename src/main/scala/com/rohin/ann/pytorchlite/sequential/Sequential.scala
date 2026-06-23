package com.rohin.ann.pytorchlite.sequential

import com.rohin.ann.pytorchlite.ConsoleLogging.DebugConfig
import com.rohin.ann.pytorchlite.dense.Dense
import com.rohin.ann.pytorchlite.loss.{LossFunction, MeanSquaredError}
import com.rohin.ann.pytorchlite.optimizer.{Gradients, LayerGradient, Optimizer, SGD}
import org.apache.commons.math3.linear.RealVector

import scala.annotation.tailrec

case class Sequential(
                       layers: List[Layer],
                       loss: LossFunction = MeanSquaredError,
                       optimizer: Optimizer = SGD(0.1)
                     ) {

  def forward(input: RealVector): Double = {

    @tailrec
    def loop(layers: List[Layer], current: RealVector): RealVector =
      layers match {
        case Nil =>
          current

        case layer :: remaining =>
          val (_, a) = layer.forward(current)
          loop(remaining, a)
      }

    loop(layers, input).getEntry(0)
  }

  def forwardPass(input: RealVector): ForwardPass = {

    @tailrec
    def loop(
              layers: List[Layer],
              currentActivation: RealVector,
              zsAcc: Vector[RealVector],
              activationsAcc: Vector[RealVector]
            ): ForwardPass =
      layers match {
        case Nil => ForwardPass(zsAcc.toList, activationsAcc.toList)
        case layer :: remaining =>
          val (z, a) = layer.forward(currentActivation)
          loop(remaining, a, zsAcc :+ z, activationsAcc :+ a)
      }

    loop(layers, input, Vector.empty, Vector(input)
    )
  }

  def backward(pass: ForwardPass, target: RealVector): Gradients = {
    val outputDelta: RealVector =
      loss.outputDelta(
        pass.activations.last,
        target
      )

    @tailrec
    def loop(l: Int, accGradient: List[LayerGradient]): List[LayerGradient] =
      if (l < 0) accGradient
      else {
        val nextLayer = layers(l + 1)
        val nextDelta = accGradient.head.dB

        val propagated = nextLayer.W.transpose().operate(nextDelta)

        val activation = pass.activations(l + 1)

        val grad = activation.map(a =>
          layers(l)
            .activation
            .derivativeFromActivation(a)
        )

        val delta = propagated.ebeMultiply(grad)

        val layerGradient = LayerGradient(
          dW = delta.outerProduct(pass.activations(l)),
          dB = delta
        )

        loop(l - 1, layerGradient :: accGradient)
      }

    val startIdx = layers.length - 2
    Gradients(loop(startIdx, List(LayerGradient(
      dW = outputDelta.outerProduct(pass.activations.init.last),
      dB = outputDelta
    ))))
  }

  def trainOne(
                input: RealVector,
                target: RealVector
              ): Sequential = {

    val pass: ForwardPass = forwardPass(input)

    val grads: Gradients = backward(pass, target)

    val updatedLayers = optimizer.step(
        layers,
        grads
      )

    copy(
      layers = updatedLayers
    )
  }

  def train(
             dataset: List[(RealVector, RealVector)],
             epochs: Int
           )(using dc: DebugConfig): Sequential = {

    @tailrec
    def loop(
              remaining: Int,
              current: Sequential
            ): Sequential =

      if remaining == 0 then current
      else

        val updated =
          dataset.foldLeft(current) { case (net, (input, target)) =>
            net.trainOne(
              input,
              target
            )
          }

        loop(
          remaining - 1,
          updated
        )

    loop(epochs, this)
  }

  def updateLoss(loss: LossFunction): Sequential =
    copy(loss = loss)

  def updateOptimizer(optimizer: Optimizer): Sequential =
    copy(optimizer = optimizer)
}

object Sequential {
  def apply(ds: Dense*) =
    new Sequential(ds.toList)
}
