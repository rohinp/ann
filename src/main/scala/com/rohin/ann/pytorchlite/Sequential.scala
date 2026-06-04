package com.rohin.ann.pytorchlite

import org.apache.commons.math3.linear.RealVector
import scala.collection.mutable.ListBuffer
import org.apache.commons.math3.linear.RealMatrix
import ConsoleLogging._
import scala.annotation.tailrec

case class Sequential(layers: List[Dense])

object Sequential {

  def apply(ds: Dense*) =
    new Sequential(ds.toList)

  extension (sequential: Sequential)
    def forward(input: RealVector): Double = {
      sequential.layers
        .foldLeft(input) { case (current, layer) =>
          val (_, a) = layer.forward(current)
          a
        }
        .getEntry(0)
    }

  extension (sequential: Sequential)
    def forwardPass(input: RealVector): ForwardPass = {
      val (zs, activations) = sequential.layers
        .foldLeft((Vector.empty[RealVector], Vector(input))) {
          case ((zsAcc, activationsAcc), layer) =>
            val (z, a) = layer.forward(activationsAcc.last)
            (zsAcc :+ z, activationsAcc :+ a)
        }
      ForwardPass(zs.toList, activations.toList)
    }

  extension (sequential: Sequential)
    def backward(pass: ForwardPass, target: RealVector): Gradients = {
      val outputDelta =
        Loss.outputDelta(
          pass.activations.last,
          target
        )

      var deltas = List(outputDelta)

      for (l <- sequential.layers.length - 2 to 0 by -1) {

        val nextLayer = sequential.layers(l + 1)
        val nextDelta = deltas.head

        val propagated =
          nextLayer.W.transpose().operate(nextDelta)

        val activation =
          pass.activations(l + 1)

        val grad =
          activation.map(a =>
            sequential
              .layers(l)
              .activation
              .derivativeFromActivation(a)
          )

        val delta =
          propagated.ebeMultiply(grad)

        deltas = delta :: deltas
      }

      Gradients(deltas)
    }

  extension (sequential: Sequential)
    def applyGradients(
        pass: ForwardPass,
        grads: Gradients,
        learningRate: Double
    )(using dc: DebugConfig): Sequential = {
      val updatedLayers =
        sequential.layers.zipWithIndex.map {
          case (layer, l) => {

            log("layer = " + l)

            val prevActivation =
              pass.activations(l)

            log("prevActivation = " + prevActivation)

            val delta =
              grads.deltas(l)

            log("delta = " + delta)
            val gradW =
              delta.outerProduct(prevActivation)

            log("gradW = " + gradW)

            val newW: RealMatrix =
              layer.W.subtract(
                gradW.scalarMultiply(learningRate)
              )

            log("newW = " + newW)

            val newB: RealVector =
              layer.b.subtract(
                delta.mapMultiply(learningRate)
              )

            log("newB = " + newB)

            layer.update(newW, newB)
          }
        }

      sequential.copy(layers = updatedLayers)
    }

  extension (sequential: Sequential)
    def trainOne(
        input: RealVector,
        target: RealVector,
        learningRate: Double
    )(using dc: DebugConfig): Sequential = {

      val pass =
        sequential.forwardPass(input)

      val grads =
        sequential.backward(pass, target)

      println("grads.deltas " + grads.deltas)
      sequential.applyGradients(
        pass,
        grads,
        learningRate
      )
    }

  extension (sequential: Sequential)

    def train(
        dataset: List[(RealVector, RealVector)],
        epochs: Int,
        learningRate: Double
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
                target,
                learningRate
              )
            }

          loop(
            remaining - 1,
            updated
          )

      loop(epochs, sequential)
    }
}
