package com.rohin.ann.pytorchlite

import org.apache.commons.math3.linear.RealVector
import scala.collection.mutable.ListBuffer

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
}
