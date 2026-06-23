package com.rohin.ann.pytorchlite.optimizer

import com.rohin.ann.pytorchlite.dense.Dense
import com.rohin.ann.pytorchlite.sequential.Layer

case class SGD(
                learningRate: Double
              ) extends Optimizer {

  def step(
            layers: List[Layer],
            gradients: Gradients
          ): List[Layer] =

    layers
      .zip(gradients.layers)
      .map {
        case (layer, grad) =>

          val newW =
            layer.W.subtract(
              grad.dW.scalarMultiply(learningRate)
            )

          val newB =
            layer.b.subtract(
              grad.dB.mapMultiply(learningRate)
            )

          layer.update(
            newW,
            newB
          )
      }
}