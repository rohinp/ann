package com.rohin.ann.pytorchlite

import com.rohin.ann.FunSuiteWithLogging
import com.rohin.ann.pytorchlite.dense.ActivationFn.*
import com.rohin.ann.pytorchlite.ConsoleLogging.DebugConfig
import com.rohin.ann.pytorchlite.dense.Dense
import com.rohin.ann.pytorchlite.loss.{LossFunction, MeanSquaredError}
import com.rohin.ann.pytorchlite.sequential.Layer
import org.apache.commons.math3.linear.RealVector

class LayerSuite extends FunSuiteWithLogging {

  test("layer can perform forward pass") {

    val layer: Layer =
      Dense(2, 2, Sigmoid)

    val result =
      layer.forward(vec(1, 0))

    assertEquals(result._2.getDimension, 2)
  }

  test("layer can perform forward pass") {

    val layer: Layer =
      Dense(2, 2, Sigmoid)

    val result =
      layer.forward(vec(1, 0))

    assertEquals(result._2.getDimension, 2)
  }

  test("mse computes output delta") {

    val loss: LossFunction =
      MeanSquaredError

    val delta =
      loss.outputDelta(
        vec(0.5),
        vec(1)
      )

    assert(delta.getDimension == 1)
  }
}
