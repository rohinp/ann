package com.rohin.ann.pytorchlite

import com.rohin.ann.FunSuiteWithLogging
import com.rohin.ann.pytorchlite.Dense
import com.rohin.ann.pytorchlite.ActivationFn.*
import org.apache.commons.math3.linear.RealVector
class SequentialSuite extends FunSuiteWithLogging {
  test("sequential network forward works") {

    val net = Sequential(
      Dense(2, 2, Sigmoid),
      Dense(2, 1, Sigmoid)
    )

    val output = net.forward(vec(1, 0))

    assert(output > 0.0 && output < 1.0)
  }

  test("forward pass stores activations and zs") {

    val net = Sequential(
      Dense(2, 2, Sigmoid),
      Dense(2, 1, Sigmoid)
    )

    val pass = net.forwardPass(vec(1, 0))

    assertEquals(pass.activations.size, 3)
    assertEquals(pass.zs.size, 2)
  }

  test("backward pass computes deltas for all layers") {

    val net = Sequential(
      Dense(2, 2, Sigmoid),
      Dense(2, 1, Sigmoid)
    )

    val pass = net.forwardPass(vec(1, 0))
    val grads =
      net.backward(pass, vec(1))

    assertEquals(grads.deltas.size, 2)
  }

  test("applyGradients updates weights") {

    val net = Sequential(
      Dense(2, 2, Sigmoid),
      Dense(2, 1, Sigmoid)
    )

    val pass =
      net.forwardPass(vec(1, 0))

    val grads =
      net.backward(pass, vec(1))

    val updated =
      net.applyGradients(pass, grads, learningRate = 0.1)

    assertNotEquals(
      updated.layers.head.W,
      net.layers.head.W
    )
  }

  test("trainOne updates network") {

    val net =
      Sequential(
        Dense(2, 2, Sigmoid),
        Dense(2, 1, Sigmoid)
      )

    val updated =
      net.trainOne(
        input = vec(1, 0),
        target = vec(1),
        learningRate = 0.1
      )

    assertNotEquals(
      updated.layers.head.W,
      net.layers.head.W
    )
  }

  test("train performs multiple updates") {

    val net =
      Sequential(
        Dense(2, 2, Sigmoid),
        Dense(2, 1, Sigmoid)
      )

    val dataset =
      List(
        (vec(1, 0), vec(1))
      )

    val trained =
      net.train(
        dataset,
        epochs = 10,
        learningRate = 0.1
      )

    assertNotEquals(
      trained.layers.head.W,
      net.layers.head.W
    )
  }
}
