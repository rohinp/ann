package com.rohin.ann.pytorchlite

import com.rohin.ann.FunSuiteWithLogging
import com.rohin.ann.pytorchlite.dense.ActivationFn.*
import com.rohin.ann.pytorchlite.dense.Dense
import com.rohin.ann.pytorchlite.optimizer.SGD
import com.rohin.ann.pytorchlite.sequential.Sequential

class XORWithSequentialSuite extends FunSuiteWithLogging {

  test("network learns xor") {

    val xor = List(
        (vec(0, 0), vec(0)),
        (vec(0, 1), vec(1)),
        (vec(1, 0), vec(1)),
        (vec(1, 1), vec(0))
      )

    val net = Sequential(
        Dense(2, 4, Sigmoid),
        Dense(4, 1, Sigmoid)
      ).updateOptimizer(SGD(learningRate = 1.0))

    val trained = net.train(
      xor,
      epochs =5000,
    )

    println(
      s"trained.forward(vec(0, 0)) < 0.2 = ${trained.forward(vec(0, 0))} = " + (trained
        .forward(vec(0, 0)) < 0.2)
    )
    println(
      s"trained.forward(vec(0, 1)) > 0.8 = ${trained.forward(vec(0, 1))} = " + (trained
        .forward(vec(0, 1)) > 0.8)
    )
    println(
      s"trained.forward(vec(1, 0)) > 0.8 = ${trained.forward(vec(1, 0))} = " + (trained
        .forward(vec(1, 0)) > 0.8)
    )
    println(
      s"trained.forward(vec(1, 1)) < 0.2 = ${trained.forward(vec(1, 1))} = " + (trained
        .forward(vec(1, 1)) < 0.2)
    )

    assert(trained.forward(vec(0, 0)) < 0.2)
    assert(trained.forward(vec(0, 1)) > 0.8)
    assert(trained.forward(vec(1, 0)) > 0.8)
    assert(trained.forward(vec(1, 1)) < 0.2)
  }

  test("network learns xor with ReLU") {

    val xor = List(
      (vec(0, 0), vec(0)),
      (vec(0, 1), vec(1)),
      (vec(1, 0), vec(1)),
      (vec(1, 1), vec(0))
    )

    val net = Sequential(
      Dense(2, 2, ReLU),
      Dense(2, 1, Sigmoid)
    )

    val trained = net.train(
      xor,
      epochs = 5000,
    )

    println(
      s"trained.forward(vec(0, 0)) < 0.2 = ${trained.forward(vec(0, 0))} = " + (trained
        .forward(vec(0, 0)) < 0.2)
    )
    println(
      s"trained.forward(vec(0, 1)) > 0.8 = ${trained.forward(vec(0, 1))} = " + (trained
        .forward(vec(0, 1)) > 0.8)
    )
    println(
      s"trained.forward(vec(1, 0)) > 0.8 = ${trained.forward(vec(1, 0))} = " + (trained
        .forward(vec(1, 0)) > 0.8)
    )
    println(
      s"trained.forward(vec(1, 1)) < 0.2 = ${trained.forward(vec(1, 1))} = " + (trained
        .forward(vec(1, 1)) < 0.2)
    )

    assert(trained.forward(vec(0, 0)) < 0.2)
    assert(trained.forward(vec(0, 1)) > 0.8)
    assert(trained.forward(vec(1, 0)) > 0.8)
    assert(trained.forward(vec(1, 1)) < 0.2)
  }
}
