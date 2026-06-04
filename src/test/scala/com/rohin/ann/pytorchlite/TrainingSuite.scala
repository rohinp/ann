package com.rohin.ann.pytorchlite

import com.rohin.ann.FunSuiteWithLogging
import com.rohin.ann.pytorchlite.Dense
import com.rohin.ann.pytorchlite.ActivationFn.*
import org.apache.commons.math3.linear.RealVector
import com.rohin.ann.pytorchlite.ConsoleLogging.DebugConfig
class TrainingSuite extends FunSuiteWithLogging {

  given df: DebugConfig = DebugConfig(
    isEnabled = false
  )

  test("network learns xor") {

    val xor =
      List(
        (vec(0, 0), vec(0)),
        (vec(0, 1), vec(1)),
        (vec(1, 0), vec(1)),
        (vec(1, 1), vec(0))
      )

    val net =
      Sequential(
        Dense(2, 2, Sigmoid),
        Dense(2, 1, Sigmoid)
      )

    val trained =
      net.train(
        xor,
        epochs = 5000,
        learningRate = 0.1
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
