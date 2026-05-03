package com.rohin.ann

import munit.FunSuite
import org.apache.commons.math3.linear.{ArrayRealVector, RealVector}
import com.rohin.ann.Perceptron

class PerceptronSuite extends FunSuite {
  import Perceptron.*

  test("predict returns 1 when weighted sum is positive") {

    val perceptron = create(bias = 0, weights = vec(1, 1))

    val input = vec(1, 1)

    val prediction = perceptron.predict(input)

    assertEquals(prediction, 1)
  }

  test("predict returns 1 for boundry case") {

    val perceptron = create(bias = 0, weights = vec(1, -1))

    val input = vec(1, 1)

    val prediction = perceptron.predict(input)

    assertEquals(prediction, 1)
  }

  test("weights are updated when prediction is wrong") {

    val perceptron = create(bias = 0, weights = vec(-1, -1), learningRate = 1.0)
    val input = vec(1, 1)
    val actual = 1

    val updated = perceptron.trainOne(input, actual)

    assertEquals(updated.weights, vec(0, 0))
    assertEquals(updated.bias, 1d)
  }

  test("No update when prediction is correct") {

    val perceptron = create(bias = 0, weights = vec(1, 1), learningRate = 1.0)
    val input = vec(1, 1)
    val actual = 1

    val updated = perceptron.trainOne(input, actual)

    assertEquals(updated.weights, vec(1, 1))
    assertEquals(updated.bias, 0d)
  }

  test("perceptron learns AND gate") {

    val data = List(
      (vec(0, 0), 0),
      (vec(0, 1), 0),
      (vec(1, 0), 0),
      (vec(1, 1), 1)
    )

    val initial = create(bias = 0, weights = zeroVec(2), learningRate = 1.0)

    val trained = initial.train(data, epochs = 10)

    assertEquals(trained.predict(vec(0, 0)), 0)
    assertEquals(trained.predict(vec(0, 1)), 0)
    assertEquals(trained.predict(vec(1, 0)), 0)
    assertEquals(trained.predict(vec(1, 1)), 1)

  }

  test("perceptron learns OR gate") {

    val data = List(
      (vec(0, 0), 0),
      (vec(0, 1), 1),
      (vec(1, 0), 1),
      (vec(1, 1), 1)
    )

    val initial = create(bias = 0, weights = zeroVec(2), learningRate = 1.0)

    val trained = initial.train(data, epochs = 10)

    assertEquals(trained.predict(vec(0, 0)), 0)
    assertEquals(trained.predict(vec(0, 1)), 1)
    assertEquals(trained.predict(vec(1, 0)), 1)
    assertEquals(trained.predict(vec(1, 1)), 1)

  }

  /*
  A single perceptron can only learn linearly separable data.
  XOR is not linearly separable
   */
  test("perceptron learns XOR gate") {

    val data = List(
      (vec(0, 0), 0),
      (vec(0, 1), 1),
      (vec(1, 0), 1),
      (vec(1, 1), 0)
    )

    val initial = create(bias = 0, weights = zeroVec(2), learningRate = 1.0)

    val trained = initial.train(data, epochs = 10)

    val correct = trained.predict(vec(0, 0)) == 0 &&
      trained.predict(vec(0, 1)) == 1 &&
      trained.predict(vec(1, 0)) == 1 &&
      trained.predict(vec(1, 1)) == 0

    assertEquals(correct, false)

  }
}
