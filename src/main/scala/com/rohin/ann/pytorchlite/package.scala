package com.rohin.ann

import org.apache.commons.math3.linear.RealVector
import org.apache.commons.math3.linear.ArrayRealVector
import scala.util.Random
import org.apache.commons.math3.linear.RealMatrix
import org.apache.commons.math3.linear.Array2DRowRealMatrix

package object pytorchlite {
  def vec(xs: Double*): RealVector =
    new ArrayRealVector(xs.toArray)

  def zeroVec(size: Int): RealVector =
    new ArrayRealVector(size)

  def randomVec(size: Int): RealVector = {
    val rng = new Random(42)
    new ArrayRealVector(Array.fill(size)(rng.nextDouble() - 0.5))
  }

  def randomMatrix(rows: Int, cols: Int): RealMatrix = {
    val matrix = Array2DRowRealMatrix(rows, cols)
    for i <- 0 until rows do matrix.setRowVector(i, randomVec(cols))
    matrix
  }
}
