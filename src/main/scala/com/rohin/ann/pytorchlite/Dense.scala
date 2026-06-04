package com.rohin.ann.pytorchlite

import org.apache.commons.math3.linear.RealMatrix
import org.apache.commons.math3.linear.RealVector
import org.apache.commons.math3.analysis.UnivariateFunction
import org.apache.commons.math3.linear.Array2DRowRealMatrix
import org.apache.commons.math3.linear.ArrayRealVector

final case class Dense private (
    W: RealMatrix,
    b: RealVector,
    activation: ActivationFn
) {
  override def toString: String = {
    def formatMatrix(m: RealMatrix): String =
      m.getData
        .map(_.map(v => f"$v% .4f").mkString("[", ", ", "]"))
        .mkString("\n    ")

    def formatVector(v: RealVector): String =
      v.toArray.map(x => f"$x% .4f").mkString("[", ", ", "]")

    s"""Dense(
       |  in=${W.getColumnDimension},
       |  out=${W.getRowDimension},
       |  activation=$activation,
       |  W=
       |    ${formatMatrix(W)},
       |  b=${formatVector(b)}
       |)""".stripMargin
  }
}

object Dense {

  extension (dense: Dense)
    def update(w: RealMatrix, b: RealVector): Dense = {
      dense.copy(W = w, b = b)
    }
  extension (dl: Dense)
    def forward(input: RealVector): (RealVector, RealVector) = {
      val z = dl.W.operate(input).add(dl.b)
      val a = z.map(dl.activation.forwardFn)
      (z, a)
    }

  def apply(inputSize: Int, outputSize: Int, act: ActivationFn): Dense = {
    val rand = new scala.util.Random()
    val W = randomMatrix(outputSize, inputSize)
    val b = randomVec(outputSize)
    new Dense(W, b, act)
  }
}
