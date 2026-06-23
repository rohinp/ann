package com.rohin.ann.pytorchlite.dense

import com.rohin.ann.pytorchlite.sequential.Layer
import com.rohin.ann.pytorchlite.{randomMatrix, zeroVec}
import org.apache.commons.math3.linear.{RealMatrix, RealVector}

final case class Dense private (
    W: RealMatrix,
    b: RealVector,
    activation: ActivationFn
) extends Layer {
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

   def forward(input: RealVector): (RealVector, RealVector) = {
    val z = W.operate(input).add(b)
    val a = z.map(activation.forwardFn)
    (z, a)
  }

  override def update(w: RealMatrix, b: RealVector): Layer = copy(W = w, b = b)
}

object Dense {

  def apply(inputSize: Int, outputSize: Int, act: ActivationFn): Dense = {
    val W = randomMatrix(outputSize, inputSize)
    val b = zeroVec(outputSize)
    new Dense(W, b, act)
  }
}
