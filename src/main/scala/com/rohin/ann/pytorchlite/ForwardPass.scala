package com.rohin.ann.pytorchlite

import org.apache.commons.math3.linear.RealVector

case class ForwardPass(
    zs: List[RealVector],
    activations: List[RealVector]
) {
  override def toString: String = {
    def formatVector(v: RealVector): String =
      v.toArray.map(x => f"$x% .4f").mkString("[", ", ", "]")

    val zsStr =
      zs.zipWithIndex
        .map { case (z, i) =>
          s"z[$i]=${formatVector(z)}"
        }
        .mkString("\n    ")

    val activationsStr =
      activations.zipWithIndex
        .map { case (a, i) =>
          s"a[$i]=${formatVector(a)}"
        }
        .mkString("\n    ")

    s"""ForwardPass(
         |  zs=
         |    $zsStr,
         |
         |  activations=
         |    $activationsStr
         |)""".stripMargin
  }
}
