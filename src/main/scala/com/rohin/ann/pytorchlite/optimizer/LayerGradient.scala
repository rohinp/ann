package com.rohin.ann.pytorchlite.optimizer

import org.apache.commons.math3.linear.{RealMatrix, RealVector}

case class LayerGradient(
                          dW: RealMatrix,
                          dB: RealVector
                        )