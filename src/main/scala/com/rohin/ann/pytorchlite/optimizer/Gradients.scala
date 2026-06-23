package com.rohin.ann.pytorchlite.optimizer

import org.apache.commons.math3.linear.RealVector

case class Gradients(
                      layers: List[LayerGradient]
                    )