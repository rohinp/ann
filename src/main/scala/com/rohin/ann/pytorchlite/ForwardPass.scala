package com.rohin.ann.pytorchlite

import org.apache.commons.math3.linear.RealVector

case class ForwardPass(
    zs: List[RealVector],
    activations: List[RealVector]
)
