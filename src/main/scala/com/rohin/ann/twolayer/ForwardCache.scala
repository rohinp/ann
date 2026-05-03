package com.rohin.ann.twolayer

import org.apache.commons.math3.linear.RealVector

case class ForwardCache(
    z1: RealVector,
    a1: RealVector,
    z2: Double,
    a2: Double
)
