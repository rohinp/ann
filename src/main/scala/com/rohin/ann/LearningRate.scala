package com.rohin.ann

opaque type LearningRate = Double

object LearningRate:
  def of(rate: Double): Either[String, LearningRate] =
    if rate > 0.0 && rate <= 1.0 then Right(rate)
    else Left(s"Learning rate must be in (0.0, 1.0], got $rate")

  extension (lr: LearningRate)
    def value: Double = lr
