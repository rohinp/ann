package com.rohin.ann.pytorchlite.optimizer

import com.rohin.ann.pytorchlite.dense.Dense
import com.rohin.ann.pytorchlite.sequential.Layer

trait Optimizer {

  def step(
            layers: List[Layer],
            gradients: Gradients
          ): List[Layer]

}