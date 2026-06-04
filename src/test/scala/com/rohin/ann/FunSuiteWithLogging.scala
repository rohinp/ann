package com.rohin.ann

import munit.FunSuite
import com.rohin.ann.pytorchlite.ConsoleLogging.DebugConfig
import com.rohin.ann.pytorchlite.ConsoleLogging.Position

trait FunSuiteWithLogging extends FunSuite {
  given dc: DebugConfig = DebugConfig(
    isEnabled = true,
    position = Position.Both
  )
}
