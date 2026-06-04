package com.rohin.ann.pytorchlite

object ConsoleLogging {
  enum Position:
    case Top, Bottom, Both, No

  case class DebugConfig(
      isEnabled: Boolean = false,
      sapatator: String = "_",
      position: Position = Position.No,
      sapatatorCount: Int = 100
  )

  def log(str: String)(using ds: DebugConfig): Unit = {
    ds match {
      case DebugConfig(false, sapatator, position, sapatatorCount) =>
        ()
      case DebugConfig(true, sapatator, Position.No, sapatatorCount) =>
        println(str)
      case DebugConfig(true, sapatator, Position.Top, sapatatorCount) =>
        println(s"""
            |${sapatator * sapatatorCount}
            |$str
            |""".stripMargin)
      case DebugConfig(true, sapatator, Position.Bottom, sapatatorCount) =>
        println(s"""
            |$str
            |${sapatator * sapatatorCount}
            |""".stripMargin)
      case DebugConfig(true, sapatator, Position.Both, sapatatorCount) =>
        println(s"""
            |${sapatator * sapatatorCount}
            |$str
            |${sapatator * sapatatorCount}
            |""".stripMargin)
    }
  }

}
