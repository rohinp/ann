package com.rohin.ann

enum Label(val value: Int):
  case Zero extends Label(0)
  case One  extends Label(1)

object Label:
  def fromInt(i: Int): Either[String, Label] = i match
    case 0 => Right(Zero)
    case 1 => Right(One)
    case _ => Left(s"Label must be 0 or 1, got $i")
