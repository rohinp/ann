val scala3Version = "3.8.3"

lazy val root = project
  .in(file("."))
  .settings(
    name := "ann",
    version := "0.1.0-SNAPSHOT",

    scalaVersion := scala3Version,

    libraryDependencies ++= Seq(
      "org.scalameta" %% "munit" % "1.3.0" % Test,
      "org.apache.commons" % "commons-math3" % "3.6.1"
    )
  )
