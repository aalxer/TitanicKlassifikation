ThisBuild / version := "1.0.0-SNAPSHOT"

/*
javaOptions in Runtime ++= Seq(
  // -J params will be added as jvm parameters
  "-Djava.io.tmpdir=/home/hendrik/testi")
*/
lazy val osName = System.getProperty("os.name") match {
  case n if n.startsWith("Linux") => "linux"
  case n if n.startsWith("Mac") => "mac"
  case n if n.startsWith("Windows") => "win"
  case _ => throw new Exception("Unknown platform!")
}
def sysPropOrDefault(propName:String,default:String):String = Option(System.getProperty(propName)).getOrElse(default)
lazy val javaiotmpdir = sysPropOrDefault("java.io.tmpdir","./tmp/")
lazy val javaFXModules = Seq("base", "controls", "fxml", "media", "swing", "web")
lazy val root = (project in file("."))
  .settings(
    name := "Klassifikation_Aufgabe",
    scalaVersion := "2.11.12",
    //Runtime / javaOptions ++= Seq(
      // -J params will be added as jvm parameters
      //"java.io.tmpdir=/home/hendrik/testi"),
    Compile / scalacOptions ++= Seq("-deprecation"),
    Compile / console / scalacOptions --= Seq("-Ywarn-unused", "-Ywarn-unused-import"),
    Test / fork := true,
    libraryDependencies ++=Seq("org.scalactic" %% "scalactic" % "3.2.10" % "test",
	"org.scalatest" %% "scalatest" % "3.2.10" % "test", "org.vegas-viz" %% "vegas" % "0.3.11",
      "org.scalafx" %% "scalafx" % "15.0.1-R21") ++ javaFXModules.map(m =>
      "org.openjfx" % s"javafx-$m" % "12.0.2" classifier osName
    )



// Add dependency on JavaFX libraries, OS dependent



)
run := Defaults.runTask(fullClasspath in Runtime, mainClass in run in Compile, runner in run).evaluated
/*
libraryDependencies ++= Seq(
  "org.scalactic" %% "scalactic" % "3.2.10" % "test",
	"org.scalatest" %% "scalatest" % "3.2.10" % "test"
//  "org.vegas-viz" %% "vegas" % "0.3.11"
)

run := Defaults.runTask(fullClasspath in Runtime, mainClass in run in Compile, runner in run).evaluated


//https://github.com/scalafx/scalafx
// required, because javaFX no longer included after Java 11, but required by vegas

// Add dependency on ScalaFX library
//libraryDependencies += "org.scalafx" %% "scalafx" % "12.0.2-R18"

// Determine OS version of JavaFX binaries
*/



