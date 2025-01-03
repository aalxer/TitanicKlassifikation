package titanic

import java.io.PrintWriter

import scala.util.Try


object Utils {

  // Regular Expressions for extracting the information
  val DATA_ACCESS_PATTERN_test = """(\d+),(\d),"(.+)",(male|female),([0-9]*\.[0-9]+|[0-9]+|d*),(\d*),(\d*),(.*),([0-9]*\.[0-9]+|[0-9]+|d*),(.*),(\w*)""".r
  val DATA_ACCESS_PATTERN_train = """(\d+),(\d),(\d),"(.+)",(male|female),([0-9]*\.[0-9]+|[0-9]+|d*),(\d*),(\d*),(.*),([0-9]*\.[0-9]+|[0-9]+|d*),(.*),(\w*)""".r

  // Reading text file
  // Stores the information in a map consisting of a property name (key) and its value
  def loadDataCSV(filename: String): List[Map[String, Any]] = {

    val stream = getClass.getResourceAsStream("/" + filename)
    val src = scala.io.Source.fromInputStream(stream)
    val iter = src.getLines().drop(1) //skip first line (property names)

    val result = (for (row <- iter) yield readData(row)).toList

    src.close
    result.flatMap(_ match { case p: Option[Map[String, Any]] => p })
  }


  // Extracting all information storing it into a Map[String,Any]
  def readData(line: String): Option[Map[String, Any]] = {

    def toInt(key: String, s: String): Option[(String, Int)] = Try(s.toInt).toOption.map((key, _))

    def toFloat(key: String, s: String): Option[(String, Float)] = Try(s.toFloat).toOption.map((key, _))

    def toString(key: String, s: String): Option[(String, String)] =
      if (s.nonEmpty) Some((key, s)) else None

    def createPassengerMap(t1: String, t2: String, t3: String, t4: String, t5: String, t6: String, t7: String,
                           t8: String, t9: String, t10: String, t11: String, t12: String): Option[Map[String, Any]] = {

      val l = List(
        toInt("passengerID", t1),
        toInt("survived", t2),
        toInt("pclass", t3),
        toString("name", t4),
        toString("sex", t5),
        toFloat("age", t6),
        toInt("sibsp", t7),
        toInt("parch", t8),
        toString("ticket", t9),
        toFloat("fare", t10),
        toString("cabin", t11),
        {
          if (t12.length > 0) Some(("embarked", t12(0))) else None
        })
      Some(l.flatMap(_ match { case p: Option[(String, Any)] => p }).toMap)
    }

    val result = line match {
      case DATA_ACCESS_PATTERN_test(t1, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12) =>
        createPassengerMap(t1, "-1", t3, t4, t5, t6, t7, t8, t9, t10, t11, t12)

      case DATA_ACCESS_PATTERN_train(t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12) => {
        createPassengerMap(t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12)
      }

    }
    result
  }

  // Method for printing a passenger in a readable manner
  def printPassenger(p: Map[String, Any]): Unit = {

    println("\n---------------------------------------------------------------------")
    println("passengerID:" + p.getOrElse("passengerID", -1))
    println("survived:" + p.getOrElse("survived", -1))
    println("pclass:" + p.getOrElse("pclass", -1))
    println("name:" + p.getOrElse("name", "-"))
    println("sex:" + p.getOrElse("sex", "-"))
    println("age:" + p.getOrElse("age", -1))
    println("sibsp:" + p.getOrElse("sibsp", -1))
    println("parch:" + p.getOrElse("parch", -1))
    println("ticket:" + p.getOrElse("ticket", "-"))
    println("fare:" + p.getOrElse("fare", -1))
    println("cabin:" + p.getOrElse("cabin", -1))
    println("embarked:" + p.getOrElse("embarked", '-'))
    println("---------------------------------------------------------------------\n")
  }

  def createSubmitFile[ID, CLASS](filename: String, data: Seq[(ID, CLASS)], header: String): Unit = {
    val pw = new PrintWriter(filename)
    pw.println(header)
    data.foreach(e => pw.println(e._1.toString + "," + e._2.toString))
    pw.close()
  }
} 
  