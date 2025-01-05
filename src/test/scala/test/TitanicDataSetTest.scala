package test

import org.scalatest.funsuite.AnyFunSuite
import titanic._

class TitanicDataSetTest extends AnyFunSuite {

  val trainingData = Utils.loadDataCSV("train.csv")

  test("Test Model with 1 person") {

    val person = Map[String, Any](
      "passengerID" -> 6,
      "sex" -> "male",
      "pclass" -> 3,
      "age" -> 5, // muss kategorisiert sein !
      "fare" -> 8.5
    )
    val model = TitanicDataSet.createModelWithTitanicTrainingData(trainingData, "survived")
    val predict = model(person, "passengerID")

    println(s"PassengerID : ${predict._1} \nSurvived? : ${if (predict._2 == 0) "No" else "Yes"}")
  }
}