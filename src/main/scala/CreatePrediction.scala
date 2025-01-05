import titanic.{NaiveBayes, TitanicDataSet, Utils}

object CreatePrediction extends App {

  val trainingData = Utils.loadDataCSV("train.csv")
  val testData = Utils.loadDataCSV("test.csv")

  println("Train Dataset:" + trainingData.size + " Elements")
  println("Test Dataset:" + testData.size + " Elements")

  val model = TitanicDataSet.createModelWithTitanicTrainingData(trainingData, "survived")
  val evaluation = TitanicDataSet.createDataSetForTraining(testData)
  val evalData = evaluation.map(map => map - ("survived"))
  val prediction = NaiveBayes.applyModel(model, evalData, "passengerID")

  Utils.createSubmitFile("TitanicPrediction.csv", prediction, "passengerID,survived")
  println("TitanicPrediction.csv created !")
}
