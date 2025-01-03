import titanic.{NaiveBayes, TitanicDataSet, Utils}

object CreatePrediction extends App{

  val train = Utils.loadDataCSV("train.csv")
  val test = Utils.loadDataCSV("test.csv")

  println("Train Dataset:" + train.size + " Elements")
  println("Test Dataset:" + test.size + " Elements")

  val model= TitanicDataSet.createModelWithTitanicTrainingData(train,"survived")
  val evaluation= TitanicDataSet.createDataSetForTraining(test)
  val evalData= evaluation.map(map=>map-("survived"))

  val prediction= NaiveBayes.applyModel(model,evalData,"passengerID")
  Utils.createSubmitFile("TitanicPrediction.csv",prediction,"passengerID,survived")
  println(prediction)
}
