package titanic

import titanic.NaiveBayes.findBestFittingClass

object TitanicDataSet {


  /**
   * Creates a model that predicts 1 (survived) if the person of the certain record
   * is female and 0 (deceased) otherwise
   *
   * @return The model represented as a function
   */
  def simpleModel: (Map[String, Any], String) => (Any, Any) = {
    (map, passengerID) => (map(passengerID), if(map("sex") == "female") 1 else 0)
  }

  /**
   * This function should count for a given attribute list, how often an attribute is
   * not present in the data records of the data set
   *
   * @param data    The DataSet where the counting takes place
   * @param attList List of attributes where the missings should be counted
   * @return A Map that contains the attribute names (key) and the number of missings (value)
   */
  def countAllMissingValues(data: List[Map[String, Any]], attList: List[String]): Map[String, Int] = {

    data.foldLeft(Map[String, Int]()) { (base, passenger) =>
      attList.foldLeft(base) { (acc, att) =>
        passenger.get(att) match {
          case Some(_) => acc
          case None => acc.updated(att, 1 + acc.getOrElse(att, 0))
        }
      }
    }
  }

  /**
   * This function should extract a set of given attributes from a record
   *
   * @param record  Record that should be extracted
   * @param attList List of attributes that should be extracted
   * @return A Map that contains only the attributes that should be extracted
   *
   */
  def extractTrainingAttributes(record: Map[String, Any], attList: List[String]): Map[String, Any] = {
    record.filter(att => attList.contains(att._1))
  }

  /**
   * This function should create the training data set. It extracts the necessary attributes,
   * categorize them and deals with the missing values. You could find some hints in the description
   * and the lectures
   *
   * @param data Training Data Set that needs to be prepared
   * @return Prepared Data Set for using it with Naive Bayes
   */
  def createDataSetForTraining(data: List[Map[String, Any]]): List[Map[String, Any]] = {

    val attributes = List("passengerID","sex", "age", "pclass", "survived")
    val preparedData = data.map(passenger => passenger.get("age") match {
      case Some(age) => passenger.updated("age", this.categorizeAge(age.asInstanceOf[Float]))
      case None =>
        passenger.updated("age",this.predictAge(passenger, data))
        //passenger.updated("age",5)
    })
    preparedData.map(passenger => this.extractTrainingAttributes(passenger, attributes))


  }

  def categorizeAge(age: Any): Int = {
    age.asInstanceOf[Float] match {
      case age if 0 < age && age < 4 => 1
      case age if 4 <= age && age < 12 => 2
      case age if 12 <= age && age < 17 => 3
      case age if 17 <= age && age < 24 => 4
      case age if 24 <= age && age < 34 => 5
      case age if 34 <= age && age < 44 => 6
      case age if 44 <= age && age < 54 => 7
      case age if 54 <= age && age < 64 => 8
      case age if 64 <= age => 9
    }
  }

  def predictAge(passenger: Map[String, Any], data: List[Map[String, Any]]): Int = {
    // alle die keinen Alter haben entfernen:
    data.filter(_.getOrElse("age",0) != 0).map(passenger =>
      // nur die Keys "pclass", "age", "fare" behalten:
      this.extractTrainingAttributes(passenger, List("pclass", "age", "fare"))).
      // die Map in Tupel umwandeln und Alter kategorisieren:
      map(x => (x("pclass"), this.categorizeAge(x("age")), x.getOrElse("fare", 0.0))).
      // groupieren nach (Klasse , Alter):
      groupBy(tupel => (tupel._1, tupel._2)).
      // durchschnittlicher Preis für die Altargruppe und Klasse ermittle:
      map(x => (x._1._1, x._1._2, x._2.map(y => y._3.toString.toDouble).sum/x._2.size)).
      // die Klasse und der bezahle Preis werden verglichen:
      foldLeft(1)((base, x) => {
        //println(passenger , "current base: " , base , "current tupel: " , x)
        if(passenger("pclass") == x._1 && (passenger("fare").asInstanceOf[Float] - x._3).abs <= 1 ) x._2
        //else if ((passenger("fare").asInstanceOf[Double] - x._3).abs <= 3) x._2
        else base
      })
  }

  /**
   * This function builds the model. It is represented as a function that maps a data record
   * and the name of the id-attribute to the value of the id attribute and the predicted class
   * (similar to the model building process in the train example)
   *
   * @param trainDataSet Training Data Set
   * @param classAttrib  name of the attribute that contains the class
   * @return A tuple consisting of the id (first element) and the predicted class (second element)
   */
  def createModelWithTitanicTrainingData(tdata: List[Map[String, Any]], classAttrib: String):
  (Map[String, Any], String) => (Any, Any) = {
    val trainingData = this.createDataSetForTraining(tdata)

    val classVals= NaiveBayes.countAttributeValues(trainingData,classAttrib)
    val data= NaiveBayes.calcAttribValuesForEachClass(trainingData,classAttrib)
    val condProp = NaiveBayes.calcConditionalPropabilitiesForEachClass(data,classVals)
    val prior= NaiveBayes.calcPriorPropabilities(trainingData,classAttrib)

    (record, idKey) => {
      val preparedRecord = extractTrainingAttributes(record, List("age", "sex", "pclass"))
      val classPrediction = NaiveBayes.findBestFittingClass(NaiveBayes.calcClassValuesForPrediction(preparedRecord, condProp, prior))
      (record(idKey), classPrediction)
    }

  }
}