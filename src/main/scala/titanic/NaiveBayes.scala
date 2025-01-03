package titanic

import mapreduce.MapReduceOperations.mapReduce

object NaiveBayes {

  /**
   * Counts the number of distinct attribute values for each attribute
   * given in the attribute list
   *
   * @param data    Data Set for counting
   * @param attribList List of the attributes that should be counted
   * @return A Map with the attribute name as the key and the number of distinct
   *         values as the value
   */
  def countAttributeValues(data:List[Map[String, Any]], attribList:String): Map[Any,Int] = {
    val prepringData = data.flatMap(x => (x.filter(y => y._1==attribList).toList)).asInstanceOf[List[(String, String)]]
    mapReduce[String, String, String, Int, String, Int](
      (x => List((x._2, 1))) ,
      (x => List((x._1, x._2.sum))) ,
      prepringData
    ).toMap
  }

  /**
   * Extracts all attribute names that occur in a data set
   *
   * @param data    Data Set to be searched
   * @return A List of the attribute names that appear in the data set
   */
  def getAttributes(data:List[Map[String, Any]]):Set[String] = {
    data.flatMap(x => x.keys).toSet
  }

  /**
   * Extracts all attribute values that occur in a data set.
   *
   * @param data    Data Set to be extracted
   * @return A Map that consists of all attributes and their corresponding attribute values.
   *         The attribute values are stored in a Set.
   */
  def getAttributeValues(data:List[Map[String, Any]]):Map[String,Set[Any]]={

    val attribs= getAttributes(data)
    attribs.map(a => (a,data.map(_(a)).groupBy(identity).keys.toSet)).toMap
  }

  /**
   * Calculate the prior propabilities of each class.
   *
   * @param data    Data Set to be used for calculation
   * @param classAttrib Name of the attribute that contains the class assignment
   * @return A Map that consists of all classes (as key) and their corresponding prior propabilities.
   *
   */
  def calcPriorPropabilities(data:List[Map[String, Any]], classAttrib:String):Map[Any,Double] = {
    this.countAttributeValues(data, classAttrib).foldLeft(Map[Any, Double]())((base, x) =>{
      base.updated(x._1, x._2.toDouble / data.size)
    })
  }

  /**
   * This function should count for each class and attribute how often an
   * attribute value occurs in the data set. The result should be a Map that consists of
   * a class as the key element. The value of the Map should be a Set of 2-tuples where
   * the first element is the attribute name and the second element is a map. This map
   * stores the attribute value as the key and the number of occurrences as the value (see
   * test for further details
   *
   * @param data    Data Set to be used for calculation
   * @param classAttrib Name of the attribute that contains the class assignment
   * @return A Map that consists of all classes (as key) and a set of tuples (as value)
   *         that contains all attributes with their name (first element) and the corresponding
   *         number of occurrences stored in a Map(second element).
   *
   */
  def calcAttribValuesForEachClass(data:List[Map[String, Any]], classAttrib:String):
  Map[Any, Set[(String, Map[Any, Int])]] = {
    // 1. wir erhalten alle Werte, für die wir berechnen wollen:
    this.getAttributeValues(data)(classAttrib).
      // 2. wir gehen die Werte durch und fangen an unser Map fürs Ergebnis auszufüllen,
      // Key der Map ist die Klasse selber:
      map(cls => (cls ,
        // Value der Map ist ein Set von Tupels mit den (Attribut, die Werte mit der Häufigkeit als Map)
        // 3. wir wollen die Attribute außer classAttrib haben:
        this.getAttributes(data).filter(att => att != classAttrib).
          // 4. wir gehen diese Attribute durch und holen wir die Werte mit der Häufigkeit
          // Ergbnis soll das aktuelle Attribute mit der Set von Werte und deren Häufigkeit:
          map(attribute => (attribute, this.countAttributeValues(data.filter(map => map(classAttrib) == cls), attribute))))
      ).toMap
  }

  /**
   * This function should calculate the conditional propabilities for each class and attribute.
   * It takes the number of occurences of each attribute value for each class
   * (result of calcAttribValuesForEachClass) and the number of occurences of
   * each class in the training data set.
   * During caluculation it divides the number of occurences of each attribute value by
   * the number of class occurences.
   *
   * @param data    Attribute counts for each class and attribute calculated
   *                by calcAttribValuesForEachClass
   * @param classCounts Number of occurences of each class in the training data set
   * @return A Map that consists of all classes (as key) and a set of tuples (as value)
   *         that contains all attributes with their name (first element) and the corresponding
   *         conditional propability stored in a Map(second element).
   */
  def calcConditionalPropabilitiesForEachClass(data: Map[Any, Set[(String, Map[Any, Int])]],classCounts:Map[Any,Int]):
  Map[Any,Set[(String, Map[Any, Double])]] = {
    data.map(cls => (cls._1 ,
      cls._2.map(att => (att._1,
        att._2.map(map => (map._1,
          map._2.toDouble / classCounts(cls._1)))
      ))
    ))
  }

  /**
   * This function should calculate the class propability values for each class.
   * It takes a data record that should be classified. Furthermore it takes the
   * conditional propabilties (calculated by calcConditionalPropabilitiesForEachClass)
   * and the prior propabilities (calculated by calcPriorPropabilities).
   * It multiplies the corresponding conditional propabilities and the prior propability of
   * the class.
   * The result of the function is the Naive Bayes Propability for each class.
   *
   * @param record  Data record that should be classified
   * @param conditionalProps Conditional propabilities
   * @param priorProps Prior Propabilities
   * @return A Map that consists of all classes (as key) and their corresponding propability
   */
  def calcClassValuesForPrediction(record:Map[String,Any],
                                   conditionalProps: Map[Any,Set[(String, Map[Any, Double])]],
                                   priorProps:Map[Any,Double]): Map[Any,Double] = {

    priorProps.map(cls => {
      val product = record.keys.toList.map {
        attribute => conditionalProps(cls._1).find(_._1 == attribute).map(_._2).getOrElse(Map.empty).getOrElse(record.getOrElse(attribute, ""), 0.0)
      }.product
      (cls._1, cls._2 * product)
    })

    // priorProps.map(cls => {
    //      val product = record.keys.toList.map {
    //        attribute => conditionalProps.getOrElse(cls._1, Set.empty).find(_._1 == attribute).map(_._2).getOrElse(Map.empty).getOrElse(record.getOrElse(attribute, ""), 0.0)
    //      }.product
    //      (cls._1, cls._2 * product)
    //    })

  }
  /**
   * This function finds the class with the highest propability
   *
   * @param classProps  Map that contains the class (key) and the corresponding propability
   * @return The class wit the highest propability
   */
  def findBestFittingClass(classProps:Map[Any,Double]):Any = {
    classProps.reduce((cls1, cls2) => if (cls1._2 > cls2._2) cls1 else  cls2)._1
  }

  /**
   * This function builds the model. It is represented as a function that maps a data record
   * and the name of the id-attribute to the value of the id attribute and the predicted class
   *
   * @param trainDataSet  Training Data Set
   * @param classAttrib name of the attribute that contains the class
   * @return A tuple consisting of the id (first element) and the predicted class (second element)
   */
  def modelForTrainExample(trainDataSet: List[Map[String, Any]], classAttrib:String):
  (Map[String, Any], String) => (Any, Any) = {

    val classVals= NaiveBayes.countAttributeValues(trainDataSet,classAttrib)
    val data= NaiveBayes.calcAttribValuesForEachClass(trainDataSet,classAttrib)
    val condProp = NaiveBayes.calcConditionalPropabilitiesForEachClass(data,classVals)
    val prior= NaiveBayes.calcPriorPropabilities(trainDataSet,classAttrib)
    (map,id_key) => (map(id_key),
      findBestFittingClass(NaiveBayes.calcClassValuesForPrediction(map-id_key,condProp,prior)))
  }

  /**
   * This function applies the model to a data set.
   *
   * @param model  Machine Learning Model
   * @param testdata Set of data records where the class should be predicted
   * @return A set/sequence of predictions represented by tuples consisting
   *         of the id (first element) and the predicted class (second element)
   */
  def applyModel[CLASS, ID](model: (Map[String, Any], String) => (ID, CLASS),
                            testdata: Seq[Map[String, Any]], idKey: String): Seq[(ID, CLASS)] = {

    testdata.map(d => model(d, idKey))
  }

  /**
   * This function should calculate the conditional propabilities for each class and attribute.
   * Does the same as calcConditionalPropabilitiesForEachClass. The difference is that it should
   * use the add one smoothing technique described in the lecture
   * It takes an additional parameter - the attribute values for each attribute. This is necessary,
   * when an attribute value does not appear in a certain class.
   *
   * @param data    Attribute counts for each class and attribute calculated
   *                by calcAttribValuesForEachClass
   * @param attValues Contains the occuring attribute values for each attribute
   * @param classCounts Number of occurences of each class in the training data set
   * @return A Map that consists of all classes (as key) and a set of tuples (as value)
   *         that contains all attributes with their name (first element) and the corresponding
   *         conditional propability stored in a map(second element).
   */
  def calcConditionalPropabilitiesForEachClassWithSmoothing
  (data: Map[Any, Set[(String, Map[Any, Int])]],
   attValues:Map[String,Set[Any]],
   classCounts:Map[Any,Int]): Map[Any,Set[(String, Map[Any, Double])]] = {

    data.map(cls => (cls._1 ,
      cls._2.map(att => (att._1,
        attValues(att._1).map(value => (value,
          (att._2.getOrElse(value, 0) + 1) / (classCounts(cls._1).toDouble + attValues(att._1).size))).toMap
      ))
    ))
  }

  /**
   * This function builds the model using add one smoothing.
   * It is represented as a function that maps a data record
   * and the name of the id-attribute to the value of the id attribute and the predicted class
   *
   * @param trainDataSet  Training Data Set
   * @param classAttrib name of the attribute that contains the class
   * @return A tuple consisting of the id (first element) and the predicted class (second element)
   */

  def modelwithAddOneSmoothing(trainDataSet: List[Map[String, Any]], classAttrib:String):
  (Map[String, Any], String) => (Any, Any) ={

    val classVals= NaiveBayes.countAttributeValues(trainDataSet,classAttrib)
    val aValues = getAttributeValues(trainDataSet).asInstanceOf[ Map[String, Set[Any]]]
    val data: Map[Any, Set[(String, Map[Any, Int])]] = NaiveBayes.calcAttribValuesForEachClass(trainDataSet,classAttrib)
    val condProp =calcConditionalPropabilitiesForEachClassWithSmoothing(data,aValues,classVals)
    val prior= NaiveBayes.calcPriorPropabilities(trainDataSet,classAttrib)
    (map,id_key) => (map(id_key),findBestFittingClass(NaiveBayes.calcClassValuesForPrediction(map-id_key,condProp,prior)))
  }

  /*****************************************************************************
   *                            Utils                                          *
   *   helper Functions used by the tests                                      *
   *****************************************************************************/

  def extractValues(data: Map[Any, Set[(String, Map[Any, Any])]]):Set[(Any,Any,Any)]={

    val x= for ((key,classVals) <- data ; (att,attVals) <-classVals ; (k,v)<-attVals) yield(
      key,k,v
    )
    x.toSet
  }
}