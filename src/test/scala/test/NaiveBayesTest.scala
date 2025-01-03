package test

import org.scalatest.funsuite.AnyFunSuite
import titanic.NaiveBayes.{calcConditionalPropabilitiesForEachClassWithSmoothing, getAttributeValues}
import titanic._

class NaiveBayesTest extends AnyFunSuite {

  val trainDataSet= List(
    Map[String,String]("day"-> "weekday", "season"->"spring", "wind"->"none", "rain"->"none", "class"->"on time"),
    Map[String,String]("day"-> "weekday", "season"->"winter", "wind"->"none", "rain"->"slight", "class"->"on time"),
    Map[String,String]("day"-> "weekday", "season"->"winter", "wind"->"none", "rain"->"slight", "class"->"on time"),
    Map[String,String]("day"-> "weekday", "season"->"winter", "wind"->"high", "rain"->"heavy", "class"->"late"),
    Map[String,String]("day"-> "saturday", "season"->"summer", "wind"->"normal", "rain"->"none", "class"->"on time"),
    Map[String,String]("day"-> "weekday", "season"->"autumn", "wind"->"normal", "rain"->"none", "class"->"very late"),
    Map[String,String]("day"-> "holiday", "season"->"summer", "wind"->"high", "rain"->"slight", "class"->"on time"),
    Map[String,String]("day"-> "sunday", "season"->"summer", "wind"->"normal", "rain"->"none", "class"->"on time"),
    Map[String,String]("day"-> "weekday", "season"->"winter", "wind"->"high", "rain"->"heavy", "class"->"very late"),
    Map[String,String]("day"-> "weekday", "season"->"summer", "wind"->"none", "rain"->"slight", "class"->"on time"),
    Map[String,String]("day"-> "saturday", "season"->"spring", "wind"->"high", "rain"->"heavy", "class"->"cancled"),
    Map[String,String]("day"-> "weekday", "season"->"summer", "wind"->"high", "rain"->"slight", "class"->"on time"),
    Map[String,String]("day"-> "saturday", "season"->"winter", "wind"->"normal", "rain"->"none", "class"->"late"),
    Map[String,String]("day"-> "weekday", "season"->"summer", "wind"->"high", "rain"->"none", "class"->"on time"),
    Map[String,String]("day"-> "weekday", "season"->"winter", "wind"->"normal", "rain"->"heavy", "class"->"very late"),
    Map[String,String]("day"-> "saturday", "season"->"autumn", "wind"->"high", "rain"->"slight", "class"->"on time"),
    Map[String,String]("day"-> "weekday", "season"->"autumn", "wind"->"none", "rain"->"heavy", "class"->"on time"),
    Map[String,String]("day"-> "holiday", "season"->"spring", "wind"->"normal", "rain"->"slight", "class"->"on time"),
    Map[String,String]("day"-> "weekday", "season"->"spring", "wind"->"normal", "rain"->"none", "class"->"on time"),
    Map[String,String]("day"-> "weekday", "season"->"spring", "wind"->"normal", "rain"->"slight", "class"->"on time")
  )

  test("Count Classes"){

    val resClass= NaiveBayes.countAttributeValues(trainDataSet, "class").asInstanceOf[Map[String,Int]]
    val expClass= Map("late" -> 2, "cancled" -> 1, "very late" -> 3, "on time" -> 14)
    assert(resClass===expClass)
  }

  test("Count Rain"){

    val resClass= NaiveBayes.countAttributeValues(trainDataSet, "rain").asInstanceOf[Map[String,Int]]
    val expClass= Map("slight" -> 8, "none" -> 7, "heavy" -> 5)
    assert(resClass===expClass)
  }
  test("Count Wind"){

    val resClass= NaiveBayes.countAttributeValues(trainDataSet, "wind").asInstanceOf[Map[String,Int]]
    val expClass= Map("normal" -> 8, "high" -> 7, "none" -> 5)
    assert(resClass===expClass)
  }

  test("Count Season"){

    val resClass= NaiveBayes.countAttributeValues(trainDataSet, "season").asInstanceOf[Map[String,Int]]
    val expClass= Map("winter" -> 6, "autumn" -> 3, "spring" -> 5, "summer" -> 6)
    assert(resClass===expClass)
  }

  test("Count Day"){

    val resClass= NaiveBayes.countAttributeValues(trainDataSet, "day").asInstanceOf[Map[String,Int]]
    val expClass= Map("sunday" -> 1, "saturday" -> 4, "weekday" -> 13, "holiday" -> 2)
    assert(resClass===expClass)
  }

  test("Get Attributes"){

    val res= NaiveBayes.getAttributes(trainDataSet)
    println(res)
  }

  test("Get Attributes with missing Value"){

    val testSet= List(
      Map[String,Any]("a"-> 1, "b"->"hello", "c"-> 4.5),
      Map[String,Any]("a"-> 2, "b"->"test", "d"-> "none"))
    val res= NaiveBayes.getAttributes(testSet)
    assert(res===Set("a","b","c","d"))
  }

  test("Get Attribute Values"){

    val res= NaiveBayes.getAttributeValues(trainDataSet)
    val exp= Map("season" -> Set("winter", "autumn", "spring", "summer"),
      "rain" -> Set("slight", "none", "heavy"), "wind" -> Set("normal", "high", "none"),
      "class" -> Set("late", "cancled", "very late", "on time"),
      "day" -> Set("sunday", "saturday", "weekday", "holiday"))
    assert(res===exp)
  }

  test ("Calc PriorPropablities"){

    val res=NaiveBayes.calcPriorPropabilities(trainDataSet,"class")
    val exp= Map("late" -> 0.1, "cancled" -> 0.05, "very late" -> 0.15, "on time" -> 0.7)
    assert(res===exp)
  }

  test("Count Attribute Values for Each class"){

    val res = NaiveBayes.calcAttribValuesForEachClass(trainDataSet,"class")
    val exp= Map(
      "late" -> Set(("season",Map("winter" -> 2)), ("rain",Map("none" -> 1, "heavy" -> 1)),
      ("wind",Map("normal" -> 1, "high" -> 1)), ("day",Map("saturday" -> 1, "weekday" -> 1))),

        "cancled" -> Set(("season",Map("spring" -> 1)), ("rain",Map("heavy" -> 1)), ("wind",Map("high" -> 1)),
          ("day",Map("saturday" -> 1))),

      "very late" -> Set(("season",Map("winter" -> 2, "autumn" -> 1)),
        ("rain",Map("none" -> 1, "heavy" -> 2)), ("wind",Map("normal" -> 2, "high" -> 1)), ("day",Map("weekday" -> 3))),

      "on time" -> Set(("season",Map("winter" -> 2, "autumn" -> 2, "spring" -> 4, "summer" -> 6)),
        ("rain",Map("slight" -> 8, "none" -> 5, "heavy" -> 1)), ("wind",Map("normal" -> 5, "high" -> 4, "none" -> 5)),
        ("day",Map("sunday" -> 1, "saturday" -> 2, "weekday" -> 9, "holiday" -> 2)))
    )
    assert(res=== exp)
  }
  test("Calculate Conditional Propabilities For Each Class"){

    val classVals= NaiveBayes.countAttributeValues(trainDataSet,"class")
    val data= NaiveBayes.calcAttribValuesForEachClass(trainDataSet,"class")
    val res = NaiveBayes.calcConditionalPropabilitiesForEachClass(data,classVals)
      .asInstanceOf[ Map[Any, Set[(String, Map[Any, Any])]]]
    val resext= NaiveBayes.extractValues(res).asInstanceOf[Set[(String,String,Double)]].toList.sorted

    val exp= Map("late" -> Set(("season",Map("winter" -> 1.0)), ("rain",Map("none" -> 0.5, "heavy" -> 0.5)),
      ("wind",Map("normal" -> 0.5, "high" -> 0.5)), ("day",Map("saturday" -> 0.5, "weekday" -> 0.5))),

      "cancled" -> Set(("season",Map("spring" -> 1.0)), ("rain",Map("heavy" -> 1.0)), ("wind",Map("high" -> 1.0)),
        ("day",Map("saturday" -> 1.0))),

      "very late" -> Set(("season",Map("winter" -> 0.667,
        "autumn" -> 0.333)), ("rain",Map("none" -> 0.333, "heavy" -> 0.667)), ("wind",Map("normal" -> 0.667,
        "high" -> 0.333)), ("day",Map("weekday" -> 1.0))),

      "on time" -> Set(("season",Map("winter" -> 0.143, "autumn" -> 0.143,
        "spring" -> 0.286, "summer" -> 0.429)), ("rain",Map ("slight" -> 0.571, "none" -> 0.357, "heavy" -> 0.071)),
        ("wind",Map("normal" -> 0.357, "high" -> 0.285, "none" -> 0.357)), ("day",Map("sunday" -> 0.071,
          "saturday" -> 0.142, "weekday" -> 0.642, "holiday" -> 0.143)))).
      asInstanceOf[Map[Any,Set[(String, Map[Any,Any])]]]
    val expres= NaiveBayes.extractValues(exp).asInstanceOf[Set[(String,String,Double)]].toList.sorted
    val compare= resext.zip(expres)
    assert(compare.forall(x=>math.abs(x._1._3-x._2._3)<0.01)===true)
  }

  test("Calc Class Values with Conditional Propablilities"){

    val classVals= NaiveBayes.countAttributeValues(trainDataSet,"class")
    val data= NaiveBayes.calcAttribValuesForEachClass(trainDataSet,"class")
    val condProp = NaiveBayes.calcConditionalPropabilitiesForEachClass(data,classVals)
    val prior= NaiveBayes.calcPriorPropabilities(trainDataSet,"class")
    val el= Map[String,String]("day"->"weekday", "season"->"winter", "wind"->"high", "rain"->"heavy")
    val res= NaiveBayes.calcClassValuesForPrediction(el,condProp,prior)
    val resExt= res.toList.asInstanceOf[List[(String,Double)]].sorted
    val exp= List(("cancled",0.0), ("late",0.0125), ("on time",0.0013), ("very late",0.0222))
    assert(resExt.zip(exp).forall(x=>math.abs(x._1._2-x._2._2)<0.01))
    println(res)
  }

  test("Find best fitting class"){

    val app= Map[Any,Double]("cancled"->0.0, "late"->0.0125, "on time"->0.0013, "very late"->0.0222)
    val res=NaiveBayes.findBestFittingClass(app)
    assert(res==="very late")
  }

  test("apply Model One"){

    val classificationData= Seq(Map[String,String]("id"->"1","day"->"weekday", "season"->"winter", "wind"->"high",
      "rain"->"heavy"))
    val model= NaiveBayes.modelForTrainExample(trainDataSet,"class")
    val res= NaiveBayes.applyModel(model,classificationData,"id")
    val exp= List(("1","very late"))
    assert(res===exp)
  }

  test("apply Model Set"){

    val classificationData= Seq(Map[String,String]("id"->"1","day"->"weekday", "season"->"winter", "wind"->"high",
      "rain"->"heavy"),Map[String,String]("id"->"2","day"->"sunday", "season"->"summer", "wind"->"high",
      "rain"->"heavy"),Map[String,String]("id"->"3","day"->"weekday", "season"->"autumn", "wind"->"normal",
      "rain"->"none"),Map[String,String]("id"->"4","day"->"holiday", "season"->"winter", "wind"->"high",
      "rain"->"heavy"))
    val model = NaiveBayes.modelForTrainExample(trainDataSet,"class")
    val res= NaiveBayes.applyModel(model,classificationData,"id").toSet
    val exp= Set(("1","very late"), ("2","on time"), ("3","very late"), ("4","on time"))
    assert(res===exp)
  }

  test("Conditional Properties with Add One Smoothing"){

    val classVals: Map[Any, Int] = NaiveBayes.countAttributeValues(trainDataSet,"class")
    //println(classVals)
    val aValues = NaiveBayes.getAttributeValues(trainDataSet).asInstanceOf[ Map[String, Set[Any]]]
    //println(aValues)
    val data: Map[Any, Set[(String, Map[Any, Int])]] = NaiveBayes.calcAttribValuesForEachClass(trainDataSet,"class")
    //println(data)
    val condProp = NaiveBayes.calcConditionalPropabilitiesForEachClassWithSmoothing(data,aValues,classVals)
    //println(condProp)
    val exp= Map("late" -> Set(("season",Map("winter" -> 0.5, "autumn" -> 0.1667, "spring" -> 0.1667,
      "summer" -> 0.1667)), ("rain",Map("slight" -> 0.2, "none" -> 0.4, "heavy" -> 0.4)),
      ("wind",Map("normal" -> 0.4, "high" -> 0.4, "none" -> 0.2)),
      ("day",Map("sunday" -> 0.1667, "saturday" -> 0.333, "weekday" -> 0.3333, "holiday" -> 0.1667))),
      "cancled" -> Set(("season",Map("winter" -> 0.2, "autumn" -> 0.2, "spring" -> 0.4, "summer" -> 0.2)),
        ("rain",Map("slight" -> 0.25, "none" -> 0.25, "heavy" -> 0.5)),
        ("wind",Map("normal" -> 0.25, "high" -> 0.5, "none" -> 0.25)),
        ("day",Map("sunday" -> 0.2, "saturday" -> 0.4, "weekday" -> 0.2, "holiday" -> 0.2))),
      "very late" -> Set(("season",Map("winter" -> 0.4286, "autumn" -> 0.2857, "spring" -> 0.1429, "summer" -> 0.1429)),
        ("rain",Map("slight" -> 0.16667, "none" -> 0.3333, "heavy" -> 0.5)),
        ("wind",Map("normal" -> 0.5, "high" -> 0.3333, "none" -> 0.1667)),
        ("day",Map("sunday" -> 0.1429, "saturday" -> 0.1429, "weekday" -> 0.5714, "holiday" -> 0.1429))),
      "on time" -> Set(("season",Map("winter" -> 0.1667, "autumn" -> 0.16667, "spring" -> 0.27778, "summer" -> 0.3889)),
        ("rain",Map("slight" -> 0.5295, "none" -> 0.3529, "heavy" -> 0.1176)),
        ("wind",Map("normal" -> 0.3529, "high" -> 0.2941, "none" -> 0.3529)),
        ("day",Map("sunday" -> 0.11111, "saturday" -> 0.16667, "weekday" -> 0.5556, "holiday" -> 0.16667)))).
      asInstanceOf[Map[Any, Set[(String, Map[Any, Any])]]]
    val expExt= NaiveBayes.extractValues(exp).asInstanceOf[Set[(String,String,Double)]].toList.sorted
    val resExt= NaiveBayes.extractValues(condProp.asInstanceOf[Map[Any, Set[(String, Map[Any, Any])]]]).
      asInstanceOf[Set[(String,String,Double)]].toList.sorted
    val compare= expExt.zip(resExt)
    assert(compare.forall(x=>math.abs(x._1._3-x._2._3)<0.01)===true)
  }

  test("Model with add One-Smoothing"){

    val classificationData= Seq(Map[String,String]("id"->"1","day"->"weekday", "season"->"winter", "wind"->"high",
      "rain"->"heavy"),Map[String,String]("id"->"2","day"->"sunday", "season"->"summer", "wind"->"high",
      "rain"->"heavy"),Map[String,String]("id"->"3","day"->"weekday", "season"->"autumn", "wind"->"normal",
      "rain"->"none"),Map[String,String]("id"->"4","day"->"holiday", "season"->"winter", "wind"->"high",
      "rain"->"heavy"))
    val model = NaiveBayes.modelwithAddOneSmoothing(trainDataSet,"class")
    val res= NaiveBayes.applyModel(model,classificationData,"id").toSet
    val exp= Set(("1","very late"), ("2","on time"), ("3","on time"), ("4","very late"))
    assert(res===exp)
  }

  test("Extract Values"){

    val data= Map("late" -> Set(("season",Map("winter" -> 1.0)), ("rain",Map("none" -> 0.5, "heavy" -> 0.5)),
      ("wind",Map("normal" -> 0.5, "high" -> 0.5)), ("day",Map("saturday" -> 0.5, "weekday" -> 0.5))),
      "cancled" -> Set(("season",Map("spring" -> 1.0)), ("rain",Map("heavy" -> 1.0)), ("wind",Map("high" -> 1.0)),
        ("day",Map("saturday" -> 1.0)))).asInstanceOf[Map[Any, Set[(String, Map[Any, Any])]]]
    val res= NaiveBayes.extractValues(data)
    val exp=Set(("late","heavy",0.5), ("late","normal",0.5), ("late","none",0.5), ("late","weekday",0.5),
      ("late","winter",1.0), ("late","high",0.5), ("late","saturday",0.5), ("cancled","spring",1.0),
      ("cancled","heavy",1.0), ("cancled","high",1.0), ("cancled","saturday",1.0))
    assert(exp===res)
  }
}