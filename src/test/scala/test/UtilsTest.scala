package test

import org.scalatest.funsuite.AnyFunSuite
import titanic._

class UtilsTest extends AnyFunSuite {

  // load datsets                    
  val train = Utils.loadDataCSV("train.csv")
  val test = Utils.loadDataCSV("test.csv")
  val all = train ++ test

  test("Test size of the datesets") {

    assert(train.size === 891)
    assert(test.size === 418)
  }
}