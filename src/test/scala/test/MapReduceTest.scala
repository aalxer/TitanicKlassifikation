package test

import java.text.SimpleDateFormat

import org.scalatest.funsuite.AnyFunSuite
import mapreduce.MapReduce
import java.text.SimpleDateFormat

class MapReduceTest extends AnyFunSuite{

  val dateFormatPattern= "yyyy.MM.dd G 'at' HH:mm:ss z"
  val sdf= new SimpleDateFormat(dateFormatPattern)

  /*
      In diesem Anwendungsfall geht es um den Log-File eines Job-Schedulers
      Er beinhaltet die folgenden Elemente
     (<Datum und Uhrzeit>,<Nutzer>,<Jobname>,<Dauer in Sekunden>) mit den Typen
     (DateFormat,String,String,Int)

  */

  val data: List[(Int,(String,String,String,Int))] =List((1,("2020.05.08 AD at 10:05:12 PDT","meier","wordcount",20)),(2,("2020.05.08 AD at 12:05:12 PDT","weber","apachelog",50)),
    (3,("2020.05.08 AD at 13:06:17 PDT","meier","wordcount",15)),(4,("2020.05.08 AD at 15:25:17 PDT","meier","wordcount",25)),
    (5,("2020.05.09 AD at 12:00:12 PDT","weber","apachelog",45)),(6,("2020.05.09 AD at 14:17:22 PDT","meier","wordcount",21)),
    (7,("2020.05.09 AD at 18:15:17 PDT","weber","apachelog",20)),(8,("2020.05.09 AD at 20:05:12 PDT","meier","wordcount",17)),
    (9,("2020.05.10 AD at 09:22:33 PDT","mueller","MeineAnalyse",100)),(10,("2020.05.10 AD at 12:07:02 PDT","meier","TF-IDF",2000)),
    (11,("2020.05.10 AD at 12:35:02 PDT","mueller","TF-IDF",2100)))

  test("Wie viele Jobs haben die einzelnen User abgesetzt?"){

    val exp= Map("meier" -> 6, "weber" -> 3, "mueller" -> 2)
    val res= MapReduce.numberOfJobsPerUser(data).toMap
    assert(res===exp)
  }

  test("Wie häufig haben Benutzer mit Jobs mit einem bestimmten Namen abgesetzt?"){

    val exp= Map(("meier","wordcount") -> 5, ("weber","apachelog") -> 3, ("mueller","MeineAnalyse") -> 1, ("meier","TF-IDF") -> 1,
      ("mueller","TF-IDF") -> 1)
    val res= MapReduce.numberOfJobsPerUserUsingACertainName(data).toMap
    assert(res===exp)
  }

  test("Welche Jobnamen wurden vergeben (ohne Duplikate?"){

    val exp= Set("wordcount","TF-IDF", "MeineAnalyse", "apachelog")
    val res= MapReduce.distinctNamesOfJobs(data)
    assert(res.toSet===exp)
  }

  test("Wie viele Aufträge haben über X sec gedauert und wie viele drunter"){

    val exp= Map("less" -> 4, "more" -> 7)
    val res= MapReduce.moreThan20Seconds(data).toMap
    assert(exp===res)
  }

  test("Auftraege pro Tag"){

    val exp= Map("2020-05-08" -> 3, "2020-05-09" -> 3, "2020-05-10" -> 5)
    val res= MapReduce.numberOfJobsPerDay(data).toMap
    println(res)
    assert(res==exp)
  }
}