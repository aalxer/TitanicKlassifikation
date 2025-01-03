
package titanic

import java.io.{BufferedWriter, File, FileWriter}

import vegas.DSL.SpecBuilder

object VegasUtils {
//  sys.props.update("java.io.tmpdir","/home/hendrik/testi")

  /**
   * Displays the given chart, created using Vegas().withData... ,
   * in the default browser.
   *
   * @param chart    Vegas chart
   * @param centered if the charts should be centered
   *
   */
  def showInBrowser(chart: SpecBuilder, centered: Boolean = true): Unit = {
    showAllInBrowser(List(chart), centered)
  }

  /**
   * Displays the given charts, created using Vegas().withData... ,
   * in the default browser.
   *
   * @param charts   Vegas charts
   * @param centered if the charts should be centered
   */
  def showAllInBrowser(charts: List[SpecBuilder], centered: Boolean = true): Unit = {
    val file = convertToHTMLFile(charts, centered)
    openInBrowser(file.getAbsolutePath)
  }

  private def convertToHTMLFile(charts: List[SpecBuilder], centered: Boolean = true): File = {
    // throw an IllegalArgumentException if charts are empty
    require(charts.nonEmpty)
    // create a temp file and open it for writing
    //System.setProperty("java.io.tmpdir","/home/hendrik/testi")

    println("blah"+sys.props.get("java.io.tmpdir"))
    val tempFile = File.createTempFile("tmp-", "")
    val writer = new BufferedWriter(new FileWriter(tempFile))

    //write the header, which is the same for all files
    writer.write(charts.head.html.headerHTML())

    //hide annoying meta data from all graph (note, this should be inside the <head>, but also works)
    writer.write("<style>.vega-actions {display: none}</style>" + System.lineSeparator)
    //wrap everything in an extra div, that is centered
    if (centered)
      writer.write("<div align=\"center\">" + System.lineSeparator)

    //write all plots html to the file
    charts.map(_.html.plotHTML()).foreach(writer.write)
    // write the footer
    writer.write(charts.head.html.footerHTML)
    //close the div
    if (centered)
      writer.write("</div>" + System.lineSeparator)
    //close the writer
    writer.close()
    tempFile
  }

  //Opens the given file in the system default browser
  //taken from the plot.ly library
  private def openInBrowser(filename: String): Unit = {
    val cmd = sys.props.get("os.name").map(_.toLowerCase) match {
      case Some("mac os x") =>
        Seq("open", filename)
      case Some(win) if win.startsWith("windows") =>
        Seq("cmd", s"start $filename")
      case Some(lin) if lin.indexOf("linux") >= 0 =>
        Thread.sleep(2000);Seq("xdg-open", filename)
      //let it throw a match exception if os not in list
    }
    //execute in a shell
    sys.process.Process(cmd).!
  }
}
