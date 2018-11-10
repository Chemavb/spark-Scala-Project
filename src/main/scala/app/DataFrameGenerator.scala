package app

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.log4j.{Level, Logger}

/**
  * Class for the Generation of the DataFrame
  */
object DataFrameGenerator {

  /**
    * The function generates a DataFrame from all csv.bz2 files located in the inputPath directory
    * @param inputPath Directory where csv.bz2 files are stored
    * @return a unique DataFrame with all the content from each csv.bz2 file included
    */
  def generateFromCSV(inputPath : String): DataFrame = {

    Logger.getRootLogger().setLevel(Level.WARN)
    val spark = SparkSession.builder.master("local").appName("Spark CSV Reader").getOrCreate()
    val df = spark.read.format("csv").option("header", "true").load(inputPath + "*.csv.bz2")
    return df
  }


}
