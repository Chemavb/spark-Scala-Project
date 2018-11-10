package app

import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._

/**
  * Class that includes helper methods for handling the DataFrame object
  */
object DataFrameHelper {

  /**
    * The function removes the columns of the dataframe specified in the array.
    *
    * @param initialDf The initial DataFrame
    * @param badVariables Array with the name of the colums to be removed.
    * @return A DataFrame without the unnecessary columns.
    */
  def removeUnknownColumns(initialDf: DataFrame, badVariables: Array[String]): DataFrame = {

    val df = initialDf.drop(badVariables:_*)

    return df
  }

  /**
    * The function gives each variable a proper type
    *
    * @param initialDf The initial DataFrame
    * @return A DataFrame with new variable types.
    */
  def castDataFrame(initialDf: DataFrame): DataFrame ={
    val df = initialDf.select(col("Year").cast(DoubleType),
      col("Month").cast(DoubleType),
      col("DayOfMonth").cast(DoubleType),
      col("DayOfWeek").cast(DoubleType),
      col("DepTime").cast(DoubleType),
      col("UniqueCarrier").cast(StringType),
      col("ArrDelay").cast(DoubleType),
      col("DepDelay").cast(DoubleType),
      col("Origin").cast(StringType),
      col("Dest").cast(StringType),
      col("Distance").cast(DoubleType),
      col("DayType").cast(StringType),
      col("TimeZone").cast(StringType),
      col("SeasonType").cast(StringType))

    return df
  }

  /**
    * The function indexes the categorical variables of the DataFrame
    *
    * @param initialDf The initial DataFrame
    * @return A DataFrame with our categorical variables indexed
    */
  def indexVarDataFrame(initialDf: DataFrame): DataFrame ={
    var df = initialDf
    val sIndexer = new StringIndexer().setInputCol("UniqueCarrier").setOutputCol("UniqueCarrierInt")
    df = sIndexer.fit(df).transform(df)

    val sIndexer2 = new StringIndexer().setInputCol("DayType").setOutputCol("DayTypeInt")
    df = sIndexer2.fit(df).transform(df)

    val sIndexer3 = new StringIndexer().setInputCol("TimeZone").setOutputCol("TimeZoneInt")
    df = sIndexer3.fit(df).transform(df)

    val sIndexer4 = new StringIndexer().setInputCol("SeasonType").setOutputCol("SeasonTypeInt")
    df = sIndexer4.fit(df).transform(df)

    val sIndexer5 = new StringIndexer().setInputCol("Origin").setOutputCol("OriginInt")
    df = sIndexer5.fit(df).transform(df)

    val sIndexer6 = new StringIndexer().setInputCol("Dest").setOutputCol("DestInt")
    df = sIndexer6.fit(df).transform(df)

    return df
  }

  /**
    * The function adds our 3 new variables
    *
    * @param initialDf The initial DataFrame
    * @return A DataFrame with our 3 new variables
    */
  def addNewVars(initialDf: DataFrame): DataFrame ={
    //DayType
    var df = initialDf.withColumn("DayType", when(initialDf("DayOfWeek")<(6), "Workingday").otherwise("Weekend"))

    //TimeZone
    df = df.withColumn("TimeZone", when(df("DepTime") < 800, "Downing").when(df("DepTime") < 1600, "Morning").otherwise("Afternoon"))

    // Expressions for Determining if it is Christmas (From 21th December to 7th January)
    val exprIsDayGreaterThan20 = df("DayofMonth") > 20
    val exprIsDayLowerThan7 = df("DayofMonth") < 7
    val exprIsDecember = df("Month").equalTo("12")
    val exprIsJanuary = df("Month").equalTo("1")

    val exprIsChristmasJanuary = exprIsDayLowerThan7 && exprIsJanuary
    val exprIsChristmasDecember = exprIsDayGreaterThan20 && exprIsDecember

    val exprIsChristmas = exprIsChristmasJanuary || exprIsChristmasDecember

    //Expressions for determining if it is Summer (From 21th June to 23th September)
    val exprIsDayLowerThan23 = df("DayofMonth") < 23
    val exprIsJune = df("Month").equalTo("6")
    val exprIsSeptember = df("Month").equalTo("9")

    val exprIsSummerJune = exprIsDayGreaterThan20 && exprIsJune
    val exprIsSummerSeptember = exprIsDayLowerThan23 && exprIsSeptember
    val exprIsJuly = df("Month").equalTo("7")
    val exprIsAugust = df("Month").equalTo("8")
    val exprIsSummer = exprIsSummerJune || exprIsJuly || exprIsAugust || exprIsSummerSeptember

    //SeasonType
    df = df.withColumn("SeasonType", when(exprIsChristmas, "Christmas").when(exprIsSummer, "Summer").otherwise("Normal"))

    return df
  }



}