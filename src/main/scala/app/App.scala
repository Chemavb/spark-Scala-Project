package app


/**
  * @author ${Jakub Kowalczewski, José María Vera}
  *
  *
  */

object App {
  def main(args : Array[String]): Unit = {

    // We input the dataset
    val inputFilesPath = args(0)
    var df = DataFrameGenerator.generateFromCSV(inputFilesPath)

    // We remove the 9 variables that contain information unknown at the time before taking off.
    val mandatoryVarsToRemove = Array("ArrTime", "ActualElapsedTime", "AirTime", "TaxiIn",
      "Diverted", "CarrierDelay", "WeatherDelay", "NASDelay",
      "SecurityDelay", "LateAircraftDelay")

    df = DataFrameHelper.removeUnknownColumns(df, mandatoryVarsToRemove)

    // We remove the following 7 variables that we consider useless for predicting time delay
    val ourVarsToRemove = Array("CRSArrTime", "CRSDepTime", "FlightNum", "TailNum", "CRSElapsedTime", "TaxiOut", "CancellationCode", "LateAircraftDelay")

    df = DataFrameHelper.removeUnknownColumns(df, ourVarsToRemove)

    // We keep the variable "Cancelled" in order to remove every cancelled flight (We don't need them to clasify ArrDelay)
    df = df.filter("Cancelled = 0")

    // We don't need the variable anymore
    df = df.drop("Cancelled")

    // We construct new 3 columns: Weekend/no weekend, High season/Low season, Morning/Afternoon/Downing
    df = DataFrameHelper.addNewVars(df)

    // We select our variable types taking into account the restrictions of our models
    df = DataFrameHelper.castDataFrame(df)

    // We remove rows with null values as it's a very insignificant part of the dataset (<1% of the dataset, checked previously)
    df = df.na.drop()

    // We index our categorical variables using StringIndexer
    df = DataFrameHelper.indexVarDataFrame(df)

    // We split out dataset in: 80% Training - 20% Test
    var Array(trainingData, testData) = df.randomSplit(Array(0.8, 0.2))

    // RandomForestRegressor
    MachineLearning.randomForest(trainingData,testData,"ArrDelay")

    // LinearRegression
    MachineLearning.linearRegression(trainingData, testData, "ArrDelay")

    // DecisionTreeRegressor
    MachineLearning.decisionTree(trainingData, testData, "ArrDelay")


  }
}