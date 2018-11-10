package app

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.regression.DecisionTreeRegressor

object MachineLearning {

  /**
    * The function performs a Random Forest Regressor model
    *
    * @param trainingDf The piece of our DataFrame that we are going to use as a Training DataFrame
    * @param testDf The piece of our DataFrame that we are going to use as a Test DataFrame
    * @param targetVar The variable that we want to predict
    */
  def randomForest(trainingDf: DataFrame, testDf: DataFrame, targetVar: String){

    val cont = new VectorAssembler()
      .setInputCols(Array("DepDelay", "UniqueCarrierInt", "DayTypeInt", "TimeZoneInt", "SeasonTypeInt", "OriginInt", "DestInt"))
      .setOutputCol("categ")

    val rf = new RandomForestRegressor()
      .setLabelCol(targetVar)
      .setFeaturesCol("categ").setMaxBins(500)

    val pipeline = new Pipeline()
      .setStages(Array(cont, rf))

    // Train model. This also runs the indexer.
    val model = pipeline.fit(trainingDf)

    // Make predictions.
    val predictions = model.transform(testDf)

    // Select example rows to display.
    predictions.select("prediction", targetVar).show(30)

    val evaluator = new RegressionEvaluator()
      .setLabelCol(targetVar)
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val rmse = evaluator.evaluate(predictions)
    println("Random Forest Regressor model: Root Mean Squared Error (RMSE) on test data = " + rmse)
  }

  /**
    * The function performs a Linear Regression model
    *
    * @param trainingDf The piece of our DataFrame that we are going to use as a Training DataFrame
    * @param testDf The piece of our DataFrame that we are going to use as a Test DataFrame
    * @param targetVar The variable that we want to predict
    */
  def linearRegression(trainingDf: DataFrame, testDf: DataFrame, targetVar: String){
    val cont = new VectorAssembler()
      .setInputCols(Array("Year", "Month", "DayOfMonth", "DayOfWeek",
        "DepDelay", "Distance"))
      .setOutputCol("contin")

    val lr = new LinearRegression()
      .setLabelCol(targetVar)
      .setFeaturesCol("contin")

    val pipeline = new Pipeline()
      .setStages(Array(cont, lr))


    val model = pipeline.fit(trainingDf)

    // Make predictions.
    val predictions = model.transform(testDf)

    // Select example rows to display.
    predictions.select("prediction", targetVar).show(30)

    val evaluator = new RegressionEvaluator()
      .setLabelCol(targetVar)
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val rmse = evaluator.evaluate(predictions)
    println("Linear Regression model: Root Mean Squared Error (RMSE) on test data = " + rmse)
  }

  /**
    * The function performs a Decision Tree Regressor model
    *
    * @param trainingDf The piece of our DataFrame that we are going to use as a Training DataFrame
    * @param testDf The piece of our DataFrame that we are going to use as a Test DataFrame
    * @param targetVar The variable that we want to predict
    */
  def decisionTree(trainingDf: DataFrame, testDf: DataFrame, targetVar: String){

    val cont = new VectorAssembler()
      .setInputCols(Array("DepDelay", "UniqueCarrierInt", "DayTypeInt", "TimeZoneInt", "SeasonTypeInt", "OriginInt", "DestInt"))
      .setOutputCol("categ")

    val rf = new DecisionTreeRegressor()
      .setLabelCol(targetVar)
      .setFeaturesCol("categ").setMaxBins(500)

    val pipeline = new Pipeline()
      .setStages(Array(cont, rf))

    // Train model. This also runs the indexer.
    val model = pipeline.fit(trainingDf)

    // Make predictions.
    val predictions = model.transform(testDf)

    // Select example rows to display.
    predictions.select("prediction", targetVar).show(30)

    val evaluator = new RegressionEvaluator()
      .setLabelCol(targetVar)
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val rmse = evaluator.evaluate(predictions)
    println("Decision Tree Regressor model: Root Mean Squared Error (RMSE) on test data = " + rmse)
  }

}
