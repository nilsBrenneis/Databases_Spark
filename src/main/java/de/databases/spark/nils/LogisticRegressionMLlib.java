package de.databases.spark.nils;

import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.when;

import java.util.ArrayList;
import java.util.Collections;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.ml.feature.ChiSqSelector;
import org.apache.spark.ml.feature.Imputer;
import org.apache.spark.ml.feature.ImputerModel;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

//https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/Imputer.html
//https://spark.apache.org/docs/1.5.1/api/java/org/apache/spark/sql/DataFrame.html
//https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/VectorAssembler.html
//https://medium.com/@dhiraj.p.rai/logistic-regression-in-spark-ml-8a95b5f5434c

public class LogisticRegressionMLlib {


  private static final String OUTCOME_COLUMN_NAME = "Outcome";

  private LogisticRegressionMLlib() {
  }

  private static final String FEATURES_COLUMN_NAME = "features";
  private static final String SCALED_FEATURES_COLUMN_NAME = "scaled features";

  public static void doLogisticRegression() {
    SparkSession spark = SparkSession.builder().appName("JavaLogRegDiabetes").getOrCreate();

    Dataset<Row> dataFrame = spark.read().format("csv").option("header", "true")
        .option("inferSchema", "true").load("./src/main/resources/nils/diabetes.csv");

    dataFrame.describe().select("Summary", "Pregnancies", "Glucose", "BloodPressure").show();

    dataFrame.select("Insulin", "Glucose", "BloodPressure", "SkinThickness", "BMI").show(5);
    dataFrame = fillInNan(dataFrame, "Insulin");
    dataFrame = fillInNan(dataFrame, "Glucose");
    dataFrame = fillInNan(dataFrame, "BloodPressure");
    dataFrame = fillInNan(dataFrame, "SkinThickness");
    dataFrame = fillInNan(dataFrame, "BMI");
    dataFrame.select("Insulin", "Glucose", "BloodPressure", "SkinThickness", "BMI").show(5);

    dataFrame = imputeData(dataFrame);
    dataFrame.select("Insulin", "Glucose", "BloodPressure", "SkinThickness", "BMI").show(5);

    dataFrame = createFeatureVector(dataFrame);
    dataFrame.select("features").show(false);

    dataFrame = scalerizeFeatureColumn(dataFrame);
    dataFrame.select(FEATURES_COLUMN_NAME, SCALED_FEATURES_COLUMN_NAME).show(5);

    Dataset<Row>[] splits = dataFrame.randomSplit(new double[]{0.8, 0.2}, 1234L);
    Dataset<Row> train = splits[0];
    Dataset<Row> test = splits[1];

    double datasetSize = train.count();
    double positives = train.where(OUTCOME_COLUMN_NAME + " == 1").count();
    double negatives = train.where(OUTCOME_COLUMN_NAME + " == 0").count();
    double percentageOfPositives = positives / datasetSize * 100;
    double balancingRatio = negatives / datasetSize;

    System.out.println("Number of positives is: " + positives);
    System.out.println("Percentage of positives is: " + percentageOfPositives);
    System.out.println("Balancing ratio is: " + balancingRatio);

    train = train
        .withColumn("classWeights", when(col(OUTCOME_COLUMN_NAME).equalTo("1"), balancingRatio)
            .otherwise(1 - balancingRatio));
    //train.select("classWeights", "Outcome").show(40);

    ChiSqSelector chiSqSelector = new ChiSqSelector();
    chiSqSelector.setFeaturesCol(SCALED_FEATURES_COLUMN_NAME);
    chiSqSelector.setOutputCol("Aspect");
    chiSqSelector.setLabelCol(OUTCOME_COLUMN_NAME);
    chiSqSelector.setFpr(0.05);

    train = chiSqSelector.fit(train).transform(train);
    test = chiSqSelector.fit(test).transform(test);
    //test.select("Aspect").show(5);

    LogisticRegressionModel model = getLogisticRegressionModel(train);
    Dataset<Row> predictTrain = model.transform(train);
    Dataset<Row> predictTest = model.transform(test);
    //predictTest.select("Outcome", "prediction").show(40);

    BinaryClassificationEvaluator binaryClassificationEvaluator = new BinaryClassificationEvaluator();
    binaryClassificationEvaluator.setRawPredictionCol("rawPrediction");
    binaryClassificationEvaluator.setLabelCol(OUTCOME_COLUMN_NAME);

    predictTest.select(OUTCOME_COLUMN_NAME, "rawPrediction", "prediction", "probability").show(5);

    System.out.println("The area under ROC for train set is: " + binaryClassificationEvaluator
        .evaluate(predictTrain));
    System.out.println("The area under ROC for test set is: " + binaryClassificationEvaluator
        .evaluate(predictTest));
  }

  private static LogisticRegressionModel getLogisticRegressionModel(Dataset<Row> train) {
    LogisticRegression logisticRegression = new LogisticRegression();
    logisticRegression.setLabelCol(OUTCOME_COLUMN_NAME);
    logisticRegression.setFeaturesCol("Aspect");
    logisticRegression.setWeightCol("classWeights");
    logisticRegression.setMaxIter(10);

    return logisticRegression.fit(train);
  }

  private static Dataset<Row> scalerizeFeatureColumn(Dataset<Row> dataFrame) {
    StandardScaler standardScaler = new StandardScaler();
    standardScaler.setInputCol(FEATURES_COLUMN_NAME);
    standardScaler.setOutputCol(SCALED_FEATURES_COLUMN_NAME);
    return standardScaler.fit(dataFrame).transform(dataFrame);
  }

  private static Dataset<Row> createFeatureVector(Dataset<Row> dataFrame) {
    String[] columns = dataFrame.columns();
    ArrayList<String> list = new ArrayList<>();
    Collections.addAll(list, columns);
    list.remove(OUTCOME_COLUMN_NAME);
    columns = list.toArray(new String[0]);
    VectorAssembler vectorAssembler = new VectorAssembler();
    vectorAssembler.setInputCols(columns);
    vectorAssembler.setOutputCol(FEATURES_COLUMN_NAME);
    return vectorAssembler.transform(dataFrame);
  }

  private static Dataset<Row> imputeData(Dataset<Row> dataFrame) {
    String[] imputerColumns = new String[]{"Glucose", "BloodPressure", "SkinThickness", "BMI",
        "Insulin"};
    Imputer imputer = new Imputer();
    imputer.setInputCols(imputerColumns);
    imputer.setOutputCols(imputerColumns);
    imputer.setStrategy("median"); //stronger resistance against outliers
    ImputerModel model = imputer.fit(dataFrame);
    dataFrame = model.transform(dataFrame);
    return dataFrame;
  }

  private static Dataset<Row> fillInNan(Dataset<Row> dataFrame, String columnName) {
    return dataFrame.withColumn(columnName, when(col(columnName).equalTo("0"), Float.NaN)
        .otherwise(col(columnName)));
  }
}
