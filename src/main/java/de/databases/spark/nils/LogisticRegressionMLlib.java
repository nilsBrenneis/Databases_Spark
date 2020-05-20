package de.databases.spark.nils;

import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.when;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.Imputer;
import org.apache.spark.ml.feature.ImputerModel;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

//https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/Imputer.html
//https://spark.apache.org/docs/1.5.1/api/java/org/apache/spark/sql/DataFrame.html
//https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/VectorAssembler.html
//https://medium.com/@dhiraj.p.rai/logistic-regression-in-spark-ml-8a95b5f5434c

public class LogisticRegressionMLlib {

  private static final String FEATURES_COLUMN_NAME = "features";
  private static final String SCALED_FEATURES_COLUMN_NAME = "scaled features";

  public static void doMLlib() {

    SparkSession spark = SparkSession.builder().appName("LogisticRegressionPipeline")
        .getOrCreate();

    // Prepare training documents, which are labeled.
    Dataset<Row> training = spark.createDataFrame(Arrays.asList(
        new JavaLabeledDocument(0L, "a b c d e spark", 1.0),
        new JavaLabeledDocument(1L, "b d", 0.0),
        new JavaLabeledDocument(2L, "spark f g h", 1.0),
        new JavaLabeledDocument(3L, "hadoop mapreduce", 0.0)
    ), JavaLabeledDocument.class);

// Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and lr.
    Tokenizer tokenizer = new Tokenizer()
        .setInputCol("text")
        .setOutputCol("words");
    HashingTF hashingTF = new HashingTF()
        .setNumFeatures(1000)
        .setInputCol(tokenizer.getOutputCol())
        .setOutputCol("features");
    LogisticRegression lr = new LogisticRegression()
        .setMaxIter(10)
        .setRegParam(0.001);
    Pipeline pipeline = new Pipeline()
        .setStages(new PipelineStage[]{tokenizer, hashingTF, lr});

// Fit the pipeline to training documents.
    PipelineModel model = pipeline.fit(training);

// Prepare test documents, which are unlabeled.
    Dataset<Row> test = spark.createDataFrame(Arrays.asList(
        new JavaDocument(4L, "spark i j k"),
        new JavaDocument(5L, "l m n"),
        new JavaDocument(6L, "spark hadoop spark"),
        new JavaDocument(7L, "apache hadoop")
    ), JavaDocument.class);

// Make predictions on test documents.
    Dataset<Row> predictions = model.transform(test);
    for (Row r : predictions.select("id", "text", "probability", "prediction").collectAsList()) {
      System.out.println("(" + r.get(0) + ", " + r.get(1) + ") --> probability=" + r.get(2)
          + ", prediction=" + r.get(3));
    }
  }

  public static void doIt() {
    SparkSession spark = SparkSession.builder().appName("JavaLogRegDiabetes").getOrCreate();

    Dataset<Row> dataFrame = spark.read().format("csv").option("header", "true")
        .option("inferSchema", "true").load("./src/main/resources/nils/diabetes.csv");

    String[] cols = dataFrame.columns();
    dataFrame.describe().select("Summary", "Pregnancies", "Glucose", "BloodPressure").show();

    dataFrame.select("Insulin", "Glucose", "BloodPressure", "SkinThickness", "BMI").show(5);
    dataFrame = fillInNan(dataFrame, "Glucose");
    dataFrame = fillInNan(dataFrame, "BloodPressure");
    dataFrame = fillInNan(dataFrame, "SkinThickness");
    dataFrame = fillInNan(dataFrame, "BMI");
    dataFrame = fillInNan(dataFrame, "Insulin");
    dataFrame = fillInNan(dataFrame, "Glucose");
    dataFrame.select("Insulin", "Glucose", "BloodPressure", "SkinThickness", "BMI").show(5);

    dataFrame = imputeData(dataFrame);
    dataFrame.select("Insulin", "Glucose", "BloodPressure", "SkinThickness", "BMI").show(5);

    dataFrame = createFeatureVector(dataFrame);
    dataFrame.select("features").show(false);

    dataFrame = scalerizeFeatureColumn(dataFrame);
    dataFrame.select(FEATURES_COLUMN_NAME, SCALED_FEATURES_COLUMN_NAME).show(5);

    Dataset<Row>[] splits = dataFrame.randomSplit(new double[]{0.7, 0.3}, 1234L);
    Dataset<Row> train = splits[0];
    Dataset<Row> test = splits[1];
    System.out.println(train.select("Outcome").count());
    System.out.println(test.select("Outcome").count());
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
    list.remove("Outcome");
    columns = list.toArray(new String[list.size()]);
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
