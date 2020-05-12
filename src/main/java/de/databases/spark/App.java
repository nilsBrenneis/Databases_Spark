package de.databases.spark;

import de.databases.spark.nils.HelloWorld;
import de.databases.spark.nils.MLlib;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;


public class App {

  public static void main(String[] args) {
    JavaSparkContext sc = getSparkContext();

    HelloWorld.doHelloWorld(sc);
    MLlib.doMLlib(sc);

  }

  private static JavaSparkContext getSparkContext() {
    SparkConf sparkConf = new SparkConf().setMaster("local").setAppName("spark projects");
    return new JavaSparkContext(sparkConf);
  }
}
