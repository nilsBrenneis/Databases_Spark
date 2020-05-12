package de.databases.spark.nils;

import java.util.ArrayList;
import java.util.List;
import org.apache.spark.api.java.JavaSparkContext;

public class HelloWorld {

  public static void doHelloWorld(final JavaSparkContext sc) {

    int numSamples = 10000000;

    List<Integer> l = new ArrayList<>(numSamples);
    for (int i = 0; i < numSamples; i++) {
      l.add(i);
    }

    long count = sc.parallelize(l).filter(i -> {
      double x = Math.random();
      double y = Math.random();
      return x * x + y * y < 1;
    }).count();
    System.out.println("Pi is roughly " + 4.0 * count / numSamples);
  }
}
