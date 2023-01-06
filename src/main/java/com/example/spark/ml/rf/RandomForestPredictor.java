package com.example.spark.ml.rf;


import scala.Tuple2;

import java.io.File;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.mllib.util.MLUtils;

/** RandomForest Classification Example using Spark MLlib
 * @author tutorialkart.com
 */
public class RandomForestPredictor {

    static RandomForestModel model;

    public static void main(String[] args) {
        // hadoop home dir [path to bin folder containing winutils.exe]
//        System.setProperty("hadoop.home.dir", "D:\\Arjun\\ml\\hadoop\\");

        // Configuring spark
        SparkConf sparkConf1 = new SparkConf().setAppName("RandomForestExample")
                .setMaster("local[2]")
                .set("spark.executor.memory","3g")
                .set("spark.driver.memory", "3g");

        // initializing the spark context
        JavaSparkContext jsc = new JavaSparkContext(sparkConf1);

        // loading the model, that is generated during training
        model = RandomForestModel.load(jsc.sc(),"RandForestClsfrMdl"+File.separator+"model");

        // Load and parse the test data file.
        String datapath = "data"+File.separator+"testValues.txt";
        JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(jsc.sc(), datapath).toJavaRDD();

        System.out.println("\nPredicted : Expected");

        // Evaluate model on test instances and compute test error
        JavaPairRDD<Double, Double> predictionAndLabel =
                data.mapToPair(new PairFunction<>() {
                    @Override
                    public Tuple2<Double, Double> call(LabeledPoint p) {
                        System.out.println(model.predict(p.features())+" : "+p.label());
                        return new Tuple2<>(model.predict(p.features()), p.label());
                    }
                });

        // compute error of the model to predict the categories for test samples/experiments
        double testErr =
                1.0 * predictionAndLabel.filter(new Function<Tuple2<Double, Double>, Boolean>() {
                    @Override
                    public Boolean call(Tuple2<Double, Double> pl) {
                        return !pl._1().equals(pl._2());
                    }
                }).count() / data.count();
        System.out.println("Test Error: " + testErr);

        jsc.stop();
    }

    private static PairFunction<LabeledPoint, Double, Double> pf =  new PairFunction<LabeledPoint, Double, Double>() {
        @Override
        public Tuple2<Double, Double> call(LabeledPoint p) {
            Double prediction= null;
            try {
                prediction = model.predict(p.features());
            } catch (Exception e) {
                //logger.error(ExceptionUtils.getStackTrace(e));
                e.printStackTrace();
            }
            System.out.println(prediction+" : "+p.label());
            return new Tuple2<>(prediction, p.label());
        }
    };

    private static Function<Tuple2<Double, Double>, Boolean> f = new Function<Tuple2<Double, Double>, Boolean>() {
        @Override
        public Boolean call(Tuple2<Double, Double> pl) {
            return !pl._1().equals(pl._2());
        }
    };
}
