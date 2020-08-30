package scalaspark.bigdata

import org.apache.spark.ml.clustering.{BisectingKMeans, GaussianMixture, KMeans, LDA}
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.sql.SparkSession

object BigDataApp {
  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder.appName("spark-bigdata-ml-clustering-app").config("spark.master", "local").getOrCreate()

    import org.apache.log4j._
    Logger.getLogger("org").setLevel(Level.ERROR)

    ////////////////////////////////////////////////////////////
    // Loads data.
    val dataset = spark.read.format("libsvm").load("data-kmeans.txt")
    // Trains a k-means model.
    val kmeans = new KMeans().setK(2).setSeed(1L)
    val model = kmeans.fit(dataset)
    // Make predictions
    val predictions = model.transform(dataset)
    // Evaluate clustering by computing Silhouette score
    val evaluator = new ClusteringEvaluator()
    val silhouette = evaluator.evaluate(predictions)
    println(s"Silhouette with squared euclidean distance = $silhouette")
    // Shows the result.
    println("Cluster Centers: ")
    model.clusterCenters.foreach(println)
    /////////////////////////////////////////////////////////
    // Loads data.
    val dataset2 = spark.read.format("libsvm")
      .load("data-lda-libsvm.txt")

    // Trains a LDA model.
    val lda = new LDA().setK(10).setMaxIter(10)
    val model2 = lda.fit(dataset2)

    val ll = model2.logLikelihood(dataset2)
    val lp = model2.logPerplexity(dataset2)
    println(s"The lower bound on the log likelihood of the entire corpus: $ll")
    println(s"The upper bound on perplexity: $lp")

    // Describe topics.
    val topics = model2.describeTopics(3)
    println("The topics described by their top-weighted terms:")
    topics.show(false)

    // Shows the result.
    val transformed = model2.transform(dataset2)
    transformed.show(false)
    /////////////////////////////////////////////////////////
    // Loads data.
    val dataset3 = spark.read.format("libsvm").load("data-kmeans.txt")

    // Trains a bisecting k-means model.
    val bkm = new BisectingKMeans().setK(2).setSeed(1)
    val model3 = bkm.fit(dataset3)

    // Evaluate clustering.
    val cost = model3.computeCost(dataset3)
    println(s"Within Set Sum of Squared Errors = $cost")

    // Shows the result.
    println("Cluster Centers: ")
    val centers = model3.clusterCenters
    centers.foreach(println)
    /////////////////////////////////////////////////////////
    // Loads data
    val dataset4 = spark.read.format("libsvm").load("data-kmeans.txt")
    // Trains Gaussian Mixture Model
    val gmm = new GaussianMixture()
      .setK(2)
    val model4 = gmm.fit(dataset4)
    // output parameters of mixture model model
    for (i <- 0 until model4.getK) {
      println(s"Gaussian $i:\nweight=${model4.weights(i)}\n" +
        s"mu=${model4.gaussians(i).mean}\nsigma=\n${model4.gaussians(i).cov}\n")
    }
    ///////////////////////////////////////////////////////////

    spark.stop()

  }
}
