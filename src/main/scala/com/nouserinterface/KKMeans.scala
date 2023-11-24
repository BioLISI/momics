package com.nouserinterface

import breeze.linalg._
import breeze.numerics._
import breeze.stats._
import scala.util.Random
import scala.util.control.Breaks._

object KKMeans {
  var kernel_function = (x: DenseVector[Double], y: DenseVector[Double]) => (x dot y);

  /**
   * Set the kernel function to use for the kernel k-means algorithm.
   * @param kernel_function_type The kernel function to use. Must be one of "linear", "polynomial", "rbf", or "sigmoid".
   */
  def set_kernel_function(kernel_function_type: String) : Unit = {
    kernel_function_type match {
      case "linear" => this.kernel_function = (x: DenseVector[Double], y: DenseVector[Double]) => (x dot y)
      case "polynomial" => this.kernel_function = (x: DenseVector[Double], y: DenseVector[Double]) => pow((x dot y) + 1, 3)
      case "rbf" => this.kernel_function = (x: DenseVector[Double], y: DenseVector[Double]) => exp(-pow(norm(x - y), 2))
      case "sigmoid" => this.kernel_function = (x: DenseVector[Double], y: DenseVector[Double]) => tanh((x.t * y) + 1)
      case _ => throw new IllegalArgumentException("Invalid kernel function: " + kernel_function)
    }
  }

  /**
   * Compute the kernel matrix for the given data.
   * @param data The data to compute the kernel matrix for.
   * @return The kernel matrix.
   */
  def kernel_matrix(data: DenseMatrix[Double]) : DenseMatrix[Double] = {
    val n = data.rows
    val kernel_matrix = DenseMatrix.zeros[Double](n, n)
    for (i <- 0 until n) {
      for (j <- 0 until n) {
        kernel_matrix(i, j) = kernel_function(data(i, ::).t, data(j, ::).t)
      }
    }
    kernel_matrix
  }

  /**
   * Compute the initial centroids for the kernel k-means algorithm using the k-means++ algorithm.
   * @param data The data to compute the initial centroids for.
   * @param k The number of clusters.
   * @return The initial centroids.
   */
  def k_means_pp(data: DenseMatrix[Double], k: Int) : DenseMatrix[Double] = {
    val n = data.rows
    val m = data.cols
    val centroids = DenseMatrix.zeros[Double](k, m)

    // Initialize centroids
    centroids(0, ::) := data(Random.nextInt(n), ::)

    for (i <- 1 until k) {
      val distances = DenseVector.zeros[Double](n)
      for (j <- 0 until n) {
        // Compute distance of data point j to centroid i - 1
        distances(j) = kernel_function(data(j, ::).t, data(j, ::).t) -
          2 * kernel_function(data(j, ::).t, centroids(i - 1, ::).t) +
          kernel_function(centroids(i - 1, ::).t, centroids(i - 1, ::).t)
      }
      // Compute probability of choosing data point j as centroid i
      val probabilities = distances / sum(distances)
      val cumulative_probabilities = probabilities.scanLeft(0.0)(_ + _)
      var r = Random.nextDouble()
      while(r == 1.0){
        r = Random.nextDouble()
      }
      centroids(i, ::) := data(cumulative_probabilities.toArray.indexWhere(_ > r), ::)
    }

    centroids
  }

  /**
   * Compute the kernel k-means algorithm.
   * @param data The data to compute the kernel k-means algorithm for.
   * @param k The number of clusters.
   * @param max_iterations The maximum number of iterations to run the algorithm for (by default 100).
   * @return The labels for each data point.
   */
  def kernel_k_means(data: DenseMatrix[Double], k: Int, max_iterations: Int = 100): List[Int] = {
    val n = data.rows
    val m = data.cols
    val centroids = k_means_pp(data, k)
    val labels = DenseVector.zeros[Int](n)

    // Initialize labels
    for (i <- 0 until n) {
      labels(i) = argmin(DenseVector((0 until k).map(j => {
        kernel_function(data(i, ::).t, data(i, ::).t) -
          2 * kernel_function(data(i, ::).t, centroids(j, ::).t) +
          kernel_function(centroids(j, ::).t, centroids(j, ::).t)
      }).toArray))
    }

    val new_labels = DenseVector.zeros[Int](n)

    var iterations = 0

    // This is an optimization to avoid computing the same term multiple times
    val third_term = DenseVector.zeros[Double](k)

    // Compute data clusters
    while (iterations < max_iterations) {

      // For each data point i compute the distance to each cluster and assign it to the closest cluster
      for(i <- 0 until n) {
        val distances = DenseVector.zeros[Double](k)
        for(j <- 0 until k) {
          val cluster = data(labels.toArray.toList.zipWithIndex.filter(_._1 == j).map(_._2), ::)

          val seq_cluster = Seq.tabulate(cluster.rows, cluster.cols){case (i, j) => cluster(i, j)}

          if(i==0){
            third_term(j) = sum(seq_cluster.map(x => {
              seq_cluster.foldLeft(0.0)((acc, y) => acc + kernel_function(DenseVector(x.toArray), DenseVector(y.toArray)))
            })) / (cluster.rows * cluster.rows)
          }

          val second_term = 2 * seq_cluster.foldLeft(0.0)((acc, x) => acc + kernel_function(data(i, ::).t, DenseVector(x.toArray))) / cluster.rows

          distances(j) = kernel_function(data(i, ::).t, data(i, ::).t) - second_term + third_term(j)
        }

        new_labels(i) = argmin(distances)
      }

      // If the labels are the same as the previous iteration, stop
      if(new_labels == labels) {
        iterations = max_iterations
      }
      labels := new_labels
      iterations += 1
    }

    // Return the labels as a list
    labels.toArray.toList
  }
}