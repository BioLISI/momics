package com.nouserinterface

import breeze.linalg.{DenseMatrix}
import breeze.stats.{mean, stddev}

class SNF(val dataSets: Seq[DenseMatrix[Double]]) {

  private def normalize(matrix: DenseMatrix[Double]): DenseMatrix[Double] = {
    val avg = mean(matrix)
    val dev = stddev(matrix)
    (matrix - avg) / dev
  }

  private def calculateDistance(matrix: DenseMatrix[Double]): DenseMatrix[Double] = {
    val distances = DenseMatrix.zeros[Double](matrix.rows, matrix.rows)
    for (i <- 0 until matrix.rows; j <- 0 until matrix.rows) {
      distances(i, j) = math.sqrt((0 until matrix.cols).map(k => math.pow(matrix(i, k) - matrix(j, k), 2)).sum)
    }
    distances
  }

  private def calculateNeighbors(rows: Int, percentage: Double): Int = math.max((rows * percentage / 100).toInt, 1)

  private def calculateAverageDistance(distances: DenseMatrix[Double], percentage: Double): DenseMatrix[Double] = {
    val avgDistances = DenseMatrix.zeros[Double](distances.rows, 1)
    val neighbors = calculateNeighbors(distances.rows, percentage)

    for (i <- 0 until distances.rows) {
      val sortedRow = distances(i, ::).inner.toArray.sorted
      avgDistances(i, 0) = sortedRow.slice(1, neighbors + 1).sum / neighbors
    }
    avgDistances
  }

  private def calculateE(i: Int, j: Int, avgDistances: DenseMatrix[Double], distances: DenseMatrix[Double]): Double =
    (avgDistances(i, 0) + avgDistances(j, 0) + distances(i, j)) / 3

  private def calculateKernel(distances: DenseMatrix[Double], m: Double, percentage: Double): DenseMatrix[Double] = {
    val kernel = DenseMatrix.zeros[Double](distances.rows, distances.rows)
    val avgDistances = calculateAverageDistance(distances, percentage)

    for (i <- 0 until distances.rows; j <- 0 until distances.cols) {
      kernel(i, j) = math.exp(-math.pow(distances(i, j), 2) / (calculateE(i, j, avgDistances, distances) * m))
    }
    kernel
  }

  private def calculateSummation(matrix: DenseMatrix[Double]): DenseMatrix[Double] = {
    val summation = DenseMatrix.zeros[Double](matrix.rows, 1)

    for (i <- 0 until matrix.rows) {
      summation(i, 0) = (0 until matrix.cols).filter(_ != i).map(j => matrix(i, j)).sum
    }
    summation
  }

  private def calculateCompleteKernel(kernel: DenseMatrix[Double]): DenseMatrix[Double] = {
    val completeKernel = DenseMatrix.zeros[Double](kernel.rows, kernel.cols)
    val summation = calculateSummation(kernel)

    for (i <- 0 until kernel.rows; j <- 0 until kernel.cols) {
      completeKernel(i, j) = if (i != j) kernel(i, j) / (2 * summation(i, 0)) else 0.5
    }
    completeKernel
  }

  private def calculateSparseKernel(kernel: DenseMatrix[Double], percentage: Double): DenseMatrix[Double] = {
    val sparseKernel = DenseMatrix.zeros[Double](kernel.rows, kernel.cols)
    val neighbors = calculateNeighbors(kernel.rows, percentage)

    for (i <- 0 until kernel.rows) {
      val sortedRow = kernel(i, ::).inner.toArray.sorted
      val threshold = sortedRow(sortedRow.length - neighbors - 1)
      for (j <- 0 until kernel.cols) {
        if (kernel(i, j) >= threshold) sparseKernel(i, j) = kernel(i, j) / sortedRow.filter(_ >= threshold).sum
      }
    }
    sparseKernel
  }

  private def calculateStatusMatrix(completeKernels: Seq[DenseMatrix[Double]], sparseKernel: DenseMatrix[Double]): DenseMatrix[Double] = {
    val sumMatrices = completeKernels.reduce(_ + _) / completeKernels.length.toDouble
    sparseKernel * sumMatrices * sparseKernel.t
  }

  def applySNF(m: Double, percentage: Double, iterations: Int): (DenseMatrix[Double], Seq[DenseMatrix[Double]]) = {
    val kernels = dataSets.map(dataSet => calculateKernel(calculateDistance(normalize(dataSet).t), m, percentage))
    var completeKernels = kernels.map(calculateCompleteKernel)
    var sparseKernels = kernels.map(kernel => calculateSparseKernel(kernel, percentage))
    var statusMatrices = Seq[DenseMatrix[Double]]()

    for (_ <- 1 to iterations) {
      statusMatrices = sparseKernels.zipWithIndex.map { case (sparseKernel,

      idx) =>
        val otherKernels = completeKernels.zipWithIndex.filter(_._2 != idx).map(_._1)
        calculateStatusMatrix(otherKernels, sparseKernel)
      }
      completeKernels = statusMatrices
    }

    val averageStatusMatrix = statusMatrices.reduce(_ + _) / statusMatrices.length.toDouble
    (averageStatusMatrix, statusMatrices)
  }
}
