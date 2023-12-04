package com.nouserinterface.lracluster

import breeze.linalg._
import breeze.numerics._
import com.nouserinterface.lracluster.BinaryDataProcessor.nuclearApproximation

object LRACluster {
  def processMatrix(mat: DenseMatrix[Double], dataType: String, name: String): DenseMatrix[Double] = dataType match {
    case "binary" => BinaryDataProcessor.checkMatrix(mat, name)
    case "gaussian" => GaussianDataProcessor.checkMatrix(mat, name)
    case "poisson" => PoissonDataProcessor.checkMatrix(mat, name)
    case _ => throw new IllegalArgumentException(s"Unknown type $dataType")
  }

  def cluster(data: Array[DenseMatrix[Double]], types: Array[String], dimension: Int = 2, names: Array[String] = Array()): DenseMatrix[Double] = {
    if (data.length != types.length) throw new IllegalArgumentException("Data and types must be the same length")

    val checkedData = data.zip(types).zipWithIndex.map { case ((mat, dt), idx) => processMatrix(mat, dt, names.lift(idx).getOrElse(idx.toString)) }
    val nSample = checkedData.head.cols
    val nGene = checkedData.map(_.rows).sum

    val base = DenseMatrix.zeros[Double](nGene, nSample)
    val now = DenseMatrix.zeros[Double](nGene, nSample)
    var update = DenseMatrix.zeros[Double](nGene, nSample)

    var loglmin = 0.0
    var loglmax = 0.0
    var loglu = 0.0
    var startRow = 0

    checkedData.zip(types).foreach { case (mat, dt) =>
      val endRow = startRow + mat.rows
      val indexRange = startRow until endRow
      val baseSegment = base(indexRange, ::)
      val nowSegment = now(indexRange, ::)
      val updateSegment = update(indexRange, ::)

      dt match {
        case "binary" =>
          baseSegment := BinaryDataProcessor.base(mat)
          loglmin += BinaryDataProcessor.LLmin(mat, baseSegment)
          loglmax += BinaryDataProcessor.LLmax(mat)
        case "gaussian" =>
          baseSegment := GaussianDataProcessor.base(mat)
          loglmin += GaussianDataProcessor.LLmin(mat, baseSegment)
          loglmax += GaussianDataProcessor.LLmax(mat)
        case "poisson" =>
          baseSegment := PoissonDataProcessor.base(mat)
          loglmin += PoissonDataProcessor.LLmin(mat, baseSegment)
          loglmax += PoissonDataProcessor.LLmax(mat)
      }

      startRow = endRow
    }

    var eps = 0.0
    var nIter = 0
    var thres = DenseVector.zeros[Double](3)
    var epsN = DenseVector.zeros[Double](2)
    var thr = DenseVector.zeros[Double](checkedData.length)

    do {
      startRow = 0
      checkedData.zip(types).foreach { case (mat, dt) =>
        val endRow = startRow + mat.rows
        val indexRange = startRow until endRow
        val baseSegment = base(indexRange, ::)
        val nowSegment = now(indexRange, ::)
        val updateSegment = update(indexRange, ::)

        dt match {
          case "binary" => thr(startRow / mat.rows) = BinaryDataProcessor.stop(mat, baseSegment, nowSegment, updateSegment)
          case "gaussian" => thr(startRow / mat.rows) = GaussianDataProcessor.stop(mat, baseSegment, nowSegment, updateSegment)
          case "poisson" => thr(startRow / mat.rows) = PoissonDataProcessor.stop(mat, baseSegment, nowSegment, updateSegment)
        }

        startRow = endRow
      }

      nIter += 1
      thres = DenseVector(thres(1), thres(2), sum(thr))
      epsN = DenseVector(epsN(1), eps)

      if (nIter > 5) {
        // Assuming runif generates a uniform random number between 0 and 1
        eps = if (scala.util.Random.nextDouble() < thres(0) * thres(2) / (thres(1) * thres(1) + thres(0) * thres(2))) epsN(0) + 0.05 * scala.util.Random.nextDouble() - 0.025
        else epsN(1) + 0.05 * scala.util.Random.nextDouble() - 0.025

        if (eps < -0.7) eps = 0
        if (eps > 1.4) eps = 0
      }

      if (sum(thr) < checkedData.length * 0.2) return now

      now := update
      startRow = 0
      checkedData.zip(types).foreach { case (mat, dt) =>
        val endRow = startRow + mat.rows
        val indexRange = startRow until endRow
        val baseSegment = base(indexRange, ::)
        val nowSegment = now(indexRange, ::)
        val updateSegment = update(indexRange, ::)

        dt match {
          case "binary" => updateSegment := BinaryDataProcessor.update(mat, baseSegment, nowSegment, exp(eps))
          case "gaussian" => updateSegment := GaussianDataProcessor.update(mat, baseSegment, nowSegment, exp(eps))
          case "poisson" => updateSegment := PoissonDataProcessor.update(mat, baseSegment, nowSegment, exp(eps))
        }

        startRow = endRow
      }

      update = nuclearApproximation(update, dimension)
    } while (true)

    now
  }

  def main(args: Array[String]): Unit = {
    // Example test case
    val data = Array(
      DenseMatrix((1.0, 2.0), (3.0, 4.0)), // Example binary matrix
      DenseMatrix((5.0, 6.0), (7.0, 8.0)), // Example Gaussian matrix
      DenseMatrix((9.0, 10.0), (11.0, 12.0)) // Example Poisson matrix
    )
    val types = Array("binary", "gaussian", "poisson")
    val names = Array("binaryData", "gaussianData", "poissonData")

    val result = LRACluster.cluster(data, types, dimension = 2, names = names)
    println(result)
  }
}