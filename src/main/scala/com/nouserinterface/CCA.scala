package com.nouserinterface

import breeze.linalg.{DenseMatrix, eig, inv, sum}
import breeze.numerics.{abs, pow}
import breeze.stats.{mean, stddev}

class CCA(matrix: DenseMatrix[Double]) {
  private def calculateMeanMatrix(matrix: DenseMatrix[Double]): DenseMatrix[Double] = {
    val meanMatrix = matrix.copy
    for (j <- 0 until matrix.cols) {
      val avg = mean(matrix(::, j))
      for (i <- 0 until matrix.rows) {
        meanMatrix(i, j) = avg
      }
    }
    meanMatrix
  }

  private def calculateCovarianceMatrix(x: DenseMatrix[Double], y: DenseMatrix[Double], meanMatrixX: DenseMatrix[Double], meanMatrixY: DenseMatrix[Double]): DenseMatrix[Double] = {
    ((x - meanMatrixX).t * (y - meanMatrixY)).map(l => l / x.rows)
  }

  private def normalizeData(m: DenseMatrix[Double]): DenseMatrix[Double] = {
    (m - mean(m)) / stddev(m)
  }

  private def calculateCorrelationMatrix(u: DenseMatrix[Double], v: DenseMatrix[Double]): DenseMatrix[Double] = {
    val uNorm = normalizeData(u)
    val vNorm = normalizeData(v)
    val meanMatrixU = calculateMeanMatrix(uNorm)
    val meanMatrixV = calculateMeanMatrix(vNorm)
    val covarianceU = abs(pow(sum(calculateCovarianceMatrix(uNorm, uNorm, meanMatrixU, meanMatrixU)), 0.5))
    val covarianceV = abs(pow(sum(calculateCovarianceMatrix(vNorm, vNorm, meanMatrixV, meanMatrixV)), 0.5))
    val covarianceUV = abs(sum(calculateCovarianceMatrix(uNorm, vNorm, meanMatrixU, meanMatrixV)))
    val covarianceVU = abs(sum(calculateCovarianceMatrix(vNorm, uNorm, meanMatrixV, meanMatrixU)))

    DenseMatrix((covarianceU, covarianceUV), (covarianceVU, covarianceV))
  }

  private def calculateWeights(x: DenseMatrix[Double], y: DenseMatrix[Double]): (DenseMatrix[Double], DenseMatrix[Double]) = {
    val meanMatrixX = calculateMeanMatrix(x)
    val meanMatrixY = calculateMeanMatrix(y)

    val sigmaS = List(
      calculateCovarianceMatrix(x, x, meanMatrixX, meanMatrixX),
      calculateCovarianceMatrix(x, y, meanMatrixX, meanMatrixY),
      calculateCovarianceMatrix(y, y, meanMatrixY, meanMatrixY)
    )

    val matrixSigmaX: DenseMatrix[Double] = inv(sigmaS(0)) * sigmaS(1) * inv(sigmaS(2)) * sigmaS(1).t
    val matrixSigmaY: DenseMatrix[Double] = inv(sigmaS(2)) * sigmaS(1).t * inv(sigmaS(0)) * sigmaS(1)

    (calculateEigenvalues(matrixSigmaX, sigmaS(0)), calculateEigenvalues(matrixSigmaY, sigmaS(2)))
  }

  private def calculateEigenvalues(matrixSigma: DenseMatrix[Double], covarianceMatrix: DenseMatrix[Double]): DenseMatrix[Double] = {
    val eigenvalues = eig(matrixSigma).eigenvalues.asDenseMatrix
    val constraint = eigenvalues * covarianceMatrix * eigenvalues.t
    eigenvalues.map(x => x / pow(constraint(0, 0), 0.5))
  }

  private def calculateCanonicalVariables(data: DenseMatrix[Double], weights: DenseMatrix[Double]): DenseMatrix[Double] = {
    if (data.cols != weights.rows) {
      data * weights.t
    } else {
      data * weights
    }
  }

  def applyCanonicalCorrelationAnalysis(): DenseMatrix[Double] = {
    val matrixT = matrix.t
    var max: Double = 0
    var correlationMatrixMax = matrix

    for (i <- 5 until matrixT.cols) {
      val x = matrixT(0 until matrixT.rows - 1, 0 until i)
      val y = matrixT(0 until matrixT.rows - 1, i until matrixT.cols)

      val (weightsX, weightsY) = calculateWeights(x, y)

      val u = calculateCanonicalVariables(x, weightsX)
      val v = calculateCanonicalVariables(y, weightsY)

      val correlationMatrix = calculateCorrelationMatrix(u, v)
      if (correlationMatrix(0, 1) > max) {
        max = correlationMatrix(0, 1)
        correlationMatrixMax = correlationMatrix
      }
    }
    correlationMatrixMax
  }
}
