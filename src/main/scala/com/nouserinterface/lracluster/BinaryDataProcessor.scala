package com.nouserinterface.lracluster

import breeze.linalg._
import breeze.numerics._
import breeze.stats.mean

object BinaryDataProcessor extends DataProcessor(2.0) {
  override def checkRow(arr: DenseVector[Double]): Boolean =
    arr.data.count(!_.isNaN) != 0 && !arr.data.forall(_ == 0.0) && !arr.data.forall(_ == 1.0)

  def LL(mat: DenseMatrix[Double], matB: DenseMatrix[Double], matU: DenseMatrix[Double]): Double = {
    val mu = matB + matU
    val aru = exp(mu)
    val idx1 = for {
      i <- 0 until mat.rows
      j <- 0 until mat.cols
      if !mat(i, j).isNaN && mat(i, j) == 1
    } yield (i, j)
    val idx0 = for {
      i <- 0 until mat.rows
      j <- 0 until mat.cols
      if !mat(i, j).isNaN && mat(i, j) == 0
    } yield (i, j)
    idx1.map { case (i, j) => log(aru(i, j) / (1.0 + aru(i, j))) }.sum +
      idx0.map { case (i, j) => log(1.0 / (1.0 + aru(i, j))) }.sum
  }

  def LLmax(mat: DenseMatrix[Double]): Double = 0.0

  def LLmin(mat: DenseMatrix[Double], matB: DenseMatrix[Double]): Double = {
    val aru = exp(matB)
    val idx1 = for {
      i <- 0 until mat.rows
      j <- 0 until mat.cols
      if !mat(i, j).isNaN && mat(i, j) == 1
    } yield (i, j)
    val idx0 = for {
      i <- 0 until mat.rows
      j <- 0 until mat.cols
      if !mat(i, j).isNaN && mat(i, j) == 0
    } yield (i, j)
    idx1.map { case (i, j) => log(aru(i, j) / (1.0 + aru(i, j))) }.sum +
      idx0.map { case (i, j) => log(1.0 / (1.0 + aru(i, j))) }.sum
  }

  def base(mat: DenseMatrix[Double]): DenseMatrix[Double] = {
    DenseMatrix.tabulate(mat.rows, mat.cols) { (i, j) =>
      if (mat(i, j).isNaN) 0.0 else log(mat(i, j) / (1.0 - mat(i, j)))
    }
  }

  override def baseRow(arr: DenseVector[Double]): Double = {
    val nonNa = arr.findAll(!_.isNaN)
    if (nonNa.isEmpty) 0.0 else log(mean(arr(nonNa)))
  }

  override def update(mat: DenseMatrix[Double], matB: DenseMatrix[Double], matNow: DenseMatrix[Double], eps: Double): DenseMatrix[Double] = {
    val matP = matB + matNow
    val matU = DenseMatrix.zeros[Double](mat.rows, mat.cols)
    for (i <- 0 until mat.rows; j <- 0 until mat.cols) {
      if (!mat(i, j).isNaN) {
        matU(i, j) = matNow(i, j) + eps * epsilon * (if (mat(i, j) == 1) 1.0 / (1.0 + exp(-matP(i, j))) else -exp(matP(i, j)) / (1.0 + exp(matP(i, j))))
      }
    }
    matU
  }

  override def stop(mat: DenseMatrix[Double], matB: DenseMatrix[Double], matNow: DenseMatrix[Double], matU: DenseMatrix[Double]): Double = {
    val index = mat.findAll(!_.isNaN)
    val mn = matB + matNow
    val mu = matB + matU
    val lgn = index.map { case (i, j) => if (mat(i, j) == 1) log(1.0 / (1.0 + exp(-mn(i, j)))) else log(1.0 - 1.0 / (1.0 + exp(-mn(i, j)))) }.sum
    val lgu = index.map { case (i, j) => if (mat(i, j) == 1) log(1.0 / (1.0 + exp(-mu(i, j)))) else log(1.0 - 1.0 / (1.0 + exp(-mu(i, j)))) }.sum
    lgu - lgn
  }
}
