package com.nouserinterface.lracluster

import breeze.linalg._
import breeze.numerics._
import breeze.stats.mean

object PoissonDataProcessor extends DataProcessor(0.5) {
  override def checkRow(arr: DenseVector[Double]): Boolean =
    arr.data.count(!_.isNaN) != 0 && !arr.data.exists(_ < 0)

  def LL(mat: DenseMatrix[Double], matB: DenseMatrix[Double], matU: DenseMatrix[Double]): Double = {
    val mu = matB + matU
    (0 until mat.rows).flatMap { i =>
      (0 until mat.cols).collect { case j if !mat(i, j).isNaN =>
        mat(i, j) * mu(i, j) - exp(mu(i, j))
      }
    }.sum
  }

  def LLmax(mat: DenseMatrix[Double]): Double = {
    (0 until mat.rows).flatMap { i =>
      (0 until mat.cols).collect { case j if !mat(i, j).isNaN =>
        mat(i, j) * log(mat(i, j)) - mat(i, j)
      }
    }.sum
  }

  def LLmin(mat: DenseMatrix[Double], matB: DenseMatrix[Double]): Double = {
    (0 until mat.rows).flatMap { i =>
      (0 until mat.cols).collect { case j if !mat(i, j).isNaN =>
        mat(i, j) * matB(i, j) - exp(matB(i, j))
      }
    }.sum
  }

  override def baseRow(arr: DenseVector[Double]): Double = {
    val nonNaIndices = arr.findAll(!_.isNaN)
    if (nonNaIndices.isEmpty) 0.0 else mean(log(arr(nonNaIndices)))
  }

  def base(mat: DenseMatrix[Double]): DenseMatrix[Double] = {
    val baseMat = DenseMatrix.zeros[Double](mat.rows, mat.cols)
    for (i <- 0 until mat.rows) {
      val row = mat(i, ::).t
      val nonNaIndices = row.findAll(v => !v.isNaN)
      val meanLogValue = if (nonNaIndices.isEmpty) 0.0 else mean(log(row(nonNaIndices)))
      baseMat(i, ::) := DenseVector.fill(mat.cols)(meanLogValue).t
    }
    baseMat
  }

  override def update(mat: DenseMatrix[Double], matB: DenseMatrix[Double], matNow: DenseMatrix[Double], eps: Double): DenseMatrix[Double] = {
    val matP = matB + matNow
    val matU = DenseMatrix.zeros[Double](mat.rows, mat.cols)
    for (i <- 0 until mat.rows; j <- 0 until mat.cols) {
      if (!mat(i, j).isNaN) {
        matU(i, j) = matNow(i, j) + eps * epsilon * (log(mat(i, j)) - matP(i, j))
      }
    }
    matU
  }

  override def stop(mat: DenseMatrix[Double], matB: DenseMatrix[Double], matNow: DenseMatrix[Double], matU: DenseMatrix[Double]): Double = {
    val index = mat.findAll(!_.isNaN)
    val mn = matB + matNow
    val mu = matB + matU
    val lgn = index.map { case (i, j) => mat(i, j) * mn(i, j) - exp(mn(i, j)) }.sum
    val lgu = index.map { case (i, j) => mat(i, j) * mu(i, j) - exp(mu(i, j)) }.sum
    lgu - lgn
  }
}