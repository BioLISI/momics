package com.nouserinterface.lracluster

import breeze.linalg._
import breeze.numerics._
import breeze.stats.mean

object GaussianDataProcessor extends DataProcessor(0.5) {
  override def checkRow(arr: DenseVector[Double]): Boolean =
    arr.data.count(!_.isNaN) != 0

  def LL(mat: DenseMatrix[Double], matB: DenseMatrix[Double], matU: DenseMatrix[Double]): Double = {
    val mu = matB + matU
    -0.5 * (0 until mat.rows).flatMap { i =>
      (0 until mat.cols).collect { case j if !mat(i, j).isNaN =>
        val reu = mat(i, j) - mu(i, j)
        reu * reu
      }
    }.sum
  }

  def LLmax(mat: DenseMatrix[Double]): Double = 0.0

  def LLmin(mat: DenseMatrix[Double], matB: DenseMatrix[Double]): Double = {
    -0.5 * (0 until mat.rows).flatMap { i =>
      (0 until mat.cols).collect { case j if !mat(i, j).isNaN =>
        val reu = mat(i, j) - matB(i, j)
        reu * reu
      }
    }.sum
  }

  override def baseRow(arr: DenseVector[Double]): Double = {
    val nonNaIndices = arr.findAll(!_.isNaN)
    if (nonNaIndices.isEmpty) 0.0 else mean(arr(nonNaIndices))
  }

  def base(mat: DenseMatrix[Double]): DenseMatrix[Double] = {
    val baseMat = DenseMatrix.zeros[Double](mat.rows, mat.cols)
    for (i <- 0 until mat.rows) {
      val row = mat(i, ::).t
      val nonNaIndices = row.findAll(v => !v.isNaN)
      val meanValue = if (nonNaIndices.isEmpty) 0.0 else mean(row(nonNaIndices))
      baseMat(i, ::) := DenseVector.fill(mat.cols)(meanValue).t
    }
    baseMat
  }

  override def update(mat: DenseMatrix[Double], matB: DenseMatrix[Double], matNow: DenseMatrix[Double], eps: Double): DenseMatrix[Double] = {
    val matP = matB + matNow
    val matU = DenseMatrix.zeros[Double](mat.rows, mat.cols)
    for (i <- 0 until mat.rows; j <- 0 until mat.cols) {
      if (!mat(i, j).isNaN) {
        matU(i, j) = matNow(i, j) + eps * epsilon * (mat(i, j) - matP(i, j))
      }
    }
    matU
  }

  override def stop(mat: DenseMatrix[Double], matB: DenseMatrix[Double], matNow: DenseMatrix[Double], matU: DenseMatrix[Double]): Double = {
    val index = mat.findAll(!_.isNaN)
    val mn = matB + matNow
    val mu = matB + matU
    val lgn = -0.5 * index.map { case (i, j) => pow(mat(i, j) - mn(i, j), 2) }.sum
    val lgu = -0.5 * index.map { case (i, j) => pow(mat(i, j) - mu(i, j), 2) }.sum
    lgu - lgn
  }
}
