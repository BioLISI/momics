package com.nouserinterface.lracluster

import breeze.linalg._
import breeze.numerics._

abstract class DataProcessor(val epsilon: Double) {
  def checkRow(arr: DenseVector[Double]): Boolean

  def checkMatrix(mat: DenseMatrix[Double], name: String): DenseMatrix[Double] = {
    val validRows = (0 until mat.rows).filter(i => checkRow(mat(i, ::).t))
    if (validRows.isEmpty) throw new IllegalArgumentException(s"All rows in $name are invalid.")
    mat(validRows, ::).toDenseMatrix
  }

  def LL(mat: DenseMatrix[Double], matB: DenseMatrix[Double], matU: DenseMatrix[Double]): Double

  def LLmin(mat: DenseMatrix[Double], matB: DenseMatrix[Double]): Double

  def LLmax(mat: DenseMatrix[Double]): Double

  def baseRow(arr: DenseVector[Double]): Double
  def base(mat: DenseMatrix[Double]): DenseMatrix[Double]

  def update(mat: DenseMatrix[Double], matB: DenseMatrix[Double], matNow: DenseMatrix[Double], eps: Double): DenseMatrix[Double]
  def stop(mat: DenseMatrix[Double], matB: DenseMatrix[Double], matNow: DenseMatrix[Double], matU: DenseMatrix[Double]): Double

  def nuclearApproximation(mat: DenseMatrix[Double], dimension: Int): DenseMatrix[Double] = {
    val svdResult = svd(mat)
    val d = svdResult.singularValues
    val u = svdResult.U
    val v = svdResult.Vt.t

    if (dimension < d.length) {
      val lambda = d(dimension)
      val filteredD = d.map(value => if (value > lambda) value - lambda else 0.0)
      val diagMatrix = diag(DenseVector(filteredD.toArray.take(dimension) ++ Array.fill(mat.cols - dimension)(0.0)))
      u(::, 0 until dimension) * diagMatrix * v(::, 0 until dimension).t
    } else {
      mat
    }
  }

  def process(data: DenseMatrix[Double], dimension: Int, name: String, eps: Double): DenseMatrix[Double] = {
    var dataChecked = checkMatrix(data, name)
    var dataB = DenseMatrix.tabulate(dataChecked.rows, dataChecked.cols) { (i, _) => baseRow(dataChecked(i, ::).t) }
    var dataNow = DenseMatrix.zeros[Double](dataChecked.rows, dataChecked.cols)
    var dataU = update(dataChecked, dataB, dataNow, eps)
    dataU = nuclearApproximation(dataU, dimension)
    while (true) {
      val thr = stop(dataChecked, dataB, dataNow, dataU)
      if (thr < 0.2) return dataNow
      dataNow = dataU
      dataU = update(dataChecked, dataB, dataNow, eps)
      dataU = nuclearApproximation(dataU, dimension)
    }
    dataNow
  }
}