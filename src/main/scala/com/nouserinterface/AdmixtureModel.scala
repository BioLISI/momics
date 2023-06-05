package com.nouserinterface

import com.nouserinterface.Trainer.FancyIterable

import breeze.linalg.{DenseVector, _}
import breeze.stats.distributions._
import org.apache.commons.math3.random.MersenneTwister

//import breeze.compat._
//import scala.collection.compat._
//import breeze.compat.Scala3Compat._
import breeze.compat.Scala3Compat.given_Conversion_T_U

object AdmixtureModel {
  implicit val rand: RandBasis = new RandBasis(new MersenneTwister())

  def dirichlets(k: Int, alpha: DenseVector[Double]): DenseMatrix[Double] = {
    val sample = (0 until alpha.length).map(i => Gamma(alpha(i), 1.0).sample(k)).reduceLeft(_ ++ _).toArray
    val theta = new DenseMatrix(k, alpha.length, sample)
    normalize(theta(*, ::), 1)
  }

  def dirichlet(alpha: DenseVector[Double]): DenseVector[Double] = {
    dirichlets(1, alpha)(0, ::).t
  }

  def multinomials(params: DenseMatrix[Double], k: Int = 1): DenseMatrix[Int] = {
    val sample = params(*, ::).map(dv => Multinomial[DenseVector[Double], Int](dv).sample(k)).reduceLeft(_ ++ _).toArray
    new DenseMatrix(k, params.rows, sample)
  }

  def multinomial(params: DenseVector[Double]): Int = {
    Multinomial[DenseVector[Double], Int](params).draw()
  }

  def simulateData(n: Int, k: Int, loci: Int): (DenseMatrix[Int], DenseMatrix[Double], DenseMatrix[Double], DenseMatrix[Int]) = {
    val p = dirichlets(k, DenseVector.ones[Double](loci))
    val q = dirichlets(n, DenseVector.ones[Double](k))
    val z = q(*, ::).map(qe => multinomials((qe *:* p(::, *)).t)).reduceLeft((a, b) => DenseMatrix.vertcat(a, b))


    val X = (0 until n).map(
      i => {
        val probs: DenseMatrix[Double] = p(z(i, ::).t.toScalaVector, ::).toDenseMatrix
        probs(*, ::).map(dv => Multinomial[DenseVector[Double], Int](dv).draw()).toDenseMatrix
      }).reduceLeft((a, b) => DenseMatrix.horzcat(a, b))
    (z, p, q, X)
  }

  def updateZ(n: Int, loci: Int, p: DenseMatrix[Double], q: DenseMatrix[Double], X: Option[DenseMatrix[Int]] = None): DenseMatrix[Int] = {
    (0 until n).map(
      i => {
        val rs: DenseMatrix[Double] = X match {
          case Some(v) => p(::, v(i, ::).t.toScalaVector).toDenseMatrix
          case None => p
        }
        val probs = q(i, ::).t *:* rs(::, *)
        probs(*, ::).map(dv => Multinomial[DenseVector[Double], Int](dv).draw()).toDenseMatrix
      }).reduceLeft((a, b) => DenseMatrix.horzcat(a, b))
  }

  def updateP(k: Int, loci: Int, z: DenseMatrix[Int]): DenseMatrix[Double] = {
    DenseMatrix.tabulate(k, loci) {
      case (j, l) =>
        Gamma(z(::, l).valuesIterator.count(_ == j) + 1.0, 1.0).draw()
    }
  }

  def updateQ(n: Int, k: Int, z: DenseMatrix[Int]): DenseMatrix[Double] = {
    (0 until n).map(
      i => dirichlet(
        DenseVector.tabulate(k)(
          j => z(i, ::).inner.toArray.count(_ == j) + 1.0)).toDenseMatrix)
      .reduceLeft((a, b) => DenseMatrix.horzcat(a, b))
  }

  def runMCMC(X: DenseMatrix[Int], n: Int, loci: Int, k: Int, burnIn: Int, thin: Int, iterations: Int): (DenseMatrix[Int], DenseMatrix[Double], DenseMatrix[Double]) = {
    val (z0, p0, q0, _) = simulateData(n, k, loci)
    (1 to iterations).foldOrStop((z0, p0, q0)) { case ((z, p, q), i) =>
      val nz = updateZ(n, loci, p, q, Some(X))
      if (i >= burnIn && (i - burnIn) % thin == 0) {
        Some((nz, updateP(k, loci, nz), updateQ(n, k, nz)))
      } else
        Some((nz, p, q))

    }
  }

  def rmse2(a: DenseMatrix[Double], b: DenseMatrix[Double]): Double = {
    require(a.rows == b.rows && a.cols == b.cols, "Matrices must have the same dimensions")
    val diff = a - b
    sum(diff *:* diff) / (a.rows * a.cols)
  }

  def main(args: Array[String]): Unit = {
    val n = 100
    val k = 5
    val loci = 20
    val burnIn = 100
    val thin = 10
    val iterations = 5000

    // Simulate data
    val (z, pSimulated, qSimulated, x) = AdmixtureModel.simulateData(n, k, loci)

    // Run MCMC
    val (zMCMC, pMCMC, qMCMC) = AdmixtureModel.runMCMC(x, n, loci, k, burnIn, thin, iterations)

    // Compute RMSE
    val pRMSE = rmse2(pSimulated, pMCMC)
    val qRMSE = rmse2(qSimulated, qMCMC)

    // Print RMSE
    println(s"RMSE2 for p: $pRMSE")
    println(s"RMSE2 for q: $qRMSE")
  }
}