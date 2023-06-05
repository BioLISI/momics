package com.nouserinterface

import com.nouserinterface.Trainer.FancyIterable

import breeze.linalg.{DenseVector, _}
import breeze.stats.distributions._

object AdmixtureModel {
  implicit val rand: RandBasis = RandBasis.withSeed(42)

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

  def simulateData(n: Int, k: Int, loci: Int, ploidy: Int): (Seq[DenseMatrix[Int]], Seq[DenseMatrix[Double]], DenseMatrix[Double], Seq[DenseMatrix[Int]], Seq[Int]) = {
    val numAlleles = Seq.fill(loci)(rand.randInt(5).sample() + 1)
    val p = numAlleles.map(na => dirichlets(k, DenseVector.ones[Double](na)))
    val q = dirichlets(n, DenseVector.ones[Double](k))

    val Z = Seq.fill(ploidy) {
      (0 until loci).map(l => multinomials(q(*, ::).map(dv => {
        val a = dv *:* p(1)(::, *);
        sum(a(*, ::))
      })).t).reduceLeft((a, b) => DenseMatrix.horzcat(a, b))
    }

    val X = Z.map { z =>
      (0 until loci).map(l =>
        multinomials(p(l)(z(::, l).toScalaVector, ::).toDenseMatrix).t
      ).reduceLeft((a, b) => DenseMatrix.horzcat(a, b))
    }
    (Z, p, q, X, numAlleles)
  }

  def updateZ(n: Int, k: Int, loci: Int, p: Seq[DenseMatrix[Double]], q: DenseMatrix[Double], X: Seq[DenseMatrix[Int]]): Seq[DenseMatrix[Int]] = {
    X.map(x =>
      DenseMatrix.tabulate(n, loci) { case (i, l) =>
        val allele = x(i, l)
        val probabilities = DenseVector.tabulate(k)(j => q(i, j) * p(l)(j, allele))
        Multinomial[DenseVector[Double], Int](probabilities).sample()
      }
    )
  }

  def updateP(k: Int, loci: Int, z: Seq[DenseMatrix[Int]], numAlleles: Seq[Int]): Seq[DenseMatrix[Double]] = {
    numAlleles.map { alleles =>
      DenseMatrix.tabulate(k, alleles) { case (j, a) =>
        val count = sum(z.map(zMatrix => sum(zMatrix(*, ::).map(d => d.data.count(_ == j)))))
        Gamma(count + 1.0, 1.0).draw()
      }
    }
  }

  def updateQ(n: Int, k: Int, z: Seq[DenseMatrix[Int]]): DenseMatrix[Double] = {
    DenseMatrix.tabulate(n, k) { case (i, j) =>
      val count = z.map(zMatrix => zMatrix(i, ::).inner.data.count(_ == j)).sum
      val alpha = DenseVector.tabulate(k)(_ => count + 1.0)
      dirichlet(alpha)(j)
    }
  }

  def runMCMC(X: Seq[DenseMatrix[Int]], n: Int, loci: Int, k: Int, burnIn: Int, thin: Int, iterations: Int, ploidy: Int = 2): (Seq[DenseMatrix[Int]], Seq[DenseMatrix[Double]], DenseMatrix[Double]) = {
    val (z0, p0, q0, _, numAlleles) = simulateData(n, k, loci, ploidy)
    (1 to iterations).foldOrStop((z0, p0, q0)) { case ((z, p, q), i) =>
      val nz = updateZ(n, k, loci, p, q, X)
      if (i >= burnIn && (i - burnIn) % thin == 0) {
        Some((nz, updateP(k, loci, nz, numAlleles), updateQ(n, k, nz)))
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
    val ploidy = 2

    // Simulate data
    val (z0, pSimulated, qSimulated, x, numAlleles) = AdmixtureModel.simulateData(n, k, loci, ploidy)

    // Run MCMC
    val (zMCMC, pMCMC, qMCMC) = AdmixtureModel.runMCMC(x, n, loci, k, burnIn, thin, iterations)

    // Compute RMSE
    val pRMSE1 = rmse2(pSimulated(0), pMCMC(0))
    val pRMSE2 = rmse2(pSimulated(1), pMCMC(1))
    val qRMSE = rmse2(qSimulated, qMCMC)

    // Print RMSE
    println(s"RMSE2 for p0: $pRMSE1")
    println(s"RMSE2 for p1: $pRMSE2")
    println(s"RMSE2 for q: $qRMSE")
  }
}