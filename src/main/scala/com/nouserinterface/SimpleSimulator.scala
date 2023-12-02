package com.nouserinterface

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions.{Gaussian, RandBasis}
import com.nouserinterface.Distributions.{dirichlets, multinomials}

import scala.collection.immutable.Seq

class SimpleSimulator {
  implicit val rand: RandBasis = RandBasis.systemSeed
  def generateMultinomialMatrix(n: Int, k: Int, s: Double = 1.0, ns: Double = 0.01): DenseMatrix[Double] = {
    val samples = multinomials(DenseMatrix.ones[Double](1, k) /:/ k.toDouble, n).toArray
    DenseMatrix.tabulate(n, k) { (i, j) =>
      if (samples(i) == j) s else ns
    }
  }

  def simulateData(n: Int, k: Int, loci: Int, alpha: Double = 1.0, nAlleles: Option[Seq[Int]] = None, ploidy: Int = 2): (Seq[DenseMatrix[Int]], Seq[DenseMatrix[Double]], DenseMatrix[Double], Seq[DenseMatrix[Int]], Seq[Int]) = {
    val numAlleles = nAlleles match {
      case Some(nA) => nA
      case None => Seq.fill(loci)(rand.randInt(5).sample() + 2)
    }
    val p = numAlleles.map(na => dirichlets(k, DenseVector.ones[Double](na)))
    val qParams = generateMultinomialMatrix(n, k)
    val q = dirichlets(qParams)

    val Z = Seq.fill(ploidy) {
      (0 until loci).map(l => multinomials(q).t).reduceLeft((a, b) => DenseMatrix.horzcat(a, b))
    }

    val X = Z.map(z =>
      (0 until loci).map(l => {
        val params = p(l)(z(::, l).toScalaVector, ::).toDenseMatrix
        multinomials(params).t
      }).reduceLeft((a, b) => DenseMatrix.horzcat(a, b))
    )
    val normalCopyNumber = Gaussian(0, 1)
    (Z, p, q, X, numAlleles)
  }
}
