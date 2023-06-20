package com.nouserinterface

import breeze.linalg._
import breeze.stats.distributions._

object Distributions {
  def dirichlets(k: Int, alpha: DenseVector[Double])(implicit rand: RandBasis): DenseMatrix[Double] = {
    val sample = (0 until alpha.length).map(i => Gamma(alpha(i), 1.0).sample(k)).reduceLeft(_ ++ _).toArray
    val theta = new DenseMatrix(k, alpha.length, sample)
    normalize(theta(*, ::), 1)
  }

  def dirichlet(alpha: DenseVector[Double])(implicit rand: RandBasis): DenseVector[Double] = {
    dirichlets(1, alpha)(rand)(0, ::).t
  }

  def dirichlets(alphas: DenseMatrix[Double])(implicit rand: RandBasis): DenseMatrix[Double] = {
    alphas(*, ::).map(dv => dirichlets(1, dv)).reduceLeft((a, b) => DenseMatrix.vertcat(a, b))
  }

  def multinomials(params: DenseMatrix[Double], k: Int = 1)(implicit rand: RandBasis): DenseMatrix[Int] = {
    val sample = params(*, ::).map(dv => Multinomial[DenseVector[Double], Int](dv).sample(k)).reduceLeft(_ ++ _).toArray
    new DenseMatrix(k, params.rows, sample)
  }

  def multinomial(params: DenseVector[Double])(implicit rand: RandBasis): Int = {
    Multinomial[DenseVector[Double], Int](params).draw()
  }

  //from mymath.c in Structure
  def lnGamma(z: Double): Double = {
    assert(z > 0d)
    val a = Seq(0.9999999999995183, 676.5203681218835, -1259.139216722289, 771.3234287757674, -176.6150291498386, 12.50734324009056, -0.1385710331296526, 9.934937113930748e-6, 1.659470187408462e-7)
    val lnsqrt2pi = 0.9189385332046727
    val (result, _) = a.tail.foldRight((a.head, z + 7.0d)) { case (at, (result, temp)) => (result + at / temp, temp - 1.0d)
    }
    math.log(result) + lnsqrt2pi - (z + 6.5) + (z - 0.5) * math.log(z + 6.5);
  }
}
