package com.nouserinterface

import breeze.linalg._
import breeze.stats.distributions._
import breeze.stats.mean
import com.nouserinterface.Distributions._
import com.nouserinterface.Trainer.FancyIterable

object AdmixtureModel {
  implicit val rand: RandBasis = RandBasis.systemSeed

  def main(args: Array[String]): Unit = {
    val n = 200
    val k = 4
    val loci = 7
    val burnIn = 10000
    val thin = 1
    val iterations = 20000
    val alpha = 1.0

    // Simulate data
    val (z0, pSimulated, qSimulated, x, numAlleles) = AdmixtureModel.simulateData(n, k, loci, alpha)

    // Run MCMC
    val State(zMCMC, pMCMC, qMCMC, alphaMCMC) = AdmixtureModel.runMCMC(x, k, iterations, burnIn, thin)

    // Compute RMSE

    val pRMSE = (0 until loci).map(i => rmse(pSimulated(i), pMCMC(i)))
    val qRMSE = rmse(qSimulated, qMCMC)

    // Print RMSE
    println(f"RMSE for p (${mean(pRMSE)}%.2f): ${pRMSE.zip(numAlleles).map { case (r, n) => f"$r%.2f($n%d)" }.mkString("\n", ",", "\n")}q: $qRMSE%.2f")
    println(qSimulated)
    println(qMCMC)
  }

  def simulateData(n: Int, k: Int, loci: Int, alpha: Double = 1.0, nAlleles: Option[Seq[Int]] = None): (DenseMatrix[Int], Seq[DenseMatrix[Double]], DenseMatrix[Double], DenseMatrix[Int], Seq[Int]) = {
    val numAlleles = nAlleles match {
      case Some(nA) => nA
      case None => Seq.fill(loci)(rand.randInt(5).sample() + 2)
    }
    val p = numAlleles.map(na => dirichlets(k, DenseVector.ones[Double](na)))
    val q = dirichlets(n, DenseVector.zeros[Double](k) +:+ alpha)

    val Z = (0 until loci).map(l => multinomials(q).t).reduceLeft((a, b) => DenseMatrix.horzcat(a, b))

    val X =
      (0 until loci).map(l => {
        val params = p(l)(Z(::, l).toScalaVector, ::).toDenseMatrix
        multinomials(params).t
      }).reduceLeft((a, b) => DenseMatrix.horzcat(a, b))

    (Z, p, q, X, numAlleles)
  }

  def runMCMC(X: DenseMatrix[Int], k: Int, iterations: Int, burnIn: Int, thin: Int): State = {
    val (n, loci) = (X.rows, X.cols)
    val numAlleles = (0 until loci).map(l => X(::, l).toArray.max + 1)
    val z0 =
      numAlleles.map(j => multinomials(DenseMatrix.ones[Double](1, k) /:/ k.toDouble, n)).reduceLeft((a, b) => DenseMatrix.horzcat(a, b))

    val initial = zeroState(n, k, loci, numAlleles).copy(z = z0, alpha = 1.0)
    val average = zeroState(n, k, loci, numAlleles)
    val (_, expectedStage) = (1 to iterations).foldOrStop((initial, average)) { case ((current, average), i) => {
      val np = updateP(k, current.z, X, numAlleles)
      val nq = updateQ(current.alpha, n, k, current.z)
      val nz = updateZ(np, nq, X)
      val nalpha = updateAlpha(current.alpha, nq, n, k)
      val updated = State(nz, np, nq, nalpha)
      val naverage = if (i >= burnIn && (i - burnIn) % thin == 0) {
        val nstate = average + updated
        println(s"$i: ${nstate.alpha / ((i - burnIn) / thin)}")
        nstate
      } else average
      Some(updated, naverage)
    }
    }
    expectedStage / ((iterations - burnIn) / thin)
  }

  def updateZ(p: Seq[DenseMatrix[Double]], q: DenseMatrix[Double], x: DenseMatrix[Int]): DenseMatrix[Int] = {
    p.indices.map(l => {
      val s = ((p(l).t)(x(::, l).toScalaVector, ::)).toDenseMatrix
      val r = q *:* s
      val nr = r(::, *) /:/ sum(r(*, ::))
      multinomials(nr).t
    }).reduceLeft((a, b) => DenseMatrix.horzcat(a, b))
  }

  def updateP(k: Int, z: DenseMatrix[Int], x: DenseMatrix[Int], numAlleles: Seq[Int]): Seq[DenseMatrix[Double]] = {
    numAlleles.indices.map(l => {
      val d1s = DenseMatrix.ones[Double](k, numAlleles(l))
      z(::, l).toArray.zip(x(::, l).toArray).foreach { case (ze, xe) => d1s.update(ze, xe, d1s(ze, xe) + 1.0)}
      dirichlets(d1s)
    })
  }

  def updateQ(alpha: Double, n: Int, k: Int, z: DenseMatrix[Int]): DenseMatrix[Double] = {
    val d1s = DenseMatrix.zeros[Double](n, k) +:+ alpha
    (0 until n).foreach(i => z(i, ::).t.toArray.foreach(ki => d1s.update(i, ki, d1s(i, ki) + 1.0)))
    dirichlets(d1s)
  }

  def updateAlpha(alpha: Double, q: DenseMatrix[Double], n: Int, k: Int): Double = {
    val cAlpha = Gaussian(alpha, 0.05).draw()
    if (cAlpha < 10 && cAlpha > 0){
      val rv = Uniform(0.0, 1.0).draw()
      val thresh = math.exp(logProbQ(cAlpha, q, n, k) - logProbQ(alpha, q, n, k))
      println(f"$thresh%10f $cAlpha%.2f $rv%.10f $alpha%.2f")
      if( rv< thresh) cAlpha else alpha
    }else alpha
  }

  def logProbQ(alpha: Double, q: DenseMatrix[Double], n: Int, k: Int): Double = {
    val uf: Double = 1.0e-200
    val (sum, runningTotal) = q.t.toArray.foldLeft((0.0, 1.0)) {
      case ((sum, runningTotal), v) => {
        val nr = runningTotal * (if (v > uf) v else uf)
        if (nr < uf)
          (sum + (alpha - 1.0) * math.log(nr), 1.0)
        else
          (sum, nr)
      }
    }
    sum + (alpha - 1.0) * math.log(runningTotal) + (lnGamma(k * alpha) - k * lnGamma(alpha)) * n
  }

  def zeroState(n: Int, k: Int, loci: Int, numAlleles: Seq[Int]): State = {
    State(
      DenseMatrix.zeros[Int](n, loci)
    , numAlleles.map(nl => DenseMatrix.zeros[Double](k, nl)), DenseMatrix.ones[Double](n, k) /:/ k.toDouble, 0.0f)
  }

  def rmse(a: DenseMatrix[Double], b: DenseMatrix[Double]): Double = {
    require(a.rows == b.rows && a.cols == b.cols, "Matrices must have the same dimensions")
    val diff = a - b
    math.sqrt(sum(diff *:* diff) / (a.rows * a.cols))
  }

  case class State(z: DenseMatrix[Int], p: Seq[DenseMatrix[Double]], q: DenseMatrix[Double], alpha: Double) {
    def +(other: State): State = {
      State(this.z + other.z, this.p.zip(other.p).map { case (a, b) => a + b }, this.q + other.q, this.alpha + other.alpha)
    }

    def /(den: Double): State = {
      State(this.z /:/ den.toInt, this.p.map(a => a /:/ den), this.q /:/ den, this.alpha / den)
    }
  }
}
