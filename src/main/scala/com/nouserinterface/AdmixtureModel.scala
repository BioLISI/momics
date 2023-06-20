package com.nouserinterface

import breeze.linalg._
import breeze.stats.distributions._
import com.nouserinterface.Distributions._
import com.nouserinterface.Trainer.FancyIterable

object AdmixtureModel {
  implicit val rand: RandBasis = RandBasis.systemSeed

  def simulateData(n: Int, k: Int, loci: Int, ploidy: Int, alpha: Double = 1.0, nAlleles: Option[Seq[Int]] = None): (Seq[DenseMatrix[Int]], Seq[DenseMatrix[Double]], DenseMatrix[Double], Seq[DenseMatrix[Int]], Seq[Int]) = {
    val numAlleles = nAlleles match {
      case Some(nA) => nA
      case None => Seq.fill(loci)(rand.randInt(5).sample() + 2)
    }
    val p = numAlleles.map(na => dirichlets(k, DenseVector.ones[Double](na)))
    val q = dirichlets(n, DenseVector.zeros[Double](k) +:+ alpha)

    val Z = Seq.fill(ploidy) {
      (0 until loci).map(l => multinomials(q).t).reduceLeft((a, b) => DenseMatrix.horzcat(a, b))
    }

    val X = Z.map { z =>
      (0 until loci).map(l => {
        multinomials(p(l)(z(::, l).toScalaVector, ::).toDenseMatrix).t
      }).reduceLeft((a, b) => DenseMatrix.horzcat(a, b))
    }
    (Z, p, q, X, numAlleles)
  }

  def updateZ(p: Seq[DenseMatrix[Double]], q: DenseMatrix[Double], X: Seq[DenseMatrix[Int]]): Seq[DenseMatrix[Int]] = {
    X.map(x => (0 until p.length).map(l => {
      val s = ((p(l).t)(x(::, l).toScalaVector, ::)).toDenseMatrix
      val r = q *:* s
      val nr = r(::, *) /:/ sum(r(*, ::))
      multinomials(nr).t
    }).reduceLeft((a, b) => DenseMatrix.horzcat(a, b)))
  }

  def updateP(k: Int, loci: Int, z: Seq[DenseMatrix[Int]], X: Seq[DenseMatrix[Int]], numAlleles: Seq[Int]): Seq[DenseMatrix[Double]] = {
    (0 until loci).map(l => {
      val d1s = DenseMatrix.ones[Double](k, numAlleles(l))
      z.zip(X).foreach { case (z, x) => z(::, l).toArray.zip(x(::, l).toArray).foreach { case (ze, xe) => d1s.update(ze, xe, d1s(ze, xe) + 1.0)
      }
      }
      dirichlets(d1s)
    })
  }

  def updateQ(alpha: Double, n: Int, k: Int, z: Seq[DenseMatrix[Int]]): DenseMatrix[Double] = {
    val d1s = DenseMatrix.zeros[Double](n, k) +:+ alpha
    (0 until n).foreach(i => z.flatMap(zm => zm(i, ::).t.toArray).foreach(ki => d1s.update(i, ki, d1s(i, ki) + 1.0)))
    dirichlets(d1s)
  }

  def logProbQ(alpha: Double, q: DenseMatrix[Double], n: Int, k: Int): Double = {
    val uf: Double = 1.0e-200
    val (sum, runningTotal) = q.toArray.foldLeft((0.0, 1.0)) { case ((sum, runningTotal), v) =>
      val nr = runningTotal * (if (v > uf) v else uf)
      if (nr < uf) (sum + (alpha - 1.0) * math.log(nr), 1.0) else (sum, nr)
    }
    sum + (alpha - 1.0) * math.log(runningTotal) + (lnGamma(k * alpha) - k * lnGamma(alpha)) * n
  }

  def updateAlpha(alpha: Double, q: DenseMatrix[Double], n: Int, k: Int): Double = {
    val cAlpha = Gaussian(alpha, 0.05).draw()
    if (cAlpha<10 && cAlpha > 0 && (Uniform(0.0, 1.0).draw() < math.exp(logProbQ(cAlpha, q, n, k) - logProbQ(alpha, q, n, k)))) cAlpha else alpha
  }

  case class State(z: Seq[DenseMatrix[Int]], p: Seq[DenseMatrix[Double]], q: DenseMatrix[Double], alpha: Double) {
    def +(other: State): State = {
      State(
        this.z.zip(other.z).map { case (a, b) => a + b },
        this.p.zip(other.p).map { case (a, b) => a + b },
        this.q + other.q,
        this.alpha + other.alpha
      )
    }

    def /(den: Double): State = {
      State(
        this.z.map(a => a /:/ den.toInt),
        this.p.map(a => a /:/ den),
        this.q /:/ den,
        this.alpha / den
      )
    }
  }

  def zeroState(n: Int, k: Int, loci: Int, ploidy: Int, numAlleles: Seq[Int]): State = {
    State(
      Seq.fill(ploidy) {
        DenseMatrix.zeros[Int](n, loci)
      },
      numAlleles.map(nl => DenseMatrix.zeros[Double](k, nl)),
      DenseMatrix.zeros[Double](n, k),
      0.0f
    )
  }

  def runMCMC(X: Seq[DenseMatrix[Int]], n: Int, loci: Int, k: Int, burnIn: Int, thin: Int, iterations: Int, ploidy: Int = 2, numAlleles: Seq[Int], alpha0: Double = 1.0): State = {
    val (z0, p0, q0, _, _) = simulateData(n, k, loci, ploidy, alpha0, Some(numAlleles))
    val initial = State(z0, p0, q0, alpha0)
    val average = zeroState(n, k, loci, ploidy, numAlleles)
    val (_, expectedStage) = (1 to iterations).foldOrStop((initial, average)) { case ((current, average), i) =>
      val np = updateP(k, loci, current.z, X, numAlleles)
      val nq = updateQ(current.alpha, n, k, current.z)
      val nz = updateZ(np, nq, X)
      val nalpha = updateAlpha(current.alpha, nq, n, k)
      val updated = State(nz, np, nq, nalpha)
      val naverage = if (i >= burnIn && (i - burnIn) % thin == 0) {
        val nstate = average + updated
        println(s"$i: ${nstate.alpha/((i-burnIn)/thin)}")
        nstate
      } else average
      Some(updated, naverage)
    }
    expectedStage / ((iterations-burnIn)/thin)
  }

  def rmse2(a: DenseMatrix[Double], b: DenseMatrix[Double]): Double = {
    require(a.rows == b.rows && a.cols == b.cols, "Matrices must have the same dimensions")
    val diff = a - b
    sum(diff *:* diff) / (a.rows * a.cols)
  }

  def main(args: Array[String]): Unit = {
    val n = 200
    val k = 4
    val loci = 7
    val burnIn = 100
    val thin = 1
    val iterations = 10000
    val ploidy = 2
    val alpha = 1.0

    // Simulate data
    val (z0, pSimulated, qSimulated, x, numAlleles) = AdmixtureModel.simulateData(n, k, loci, ploidy, alpha)

    // Run MCMC
    val State(zMCMC, pMCMC, qMCMC, alphaMCMC) = AdmixtureModel.runMCMC(x, n, loci, k, burnIn, thin, iterations, ploidy, numAlleles)

    // Compute RMSE
    val pRMSE1 = rmse2(pSimulated(0), pMCMC(0))
    //val pRMSE2 = rmse2(pSimulated(1), pMCMC(1))
    val qRMSE = rmse2(qSimulated, qMCMC)

    // Print RMSE
    println(s"RMSE2 for p0: $pRMSE1")

    //println(s"RMSE2 for p1: $pRMSE2")
    println(s"RMSE2 for q: $qRMSE")
  }
}