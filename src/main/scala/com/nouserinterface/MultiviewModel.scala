package com.nouserinterface

import breeze.io.CSVReader
import breeze.linalg.{sum, _}
import breeze.math.Ring.ringFromField
import breeze.numerics.{exp, log}
import breeze.stats.distributions._
import breeze.stats.mean
import com.nouserinterface.Distributions._
import com.nouserinterface.Trainer.FancyIterable

import java.io.{FileInputStream, InputStreamReader}

object MultiviewModel {
  implicit val rand: RandBasis = RandBasis.systemSeed

  def main(args: Array[String]): Unit = {
    // simulated microsatellite data with 200 diploid individuals from 2 populations;
    // LABEL=1, POPDATA=1, POPFLAG=1, NUMLOCI=5, PLOIDY=2, MISSING=-999, ONEROWPERIND=0.
    val mat = CSVReader.read(new InputStreamReader(new FileInputStream(s"data/testdata1.txt")), ' ', '"', '\\')
    val test = StructureRecord.readStructureInput(mat)
    val xs = StructureRecord.str2mat(test)
    val State(_, pMCMC, qMCMC, alphaMCMC) = MultiviewModel.runMCMC(xs, 2, 20000, 10000, 1)

    println(qMCMC.toString(500))
  }

  def runMCMC(X: Seq[DenseMatrix[Int]], k: Int, iterations: Int, burnIn: Int, thin: Int): State = {
    val (n, loci) = (X.head.rows, X.head.cols)
    val numAlleles = (0 until loci).map(l => X.map(x => x(::, l).toArray).reduce(_ ++ _).max + 1)
    println(numAlleles)
    val z0 = Seq.fill(X.length) {
      numAlleles.map(j => multinomials(DenseMatrix.ones[Double](1, k) /:/ k.toDouble, n)).reduceLeft((a, b) => DenseMatrix.horzcat(a, b))
    }

    val initial = zeroState(n, k, loci, numAlleles, X.length).copy(z = z0, alpha = 1.0)
    val average = zeroState(n, k, loci, numAlleles, X.length)
    val (_, expectedStage) = (1 to iterations).foldOrStop((initial, average)) { case ((current, average), i) => {
      val np = updateP(k, current.z, X, numAlleles)
      val nq = if(i % 20 != 0) updateQ(current.alpha, n, k, current.z) else updateQMetro(current.q, np, X, current.alpha, n, k)
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

  def updateZ(p: Seq[DenseMatrix[Double]], q: DenseMatrix[Double], X: Seq[DenseMatrix[Int]]): Seq[DenseMatrix[Int]] = {
    X.map(x =>
      p.indices.map(l => {
        //P: k x numAlleles(l) X: n x l
        val s = ((p(l).t)(x(::, l).toScalaVector, ::)).toDenseMatrix
        val r = q *:* s
        val nr = r(::, *) /:/ sum(r(*, ::))
        multinomials(nr).t
      }).reduceLeft((a, b) => DenseMatrix.horzcat(a, b))
    )
  }

  def updateP(k: Int, Z: Seq[DenseMatrix[Int]], X: Seq[DenseMatrix[Int]], numAlleles: Seq[Int]): Seq[DenseMatrix[Double]] = {
    numAlleles.indices.map(l => {
      val d1s = DenseMatrix.ones[Double](k, numAlleles(l))
      Z.zip(X).foreach { case (z, x) => z(::, l).toArray.zip(x(::, l).toArray).foreach { case (ze, xe) => d1s.update(ze, xe, d1s(ze, xe) + 1.0) } }
      dirichlets(d1s)
    })
  }

  def updateQ(alpha: Double, n: Int, k: Int, Z: Seq[DenseMatrix[Int]]): DenseMatrix[Double] = {
    val d1s = DenseMatrix.zeros[Double](n, k) +:+ alpha
    (0 until n).foreach(i => Z.flatMap(z => z(i, ::).t.toArray).foreach(ki => d1s.update(i, ki, d1s(i, ki) + 1.0)))
    dirichlets(d1s)
  }

  def updateQMetro(q: DenseMatrix[Double], p: Seq[DenseMatrix[Double]], X: Seq[DenseMatrix[Int]], alpha: Double, n: Int, k: Int): DenseMatrix[Double] = {
    val testQ = dirichlets(n, DenseVector.zeros[Double](k) +:+ alpha)
    val rv = DenseVector.rand[Double](n)
    val ql = individualLikelihood(q, p, X)
    val tl = individualLikelihood(testQ, p, X)
    val diff = tl - ql
    val accept = rv <:< exp(diff)
    val newQ = q.copy
    accept.mapActivePairs { (i, _) =>
      newQ(i, ::) := testQ(i, ::)
      0
    }
    println(s"UpdateQMetro rate: ${accept.activeSize.toDouble/n}")
    newQ
  }

  def individualLikelihood(q: DenseMatrix[Double], p: Seq[DenseMatrix[Double]], X: Seq[DenseMatrix[Int]]): DenseVector[Double] = {
    val uf: Double = math.sqrt(1.0e-100)
    val nn = X.map(x => p.indices.map(l => {
      val s = ((p(l).t)(x(::, l).toScalaVector, ::)).toDenseMatrix
      val r = q *:* s
      sum(r(*, ::)).toDenseMatrix.t
    }).reduceLeft((a, b) => DenseMatrix.horzcat(a, b))).reduceLeft((a, b) => DenseMatrix.horzcat(a, b))
    nn(nn <:< uf) := uf
    val lnn = log(nn)
    sum(lnn(*, ::))
  }

  def updateAlpha(alpha: Double, q: DenseMatrix[Double], n: Int, k: Int): Double = {
    val cAlpha = Gaussian(alpha, 0.05).draw()
    if (cAlpha < 10 && cAlpha > 0) {
      val rv = Uniform(0.0, 1.0).draw()
      val thresh = math.exp(logProbQ(cAlpha, q, n.toDouble, k.toDouble) - logProbQ(alpha, q, n.toDouble, k.toDouble))
      //println(f"$thresh%10f $cAlpha%.2f $rv%.10f $alpha%.2f")
      if (rv < thresh) cAlpha else alpha
    } else alpha
  }

  def logProbQ(alpha: Double, q: DenseMatrix[Double], n: Double, k: Double): Double = {
    val uf: Double = math.sqrt(1.0e-100)
    val (sum, runningTotal) = q.toArray.foldLeft((0.0, 1.0)) {
      case ((sum, runningTotal), v) => {
        val nr = runningTotal * (if (v > uf) v else {
          uf
        })
        if (nr < uf) {
          //print("*")
          (sum + (alpha - 1.0) * math.log(nr), 1.0)
        } else
          (sum, nr)
      }
    }
    val fsum = sum + (alpha - 1.0) * math.log(runningTotal) + (lnGamma(k * alpha) - k * lnGamma(alpha)) * n
    //println(sum)
    fsum
  }

  def zeroState(n: Int, k: Int, loci: Int, numAlleles: Seq[Int], ploidy: Int): State = {
    State(
      Seq.fill(ploidy) {
        DenseMatrix.zeros[Int](n, loci)
      }
      , numAlleles.map(nl => DenseMatrix.zeros[Double](k, nl)), DenseMatrix.ones[Double](n, k) /:/ k.toDouble, 0.0f)
  }

  def rmse(a: DenseMatrix[Double], b: DenseMatrix[Double]): Double = {
    require(a.rows == b.rows && a.cols == b.cols, "Matrices must have the same dimensions")
    val diff = a - b
    math.sqrt(sum(diff *:* diff) / (a.rows * a.cols))
  }

  case class State(z: Seq[DenseMatrix[Int]], p: Seq[DenseMatrix[Double]], q: DenseMatrix[Double], alpha: Double) {
    def +(other: State): State = {
      State(this.z.zip(other.z).map { case (a, b) => a + b }, this.p.zip(other.p).map { case (a, b) => a + b }, this.q + other.q, this.alpha + other.alpha)
    }

    def /(den: Double): State = {
      State(Seq(), this.p.map(a => a /:/ den), this.q /:/ den, this.alpha / den)
    }
  }
}
