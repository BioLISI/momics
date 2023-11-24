package com.nouserinterface

import breeze.linalg.operators.HasOps.{impl_Op_InPlace_V_V_Double_OpSet, m_m_UpdateOp_Double_OpSet}
import breeze.linalg.{DenseMatrix, DenseVector, csvwrite}
import breeze.stats.distributions.{Bernoulli, Density, Gaussian, Multinomial, Poisson, Rand, RandBasis}
import com.nouserinterface.Distributions.{dirichlets, multinomials}

import java.io.File
import scala.util.Random

object Simulator {
  implicit val rand: RandBasis = RandBasis.systemSeed

  def generateMultinomialMatrix(n: Int, k: Int, s: Double = 1.0, ns: Double=0.01): DenseMatrix[Double] = {
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

  sealed abstract class OmicSample(dist: Rand[Double]) {
    def dist(): Rand[Double] = dist
  }

  sealed abstract class OmicDefinition {
    val background: OmicSample
    val changes: Seq[OmicSample]
  }

  case object DNA extends OmicDefinition {
    final case object background extends OmicSample(Gaussian(0, 1))
    final case object amp extends OmicSample(Gaussian(2, 1))
    final case object del extends OmicSample(Gaussian(-2, 1))

    override val changes = Seq(amp, del)
  }

  case object Expression extends OmicDefinition {
    final case object background extends OmicSample( Poisson(2).map(_.toDouble))
    final case object inc extends OmicSample(Poisson(5).map(_.toDouble))
    final case object dec extends OmicSample(Poisson(1).map(_.toDouble))
    override val changes = Seq(inc, dec)
  }

  case object Methylation extends OmicDefinition {
    final case object background extends OmicSample(Gaussian(0, 1))
    final case object hyper extends OmicSample(Gaussian(2, 1))
    final case object hypo extends OmicSample(Gaussian(-2, 1))
    override val changes = Seq(hyper, hypo)
  }

  case object Mutation extends OmicDefinition {
    final case object background extends OmicSample(Bernoulli(0.05).map(if(_) 1f else 0f))
    final case object incMut extends OmicSample(Bernoulli(0.3).map(if(_) 1f else 0f))
    final case object highMut extends OmicSample(Bernoulli(0.5).map(if(_) 1f else 0f))
    override val changes = Seq(incMut, highMut)
  }

  case class OmicChangeSet(name: String, n: Int, changes: Seq[(Int, OmicSample)])

  def generate(omicDefs: Map[String, (Int, OmicDefinition)], omicChangeSets: Seq[OmicChangeSet]): Map[String, DenseMatrix[Double]] = {
    omicDefs.map { case (omicName, (numFeatures, omicDef)) =>
      val totalSamples = omicChangeSets.map(_.n).sum
      val omicMatrix = DenseMatrix.rand(totalSamples, numFeatures, omicDef.background.dist())
      var startIndex = 0
      omicChangeSets.foreach { omicChangeSet =>
        val sampleIndices = startIndex until startIndex + omicChangeSet.n
        startIndex += omicChangeSet.n
        omicChangeSet.changes.filter(c=>omicDef.changes.contains(c._2)).foreach { case (numInformative, change) =>
          val informativeIndices = Random.shuffle((0 until numFeatures).toList).take(numInformative)
          informativeIndices.foreach { i =>
            sampleIndices.foreach { idx => omicMatrix(idx, i) = change.dist().sample() }
          }
        }
      }
      omicName -> omicMatrix
    }
  }

  val nFeatures = 2000
  val nInformativeFeatures = 50
  val nSamples = 60
  val omicDefs = Map(
    "DNA" -> (nFeatures, DNA),
    "Expression" -> (nFeatures, Expression),
    "Methylation" -> (nFeatures, Methylation),
    "Mutation" -> (nFeatures, Mutation)
  )

  val omicChangeSets = Seq(
    OmicChangeSet("A", nSamples, Seq((nInformativeFeatures, DNA.amp), (nInformativeFeatures, Expression.inc))),
    OmicChangeSet("B", nSamples, Seq((nInformativeFeatures, Methylation.hyper), (nInformativeFeatures, Expression.dec))),
    OmicChangeSet("C", nSamples, Seq((nInformativeFeatures, Methylation.hypo), (nInformativeFeatures, Expression.inc), (nInformativeFeatures, Mutation.incMut))),
    OmicChangeSet("D", nSamples, Seq((nInformativeFeatures, DNA.del), (nInformativeFeatures, Expression.dec), (nInformativeFeatures, Mutation.highMut)))
  )

  def main(args: Array[String]): Unit = {
    val generatedData = generate(omicDefs, omicChangeSets)
    generatedData.foreach { case (omicType, matrix) =>
      println(s"$omicType:")
      println(matrix.mapValues(v => f"$v%.1f").toString())
      csvwrite(new File(s"data/generated/$omicType"), matrix)
    }
  }
}
