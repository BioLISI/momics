package com.nouserinterface

import breeze.linalg.{DenseMatrix,  csvwrite}
import breeze.stats.distributions.{Bernoulli, Gaussian, Poisson, Rand, RandBasis}

import java.io.File
import scala.util.Random

object Simulator {
  implicit val rand: RandBasis = RandBasis.systemSeed

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
      csvwrite(new File(s"data/generated2/$omicType"), matrix)
    }
  }
}
