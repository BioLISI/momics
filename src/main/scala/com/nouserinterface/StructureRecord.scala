package com.nouserinterface

import breeze.linalg.DenseMatrix

case class StructureRecord(label: String, population: Int, popFlag: Boolean, genotype: Seq[Int])

object StructureRecord {
  def str2mat(data: Seq[StructureRecord], ploidy: Int = 2, numLoci: Int = 5): Seq[DenseMatrix[Int]] = {
    val raw = data.foldLeft(Array(): Array[Int])((c, e) => c ++ e.genotype.toArray)
    val ma = new DenseMatrix[Int](data.length, numLoci * 2, raw, offset = 0, majorStride = ploidy * numLoci, isTranspose = true)
    val ma2 = (1 to ploidy).map(m => ma(::, (0 until numLoci).map(_ * ploidy + m - 1)).toDenseMatrix)
    val reps = (0 until numLoci).map(l => ma2.map(_(::, l).toArray).reduceLeft(_ ++ _).distinct.zipWithIndex.toMap)
    ma2.foreach(m =>
      m.foreachKey { case (i, j) =>
        m.update(i, j, reps(j)(m(i, j)))
      }
    )
    ma2
  }
  def readStructureInput(data: Seq[Seq[String]], hasLabel: Boolean = true, hasPopData: Boolean = true, hasPopFlag: Boolean = true, numLoci: Int = 5, ploidy: Int = 2, missing: Int = -999, oneRowPerInd: Boolean = false): Seq[StructureRecord] = {
    def intersperse[A](a: List[A]*): List[A] = a.head match {
      case first :: rest => first :: intersperse[A](a.tail :+ rest: _*)
      case _ => if (a.tail.size == 1) a.last else intersperse(a.tail: _*)
    }

    val idxPopData = if (hasLabel && hasPopData) 1 else 0
    val idxPopFlag = if (hasLabel && hasPopFlag) idxPopData + 1 else 0
    val pre = 0 + (if (hasLabel) 1 else 0) + (if (hasPopData) 1 else 0) + (if (hasPopFlag) 1 else 0)

    val inds = if (oneRowPerInd) data else data.grouped(ploidy).map(indRows => indRows.head.take(pre) ++ intersperse(indRows.map(l => l.slice(pre, pre + numLoci).toList): _*)).toSeq
    inds.zipWithIndex.map { case (rd, i) =>
      StructureRecord(if (hasLabel) rd.head else i.toString,
        if (hasPopData) rd(idxPopData).toInt else 0,
        if (hasPopFlag && rd(idxPopFlag) == "0") false else true,
        rd.slice(pre, pre + ploidy * numLoci).map(_.toInt))
    }
  }

}
