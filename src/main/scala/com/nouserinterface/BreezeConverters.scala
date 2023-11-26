package com.nouserinterface
/*
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.ml.linalg.Vector
import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV}

// Copied from https://stackoverflow.com/a/54281609

object BreezeConverters {
  implicit def toBreeze(dv: DenseVector): BDV[Double] =
    new BDV[Double](dv.values)

  implicit def toBreeze(sv: SparseVector): BSV[Double] =
    new BSV[Double](sv.indices, sv.values, sv.size)

  implicit def toBreeze(v: Vector): BV[Double] =
    v match {
      case dv: DenseVector => toBreeze(dv)
      case sv: SparseVector => toBreeze(sv)
    }

  implicit def fromBreeze(dv: BDV[Double]): DenseVector =
    new DenseVector(dv.toArray)

  implicit def fromBreeze(sv: BSV[Double]): SparseVector =
    new SparseVector(sv.length, sv.index, sv.data)

  implicit def fromBreeze(bv: BV[Double]): Vector =
    bv match {
      case dv: BDV[Double] => fromBreeze(dv)
      case sv: BSV[Double] => fromBreeze(sv)
    }
}
*/