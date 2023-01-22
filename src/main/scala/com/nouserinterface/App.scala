package com.nouserinterface

import breeze.io.{CSVReader, CSVWriter}
import breeze.linalg.DenseMatrix

import java.io.{FileInputStream, InputStreamReader}

/**
 * @author ${user.name}
 */
object App {

  def main(args: Array[String]): Unit = {
    val ctypes = Seq("aml", "breast", "colon", "gbm", "kidney", "liver", "lung", "melanoma", "ovarian", "sarcoma")
    val omics = Seq("exp", "methy", "mirna")
    val cnts = ctypes.map(c => {
      val mas = omics.map(o => csvreader(new InputStreamReader(new FileInputStream(s"data/rappoport2018/$c/$o"))))
      //val pats = mas.map(ma => ma.colNames)
      //pats.foldLeft(pats(0))(_ intersect _).size
      var hs = mas.map(ma => DenseMatrix.rand[Double](omics.length, ma.data.cols))
      var w = DenseMatrix.rand[Double](mas(0).data.cols, omics.length)

      // def muH(w:DenseMatrix[Double], h:DenseMatrix[Double], x:DenseMatrix[Double]):DenseMatrix[Double] = {
      //   h
      // }

      // def upd(w: DenseMatrix[Double], hs: Seq[DenseMatrix[Double]], xs:Seq[DenseMatrix[Double]]):(DenseMatrix[Double], Seq[DenseMatrix[Double]])={
      //   val nhs = hs.map(h=> muH(w, h, xs(0)))
      //   (w,nhs)
      // }

      // (0 to 500).foreach(i=>3)


    })
    println(cnts)
  }

  case class DataMa(rowNames: Seq[String], colNames: Seq[String], data: DenseMatrix[Double])

  def csvreader(
                 reader: java.io.Reader,
                 separator: Char = ' ',
                 quote: Char = '"',
                 escape: Char = '\\',
                 skipLines: Int = 1,
                 skipCols: Int = 1): DataMa = {
    var mat = CSVReader.read(reader, separator, quote, escape)
    mat = mat.takeWhile(line => line.nonEmpty && line.head.nonEmpty) // empty lines at the end
    reader.close()
    val ma = if (mat.isEmpty) {
      DenseMatrix.zeros[Double](0, 0)
    } else {
      DenseMatrix.tabulate(mat.length - skipLines, mat.head.length - skipCols)((i, j) => mat(i + skipLines)(j + skipCols).toDouble)
    }
    val rowNames = (0 to mat.length - 1).map(i => mat(i)(0))
    val colNames = (0 to mat.head.length - 1).map(i => mat(0)(i))
    DataMa(rowNames, colNames, ma)
  }

  def csvwriter(
                 writer: java.io.Writer,
                 mat: DenseMatrix[Double],
                 separator: Char = ',',
                 quote: Char = '\u0000',
                 escape: Char = '\\',
                 skipLines: Int = 0): Unit = {
    CSVWriter.write(writer, IndexedSeq.tabulate(mat.rows, mat.cols)(mat(_, _).toString), separator, quote, escape)
  }
}
//jNMF, Kernel, wgcna, kmeans