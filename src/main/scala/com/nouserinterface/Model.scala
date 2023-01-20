package com.nouserinterface

abstract class Model{
  def icluster(assays: Seq[Assay]) : Map[Gene, Int]{

  }
}

//molecular biology


case class Genome(species: Species, chromosomes: Seq[Chromosome])

case class Chromosome(name: String)

case class Gene(chromosome: Chromosome, start: Int, regions: Seq[Region])

case class Region(regionType: RegionType, start: Int, end: Int)

sealed abstract class RegionType

case object Enhancer extends RegionType

case object Promoter extends RegionType

case object NonCoding extends RegionType

case object Intron extends RegionType

case object Exon extends RegionType

case class Transcript(gene: Gene, sequence: String)

case class Protein(gene: Gene, sequence: String)

case class CellType(name: String)

case class Cell(cellType: CellType, id: String)

case class Tissue(name: String, cellTypes: Seq[CellType])

case class Organ(name: String, tissues: Seq[Tissue])

case class System(name: String, organs: Seq[Organ])

//case class Subject(id: String)
case class Species(name: String)

sealed abstract class Subject
case class Sample(id: String, sampleType: String) extends Subject
case class CellCulture(id: String, cellType: CellType) extends Subject

case class GenomicVariant(gene: Gene, start: Int, end: Int, kind: String, change: String)

sealed trait Assay

case class Sequencing(subject: Sample, targets: Seq[Gene], results: Seq[GenomicVariant]) extends Assay
case class ExpressionProfiling(subject: CellCulture, targets: Seq[Gene], results: Map[Gene, Int]) extends Assay
case class MethylationProfiling(subject: CellCulture, targets: Seq[Gene], results: Map[Gene, Int]) extends Assay

