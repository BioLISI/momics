
scalaVersion := "2.13.12"
organization := "com.nouserinterface"

name:="momics"

libraryDependencies += "org.scala-lang" % "scala-library" % "2.13.11" % "compile"

libraryDependencies += "org.scalanlp" % "breeze_2.13" % "2.1.0" % "compile"

//dependencyOverrides += "dev.ludovic.netlib" % "blas" % "3.0.3"
//dependencyOverrides += "dev.ludovic.netlib" % "lapack" % "3.0.3"
//dependencyOverrides += "dev.ludovic.netlib" % "arpack" % "3.0.3"


//libraryDependencies += "org.apache.spark" % "spark-core_2.13" % "3.4.0" % "compile"
 
//libraryDependencies += "org.apache.spark" % "spark-mllib_2.13" % "3.4.0" % "compile"
 
libraryDependencies += "junit" % "junit" % "4.13.2" % "test"
 
libraryDependencies += "org.scalatest" % "scalatest_2.13" % "3.2.14" % "test"
 
libraryDependencies += "org.scalatestplus" % "junit-4-13_2.13" % "3.2.14.0" % "test"
 
libraryDependencies += "org.specs2" % "specs2-core_2.13" % "4.17.0" % "test"
 
