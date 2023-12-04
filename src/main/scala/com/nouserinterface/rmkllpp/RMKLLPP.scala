package com.nouserinterface.rmkllpp

import breeze.linalg._
import breeze.numerics._
import org.ojalgo.matrix.MatrixR064
import org.ojalgo.optimisation.ExpressionsBasedModel
import org.ojalgo.optimisation.Variable
import org.ojalgo.optimisation.Optimisation.Result

import scala.collection.mutable.ArrayBuffer
import scala.jdk.CollectionConverters.SeqHasAsJava
import scala.util.control.Breaks.{break, breakable}
import scala.util.{Failure, Success, Try}


object RMKLLPP {

  def optim_A(
               kernelArray: Array[DenseMatrix[Double]],
               beta: DenseVector[Double],
               newDim: Int,
               L1: DenseMatrix[Double],
               L2: DenseMatrix[Double]
             ): (DenseMatrix[Double], Double) = {

    val thresholdZero = 0.0
    val ndim = kernelArray.head.rows
    var ensK = DenseMatrix.zeros[Double](ndim, ndim)

    // Generate ensembleKernel
    for (i <- 0 until beta.length) {
      ensK += kernelArray(i) * beta(i)
    }

    // Calculate S_W and S_D using L1 and L2
    val S_W_tmp = ensK * L1 * ensK
    val S_W = upperTriangular(S_W_tmp) + strictlyUpperTriangular(S_W_tmp).t
    val S_D_tmp = ensK * L2 * ensK
    var S_D = upperTriangular(S_D_tmp) + strictlyUpperTriangular(S_D_tmp).t

    // Check if S_D has full rank
    var pseudoCount = 1e-13
    while (rank(S_D) < ndim) {
      S_D += DenseMatrix.eye[Double](ndim) * pseudoCount
      pseudoCount *= 10
    }

    // Make sure S_W is positive semidefinite
    val evalsW = eigSym(S_W).eigenvalues
    if (min(evalsW) < 0) {
      S_W += DenseMatrix.eye[Double](ndim) * (-min(evalsW))
    }

    // Make sure S_D is positive (semi)definite
    val evalsD = eigSym(S_D).eigenvalues
    if (min(evalsD) <= 0) {
      S_D += DenseMatrix.eye[Double](ndim) * (-min(evalsD) + 1e-12)
    }

    // Generalized eigenvalue decomposition
    val (evals, evecs) = generalizedEig(S_W, S_D) match {
      case Success((eigenvalues, eigenvectors)) =>
        (eigenvalues, eigenvectors)
      case Failure(exception) =>
        println(s"Error during GEVD: ${exception.getMessage}")
        (DenseVector.zeros[Double](0), DenseMatrix.zeros[Double](0,0))
    }
    val sortedIndices = argsort(evals)
    val ind = ArrayBuffer[Int]()
    for (i <- sortedIndices) {
      if (evals(i) > thresholdZero && ind.length < newDim) {
        ind += i
      }
    }

    var A = evecs(::, ind.toSeq).toDenseMatrix

    // Check if A contains complex numbers
    /*
    if (!all(isreal(A))) {
      return (DenseMatrix.zeros[Double](0, 0), Double.NaN)
    }
    */

    // Normalize length of projection vectors in A
    for (i <- 0 until A.cols) {
      A(::, i) := A(::, i) / norm(A(::, i))
    }

    val ojVal = trace(A.t * S_W * A)

    (A, ojVal)
  }

  def generalizedEig(A: DenseMatrix[Double], B: DenseMatrix[Double]): Try[(DenseVector[Double], DenseMatrix[Double])] = Try {
    require(A.rows == A.cols, "Matrix A must be square.")
    require(B.rows == B.cols, "Matrix B must be square.")
    require(A.rows == B.rows, "Matrices A and B must have the same dimensions.")

    // Compute the Cholesky decomposition of B
    val cholB = cholesky(B)
    // Invert B using the Cholesky factorization
    val invCholB = inv(cholB)
    // Solve the standard eigenvalue problem B⁻¹A
    val transformedA = invCholB.t * A * invCholB
    val eigResult = eigSym(transformedA)

    // The eigenvalues remain the same. However, we need to transform the eigenvectors back.
    // original eigenvectors = invCholB * transformed eigenvectors
    val originalEigenvectors = invCholB * eigResult.eigenvectors

    (eigResult.eigenvalues, originalEigenvectors)
  }


  def optimBeta(kernelArray: Array[DenseMatrix[Double]], AAt: DenseMatrix[Double], L1: DenseMatrix[Double], L2: DenseMatrix[Double]): (DenseVector[Double], Double, Int) = {
    val k = kernelArray.length
    val S_W = DenseMatrix.zeros[Double](k, k)
    val S_B = DenseMatrix.zeros[Double](k, k)

    // Calculate S_W and S_B matrices
    for (i <- 0 until k; j <- i until k) {
      val tmp = kernelArray(j) * AAt * kernelArray(i).t
      S_W(i, j) = trace(L1 * tmp)
      S_W(j, i) = S_W(i, j)
      S_B(i, j) = trace(L2 * tmp)
      S_B(j, i) = S_B(i, j)
    }

    // Ensure S_W is positive definite
    if (min(eigSym(S_W).eigenvalues) < 0.0) {
      S_W := S_W - min(eigSym(S_W).eigenvalues) * DenseMatrix.eye[Double](k)
    }

    // Optimization Model
    val model = new ExpressionsBasedModel()

    // Variables & Expression for Optimization
    val betaVars = (1 to k).map(i=>model.addVariable(s"beta$i").lower(0).upper(1))

    val objective = model.addExpression("Objective")

    var solutionStatus = 1
    var objectiveValue: Double = Double.NaN
    var beta: DenseVector[Double] = DenseVector.zeros[Double](k)

    breakable {
      for (loop <- 0 until 10) { // Adjust number of loops as necessary
        objective.weight(1)
        betaVars.zipWithIndex.foreach {
          case (variable, index) => objective.set(variable, S_W(index, index))
        }

        val sumBetaConstraint = model.addExpression().level(1)
        betaVars.foreach(sumBetaConstraint.set(_, 1))

        val doubleArray: Array[Array[Double]] = new Array[Array[Double]](S_B.rows)
        for (i <- 0 until S_B.rows) {
          doubleArray(i) = S_B(i, ::).t.toArray
        }
        val quadCoeff = MatrixR064.FACTORY.rows(doubleArray:_*)
        model.addExpression("quads").setQuadraticFactors(betaVars.asJava, quadCoeff)
        val result = model.maximise()

        if (result.getState.isOptimal) {
          beta = DenseVector(betaVars.map(_.getValue.doubleValue()))
          objectiveValue = beta.t * S_W * beta
          solutionStatus = 0
          break()
        } else {
          // Optionally adjust the optimization problem for next iteration here, if required.
        }
      }
    }
    (beta, objectiveValue, solutionStatus)
  }


  def mklDr(
             kernelArray: Array[DenseMatrix[Double]], drmethod: String, kNN: Int, newDim: Int,
             init: String, maxIt: Int
           ): (DenseMatrix[Double], DenseVector[Double], ArrayBuffer[DenseMatrix[Double]],
    ArrayBuffer[DenseVector[Double]], ArrayBuffer[Double]) = {

    val ndim = kernelArray.head.rows
    val k = kernelArray.length

    var W, L1, L2, AAt = DenseMatrix.zeros[Double](ndim, ndim)
    if (drmethod == "LPP_nn") {
      val averageKernel = kernelArray.reduce(_ + _)
      val RKHSDistances = DenseMatrix.tabulate(ndim, ndim)((i, j) =>
        if (j >= i) averageKernel(i, i) - 2 * averageKernel(i, j) + averageKernel(j, j) else 0.0
      )

      RKHSDistances(*, ::).foreach(v => v := v(argtopk(v, kNN + 1)))
      W = RKHSDistances.map(v => if (v > 0.0) 1.0 else 0.0)
      W -= diag(diag(W))
      val D = diag(sum(W(*, ::)))
      L1 = 2.0 * (D - W)
      L2 = D
      AAt = DenseMatrix.eye[Double](ndim)
    }

    val As = ArrayBuffer.fill[DenseMatrix[Double]](maxIt)(DenseMatrix.zeros[Double](ndim, newDim))
    val bes = ArrayBuffer.fill[DenseVector[Double]](maxIt)(DenseVector.zeros[Double](k))
    val ojs = ArrayBuffer.fill[Double](maxIt)(0.0)
    var beta = DenseVector.zeros[Double](k)
    var A = DenseMatrix.zeros[Double](ndim, newDim)
    var oldOj = 0.0

    for (i <- 0 until maxIt) {
      init match {
        case "A" =>
          (optimBeta(kernelArray, AAt, L1, L2), optim_A(kernelArray, beta, newDim, L1, L2)) match {
            case ((b, ojB, status), (a, ojA)) if status == 0 && !a.data.exists(_.isNaN) && !b.data.exists(_.isNaN) =>
              As(i) = a;
              bes(i) = b;
              ojs(i) = ojA;
              AAt = a * a.t
              if (i > 0 && abs(oldOj - ojA) < 1e-4) return (As(i), bes(i), As.take(i + 1), bes.take(i + 1), ojs.take(i + 1))
              oldOj = ojA
            case _ => return (As(i - 1), bes(i - 1), As.take(i - 1), bes.take(i - 1), ojs.take(i - 1))
          }
        case "B" =>
          beta = DenseVector.ones[Double](k) / k.toDouble
          (optim_A(kernelArray, beta, newDim, L1, L2), optimBeta(kernelArray, AAt, L1, L2)) match {
            case ((a, ojA), (b, ojB, status)) if status == 0 && !a.data.exists(_.isNaN) && !b.data.exists(_.isNaN) =>
              As(i) = a;
              bes(i) = b;
              ojs(i) = ojB;
              AAt = a * a.t
              if (i > 0 && abs(oldOj - ojB) < 1e-4) return (As(i), bes(i), As.take(i + 1), bes.take(i + 1), ojs.take(i + 1))
              oldOj = ojB
            case _ => return (As(i - 1), bes(i - 1), As.take(i - 1), bes.take(i - 1), ojs.take(i - 1))
          }
      }
    }

    val optPos = ojs.zipWithIndex.min._2
    (As(optPos), bes(optPos), As, bes, ojs)
  }


}
