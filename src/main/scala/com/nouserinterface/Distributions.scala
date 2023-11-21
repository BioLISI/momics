package com.nouserinterface

import breeze.linalg._
import breeze.stats.distributions._

object Distributions {
  def dirichlets(k: Int, alpha: DenseVector[Double])(implicit rand: RandBasis): DenseMatrix[Double] = {
    val sample = (0 until alpha.length).map(i => breeze.stats.distributions.Gamma(alpha(i), 1.0).sample(k)).reduceLeft(_ ++ _).toArray
    val theta = new DenseMatrix(alpha.length, k, sample).t.toDenseMatrix
    normalize(theta(*, ::), 1)
  }

  def dirichlet(alpha: DenseVector[Double])(implicit rand: RandBasis): DenseVector[Double] = {
    dirichlets(1, alpha)(rand)(0, ::).t
  }

  def dirichlets(alphas: DenseMatrix[Double])(implicit rand: RandBasis): DenseMatrix[Double] = {
    alphas(*, ::).map(dv => dirichlets(1, dv)).reduceLeft((a, b) => DenseMatrix.vertcat(a, b))
  }

  def multinomials(params: DenseMatrix[Double], k: Int = 1)(implicit rand: RandBasis): DenseMatrix[Int] = {
    val sample = params(*, ::).map(dv => Multinomial[DenseVector[Double], Int](dv).sample(k)).reduceLeft(_ ++ _).toArray
    new DenseMatrix(k, params.rows, sample)
  }

  def multinomial(params: DenseVector[Double])(implicit rand: RandBasis): Int = {
    Multinomial[DenseVector[Double], Int](params).draw()
  }

  //from mymath.c in Structure
  def lnGamma(z: Double): Double = {
    Gamma.logGamma(z)
    /*
    assert(z > 0f)
    val a = Seq(0.9999999999995183, 676.5203681218835, -1259.139216722289, 771.3234287757674, -176.6150291498386, 12.50734324009056, -0.1385710331296526, 9.934937113930748e-6, 1.659470187408462e-7)
    val lnsqrt2pi = 0.9189385332046727
    val (result, _) = a.tail.foldRight((a.head, z + 7.0f)) { case (at, (result, temp)) => (result + at / temp, temp - 1.0f)
    }
    math.log(result) + lnsqrt2pi - (z + 6.5) + (z - 0.5) * math.log(z + 6.5);

     */
  }
}

import scala.math
import scala.annotation.tailrec
import java.lang.Integer

// Adapted from http://www.johndcook.com/stand_alone_code.html
// All bugs are however likely my fault
class Gamma {
  //Entry points
  def gamma(x:Double): Double = {
    val v = hoboTrampoline(x,false,((y: Double) => y))
    v
  }
  def logGamma(x:Double): Double = {
    val v = hoboTrampoline(x,true,((y: Double) => y))
    v
  }

  //Since scala doesn't support optimizing co-recursive tail-calls
  //we manually make a trampoline and make it tail recursive
  @tailrec
  private def hoboTrampoline(x: Double, log: Boolean,todo: Double => Double): Double = {
    if (!log) {
      if (x <= 0.0)
      {
        val msg = "Invalid input argument "+x+". Argument must be positive."
        throw new IllegalArgumentException(msg);
      }

      // Split the function domain into three intervals:
      // (0, 0.001), [0.001, 12), and (12, infinity)

      ///////////////////////////////////////////////////////////////////////////
      // First interval: (0, 0.001)
      //
      // For small x, 1/Gamma(x) has power series x + gamma x^2  - ...
      // So in this range, 1/Gamma(x) = x + gamma x^2 with error on the order of x^3.
      // The relative error over this interval is less than 6e-7.

      val gamma: Double = 0.577215664901532860606512090; // Euler's gamma constant
      if (x < 0.001) {
        todo(1.0/(x*(1.0 + gamma*x)));
      } else if (x < 12.0) {
        ///////////////////////////////////////////////////////////////////////////
        // Second interval: [0.001, 12)
        // The algorithm directly approximates gamma over (1,2) and uses
        // reduction identities to reduce other arguments to this interval.
        val arg_was_less_than_one: Boolean = (x < 1.0);

        // Add or subtract integers as necessary to bring y into (1,2)
        // Will correct for this below
        val (n: Integer,y: Double) =  if (arg_was_less_than_one)
        {
          (0,x + 1.0)
        } else {
          val n: Integer = x.floor.toInt - 1;
          (n,x-n)
        }

        // numerator coefficients for approximation over the interval (1,2)
        val p: Array[Double] =
          Array(
            -1.71618513886549492533811E+0,
            2.47656508055759199108314E+1,
            -3.79804256470945635097577E+2,
            6.29331155312818442661052E+2,
            8.66966202790413211295064E+2,
            -3.14512729688483675254357E+4,
            -3.61444134186911729807069E+4,
            6.64561438202405440627855E+4
          );

        // denominator coefficients for approximation over the interval (1,2)
        val q: Array[Double] =
          Array(
            -3.08402300119738975254353E+1,
            3.15350626979604161529144E+2,
            -1.01515636749021914166146E+3,
            -3.10777167157231109440444E+3,
            2.25381184209801510330112E+4,
            4.75584627752788110767815E+3,
            -1.34659959864969306392456E+5,
            -1.15132259675553483497211E+5
          );

        val z: Double = y - 1;
        val num = p.foldLeft(0: Double)({(a,b) => (b+a)*z})
        val den = q.foldLeft(1: Double)({(a,b) => a*z+b})

        val result = num/den + 1.0;

        // Apply correction if argument was not initially in (1,2)
        if (arg_was_less_than_one)
        {
          // Use identity gamma(z) = gamma(z+1)/z
          // The variable "result" now holds gamma of the original y + 1
          // Thus we use y-1 to get back the orginal y.
          todo(result / (y-1.0));
        }
        else
        {
          // Use the identity gamma(z+n) = z*(z+1)* ... *(z+n-1)*gamma(z)
          todo(List.range(0,n.toInt).map(_.toDouble).foldLeft(result)((a,b) => a*(y+b)))
        }
      } else if (x <= 171.624) {
        ///////////////////////////////////////////////////////////////////////////
        // Third interval: [12, 171.624)
        hoboTrampoline(x,true,((a: Double) => todo(math.exp(a))));
      } else {
        ///////////////////////////////////////////////////////////////////////////
        // Fourth interval: [171.624, INFINITY)
        // Correct answer too large to display.
        todo(scala.Double.PositiveInfinity)
      }
    } else {
      //log implementation
      if (x <= 0.0)
      {
        val msg = "Invalid input argument "+x+". Argument must be positive."
        throw new IllegalArgumentException(msg);
      }

      if (x < 12.0) {
        hoboTrampoline(x,false,((a: Double) => todo(math.log(math.abs(a)))));
      } else {

        // Abramowitz and Stegun 6.1.41
        // Asymptotic series should be good to at least 11 or 12 figures
        // For error analysis, see Whittiker and Watson
        // A Course in Modern Analysis (1927), page 252

        val c: Array[Double] =
          Array(
            1.0/12.0,
            -1.0/360.0,
            1.0/1260.0,
            -1.0/1680.0,
            1.0/1188.0,
            -691.0/360360.0,
            1.0/156.0
          );
        val z: Double = 1.0/(x*x);
        val sum: Double = c.foldRight(-3617.0/122400.0: Double)({(a,b) => b*z+a});
        val series: Double = sum/x;

        val halfLogTwoPi: Double = 0.91893853320467274178032973640562;
        val logGamma: Double = (x - 0.5)*math.log(x) - x + halfLogTwoPi + series;
        todo(logGamma);
      }
    }
  }


}
object Gamma extends Gamma