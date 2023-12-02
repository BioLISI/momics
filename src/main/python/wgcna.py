import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
# from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from nmf2 import NMF


class WGCNA:
    def __init__(self, ma):
        self.ma = ma

    def _tom(expn):
        # ss = 0.5 + 0.5 * np.corrcoef(expn)
        ss = np.abs(np.corrcoef(expn))
        A = np.power(ss, 6)

        d = A.shape[0]
        A[np.arange(d), np.arange(d)] = 0  # Assumption
        L = A @ A  # Can be done smarter by using the symmetry
        K = A.sum(axis=1)

        A_tom = np.zeros_like(A)
        for i in range(d):
            # I don't iterate over the diagonal elements so it is not
            # surprising that they are 0 in this case, but even if the start is
            # at i instead of i+1 the results for the diagonal aren't equal to 1
            for j in range(i + 1, d):
                numerator = L[i, j] + A[i, j]
                denominator = min(K[i], K[j]) + 1 - A[i, j]
                A_tom[i, j] = numerator / denominator

        A_tom += A_tom.T
        A_tom[np.arange(d), np.arange(d)] = 1  # Set diagonal to 1 by default

        # A_tom__wgcna = np.array(pd.read_csv("https://pastebin.com/raw/HT2gBaZC",
        #                                    sep="\t", index_col=0))
        # print(np.allclose(A_tom, A_tom__wgcna))

        pA = A_tom / A_tom.sum(axis=1)
        Shannon2 = -np.sum(pA * np.log2(A_tom), axis=1)

        return (1 - A_tom), Shannon2

    def calc_tom(self):
        L = tf.add(tf.matmul(self.coexp, self.coexp), self.coexp)
        K = tf.reduce_sum(self.coexp, axis=1)

        r = tf.tile([K], [K.shape[0], 1])
        tr = tf.transpose(r)
        om = tf.where(r > tr, tr, r)
        A_tom = tf.divide(L, tf.subtract(tf.add(om, 1), self.coexp))
        A_tom = tf.linalg.set_diag(A_tom, tf.ones(A_tom.shape[0:-1], dtype=tf.float32))
        self.tom = tf.subtract(1, A_tom)

    def calc_coexp(self):
        corr = tfp.stats.correlation(self.ma, sample_axis=1, event_axis=0)
        corr2 = tf.linalg.set_diag(corr, tf.zeros(corr.shape[0:-1], dtype=tf.float32))
        corr3 = tf.divide(tf.add(corr2, 1), 2)
        corr4 = tf.pow(corr3, 12)
        self.coexp = corr4

    def calc(self):
        self.calc_coexp()
        self.calc_tom()
        return self.tom

if __name__ == '__main__':
    ma = np.asmatrix((pd.read_csv("data/qsample0.f.tsv", header=0, index_col=0, sep="\t")).dropna(1).to_numpy(),
                     dtype=np.float32)
    wgcna = WGCNA(ma)
    Ht = wgcna.nmf_rep()
    Htp = Ht.numpy()
    plt.hist(Htp.reshape([-1]))
    plt.show()
    tf.where(Ht == 0).shape
