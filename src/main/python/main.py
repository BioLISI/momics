import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import pickle

from ldamm import LDAmm
from nmf2 import NMF
from wgcna import WGCNA


def exp_data(file, pre="", max_iter=10000, r_components=7, scale=10):
    # wgcna, nmf
    ma_exp = np.asmatrix((pd.read_csv(file, header=0, index_col=0, sep="\t")).dropna(1).to_numpy(),
                         dtype=np.float32)
    wgcna = WGCNA(ma_exp)
    tom = wgcna.calc()
    nmf = NMF(max_iter=max_iter, display_step=100, optimizer="mu")
    nmf.fit_transform(tom, r_components=r_components)
    It = nmf.inverse_transform()
    corr_nmf = tf.reduce_mean(tfp.stats.correlation(tom, It, event_axis=None, sample_axis=0))
    print(corr_nmf)
    return nmf.truncate_H(scale)


def gen_data(file, pre="", max_iter=5000, learning_rate=0.00001, r_components=114, scale=10):
    # nmf
    file = "data/cdsvarsum.f.ma"
    ma_gen = np.asmatrix(pd.read_table(file, header=None, index_col=0).dropna(1).to_numpy(),
                         dtype=np.float32)

    tM = tf.add(1., ma_gen)
    pG = tf.divide(tM, tf.reduce_sum(tM, axis=0))
    lM = tf.math.log(pG)
    sE = tf.subtract(0, tf.reduce_sum(tf.multiply(lM, pG), axis=0))
    ssE = tf.slice(tf.argsort(sE, direction="ASCENDING"), [0], [500])
    selG = tf.cast(tf.gather(tf.math.log1p(ma_gen), ssE, axis=1), tf.float32).numpy()

    # ma_genT = tf.math.log1p(tf.transpose(ma_gen))
    ma_genT = tf.transpose(selG)

    nmf = NMF(max_iter=max_iter, learning_rate=learning_rate, display_step=100, optimizer="pg")
    nmf.fit_transform(ma_genT, r_components=r_components)
    It = nmf.inverse_transform()
    tf.where(tf.math.is_nan(ma_genT))
    corr_nmf = tf.reduce_mean(tfp.stats.correlation(ma_genT, It, event_axis=None, sample_axis=1))
    print(corr_nmf)
    return nmf.truncate_H(scale)


def lda(data, k=10, R=5000, keep=10):
    lda = LDAmm(k=k)
    lda.load(data)
    lda.idx()
    lda.mcmc(R, keep)

    for i in range(len(data)):
        recon0 = tf.matmul(lda.thetaE, lda.phiE[i])
        corr_lda = tf.reduce_mean(
            tfp.stats.correlation(recon0, tf.cast(data[i], tf.float32), event_axis=None, sample_axis=1))
        print(corr_lda)
    return lda


if __name__ == '__main__':
    print("ready")
    prefix = "mk0"
    gen = gen_data("data/cdsvarsum.f.ma", prefix, max_iter=5000, learning_rate=0.0001, r_components=7, scale=10)
    exp = exp_data("data/qsample0.f.tsv", prefix, max_iter=10000, r_components=7, scale=10)
    ll = lda([gen, exp], 7)
    tE = ll.thetaE.numpy()
    p0 = ll.phiE[0].numpy()
    p1 = ll.phiE[1].numpy()
