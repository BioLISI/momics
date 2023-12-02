import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import pickle

from tensorflow.python.ops.ragged.ragged_util import repeat


class LDAmm:
    def __init__(self, k=8):
        self.k = tf.constant(k, dtype=tf.int32)

    def sample_params(self, d=2500, m=3, v=[250, 200, 150], tr=0.5, alpha_p=[0.25, 0.3, 0.3]):
        self.d = tf.constant(d, dtype=tf.int32)
        self.m = tf.constant(m, dtype=tf.int32)
        self.v = tf.constant(v, dtype=tf.int32)
        self.theta = tfp.distributions.Dirichlet(self.tf_repeat([tr], self.k)).sample(self.d)
        alpha_p = tf.constant(alpha_p, dtype=tf.float32)

        # phi = stack_ragged([tfp.distributions.Dirichlet(tf_repeat([alpha_p[i]], v[i])).sample(k) for i in range(m)])
        self.phi = [tf.constant(tfp.distributions.Dirichlet(self.tf_repeat([alpha_p[i]], self.v[i])).sample(self.k)) for
                    i in range(self.m)]

    @staticmethod
    def stack_ragged(tensors):
        values = tf.concat([tf.concat(tf.unstack(t), 0) for t in tensors], 0)
        lens0 = tf.concat(
            [tf.tile([tf.shape(t, out_type=tf.int32)[1]], [tf.shape(t, out_type=tf.int32)[0]]).numpy().tolist() for t in
             tensors], 0)
        lens1 = tf.stack([tf.shape(t, out_type=tf.int32)[0] for t in tensors]).numpy().tolist()
        return tf.RaggedTensor.from_nested_row_lengths(values, [lens1, lens0])

    @staticmethod
    @tf.function
    def tf_repeat(y, repeat_num):
        return tf.reshape(tf.tile(tf.expand_dims(y, axis=-1), [1, repeat_num]), [-1])

    @staticmethod
    @tf.function
    def gather_repeat(values, repeats, axis=0):
        indices = repeat(tf.range(tf.shape(values)[axis]), repeats, 0)
        return tf.gather(values, indices, axis=axis)

    @staticmethod
    @tf.function
    def genv(k, wa, phia, thetak):
        z = tf.cast(tfp.distributions.Multinomial(1, probs=thetak).sample(wa), tf.int32)
        zn = tf.squeeze(tf.tensordot(z, tf.range(k), 1))
        ss = tfp.distributions.Multinomial(1, probs=tf.gather(phia, zn)).sample()
        # dss = tf.RaggedTensor.from_row_lengths(ss, wa)
        wdn = tf.cast(tf.reduce_sum(ss, 0), tf.int32)
        return wdn

    @staticmethod
    @tf.function
    def parg(arg):
        return tf.map_fn(LDAmm.genv, arg, tf.int32, parallel_iterations=1)

    def sample_documents(self, iw=[150, 125, 100]):
        self.W = []
        self.w = tf.constant(tf.stack([tf.cast(tfp.distributions.Poisson(r).sample(self.d), tf.int32) for r in iw]))
        for j in tf.range(self.m):
            self.W.append(tf.Variable(tf.zeros([self.d, self.v[j]], tf.int32)))
        for i in range(self.d):
            for j in tf.range(self.m):
                s = self.genv(self.k, self.w[j][i], self.phi[j], self.theta[i])
                self.W[j][i].assign(s)
            print(i)

    def idx(self):
        self.word_list = []
        self.doc_list = []
        self.wd = []
        for j in tf.range(self.m):
            mr = tf.range(tf.reduce_sum(self.w[j]))
            rt = tf.ragged.stack_dynamic_partitions(mr, repeat(tf.range(self.d), self.w[j], 0), self.d)
            self.doc_list.append(rt)

            Wdi = tf.concat(tf.unstack(tf.tile([tf.range(self.v[j])], [self.d, 1])), 0)
            Wdr = tf.concat(tf.unstack(self.W[j]), 0)
            Wfg = tf.RaggedTensor.from_row_lengths(repeat(Wdi, Wdr, 0), self.w[j])
            self.wd.append(Wfg.flat_values)
            self.word_list.append(tf.ragged.stack_dynamic_partitions(mr, self.wd[j], self.v[j], 0))

    def mcmc(self, R=10000, keep=10):
        RS = int(R / keep)
        burnin = int((1000 / keep) + 0.5)
        z_range = len(range(burnin, RS))

        # init
        alpha0 = tf.constant(tf.ones([self.d, self.k]))
        beta0 = [tf.constant(tf.fill([i, self.k], 0.5)) for i in self.v]

        theta0 = tf.Variable(
            tfp.distributions.Dirichlet(tfp.distributions.Uniform(0.5, 2).sample(self.k)).sample(self.d))
        phi0 = [tf.Variable(tfp.distributions.Dirichlet(tfp.distributions.Uniform(0.5, 1).sample(i)).sample(self.k)) for
                i in self.v]

        self.thetaS = tf.Variable(tf.zeros([int(R / keep + 0.5), self.d, self.k]))
        self.phiS = [tf.Variable(tf.zeros([int(R / keep + 0.5), self.k, i])) for i in self.v]
        self.wS = [tf.Variable(tf.zeros([sum(i), self.k])) for i in self.w]

        wsum = [tf.Variable(tf.zeros([self.d, self.k])) for i in self.v]
        vf = [tf.Variable(tf.zeros([i, self.k])) for i in self.v]

        vec = tf.constant(tf.reshape(tf.cast(tf.divide(1, tf.range(1, self.k + 1)), tf.float32), [1, self.k]))

        for r in range(R):
            Zj = self.sample(theta0, phi0, wsum, vf, alpha0, beta0, vec)

            if r % keep == 0:
                mkeep = int((r / keep) + 0.5)
                self.thetaS[mkeep].assign(theta0)
                for j in range(self.m):
                    self.phiS[j][mkeep].assign(phi0[j])
                    if r > burnin:
                        self.wS[j].assign_add(Zj[j])
                print(r)

        self.thetaE = tf.reduce_mean(self.thetaS[burnin:RS], 0)
        self.wE = []
        self.phiE = []
        for i in range(self.m):
            self.wE.append(self.wS[i] / tf.reduce_sum(self.wS[i], 0))
            self.phiE.append(tf.reduce_mean(self.phiS[i][burnin:RS], 0))

    @staticmethod
    @tf.function
    def sample_view(theta0, phi, wsum, vf, word_list, doc_list, wd, w, beta0, k, vec):
        # j = 0
        # phi = phi0[j]
        # wsum = wsum[j]
        # vf = vf[j]
        # word_list = word_list[j]
        # doc_list = doc_list[j]
        # wd = wd[j]
        # w = w[j]
        # beta0 = beta0[j]

        p1 = repeat(theta0, w, 0)
        p2 = tf.gather(tf.transpose(phi), wd)
        bur = tf.multiply(p1, p2)

        # r = tf.reduce_sum(br,1)/tf.reduce_sum(br)

        word_rate = tf.divide(tf.transpose(bur), tf.reduce_sum(bur, 1))
        word_cumsums = tf.transpose(tf.cumsum(word_rate, 0))
        rand = tf.transpose(tf.tile([tfp.distributions.Uniform(0, 1).sample([word_rate.shape[1]])], [k, 1]))
        Zi = tf.matmul(tf.subtract(1. + tf.cast(k, tf.float32), tf.matmul(tf.where(word_cumsums > rand, 1., 0.),
                                                                          tf.cast(tf.fill([k, 1], 1),
                                                                                  tf.float32))), vec)
        Zj = tf.where(Zi != 1., 0., Zi)

        vff = tf.reduce_sum(tf.gather(Zj, word_list), 1)
        vf.assign(tf.stack(vff))

        phi.assign(tfp.distributions.Dirichlet(tf.add(tf.transpose(vf), tf.transpose(beta0))).sample(1)[0])
        wsum.assign(tf.stack(tf.reduce_sum(tf.gather(Zj, doc_list), 1)))

        return Zj

    def sample(self, theta0, phi0, wsum, vf, alpha0, beta0, vec):
        Zj = []
        for j in tf.range(self.m):
            Zj.append(self.sample_view(theta0, phi0[j], wsum[j], vf[j], self.word_list[j], self.doc_list[j], self.wd[j],
                                       self.w[j], beta0[j], self.k, vec))
        tp = tf.add(tf.add_n(wsum), alpha0)
        theta0.assign(tfp.distributions.Dirichlet(tp).sample())
        return Zj

    @staticmethod
    def test():
        print("gen")
        lda = LDAmm(k=8)
        lda.sample_params(d=500, m=2, v=[150, 100], tr=0.5, alpha_p=[0.25, 0.3])
        print("gen")
        lda.sample_documents(iw=[600, 350])
        print("idx")
        lda.idx()
        print("sampling")
        lda.mcmc(R=2000, keep=2)

        print(tf.sqrt(tf.reduce_mean((lda.theta - lda.thetaE) ** 2)))
        print([tf.sqrt(tf.reduce_mean((lda.phi[i] - lda.phiE[i]) ** 2)) for i in range(len(lda.phiE))])
        return lda

    @staticmethod
    def test_ext(W):
        print("gen")
        lda = LDAmm(k=8)
        lda.load(W)
        print("idx")
        lda.idx()
        print("sampling")
        lda.mcmc(2000, 2)

        return lda

    def load(self, W):
        self.m = tf.constant(len(W), tf.int32)
        self.W = []
        for i in range(len(W)):
            self.W.append(tf.Variable(W[i]))
        self.d = tf.constant(self.W[0].shape[0])
        self.v = tf.constant([self.W[i].shape[1] for i in tf.range(self.m)])
        self.w = tf.constant(np.array([tf.reduce_sum(self.W[i], 1) for i in tf.range(self.m)]))

    def plots(self):
        plt.plot(self.thetaE[1])
        plt.plot(self.thetas[100])
        plt.plot(self.thetas[1000])
        plt.plot(self.thetas[2000])
        plt.plot(self.phis[0][2400])
        plt.plot(self.phis[1][4500])
        plt.plot(self.phis[2][4500])

    def saveW(self, file):
        pickle.dump(self.W, open(file, "wb"))

    def loadW(self, file):
        W = pickle.load(open(file, "rb"))
        self.load(W)
