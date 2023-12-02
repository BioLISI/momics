""" Non-negative matrix factorization (tensorflow)"""
# Author: Eesung Kim <eesungk@gmail.com>

import numpy as np
import tensorflow as tf


class NMF:
    """Compute Non-negative Matrix Factorization (NMF)"""

    def __init__(self, max_iter=200, learning_rate=0.01, display_step=10, optimizer='mu'):

        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.display_step = display_step
        self.optimizer = optimizer

    @staticmethod
    @tf.function
    def mu(W, H, X):
        Wt = tf.transpose(W)
        H_new = H * tf.matmul(Wt, X) / tf.matmul(tf.matmul(Wt, W), H)
        H.assign(H_new)

        Ht = tf.transpose(H)
        W_new = W * tf.matmul(X, Ht) / tf.matmul(W, tf.matmul(H, Ht))
        W.assign(W_new)

    @staticmethod
    @tf.function
    def pg(W, H, X, learning_rate):
        # cost = tf.reduce_sum(tf.square(X - tf.matmul(W, H)))
        # dW, dH = tf.gradients(xs=[W, H], ys=cost)
        with tf.GradientTape() as g:
            g.watch(W)
            g.watch(H)
            g.watch(X)
            cost = tf.reduce_sum(tf.square(tf.subtract(X, tf.matmul(W, H))))
        dW, dH = g.gradient(cost, [W, H])
        # print((cost, tf.reduce_sum(learning_rate * dH), tf.reduce_sum(learning_rate * dW)))
        H.assign_sub(learning_rate * dH)
        H.assign(tf.where(tf.less(H, 0), tf.zeros_like(H), H))
        W.assign_sub(learning_rate * dW)
        W.assign(tf.where(tf.less(W, 0), tf.zeros_like(W), W))
        # H_update_ = H - learning_rate * dH
        # Ha = tf.where(tf.less(H_update_, 0), tf.zeros_like(H_update_), H_update_)
        # H.assign(Ha)
        # W_update_ = W - learning_rate * dW
        # Wa = tf.where(tf.less(W_update_, 0), tf.zeros_like(W_update_), W_update_)
        # W.assign(Wa)
        # print((cost.numpy(), W_update_.shape, H_update_.shape))

    def NMF(self, X, r_components, learning_rate, max_iter, display_step, optimizer="mu"):
        m, n = np.shape(X)
        initializer = tf.random_uniform_initializer(0, 1)
        self.W = tf.Variable(initializer(shape=[m, r_components], dtype=tf.float32), name="W", trainable=True)
        self.H = tf.Variable(initializer(shape=[r_components, n], dtype=tf.float32), name="H", trainable=True)
        self.X = tf.constant(X, dtype=tf.float32)
        for idx in range(max_iter):
            if optimizer == "mu":
                NMF.mu(self.W, self.H, self.X)
            elif optimizer == "pg":
                NMF.pg(self.W, self.H, self.X, learning_rate)
            if idx % display_step == 0:
                cost = tf.reduce_sum(tf.square(X - tf.matmul(self.W, self.H)))
                print(cost)
        return self.W, self.H

    def NMF2(self, X, r_components, learning_rate, max_iter, display_step, optimizer="mu"):
        m, n = np.shape(X)
        initializer = tf.random_uniform_initializer(0, 1)
        W = tf.Variable(initializer(shape=[m, r_components], dtype=tf.float32), name="W")
        H = tf.Variable(initializer(shape=[r_components, n], dtype=tf.float32), name="H")

        for idx in range(max_iter):
            cost = tf.reduce_sum(tf.square(X - tf.matmul(W, H)))
            if optimizer == "mu":
                Wt = tf.transpose(W)
                H_new = H * tf.matmul(Wt, X) / tf.matmul(tf.matmul(Wt, W), H)
                H.assign(H_new)

                Ht = tf.transpose(H)
                W_new = W * tf.matmul(X, Ht) / tf.matmul(W, tf.matmul(H, Ht))
                W.assign(W_new)
            elif optimizer == "pg":
                dW, dH = tf.gradients(xs=[W, H], ys=cost)
                H_update_ = H.assign(H - learning_rate * dH)
                H.assign(tf.where(tf.less(H_update_, 0), tf.zeros_like(H_update_), H_update_))
                W_update_ = W.assign(W - learning_rate * dW)
                W.assign(tf.where(tf.less(W_update_, 0), tf.zeros_like(W_update_), W_update_))

            if (idx % display_step) == 0:
                print("|Epoch:", "{:4d}".format(idx), " Cost=", "{:.3f}".format(cost))

        return W, H

    def fit_transform(self, X, r_components):
        """Transform input data to W, H matrices which are the non-negative matrices."""
        W, H = self.NMF(X=X, r_components=r_components, learning_rate=self.learning_rate,
                        max_iter=self.max_iter, display_step=self.display_step, optimizer=self.optimizer)
        return W, H

    def inverse_transform(self):
        """Transform data back to its original space."""
        return tf.matmul(self.W, self.H)

    def truncate_H(self, c=10):
        Wp = self.H
        miWc = tf.math.subtract(Wp, tf.reduce_min(Wp, axis=0))
        maWc = tf.math.divide_no_nan(miWc, tf.reduce_max(miWc, axis=0))
        self.minRep = tf.cast(tf.multiply(maWc, c), dtype=tf.int32)
        return tf.transpose(self.minRep)


def main():
    V = np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]], dtype=np.float32)
    model = NMF(max_iter=200, learning_rate=0.01, display_step=10, optimizer="pg")
    W, H = model.fit_transform(V, r_components=2)
    print(W)
    print(H)
    print(V)
    print(model.inverse_transform(W, H))


if __name__ == '__main__':
    main()
