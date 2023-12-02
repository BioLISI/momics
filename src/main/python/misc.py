def shannon(A_tom):
    pA = A_tom / A_tom.sum(axis=1)
    return -np.sum(pA * np.log2(A_tom), axis=1)


def load_data1():
    exp = pd.read_table("data/qsample0.f.tsv", header=0, index_col=0).dropna(1)
    gen = pd.read_table("data/cdsvarsum.f.ma", header=None, index_col=0).dropna(1)
    # gen = gen.loc[(gen!=0).any(1), (gen!=0).any(0)]

    Etom, Eshan = tom(exp.to_numpy())
    # ada = pd.DataFrame(Eshan)
    # ada.hist()
    # plt.show()

    ie = Eshan.argsort()[::-1][:1000]
    sc = Eshan[ie]
    cexp = Etom[:, ie]

    fff = np.transpose(cexp) - np.median(cexp, axis=1)
    fff[fff < 0] = 0
    # pd.DataFrame(np.reshape(fff, [-1])).hist()
    # plt.show()

    gg = np.transpose(np.asarray(fff[np.sum(fff, axis=1).argsort()[::-1][:200], :] * 100, int))

    gar = gen.to_numpy()

    far = gar + 1
    pA = far / far.sum(axis=0)
    fds = -np.sum(pA * np.log2(far), axis=1)
    sg = shannon(gar)

    train_gen = gen.sample(frac=0.9, random_state=200)
    test_gen = gen.drop(train_gen.index)

    exp2 = pd.DataFrame(gg, index=exp.index)
    train_exp = exp2.sample(frac=0.9, random_state=200)
    test_exp = exp2.drop(train_exp.index)

    trg, teg, tre, tee = train_gen.to_numpy(), test_gen.to_numpy(), train_exp.to_numpy(), test_exp.to_numpy()

    trg = np.asarray(np.log2(1 + trg), dtype=int)
    teg = np.asarray(np.log2(1 + teg), dtype=int)
    tre = np.asarray(tre, dtype=int)
    tee = np.asarray(tee, dtype=int)

    # pickle.dump((trg, teg, tre, tee), open("data/guach.p", "wb"))

    return trg, teg, tre, tee


def load_saved(file="data/guach.p"):
    return pickle.load(open(file, "rb"))


def pandas_entropy(column, base=None):
    vc = pd.Series(column).value_counts(normalize=True, sort=False)
    base = np.e if base is None else base
    return -(vc * np.log(vc) / np.log(base)).sum()


def load_data():
    exp = (pd.read_csv("data/iexp.tsv", header=None, index_col=0, sep="\t"))
    gen = pd.read_csv("data/igen.tsv", header=None, index_col=0, sep="\t")
    # gen = gen.loc[(gen!=0).any(1), (gen!=0).any(0)]

    gent = gen.apply(pandas_entropy)
    eent = exp.apply(pandas_entropy)

    genq = gen.reindex(gent.sort_values(ascending=False).index, axis=1).iloc[:, 0:400]
    expq = exp.reindex(eent.sort_values(ascending=False).index, axis=1).iloc[:, 0:400]

    train_gen = genq.sample(frac=0.9, random_state=200)
    test_gen = genq.drop(train_gen.index)

    train_exp = expq.sample(frac=0.9, random_state=200)
    test_exp = expq.drop(train_exp.index)

    trg, teg, tre, tee = train_gen.to_numpy(), test_gen.to_numpy(), train_exp.to_numpy(), test_exp.to_numpy()

    trg = np.asarray(trg, dtype=int)
    teg = np.asarray(teg, dtype=int)
    tre = np.asarray(tre, dtype=int)
    tee = np.asarray(tee, dtype=int)

    return trg, teg, tre, tee


def test_lda():
    lda = LDAmm(k=8)
    lda.loadW("data/testWk8m2d500.p")
    lda.idx()
    lda.mcmc(2000, 2)


def qa():
    #trg, teg, tre, tee = load_data()
    trg = tf.cast(trg, tf.float32)
    teg = tf.cast(teg, tf.float32)

    # test
    repGr = tf.matmul(tf.cast(teg, tf.float32), tf.transpose(lda.phiE[0]))
    repG = tf.transpose(tf.math.divide_no_nan(tf.transpose(repGr), tf.reduce_sum(repGr, 1)))

    repEr = tf.matmul(tf.cast(tee, tf.float32), tf.transpose(lda.phiE[1]))
    repE = tf.transpose(tf.math.divide_no_nan(tf.transpose(repEr), tf.reduce_sum(repEr, 1)))

    tf.reduce_mean(tfp.stats.correlation(repG, repE))
    tf.reduce_mean(tfp.stats.correlation(repGr, repEr))

    tf.sqrt(tf.reduce_mean(tf.pow((repG - repE), 2)))
    tf.reduce_mean(tf.abs(tfp.stats.correlation(repG, repE, event_axis=None, sample_axis=0)))

    td = squared_dist(repG)
    tf.reduce_mean(td)

    dist = tf.norm(repE - repG, axis=1)
    tf.reduce_mean(dist)

    # train
    trepGr = tf.matmul(tf.cast(trg, tf.float32), tf.transpose(lda.phiE[0]))
    trepEr = tf.matmul(tf.cast(tre, tf.float32), tf.transpose(lda.phiE[1]))
    trepG = tf.transpose(tf.math.divide_no_nan(tf.transpose(trepGr), tf.reduce_sum(trepGr, 1)))
    trepE = tf.transpose(tf.math.divide_no_nan(tf.transpose(trepEr), tf.reduce_sum(trepEr, 1)))

    tf.reduce_mean(trepE)
    tf.reduce_mean(tfp.stats.stddev(trepE, 1))
    tf.sqrt(tf.reduce_mean(tf.pow((lda.thetaE - trepE), 2)))

    tf.reduce_mean(trepG)
    tf.reduce_mean(tfp.stats.stddev(trepG, 1))
    tf.sqrt(tf.reduce_mean(tf.pow((lda.thetaE - trepG), 2)))

    # tf.sqrt(tf.reduce_mean(tf.pow((trepE - trepG), 2)))

    tf.sqrt(tf.norm(trepE - trepG))
    tf.sqrt(tf.norm(lda.thetaE - trepG))
    tf.sqrt(tf.norm(lda.thetaE - trepE))

    dist = tf.sqrt(tf.norm(trepE - trepG, axis=1))
    tf.reduce_mean(dist)


@tf.function
def squared_dist(A):
    expanded_a = tf.expand_dims(A, 1)
    expanded_b = tf.expand_dims(A, 0)
    distances = tf.reduce_sum(tf.math.squared_difference(expanded_a, expanded_b), 2)
    return distances


def squared_dist2(A, B):
    assert A.shape.as_list() == B.shape.as_list()

    row_norms_A = tf.reduce_sum(tf.square(A), axis=1)
    row_norms_A = tf.reshape(row_norms_A, [-1, 1])  # Column vector.

    row_norms_B = tf.reduce_sum(tf.square(B), axis=1)
    row_norms_B = tf.reshape(row_norms_B, [1, -1])  # Row vector.

    return tf.sqrt(row_norms_A - 2 * tf.matmul(A, tf.transpose(B)) + row_norms_B + 1e-6)


def squared_dist3(A):
    # A = tf.constant([[1, 1], [2, 2], [3, 3]])
    r = tf.reduce_sum(A * A, 1)

    # turn r into column vector
    r = tf.reshape(r, [-1, 1])
    D = tf.sqrt(r - 2 * tf.matmul(A, tf.transpose(A)) + tf.transpose(r) + 1e-6)
    return D


def cart(a, b):
    N = a.shape[0]
    M = b.shape[0]
    T = b.shape[1]
    tile_a = tf.tile(tf.expand_dims(a, 1), [1, M, 1])
    tile_b = tf.tile(tf.expand_dims(b, 0), [N, 1, 1])

    cartesian_product = tf.concat([tile_a, tile_b], axis=2)
    cartesian = tf.reshape(cartesian_product, [N * M, 2, T])
    return cartesian


def lda_mo():
    trg, teg, tre, tee = load_saved("data/guach.p")
    lda = LDAmm(k=8)
    lda.load([trg,tre])
    lda.idx()
    lda.mcmc(5000, 10)
    return lda

# lda = pickle.load(open("data/ldagtk8R1000kp10.p", "rb"))

#tM = tf.add(1., ma_gen)
#pG = tf.divide(tM, tf.reduce_sum(tM, axis=0))
#lM = tf.math.log(pG)
#sE = tf.subtract(0, tf.reduce_sum(tf.multiply(lM, pG), axis=0))
#ssE = tf.slice(tf.argsort(sE, direction="ASCENDING"), [0], [500]).numpy()
#selG = tf.cast(tf.gather(tf.math.log1p(ma_gen), ssE, axis=1), tf.int32)

#selG = tf.cast(tf.math.log1p(ma_gen), tf.float32)