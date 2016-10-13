import numpy as np
from mult import read_mult
import codecs
from gensim.models import LdaModel
from gensim import corpora
from gensim.corpora import Dictionary


def get_mult():
    X = read_mult('./text',3659).astype(np.float32)
    return X

def get_dummy_mult():
    X = np.random.rand(100,100)
    X[X<0.9] = 0
    return X

def read_user(f_in='./fb_train.dat',num_u=1057,num_v=281):
    fp = open(f_in)
    R = np.mat(np.zeros((num_u,num_v)))
    for i,line in enumerate(fp):
        segs = line.strip().split(' ')[1:]
        for seg in segs:
            R[i,int(seg)] = 1
    return R

def read_dummy_user():
    R = np.mat(np.random.rand(100,100))
    R[R<0.9] = 0
    R[R>0.8] = 1
    return R

def transform_data(input_file, output_file, num_u, num_v):
    R = read_user(input_file, num_u, num_v)
    R = R.T
    f = open(output_file, 'w')
    for i in range(R.shape[0]):
        index = np.where(R[i, :] > 0)[1].A1.tolist()
        f.write(str(len(index)) + ' ' + ' '.join(str(s) for s in index) + '\n')
    f.close()

def psi(x, out=None): # real signature unknown; restored from __doc__
    pass

def dirichlet_expectation(alpha):
    if (len(alpha.shape) == 1):
        result = psi(alpha) - psi(np.sum(alpha))
    else:
        result = psi(alpha) - psi(np.sum(alpha, 1))[:, np.newaxis]
    return result.astype(alpha.dtype)  # keep the same precision as input

def get_theta_beta(lda, corpus, theta_file, beta_file):
    chunksize = lda.chunksize
    import itertools
    chunker = itertools.groupby(enumerate(corpus), key=lambda (docno, doc): docno/chunksize)
    all_gamma = []
    for chunk_no, (key, group) in enumerate(chunker):
        chunk = np.asarray([np.asarray(doc) for _, doc in group])
        (gamma, sstats) = lda.inference(chunk)
        # Elogthetad = dirichlet_expectation(gamma)
        all_gamma.append(gamma)
    theta = np.vstack(all_gamma)
    beta = np.log(lda.expElogbeta)
    f_theta = open(theta_file, 'w')
    f_beta = open(beta_file, 'w')
    for i in range(theta.shape[0]):
        f_theta.write(' '.join(str(s) for s in theta[i, :].tolist()) + '\n')
    for i in range(beta.shape[0]):
        f_beta.write(' '.join(str(s) for s in beta[i, :].tolist()) + '\n')
    f_theta.close()
    f_beta.close()

def train_lda(train_file, theta_file, beta_file):
    train = []
    fp = codecs.open(train_file, 'r', encoding='utf8')
    for line in fp:
        line = line.split()
        train.append([w for w in line])
    dictionary = corpora.Dictionary(train)
    corpus = [dictionary.doc2bow(text) for text in train]
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=100)
    get_theta_beta(lda, corpus, theta_file, beta_file)


def read_ctr_result(f_v='fa_v.dat', f_u='fb_u.dat', a=100, b=100, c=100):
    fv = open(f_v)
    V = np.mat(np.zeros((a, b)))
    for i,line in enumerate(fv):
        segs = line.strip().split(' ')
        for j in range(len(segs)):
            V[i, j] = segs[j]
    fu = open(f_u)
    U = np.mat(np.zeros((c, b)))
    for i,line in enumerate(fu):
        segs = line.strip().split(' ')
        for j in range(len(segs)):
            U[i, j] = segs[j]
    return V, U


import lda

def train_lda2(train_file, theta_file, beta_file):
    train = []
    fp = codecs.open(train_file, 'r', encoding='utf8')
    for line in fp:
        line = line.split()
        train.append([w for w in line])
    dictionary = corpora.Dictionary(train)
    corpus = [dictionary.doc2bow(text) for text in train]
    model = lda.LDA(n_topics=100, random_state=0, n_iter=100)
    X = np.zeros((dictionary.num_docs, dictionary.num_pos), dtype=int)
    for i in range(len(corpus)):
        for fre_tuple in corpus[i]:
            X[i, fre_tuple[0]] = fre_tuple[1]
    model.fit(X)
    theta = model.doc_topic_
    beta = np.log(model.components_)
    f_theta = open(theta_file, 'w')
    f_beta = open(beta_file, 'w')
    for i in range(theta.shape[0]):
        f_theta.write(' '.join(str(s) for s in theta[i, :].tolist()) + '\n')
    for i in range(beta.shape[0]):
        f_beta.write(' '.join(str(s) for s in beta[i, :].tolist()) + '\n')
    f_theta.close()
    f_beta.close()