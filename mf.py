# pylint: skip-file
import numpy as np
from data import read_user, transform_data, train_lda,train_lda2,  read_ctr_result


def cal_precision(cut,R_true, R, rec_file):
    num_u = R.shape[0]
    num_hit = 0
    fp = open(rec_file, 'w')
    for i in range(num_u):
        if i!=0 and i%100==0:
            print 'Iter '+str(i)+':'+str(float(num_hit)/i/cut)
        l_score = R[i,:].A1.tolist()
        pl = sorted(enumerate(l_score),key=lambda d:d[1],reverse=True)
        l_rec = list(zip(*pl)[0])[:cut]
        s_rec = set(l_rec)
        s_true = set(np.where(R_true[i,:]>0)[1].A1.tolist())
        cnt_hit = len(s_rec.intersection(s_true))
        num_hit += cnt_hit
        fp.write('%d:' % cnt_hit)
        fp.write(' '.join(map(str,l_rec)))
        fp.write('\n')
    fp.close()
    print 'Precision: %.3f' % (float(num_hit)/num_u/cut)


def mf_train(file, num_v = 281, num_u = 1057):
    num_iter = 10
    K = 4
    lambda_u = 100
    lambda_v = 0.1
    a = 1
    b = 0.01
    a_m_b = a-b
    theta = np.mat(np.random.rand(K,num_v))
    V = np.mat(np.random.rand(K,num_v))
    U = np.mat(np.random.rand(K,num_u))
    # R = np.mat(np.random.rand(num_u,num_v))
    # R[R<0.9992] = 0
    R = read_user(file, num_u, num_v)
    I_u = np.mat(np.eye(K)*lambda_u)
    I_v = np.mat(np.eye(K)*lambda_v)
    C  = np.mat(np.ones(R.shape))*b
    C[np.where(R>0)] = a
    print 'I: %d, J: %d, K: %d' % (num_u,num_v,K)
    for it in range(num_iter):
        print 'iter %d' % it
        V_sq = V*V.T*b
        for i in range(num_u):
            idx_a = np.where(R[i,:]>0)[1].A1
            V_cut = V[:,idx_a]
            U[:,i] = np.linalg.pinv(V_sq+V_cut*V_cut.T*a_m_b+I_u)*(V_cut*R[i,idx_a].T)
        U_sq = U*U.T*b
        for j in range(num_v):
            idx_a = np.where(R[:,j]>0)[0].A1
            U_cut = U[:,idx_a]
            V[:,j] = np.linalg.pinv(U_sq+U_cut*U_cut.T*a_m_b+I_v)*(U_cut*R[idx_a,j]+lambda_v*theta[:,j])
        if it%1==0:
            E = U.T*V-R
            E = np.sum(np.multiply(C,np.multiply(E,E)))
            print 'E: %.3f' % E

    Rc = U.T*V
    cut = 10
    rec_file = 'rec.txt'
    cal_precision(cut, R, Rc, rec_file)
    # for i in range(R.shape[0]):
    #     I = np.where(R[i, :] > 0)[1]
    #     num = np.sum(I[0, :], axis=0)[0][0]

    return V,U

def mf_test(file, V, U, num_v = 281, num_u = 1057):
    R = read_user(file, num_u, num_v)
    R_c = R
    cut = 10
    rec_file = 'rec.txt'
    if U.shape[0] == V.shape[1] or U.shape[0] == V.shape[0]:
        R_c = U.T * V
    else:
        R_c = U * V.T
    cal_precision(cut, R, R_c, rec_file)


if __name__ == '__main__':
    train_file = 'fb_train.dat'
    test_file = 'fb_test.dat'
    num_v = 426
    num_u = 1956

    # test mf
    # V, U = mf_train(train_file, num_v, num_u)
    # mf_test(test_file, V, U, num_v, num_u)

    # transform data from event-user matrix to user_event
    # transform_data('fb_train.dat', 'fb_train_items.dat', num_u, num_v)

    # train_file = 'text'
    # theta_file = 'fb_theta.txt'
    # beta_file = 'fb_beta.txt'
    # train_lda2(train_file, theta_file, beta_file)

    # test ctr
    num_t = 100
    V_file = 'final-V.dat'
    U_file = 'final-U.dat'
    V, U = read_ctr_result(V_file, U_file, num_v, num_t, num_u)
    mf_test(test_file, V, U, num_v, num_u)
