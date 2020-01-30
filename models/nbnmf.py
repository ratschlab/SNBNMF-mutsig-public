"""
Copyright (c) 2020
Author: Xinrui Lyu
Institution: Biomedical Informatics group, ETH Zurich
License: MIT License
"""

import pickle
import gc

import numpy as np
from numpy.matlib import repmat
import pandas as pd
import tensorflow as tf

import sys
from sklearn.decomposition import NMF as sk_NMF


from os.path import join, exists
from os import makedirs
from time import time


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--label_type', default='project_code_no_country')
parser.add_argument('--data_dir', default='data_all')
parser.add_argument('--dataset', default='all', choices=['train', 'val', 'all'])
parser.add_argument('--ifold', type=int, default=0)

parser.add_argument('--numAtoms', type=int, default=35)
parser.add_argument('--alpha', type=float, default=1e+8)
parser.add_argument('--epochs', type=int, default=1000000)
parser.add_argument('--seed', type=int, default=2019)

parser.add_argument('--convergence_tol', type=float, default=1e-6)
parser.add_argument('--convergence_epochs', type=int, default=10)

args = parser.parse_args()

label_type = args.label_type
data_dir = args.data_dir
dataset = args.dataset
ifold = args.ifold

numAtoms = args.numAtoms
alpha = args.alpha
epochs = args.epochs
seed = args.seed

convergence_tol = args.convergence_tol
convergence_epochs = args.convergence_epochs

method = 'nbnmf'
icgc_path = '/cluster/work/grlab/projects/projects2019_mutsig-SNBNMF'
icgc_path = join(icgc_path, label_type+'_labels')


# create result path from the hyperparameter values
filename = 'ep-%d_a-%g'%(epochs, alpha)
res_path = join(icgc_path, 
                data_dir.replace('data','res'),
                method, 
                'numAtoms_%d'%numAtoms, 
                '%s%d'%(dataset, ifold),
                filename,
                'seed-%s'%seed)
if not exists(res_path):
    makedirs(res_path)

with open(join(icgc_path, data_dir, 'run%d.pkl'%ifold), 'rb') as f:
    X, Y, _, G, _ = pickle.load(f)

# if the data is splited into train, vallidation and test set, combine them 
# together as the whole training set for dictionary learning.
if 'data_all' in data_dir:
    Y = Y.T
else:
    for key, val in Y.items():
        Y[key] = val.T
    X = np.hstack((X['train'], X['val'], X['test']))
    Y = np.hstack((Y['train'], Y['val'], Y['test']))
Y[Y==0] = -1
Y = Y.astype(float)
gc.collect()

numFeatures = X.shape[0]

### Initialization
np.random.seed(seed)
D_0 = np.random.rand(X.shape[0], numAtoms)
D_0 /= np.sum(D_0, axis=0).reshape((1,-1))
Z_0 = np.dot(np.linalg.pinv(D_0), X)
Z_0[Z_0<0] = 0
Z_0 = np.round(Z_0)


g = tf.Graph()
with g.as_default():
    X_ = tf.placeholder(tf.float64, shape=[numFeatures, None], name='Data')
    alpha_ = tf.placeholder(tf.float64, shape=[], name='Dispersion_coeff')
    
    D = tf.Variable(D_0.copy(), dtype=tf.float64)
    Z = tf.Variable(Z_0.copy(), dtype=tf.float64)
    
    
    D_eps = tf.constant(np.ones(shape=D_0.shape)*1e-16, dtype=tf.float64, name='D_const_eps')
    Z_eps = tf.constant(np.ones(shape=Z_0.shape)*1e-16, dtype=tf.float64, name='Z_const_eps')
    
    X_rec = tf.matmul(D, Z)
    
    X_div_X_rec = X_ / X_rec
    X_div_X_rec_alpha = (X_ + alpha_) / (X_rec + alpha_)

    Dn = D * tf.matmul(X_div_X_rec, Z, transpose_b=True) / tf.matmul(X_div_X_rec_alpha, Z, transpose_b=True)
    update_D = tf.assign(D, tf.math.maximum(Dn, D_eps))

    with g.control_dependencies([update_D]):
        Zn = Z * tf.matmul(D, X_div_X_rec, transpose_a=True) / tf.matmul(D, X_div_X_rec_alpha, transpose_a=True)
        update_Z = tf.assign(Z, tf.math.maximum(Zn, Z_eps))
    
    D_l1norm = tf.reduce_sum(D, axis=0)
    scale_Z = tf.assign(Z, Z * tf.reshape(D_l1norm, [-1,1]))
    normalize_D = tf.assign(D, D / D_l1norm)

    tf_loss = tf.reduce_sum((X_ + alpha_)*tf.math.log(X_rec + alpha_)-X_*tf.math.log(X_rec)) 

    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=1)

    feed_dict = {X_:X, alpha_:alpha}
    loss = sess.run(tf_loss, feed_dict=feed_dict)
    sys.stdout.write('Init -- Loss: %16.16g;\n'%loss)
    sys.stdout.flush()
    i = 0
    cnt_converged = 0
    t_start = time()
    with open(join(res_path,'training_info.csv'), 'w') as f:
        f.write('iter,loss,time\n')
        f.write('%d,%g,%g\n'%(i, loss, time()-t_start))


    Z_f = Z.eval(session=sess)
    max_factor = np.argmax(Z_f, axis=0)
    conn = ((repmat(max_factor.reshape((-1,1)), 1, Z_f.shape[1]) 
             - repmat(max_factor.reshape((1,-1)), Z_f.shape[1], 1))==0).astype(int)

    while i < epochs and cnt_converged < convergence_epochs:
        sess.run([update_D, update_Z], feed_dict=feed_dict)
            
        i += 1

        if i%1000==0:
            prev_loss = loss
            loss = sess.run(tf_loss, feed_dict=feed_dict)

            # if np.abs(loss-prev_loss) < convergence_tol:
            #     cnt_converged += 1
            sys.stdout.write('Iter %d -- Loss: %16.16g; Time: %16.16g sec\n'%(i, loss, time()-t_start))
            sys.stdout.flush()
            
            with open(join(res_path,'training_info.csv'), 'a') as f:
                f.write('%d,%g,%g\n'%(i, loss, time()-t_start))
            saver.save(sess, join(res_path, 'model'), global_step=i)

            prev_conn = conn
            Z_f = Z.eval(session=sess)
            max_factor = np.argmax(Z_f, axis=0)
            conn = ((repmat(max_factor.reshape((-1,1)), 1, Z_f.shape[1]) 
                     - repmat(max_factor.reshape((1,-1)), Z_f.shape[1], 1))==0).astype(int)

            if np.sum(np.abs(prev_conn - conn)) == 0:
                cnt_converged += 1
            else:
                cnt_converged = 0

    sess.run(scale_Z, feed_dict=feed_dict)
    sess.run(normalize_D, feed_dict=feed_dict)

    D_f = D.eval(session=sess)
    Z_f = Z.eval(session=sess)
    Z_f[Z_f<1e-15] = 0

    X_nan = X.astype(float)
    X_nan[X_nan==0] = float('NaN')
    
    MAE = np.nanmean(np.abs(X - np.round(np.dot(D_f, Z_f)))/X_nan)
    print('Alpha', alpha, 'NumAtoms', numAtoms)
    print('Min Sparsity', (Z_f>0).sum(axis=0).min(), 'Max Sparsity', (Z_f>0).sum(axis=0).max())
    print('End: relative MAE', MAE)

    saver.save(sess, join(res_path, 'model'), global_step=i)
    np.savez(join(res_path, 'results.npz'), dictionary=D_f, z=Z_f, mae=MAE)
    print('All results saved in -->', res_path)
    sess.close()


