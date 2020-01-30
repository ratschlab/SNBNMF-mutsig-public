"""
Copyright (c) 2020
Author: Xinrui Lyu
Institution: Biomedical Informatics group, ETH Zurich
License: MIT License
"""

import pickle
import gc
import numpy.matlib as np_matlib

import numpy as np
import pandas as pd
import tensorflow as tf

import sys

from os.path import join, exists
from os import makedirs
from time import time

import matplotlib.pyplot as plt
import seaborn as sns


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--labels_identifier', default='histology_tier3')
parser.add_argument('--data_dir', default='data_excl_10')
parser.add_argument('--dataset', default='all', choices=['train', 'val', 'all'])
parser.add_argument('--method', default='NB_SUDL')
parser.add_argument('--ifold', type=int, default=0)

parser.add_argument('--numAtoms', type=int, default=40)
parser.add_argument('--alpha', type=float, default=1)

parser.add_argument('--convergence_tol', type=float, default=1e-6)
parser.add_argument('--convergence_epochs', type=int, default=1000)

args = parser.parse_args()

labels_identifier = args.labels_identifier
data_dir = args.data_dir
dataset = args.dataset
method = args.method
ifold = args.ifold

numAtoms = args.numAtoms
alpha = args.alpha

convergence_tol = args.convergence_tol
convergence_epochs = args.convergence_epochs


# labels_identifier = 'histology_abbreviation'
# data_dir = 'data_excl_11_cnt'
# dataset = 'all'
# method = 'sudl'
# ifold = 0

# numAtoms = 15

# convergence_tol = 1e-6
# convergence_epochs = 10000


icgc_path = '/cluster/work/grlab/projects/projects2019_mutsig-SNBNMF'
dict_path = join(icgc_path, labels_identifier+'_labels', 
                 data_dir.replace('data', 'opt_SuDL_res'), method, 
                 'numAtoms_%d_%s_a-%g.npz'%(numAtoms, dataset, alpha))
res = np.load(dict_path)


D_0 = res['dictionary']
res_path = str(res['path'])


hyper_vals = np.array([x.split('-') for x in res_path.split('/')[-1].split('_')])
hyper_vals = {hyper_vals[i][0]: float('-'.join(hyper_vals[i][1:])) for i in range(len(hyper_vals))}


epochs = int(hyper_vals['ep']) 
epochs = 1000000
alpha = hyper_vals['a']
# lambda_c = hyper_vals['lc']
# lambda_w = hyper_vals['lw']
# decay_steps = hyper_vals['ds']


X, _, _, _, _ = pickle.load(open(join(icgc_path, labels_identifier+'_labels', data_dir, 'run%d.pkl'%ifold), 'rb'))

if 'data_all' not in data_dir:
    X = np.hstack((X['train'], X['val'], X['test']))

gc.collect()
numFeatures = X.shape[0]


### Initialization
Z_0 = np.dot(np.linalg.pinv(D_0), X)
Z_0[Z_0<0] = 0
Z_0 = np.round(Z_0)


g = tf.Graph()
with g.as_default():
    # Input and output placeholders
    X_ = tf.placeholder(tf.float64, shape=[numFeatures, None], name='Data')
    
    # Hyperparameter placeholders
    alpha_ = tf.placeholder(tf.float64, shape=[], name='Dispersion_coeff')
   
    # Variables to learn
    D = tf.constant(D_0.copy(), dtype=tf.float64, name='dictionary')
    Z = tf.Variable(Z_0.copy(), dtype=tf.float64, name='exposures')
    
    Z_eps = tf.constant(np.ones(shape=Z_0.shape)*1e-16, dtype=tf.float64, name='Z_const_eps')
    Z_zeros = tf.constant(np.zeros(shape=Z_0.shape), dtype=tf.float64, name='Z_const_zeros')
    Z_1p =  tf.constant(np.ones(shape=Z_0.shape)*0.01, dtype=tf.float64, name='Z_const_1p')
    
    # Operations
    X_rec = tf.matmul(D, Z)
    
    X_div_X_rec = X_ / X_rec
    X_div_X_rec_alpha = (X_ + alpha_) / (X_rec + alpha_)
    
    
    
    # Update steps
    Zn = Z * tf.matmul(tf.transpose(D), X_div_X_rec) / tf.matmul(tf.transpose(D), X_div_X_rec_alpha)
    update_Z = tf.assign(Z, tf.math.maximum(Zn, Z_eps))
    with g.control_dependencies([update_Z]):
        # Trim small values
        trim_Z = tf.assign(Z, tf.where(tf.less(Z / tf.reduce_sum(Z, axis=0), Z_1p), Z_eps, Z))
    
    # Adjust zero-value to small values
    
    # Loss values
    tf_loss = tf.reduce_sum((X_ + alpha_)*tf.math.log(X_rec+alpha_) - X_*tf.math.log(X_rec))
    
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    feed_dict = {X_:X, alpha_:alpha}
    loss = sess.run(tf_loss, feed_dict=feed_dict)
    sys.stdout.write('Init -- Loss: %1.16e;\n'%loss)
    sys.stdout.flush()

    i = 0
    cnt_converged = 0
    t_start = time()
    with open(join(res_path,'training_info.csv'), 'w') as f:
        f.write('iter,loss,time\n')
        f.write('%d,%1.16e,%d\n'%(i, loss, time()-t_start))
        

    Z_f = Z.eval(session=sess)
    max_factor = np.argmax(Z_f, axis=0)
    conn = ((np_matlib.repmat(max_factor.reshape((-1,1)), 1, Z_f.shape[1]) 
             - np_matlib.repmat(max_factor.reshape((1,-1)), Z_f.shape[1], 1))==0).astype(int)
    while i < epochs and cnt_converged < convergence_epochs:
        
        sess.run([update_Z], feed_dict=feed_dict)
            
        i += 1
        if i%1000==0:
            # prev_loss = loss

            # loss = sess.run(tf_loss, feed_dict=feed_dict)

            # if np.abs(loss-prev_loss) < convergence_tol:
            #     cnt_converged += 1
            # else:
            #     cnt_converged = 0

            sys.stdout.write('Iter %d -- Loss: %1.16e; Time: %d sec\n'%(i, loss, time()-t_start))
            sys.stdout.flush()
            
            with open(join(res_path,'training_info.csv'), 'a') as f:
                f.write('%d,%1.16e,%d\n'%(i, loss, time()-t_start))
            
            prev_conn = conn
            Z_f = Z.eval(session=sess)
            max_factor = np.argmax(Z_f, axis=0)
            conn = ((np_matlib.repmat(max_factor.reshape((-1,1)), 1, Z_f.shape[1]) 
                     - np_matlib.repmat(max_factor.reshape((1,-1)), Z_f.shape[1], 1))==0).astype(int)

            if np.sum(np.abs(prev_conn - conn)) == 0:
                cnt_converged += 1
            else:
                cnt_converged = 0

    # sess.run(trim_Z, feed_dict=feed_dict)
    X_nan = X.astype(float)
    X_nan[X_nan==0] = float('NaN')


    Z_f = Z.eval(session=sess)

    MAE = np.nanmean(np.abs(X - np.round(np.dot(D_0, Z_f)))/X_nan)

    print('Alpha', alpha, 'NumAtoms', numAtoms)
    print('Min Sparsity', (Z_f>0).sum(axis=0).min(), 'Max Sparsity', (Z_f>0).sum(axis=0).max())
    print('End: relative MAE', MAE )


def remove_signature(X, dictionary, z):
    X_norm = X / np.sqrt(np.sum(X**2, axis=0))

    for j in range(z.shape[1]):
        idx_importance = np.argsort(z[:,j])[np.sort(z[:,j])>0][::-1]

        z_tmp = np.zeros(shape=z[:,j].shape)
        k = idx_importance[0]
        z_tmp[k] = z[k,j]
        X_rec = np.dot(dictionary, z_tmp)
        X_rec /= np.sqrt(np.sum(X_rec**2))
        prev_cos_siml = np.dot(X_norm[:,j], X_rec)

        stop_remove = False
        for k in idx_importance[1:]:
            z_tmp[k] = z[k,j]
            X_rec = np.dot(dictionary, z_tmp)
            X_rec /= np.sqrt(np.sum(X_rec**2))
            curr_cos_siml = np.dot(X_norm[:,j], X_rec)        
            if curr_cos_siml - prev_cos_siml < 0.01:
                z_tmp[k] = 0
    #             print('remove', k, prev_cos_siml, curr_cos_siml)
            else:
                pass
    #             print('keep', k, prev_cos_siml, curr_cos_siml)
            prev_cos_siml = curr_cos_siml
        z[:,j] = z_tmp
    return z

Z_f_sparse = remove_signature(X, D_0, Z_f.copy())

MAE_sparse = np.nanmean(np.abs(X - np.round(np.dot(D_0, Z_f_sparse)))/X_nan)


print('MAE', MAE, 'MAE_sparse', MAE_sparse, '; Sparsity %d'%np.median((Z_f_sparse>0).sum(axis=0)))

if 'oxog' in method:
    np.savez(dict_path, 
             dictionary=res['dictionary'], 
             z=Z_f,
             z_sparse=Z_f_sparse,
             sil_score=res['sil_score'], 
             acc=res['acc'],
             acc_oxog=res['acc_oxog'],
             mae=MAE,
             mae_sparse=MAE_sparse,
             path=res['path'],
             lst_path=res['lst_path'],
             lst_seed=['tmp_lst_seed'])
elif 'snbnmf' in method:
    np.savez(dict_path, 
             dictionary=res['dictionary'], 
             z=Z_f,
             z_sparse=Z_f_sparse,
             sil_score=res['sil_score'], 
             acc=res['acc'],
             mae=MAE,
             mae_sparse=MAE_sparse,
             path=res['path'],
             lst_path=res['lst_path'],
             lst_seed=['tmp_lst_seed'])
    
else:
    np.savez(dict_path, 
             dictionary=res['dictionary'], 
             z=Z_f,
             z_sparse=Z_f_sparse,
             sil_score=res['sil_score'], 
             mae=MAE,
             mae_sparse=MAE_sparse,
             path=res['path'],
             lst_seed=['tmp_lst_seed'])
