"""
Copyright (c) 2020
Author: Xinrui Lyu
Institution: Biomedical Informatics group, ETH Zurich
License: MIT License
"""

import pickle

import numpy as np
import numpy.matlib as matlib
import pandas as pd
import tensorflow as tf

import sys

from os.path import join, exists
from os import makedirs, listdir
from time import time


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--label_type', default='project_code_no_country')
parser.add_argument('--data_dir', default='data_all')
parser.add_argument('--dataset', default='all', choices=['train', 'val', 'all'])
parser.add_argument('--ifold', type=int, default=0)

parser.add_argument('--numAtoms', type=int, default=35)
parser.add_argument('--alpha', type=float, default=1e+08)
parser.add_argument('--epochs', type=int, default=1000000)
parser.add_argument('--seed', type=int, default=0)

parser.add_argument('--convergence_tol', type=float, default=1e-6)
parser.add_argument('--convergence_epochs', type=int, default=10)

parser.add_argument('--lambda_c', type=float, default=10)
parser.add_argument('--lambda_w', type=float, default=1e-4)
parser.add_argument('--lambda_g', type=float, default=10)
parser.add_argument('--learning_rate_lbl', type=float, default=1e-3)
parser.add_argument('--learning_rate_constr', type=float, default=1e-3)
parser.add_argument('--decay_steps', type=float, default=1000)
parser.add_argument('--decay_rate', type=float, default=0.96)

parser.add_argument('--norm_exp', action='store_true')
parser.add_argument('--weight_sample', action='store_true')
parser.add_argument('--constr_sig1', action='store_true')
parser.add_argument('--reg4constr', action='store_true')
parser.add_argument('--trimming', action='store_true')
parser.add_argument('--update_constr_bias', action='store_true')

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


lambda_c = args.lambda_c
lambda_w = args.lambda_w
lambda_g = args.lambda_g
learning_rate_lbl = args.learning_rate_lbl
learning_rate_constr = args.learning_rate_constr
decay_steps = args.decay_steps
decay_rate = args.decay_rate

norm_exp = args.norm_exp
weight_sample = args.weight_sample
constr_sig1 = args.constr_sig1
reg4constr = args.reg4constr
trimming = args.trimming
update_constr_bias = args.update_constr_bias


icgc_path = '/cluster/work/grlab/projects/projects2019_mutsig-SNBNMF'
icgc_path = join(icgc_path, label_type+'_labels')

method = 'snbnmf'
if norm_exp:
    method += '_norm'
if trimming:
    method += '_trim'
if constr_sig1:
    method += '_constr'
if weight_sample:
    method += '_weighted'


# create result path from the hyperparameter values
filename = ('ep-%d'%epochs+'_'+
            'a-%g'%alpha+'_'+
            'lc-%g'%lambda_c+'_'+
            'lw-%g'%lambda_w+'_'+
            'lrw-%g'%learning_rate_lbl+'_'+
            'dr-%g'%decay_rate+'_'+
            'ds-%d'%decay_steps)
if constr_sig1:
    filename = (filename+'_'+
                'lg-%g'%lambda_g+'_'+
                'lrg-%g'%learning_rate_constr)
res_path = join(icgc_path, 
                data_dir.replace('data', 'res'),
                method, 
                'numAtoms_%d'%numAtoms, 
                '%s%d'%(dataset, ifold), 
                filename, 
                'seed-%s'%seed)
if not exists(res_path):
    makedirs(res_path)



with open(join(icgc_path, data_dir, 'run%d.pkl'%ifold), 'rb') as f:
    X, Y, _, G, _ = pickle.load(f)

if 'data_all' in data_dir:
    Y = Y.T
    G = G.reshape((1,-1))
else:
    for key, val in Y.items():
        Y[key] = val.T
    X = np.hstack((X['train'], X['val'], X['test']))
    Y = np.hstack((Y['train'], Y['val'], Y['test']))
    G = np.concatenate((G['train'], G['val'], G['test'])).reshape((1,-1))
Y[Y==0] = -1
Y = Y.astype(float)

if constr_sig1 and 'apobec' not in data_dir and 'oxog' not in data_dir:
    raise Exception('this file does not contain constraint info.')

# if the confounder constraint is regression not classification problem
# and the confounder value is the apobec score, compute the log scale of the 
# regression
if 'apobec' in data_dir and reg4constr:
    G = np.log2(G)

# if the confounder constraint is a classification problem, for oxog is binary
# for apobec is multi-class classification
if constr_sig1 and not reg4constr:
    if 'apobec' in data_dir:
        percentile_values = [np.percentile(G, i) for i in np.arange(10, 100, 10)]
        G_label = -np.ones((10, G.shape[1]))
        G_lalel[0,G<=percentile_values[0]] = 1
        for i in range(len(percentile_values)-1):
            G_label[i+1,np.logical_and(G>percentile_values[i], 
                                       G<=percentile_values[i+1])] = 1
        G_label[-1,G>percentile_values[-1]] = 1
        G = G_label
    elif 'oxog' in data_dir:
        G[G<=80] = -1
        G[G>80] = 1
  

numFeatures = X.shape[0]
numLabels = Y.shape[0]
numSamples = X.shape[1]
print('# Features', numFeatures)
print('# Labels', numLabels)
print('# Samples', numSamples)

### Compute sample and constraint weights
# sample weights
SW = np.ones(shape=Y.shape)
for i in range(len(Y)):
    for j in np.where(Y[i]>0)[0]:
        SW[i,j] = (Y[i]<0).sum()/(Y[i]>0).sum()

# constraint weights
CW = np.ones(shape=G.shape)
for i in range(len(G)):
    for j in np.where(G[i]>0)[0]:
        CW[i,j] = (G[i]<0).sum()/(G[i]>0).sum()

### Initialization
np.random.seed(seed)

# NMF intialization
D_0 = np.random.rand(numFeatures, numAtoms)
D_0 /= np.sum(D_0, axis=0).reshape((1,-1))
Z_0 = np.dot(np.linalg.pinv(D_0), X)
Z_0[Z_0<0] = 0

# Label classifier initialization
if constr_sig1:
    W_0 = np.random.randn(numAtoms-1, numLabels)
else:
    W_0 = np.random.randn(numAtoms, numLabels)

b_0 = np.zeros((numLabels, 1))

### constr classifier initialization
mu_0 = np.random.rand(1, G.shape[0])
nu_0 = np.zeros((G.shape[0], 1))

g = tf.Graph()
with g.as_default():
    # Inputs
    X_ = tf.placeholder(tf.float64, shape=[numFeatures, None], name='data')
    Y_ = tf.placeholder(tf.float64, shape=[numLabels, None], name='label')

    # Variables
    D = tf.Variable(D_0.copy(), dtype=tf.float64, name='dictionary')
    Z = tf.Variable(Z_0.copy(), dtype=tf.float64, name='exposures')
    W = tf.Variable(W_0.copy(), dtype=tf.float64, name='svm_classifier_weight')
    b = tf.Variable(b_0.copy(), dtype=tf.float64, name='svm_classifier_bias')
    global_step_w = tf.Variable(0., dtype=tf.float64, trainable=False, 
                                name='global_step_w')

    # Hyperparamters
    alpha_ = tf.placeholder(tf.float64, shape=[], name='dispersion_coeff')
    lambda_c_ = tf.placeholder(tf.float64, shape=[], name='clf_regularization')
    lambda_w_ = tf.placeholder(tf.float64, shape=[], 
                               name='weight_regularization')

    learning_rate_lbl_ = tf.placeholder(tf.float64, shape=[], 
                                        name='learning_rate_lbl')
    decay_rate_ = tf.placeholder(tf.float64, shape=[], name='decay_rate')
    decay_steps_ = tf.placeholder(tf.float64, shape=[], name='decay_step')
    learning_rate_w = ( learning_rate_lbl_ *  
                        tf.pow( decay_rate_, global_step_w / decay_steps_) )

    # Constants
    D_eps = tf.constant(np.ones(shape=D_0.shape)*1e-16, dtype=tf.float64, 
                        name='D_const_eps')
    Z_eps = tf.constant(np.ones(shape=Z_0.shape)*1e-16, dtype=tf.float64, 
                        name='Z_const_eps')
    Z_zeros = tf.constant(np.zeros(shape=Z_0.shape), dtype=tf.float64, 
                          name='Z_const_zeros')

    if constr_sig1:
        Z_eps1 = tf.constant(np.ones(shape=Z_0[:1].shape)*1e-16, 
                             dtype=tf.float64, name='Z_const_eps1')
        Z_epsnot1 = tf.constant(np.ones(shape=Z_0[1:].shape)*1e-16, 
                                dtype=tf.float64, name='Z_const_epsnot1')
        Z_zeros1 = tf.constant(np.zeros(shape=Z_0[:1].shape), 
                               dtype=tf.float64, name='Z_const_zeros1')
        Z_zerosnot1 = tf.constant(np.zeros(shape=Z_0[1:].shape), 
                                  dtype=tf.float64, name='Z_const_zerosnot1')
        Z_1pnot1 = tf.constant(np.ones(shape=Z_0[1:].shape)*1e-2, 
                               dtype=tf.float64, name='Z_const_1pnot1')
        Z_5pnot1 = tf.constant(np.ones(shape=Z_0[1:].shape)*5e-2, 
                               dtype=tf.float64, name='Z_const_5pnot1')
    else:
        Z_1p = tf.constant(np.ones(shape=Z_0.shape)*1e-2, 
                           dtype=tf.float64, name='Z_const_1p')
        Z_5p = tf.constant(np.ones(shape=Z_0.shape)*5e-2, 
                           dtype=tf.float64, name='Z_const_5p')

    
    Y_zeros = tf.constant(np.zeros(shape=Y.shape), 
                          dtype=tf.float64, name='Y_const_zeros')
    Y_ones = tf.constant(np.ones(shape=Y.shape), 
                         dtype=tf.float64, name='Y_const_ones')

    if weight_sample:        
        SW_ = tf.placeholder(tf.float64, shape=[numLabels, None], 
                             name='sample_weight')

    if constr_sig1:
        G_ = tf.placeholder(tf.float64, 
                            shape=[1 if reg4constr else len(G), None], 
                            name='constr_score')        

        # constr variables
        # constraint regressor weights
        mu = tf.Variable(mu_0.copy(), dtype=tf.float64, name='constr_regressor')
        # constraint regressor bias
        if update_constr_bias:
            nu = tf.Variable(nu_0.copy(), dtype=tf.float64, name='constr_bias') 
        else:
            nu = tf.constant(nu_0.copy(), dtype=tf.float64, name='constr_bias')
        global_step_c = tf.Variable(0., dtype=tf.float64, trainable=False, 
                                    name='global_step_c')
        
        # constr constants
        G_zeros = tf.constant(np.zeros(shape=G.shape), 
                              dtype=tf.float64, name='G_const_zeros')
        G_ones = tf.constant(np.ones(shape=G.shape), 
                             dtype=tf.float64, name='G_const_ones')
        
        # constr hyperparamters
        lambda_g_ = tf.placeholder(tf.float64, shape=[], 
                                   name='constr_regularization')
        learning_rate_c_ = tf.placeholder(tf.float64, shape=[], 
                                          name='learning_rate_constr')
        learning_rate_c = ( learning_rate_c_ * 
                            tf.pow( decay_rate_, global_step_c / decay_steps_) )

        if weight_sample:
            CW_ = tf.placeholder(tf.float64, shape=[len(G), None], 
                                 name='constr_score_weight')

    X_rec = tf.matmul(D, Z)
    X_div_X_rec = X_ / X_rec
    X_div_X_rec_alpha = (X_ + alpha_) / (X_rec + alpha_)

    if norm_exp:
        X_sum = tf.reduce_sum(X_, axis=0)
        Z_norm = Z / X_sum
    else:
        Z_norm = Z / 1
    # Z_norm = tf.where(tf.less(Z_norm_tmp, Z_eps), Z_zeros, Z_norm_tmp)

    if constr_sig1:
        Znot1_norm = tf.slice(Z_norm, [1,0], [numAtoms-1,numSamples])
        Z1_norm = tf.slice(Z_norm, [0,0], [1,numSamples])


        WTZb = tf.matmul(W, Znot1_norm, transpose_a=True) + b
    else:
        WTZb = tf.matmul(W, Z_norm, transpose_a=True) + b
        
    YWTZb = Y_ * WTZb
    is_sv = tf.less(YWTZb, Y_ones) # true if sample is a support vector for each label
    N_s = tf.reduce_sum(tf.cast(is_sv, tf.float64), axis=1) # counts of support vectors for each label
    Y_s = tf.where(is_sv, SW_*Y_ if weight_sample else Y_, Y_zeros) # set label of non-sv to zeros
    
    
    Dn = D * ( tf.matmul(X_div_X_rec, Z, transpose_b=True) / 
               tf.matmul(X_div_X_rec_alpha, Z, transpose_b=True) )
    update_D = tf.assign(D, tf.math.maximum(Dn, D_eps))
    
    aux_update_z = lambda_c_ * tf.matmul(W, Y_s)
    if norm_exp:
        aux_update_z /= X_sum

    if constr_sig1:
        Znot1 = tf.slice(Z, [1,0], [numAtoms-1,numSamples])
        if reg4constr:

            muZ1nu = mu * Z1_norm + nu

            D1 = tf.slice(D, [0,0], [numFeatures,1])
            if norm_exp:
                aa = lambda_g_ * ((mu/X_sum) ** 2)
                bb = ( tf.matmul(D1, X_div_X_rec_alpha, transpose_a=True) +
                       lambda_g_*mu*(nu-G_) / X_sum )
            else:
                aa = lambda_g_ * (mu ** 2)
                bb = ( tf.matmul(D1, X_div_X_rec_alpha, transpose_a=True) + 
                       lambda_g_*mu*(nu-G_) )

            cc = tf.matmul(D1, X_div_X_rec, transpose_a=True) * tf.slice(Z, [0,0], [1,numSamples])
            Zn_1 = ( tf.sqrt(bb**2+4*aa*cc) - bb ) / (2 * aa)

            Dnot1 = tf.slice(D, [0,1], [numFeatures,numAtoms-1])
            Zn_not1 = Znot1 * ( tf.matmul(Dnot1, X_div_X_rec, transpose_a=True) / 
                                (tf.matmul(Dnot1, X_div_X_rec_alpha, transpose_a=True) - aux_update_z) )

            update_Z1 = tf.scatter_update(Z, tf.range(0,1), tf.math.maximum(Zn_1, Z_eps1))
            update_Znot1 = tf.scatter_update(Z, tf.range(1,numAtoms), tf.math.maximum(Zn_not1, Z_epsnot1))
            update_Z = tf.group([update_Z1, update_Znot1])

        else:
            Z1 = tf.slice(Z_norm, [0,0], [1,numSamples])
            GmuZ1nu = G_ * (tf.matmul(mu, Z1_norm, transpose_a=True) + nu)
            is_sv_constr = tf.less(GmuZ1nu, G_ones)
            NG_s = tf.reduce_sum(tf.cast(is_sv_constr, tf.float64)) 
            G_s = tf.where(is_sv_constr, CW_*G_ if weight_sample else G_, G_zeros)
            # G_s = tf.where(is_sv_constr, G_, G_zeros)

            aux_update_z1 = lambda_g_ * tf.matmul(mu, G_s)
            aux_update_zall = tf.concat([aux_update_z1, aux_update_z], axis=0)

            Zn = Z * ( tf.matmul(D, X_div_X_rec, transpose_a=True) / 
                       (tf.matmul(D, X_div_X_rec_alpha, transpose_a=True) - aux_update_zall) )
            update_Z = tf.assign(Z, tf.math.maximum(Zn, Z_eps))
            

    else:

        Zn = Z * ( tf.matmul(D, X_div_X_rec, transpose_a=True) / 
                   (tf.matmul(D, X_div_X_rec_alpha, transpose_a=True) - aux_update_z) )
        update_Z = tf.assign(Z, tf.math.maximum(Zn, Z_eps))



    l1norm_D = tf.reduce_sum(D, axis=0)

    scale_Z = tf.assign(Z, Z * tf.reshape(l1norm_D, [-1,1]))

    normalize_D = tf.assign(D, D / l1norm_D)

    if constr_sig1:
        Znot1_norm_self = Znot1 / tf.reduce_sum(Znot1, axis=0)
        trim_Z = tf.scatter_update(Z, tf.range(1,numAtoms), tf.where(tf.less(Znot1_norm_self, Z_1pnot1), Z_zerosnot1, Znot1))
    else:
        Z_norm_self = Z / tf.reduce_sum(Z, axis=0)
        trim_Z = tf.assign(Z, tf.where(tf.less(Z_norm_self, Z_1p), Z_zeros, Z))
    
    if constr_sig1:
        gradW = - tf.matmul(Znot1_norm, Y_s, transpose_b=True) + lambda_w_ * W  
    else:
        gradW = - tf.matmul(Z_norm, Y_s, transpose_b=True) + lambda_w_ * W  

    gradb = - tf.reduce_sum(Y_s, axis=1, keepdims=True)

    update_W = tf.assign(W, W - learning_rate_w * gradW )
    update_b = tf.assign(b, b - learning_rate_w * gradb )

    if constr_sig1:
        
        if reg4constr:
            gradmu = tf.squeeze(tf.matmul(Z1_norm, muZ1nu - G_, transpose_b=True))
            if update_constr_bias:
                update_nu = tf.assign(nu, tf.reduce_mean(G_-muZ1nu, axis=1, keep_dims=True))
        else:            
            gradmu = - tf.squeeze(tf.matmul(G_s, Z1_norm, transpose_b=True))
            if update_constr_bias:
                update_nu = tf.assign(nu, nu + learning_rate_c * tf.reduce_sum(G_s, axis=1, keepdims=True))

        update_mu = tf.assign(mu, tf.maximum(mu - learning_rate_c * gradmu, 1e-16) )
        update_clf = tf.group([update_W, update_b, update_mu])
    else:
        update_clf = tf.group([update_W, update_b])


    update_global_step_w = tf.assign(global_step_w, global_step_w + decay_steps)
    if constr_sig1:
        update_global_step_c = tf.assign(global_step_c, global_step_c + decay_steps)
        
    tf_loss_kl = tf.reduce_sum((X_ + alpha_)*tf.math.log(X_rec+alpha_) - X_*tf.math.log(X_rec))
    tf_loss_clf = tf.reduce_sum((SW_ if weight_sample else 1) * tf.math.maximum(Y_ones-YWTZb, Y_zeros))    
    tf_loss_reg = tf.reduce_sum(W**2)

    tf_loss = tf_loss_kl + lambda_c_ * (tf_loss_clf + lambda_w_/2 * tf_loss_reg)
    
    W_norm = tf.transpose(tf.norm(W, ord=2, axis=0, keepdims=True))
    pred = tf.math.argmax( WTZb / W_norm, axis=0)
    gt = tf.math.argmax(Y_, axis=0)
    tf_acc = tf.reduce_mean(tf.cast(tf.equal(pred, gt), tf.float64))

    if constr_sig1:
        if reg4constr:
            tf_loss_constr = tf.reduce_sum((muZ1nu-G_)**2)
            tf_loss += lambda_g_ / 2 * tf_loss_constr

            tf_res_constr = tf.reduce_mean((G_-muZ1nu)**2)
        else:
            tf_loss_constr = tf.reduce_sum((CW_ if weight_sample else 1)*tf.math.maximum(G_ones-GmuZ1nu, G_zeros))
            # tf_loss_constr = tf.reduce_sum(tf.math.maximum(G_ones-GmuZ1nu, G_zeros))
            tf_loss += lambda_g_ * tf_loss_constr
            
#             Mu_norm = tf.transpose(tf.norm(mu, ord=2, axis=0, keepdims=True))
            tf_res_constr = tf.reduce_mean(tf.cast(tf.equal(tf.math.sign(GmuZ1nu), G_), tf.float64), axis=1)

    sess = tf.Session()
    feed_dict = {X_:X, Y_:Y, 
                 alpha_:alpha, lambda_c_:lambda_c, lambda_w_:lambda_w,
                 learning_rate_lbl_: learning_rate_lbl,
                 decay_rate_:decay_rate, decay_steps_:decay_steps}

    if weight_sample:
        feed_dict.update({SW_: SW})

    if constr_sig1:
        feed_dict.update({G_:G, lambda_g_:lambda_g, learning_rate_c_: learning_rate_constr})
        if weight_sample:
            feed_dict.update({CW_: CW})


            
    if exists(join(res_path,'training_info.csv')) and not exists(join(res_path,'results.npz')):        
        print(res_path)
        i = int([f for f in listdir(res_path) if 'model' in f and 'meta' in f][0].split('.')[0].split('-')[1])

        saver = tf.train.Saver(max_to_keep=1)
        saver.restore(sess, join(res_path, 'model-%g'%i))

        if constr_sig1:
            loss, loss_kl, loss_clf, loss_constr, acc = sess.run([tf_loss, 
                                                                  tf_loss_kl, 
                                                                  tf_loss_clf, 
                                                                  tf_loss_constr, 
                                                                  tf_acc], 
                                                                 feed_dict=feed_dict)
        else:
            loss, loss_kl, loss_clf, acc = sess.run([tf_loss, 
                                                     tf_loss_kl, 
                                                     tf_loss_clf, 
                                                     tf_acc], 
                                                    feed_dict=feed_dict)
            loss_constr = float('NaN')

        t_offset = pd.read_csv(join(res_path,'training_info.csv')).iloc[-1].time
        t_start = time()
    else:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=1)

        if constr_sig1:
            loss, loss_kl, loss_clf, loss_constr, acc = sess.run([tf_loss, 
                                                                  tf_loss_kl, 
                                                                  tf_loss_clf,
                                                                  tf_loss_constr, 
                                                                  tf_acc], 
                                                                 feed_dict=feed_dict)
        else:
            loss, loss_kl, loss_clf, acc = sess.run([tf_loss, 
                                                     tf_loss_kl, 
                                                     tf_loss_clf, 
                                                     tf_acc], 
                                                    feed_dict=feed_dict)
            loss_constr = float('NaN')

        sys.stdout.write('Init - Total: %1.16e; KL: %1.9e; Clf: %1.9e; constr: %1.9e; Acc: %3.3f;\n'%(loss, loss_kl, loss_clf, loss_constr, acc))
        sys.stdout.flush()

        i = 0
        t_offset = 0
        t_start = time()
        with open(join(res_path,'training_info.csv'), 'w') as f:
            f.write('iter,loss,loss_kl,loss_clf,loss_constr,acc,time\n')
            f.write('%d,%1.16e,%1.16e,%1.9e,%1.9e,%3.3f,%d\n'%(i, loss, loss_kl, loss_clf, loss_constr, acc, time()-t_start))
                

    cnt_converged = 0
    nan_loss = False

    Z_f = Z.eval(session=sess)
    if constr_sig1:
        max_factor = np.argmax(Z_f[1:], axis=0)
        conn = ((matlib.repmat(max_factor.reshape((-1,1)), 1, Z_f.shape[1]) 
                 - matlib.repmat(max_factor.reshape((1,-1)), Z_f.shape[1], 1))==0).astype(int)
    else:
        max_factor = np.argmax(Z_f, axis=0)
        conn = ((matlib.repmat(max_factor.reshape((-1,1)), 1, Z_f.shape[1]) 
                 - matlib.repmat(max_factor.reshape((1,-1)), Z_f.shape[1], 1))==0).astype(int)

    while i < epochs and cnt_converged < convergence_epochs and not nan_loss:
        
        sess.run(update_D, feed_dict=feed_dict)
        sess.run(update_Z, feed_dict=feed_dict)
        sess.run(scale_Z)

        if trimming:
            sess.run([normalize_D, trim_Z])
            sess.run(update_clf, feed_dict=feed_dict)
        else:
            sess.run([normalize_D, update_clf], feed_dict=feed_dict)

        if update_constr_bias:
            sess.run(update_nu, feed_dict=feed_dict)        

        if i%decay_steps==0:
            sess.run(update_global_step_w)
            if constr_sig1:
                sess.run(update_global_step_c)

        i += 1

        if i%1000 == 0:
            prev_loss = loss
            prev_loss_kl = loss_kl
            prev_loss_clf = loss_clf
            prev_loss_constr = loss_constr

            if constr_sig1:
                loss, loss_kl, loss_clf, loss_constr, acc = sess.run([tf_loss, tf_loss_kl, tf_loss_clf, tf_loss_constr, tf_acc], 
                                                                   feed_dict=feed_dict)
            else:
                loss, loss_kl, loss_clf, acc = sess.run([tf_loss, tf_loss_kl, tf_loss_clf, tf_acc], 
                                                        feed_dict=feed_dict)
                loss_constr = float('NaN')
                
            if np.isnan(loss):
                print('NAN Loss!!!')
                nan_loss = True
                exit(0)
                continue

            # if np.abs(loss-prev_loss) < convergence_tol:
            #     cnt_converged += 1
            # else:
            #     cnt_converged = 0

            sys.stdout.write('%d - Total: %1.16e; KL: %1.9e; Clf: %1.9e; constr: %1.9e; Acc: %3.3f; Time: %d s\n'%(i, loss, loss_kl, loss_clf, loss_constr , acc, t_offset+time()-t_start))
            sys.stdout.flush()

            with open(join(res_path,'training_info.csv'), 'a') as f:
                f.write('%d,%1.16e,%1.16e,%1.9e,%1.9e,%3.3f,%d\n'%(i, loss, loss_kl, loss_clf, loss_constr, acc, t_offset+time()-t_start))

            saver.save(sess, join(res_path, 'model'), global_step=i)

            prev_conn = conn
            Z_f = Z.eval(session=sess)
            if constr_sig1:
                max_factor = np.argmax(Z_f[1:], axis=0)
                conn = ((matlib.repmat(max_factor.reshape((-1,1)), 1, Z_f.shape[1]) 
                         - matlib.repmat(max_factor.reshape((1,-1)), Z_f.shape[1], 1))==0).astype(int)
            else:
                max_factor = np.argmax(Z_f, axis=0)
                conn = ((matlib.repmat(max_factor.reshape((-1,1)), 1, Z_f.shape[1]) 
                         - matlib.repmat(max_factor.reshape((1,-1)), Z_f.shape[1], 1))==0).astype(int)

            if np.sum(np.abs(prev_conn - conn)) == 0:
                cnt_converged += 1
            else:
                cnt_converged = 0

    saver.save(sess, join(res_path, 'model'), global_step=i)
    sess.run(scale_Z)
    if trimming:
        sess.run([normalize_D, trim_Z])
    else:
        sess.run([normalize_D])

    D_f = D.eval(session=sess)
    Z_f = Z.eval(session=sess)
    W_f = W.eval(session=sess)
    b_f = b.eval(session=sess)
    
    X_nan = X.astype(float)
    X_nan[X_nan==0] = float('NaN')
    
    MAE = np.nanmean(np.abs(X - np.round(np.dot(D_f, Z_f)))/X_nan)
    print('Alpha', alpha, 'NumAtoms', numAtoms)
    print('Min Sparsity', (Z_f>0).sum(axis=0).min(), 'Max Sparsity', (Z_f>0).sum(axis=0).max())
    print('MAE', MAE)

    if constr_sig1:
        mu_f = mu.eval(session=sess)
        nu_f = nu.eval(session=sess)
        acc, res_constr = sess.run([tf_acc, tf_res_constr], feed_dict=feed_dict)
        if reg4constr:
            print('ACC', '%2.2f'%acc, 'MSE constr', '%6.6f'%res_constr)
        else:
            print('ACC', '%2.2f'%acc, 'ACC constr', res_constr)
    else:
        acc = sess.run(tf_acc, feed_dict=feed_dict)
        print('ACC', '%2.2f'%acc)
    
    if constr_sig1:
        np.savez(join(res_path, 'results.npz'),
                 mae=MAE,
                 dictionary=D_f, 
                 z=Z_f, 
                 w=W_f, 
                 b=b_f, 
                 acc=acc, 
                 mu=mu_f, 
                 nu=nu_f, 
                 acc_constr=res_constr)
    else:
        np.savez(join(res_path, 'results.npz'), 
                 mae = MAE,
                 dictionary=D_f, 
                 z=Z_f, 
                 w=W_f, 
                 b=b_f, 
                 acc=acc)
        
    print('All results saved in ->', res_path)
    sess.close()

