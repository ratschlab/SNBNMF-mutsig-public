"""
Copyright (c) 2020
Author: Xinrui Lyu
Institution: Biomedical Informatics group, ETH Zurich
License: MIT License
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import gc
import pickle

from os.path import join, exists
from os import listdir, makedirs
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt
import seaborn as sns

import shutil


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--labels_identifier', default='project_code_no_country')
parser.add_argument('--data_dir', default='data_all')
parser.add_argument('--method', default='snbnmf')
parser.add_argument('--numAtoms', type=int, default=35)
parser.add_argument('--train_set', default='all')
parser.add_argument('--ifold', type=int, default=0)
parser.add_argument('--alpha', type=float, default=1e+08)
args = parser.parse_args()

labels_identifier = args.labels_identifier
data_dir = args.data_dir
method = args.method
numAtoms = args.numAtoms
train_set = args.train_set
ifold = args.ifold
alpha = args.alpha

icgc_path = '/cluster/work/grlab/projects/projects2019_mutsig-SNBNMF'
max_sil_score = 0
min_mae = float('Inf')
max_acc = 0
opt_lst_acc = None
opt_lst_mae = None
best_numAtoms = None
best_res_path = None
X, Y, _, G, _ = pickle.load(open(join(icgc_path, labels_identifier+'_labels', data_dir, 
                                      'run%d.pkl'%ifold), 'rb'))

if 'trim' in data_dir:
    X = X[:,G>0]
    Y = Y[G>0]
    G = G[G>0]
if 'data_all' not in data_dir:
    X = np.hstack((X['train'], X['val'], X['test']))
    Y = np.vstack((Y['train'], Y['val'], Y['test']))
    G = np.concatenate((G['train'], G['val'], G['test']))
    
G[G<80] = 0
G[G>=80] = 1
Y = np.argmax(Y, axis=1)
res_path = join(icgc_path, labels_identifier+'_labels', data_dir.replace('data', 'res'), method, 
                'numAtoms_%d'%numAtoms, '%s%d'%(train_set, ifold))
out_path = join(icgc_path, labels_identifier+'_labels', data_dir.replace('data', 'opt_SuDL_res'), method)
if not exists(out_path):
    makedirs(out_path)
    

def compute_mae(X, D, Z):
    X_nan = X.astype(float)
    X_nan[X==-0] = float('NaN')
    X_rec = np.dot(D, Z)
    err = (np.abs(X_rec - X_nan) / X_nan).reshape((-1,))
    err = err[~np.isnan(err)]
    err90 = err[np.logical_and(err<=np.percentile(err, 95), err>=np.percentile(err, 5))]
    return np.mean(err90)

prev_G = (G==0).sum() / len(G)


lst_seed = set()
lst_foi = []
for ff in listdir(res_path):
    if 'a-%g_'%alpha not in ff:
        continue
    # if 'ep-1000000' not in ff:
    #     continue
    if float(ff.split('_')[2].split('-')[1]) < 1:
        continue
    if len(lst_seed)==0:
        lst_seed = set([int(f.split('-')[-1]) for f in listdir(join(res_path, ff)) if 'seed' in f])
    else:
        lst_seed = lst_seed & set([int(f.split('-')[-1]) for f in listdir(join(res_path, ff)) if 'seed' in f])
    lst_foi.append(ff)

lst_seed = np.sort(list(lst_seed))
# lst_seed = [0,1]

lst_av_acc = []
lst_av_gacc = []
lst_av_mae = []
lst_f = []
for ff in lst_foi:
    dictionary = []
    constr_sig = []
    acc = []
    gacc = []
    mae = []
    for seed in lst_seed:
        f = 'seed-%d'%seed
        try:
            results = np.load(join(res_path, ff, f, 'results.npz'))
        except:
            acc.append(float('NaN'))
            gacc.append(float('NaN'))
            mae.append(float('NaN'))
            continue
        if 'constr' in method:
            dictionary.append(results['dictionary'][:,1:])
            constr_sig.append(results['dictionary'][:,[0]])
        else:
            dictionary.append(results['dictionary'])
            
        acc.append(float(results['acc']))
        try:
            gacc.append(float(np.mean(results['acc_constr'])))
        except:
            gacc.append(float('NaN'))
        mae.append(results['mae'])
        # gacc.append(float('NaN'))
        # mae.append(compute_mae(X, results['dictionary'], results['z']))
        
    # if np.nanmax(gacc) < prev_G or np.isnan(gacc).sum()==len(gacc):
    #     continue
    # if np.nanmin(mae) > 100:
    #     continue
        
    lst_av_acc.append(acc)
    lst_av_gacc.append(gacc)
    lst_av_mae.append(mae)
    
    lst_f.append(ff)

lst_av_acc = np.array(lst_av_acc)
lst_av_gacc = np.array(lst_av_gacc)
lst_av_mae = np.array(lst_av_mae)


lst_optf = []
lst_optseed = []
for j in range(lst_av_acc.shape[1]):
    tmp_acc = lst_av_acc[:,j]
    tmp_gacc = lst_av_gacc[:,j]
    tmp_mae = lst_av_mae[:,j]
    tmp_lst_f = np.array(lst_f)[~np.isnan(tmp_acc)]
    tmp_gacc = tmp_gacc[~np.isnan(tmp_acc)]
    tmp_mae = tmp_mae[~np.isnan(tmp_acc)]
    tmp_acc = tmp_acc[~np.isnan(tmp_acc)]

    # tmp_lst_f = tmp_lst_f[tmp_gacc<10]
    # tmp_acc = tmp_acc[tmp_gacc<10]
    # tmp_mae = tmp_mae[tmp_gacc<10]
    # tmp_gacc = tmp_gacc[tmp_gacc<10]

    if len(tmp_mae) == 0:
        continue
    for k in np.arange(len(tmp_mae)):
        if 'constr' in method:
            # best_idx = set(np.argsort(tmp_acc)[-(k+1):]) &  set(np.argsort(tmp_gacc)[:k]) &  set(np.argsort(tmp_mae)[:k])
            # best_idx = set(np.argsort(tmp_gacc)[:k]) &  set(np.argsort(tmp_mae)[:k])
            best_idx = set(np.argsort(tmp_acc)[-(k+1):]) &  set(np.argsort(tmp_gacc)[-(k+1):]) &  set(np.argsort(tmp_mae)[:k+1])
            # best_idx = set(np.argsort(tmp_acc)[-(k+1):]) &  set(np.argsort(tmp_mae)[:k])
            # best_idx = set(np.argsort(tmp_acc)[-(k+1):]) &  set(np.argsort(tmp_gacc)[-(k+1):])
            # best_idx = set(np.argsort(tmp_gacc)[-(k+1):])
            # best_idx = set(np.argsort(tmp_acc)[-(k+1):])
        else:
            best_idx = set(np.argsort(tmp_acc)[-(k+1):])  &  set(np.argsort(tmp_mae)[:k+1])
            # best_idx = set(np.argsort(tmp_acc)[-(k+1):])
            print('no counfounder')
            
        if len(best_idx) > 0:
            best_idx = list(best_idx)[0]
            break

    try:
        print('Top %d'%k, 
              '; Acc %2.2f'%tmp_acc[best_idx], 
              '; Acc constr %2.2f'%tmp_gacc[best_idx] if 'constr' in method else '', 
              '; MAE %g'%tmp_mae[best_idx])
    except:
        import ipdb
        ipdb.set_trace()

    lst_optf.append(tmp_lst_f[best_idx])
    lst_optseed.append(lst_seed[j])


dictionary = []
constr_sig = []
acc = []
gacc = []
mae = []
tmp_lst_seed = []
for i, seed in enumerate(lst_optseed):
    opt_f = lst_optf[i]
    f = 'seed-%d'%seed
    results = np.load(join(res_path, opt_f, f, 'results.npz'))

    if 'constr' in method:
        dictionary.append(results['dictionary'][:,1:])
        constr_sig.append(results['dictionary'][:,[0]])
    else:
        dictionary.append(results['dictionary'])
        
    tmp_lst_seed.append(seed)
    acc.append(results['acc'])
    # try:
    gacc.append(float('NaN'))
    # gacc.append(np.mean(results['acc_constr']))
    # except:
    #     gacc.append(float('NaN'))
            
    # mae.append(results['mae'])
    mae.append(compute_mae(X, results['dictionary'], results['z']))


dictionary = np.hstack(tuple(dictionary))

if 'constr' in method:
    constr_sig = np.hstack(tuple(constr_sig))

if dictionary.shape[1] > numAtoms:
    model = AgglomerativeClustering(n_clusters=numAtoms-1 if 'constr' in method else numAtoms,  
                                    affinity='cosine', 
                                    linkage='complete')
    cluster_indices = model.fit_predict(dictionary.T)
    sil_score = silhouette_score(dictionary.T, cluster_indices)

    av_dictionary = np.zeros((len(dictionary), cluster_indices.max()+1))
    av_z = np.zeros((cluster_indices.max()+1, ))
    for i in range(cluster_indices.max()+1):
        av_dictionary[:,i] = np.mean(dictionary[:,cluster_indices==i], axis=1)
    if 'constr' in method:
        av_constr_sig = np.mean(constr_sig, axis=1)
        av_dictionary = np.hstack((av_constr_sig.reshape((-1,1)), av_dictionary))
else:
    av_dictionary = results['dictionary']
    sil_score = 0
print('# run', len(acc), 
      'Silhouette score', '%2.2f'%sil_score, 
      'MAE', '%2.2f (%2.2f)'%(np.mean(mae),np.std(mae)), 
      'Acc', '%2.2f (%2.2f)'%(np.mean(acc),np.std(acc)))
      # 'Acc constr', '%2.2f (%2.2f)'%(np.mean(gacc),np.std(gacc)))
best_res_path = join(res_path, opt_f)

if 'constr' in method:
    np.savez(join(out_path, 'numAtoms_%d_%s_a-%g.npz'%(numAtoms,train_set, alpha)), 
             dictionary=av_dictionary, 
             sil_score=sil_score, 
             acc=acc,
             acc_constr=gacc,
             mae=mae,
             path=best_res_path,
             lst_path=lst_optf,
             lst_seed=tmp_lst_seed)
else:
    np.savez(join(out_path, 'numAtoms_%d_%s_a-%g.npz'%(numAtoms,train_set, alpha)), 
             dictionary=av_dictionary, 
             sil_score=sil_score, 
             acc=acc,
             mae=mae,
             path=best_res_path,
             lst_path=lst_optf,
             lst_seed=tmp_lst_seed)
    

