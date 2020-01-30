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
parser.add_argument('--labels_identifier', default='project_code')
parser.add_argument('--data_dir', default='data_excl_11_cnt')
parser.add_argument('--method', default='sudl_norm')
parser.add_argument('--numAtoms', type=int, default=20)
parser.add_argument('--train_set', default='all')
parser.add_argument('--ifold', type=int, default=0)
parser.add_argument('--alpha', type=float, default=1)
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

out_path = join(icgc_path, labels_identifier+'_labels', data_dir.replace('data', 'opt_SuDL_res'), method)
if not exists(out_path):
    makedirs(out_path)

X, Y, _, _, _ = pickle.load(open(join(icgc_path, labels_identifier+'_labels', data_dir, 
                        'run%d.pkl'%ifold), 'rb'))
if 'data_all' not in data_dir:
    X = np.hstack((X['train'], X['val'], X['test']))
    Y = np.vstack((Y['train'], Y['val'], Y['test']))
    
Y = np.argmax(Y, axis=1)


all_mae = []
all_sil = []
lst_numAtoms = np.arange(35,41,1)
X_nan = X.astype(float)
X_nan[X_nan==0] = float('NaN')


for numAtoms in lst_numAtoms:
    res_path = join(icgc_path, labels_identifier+'_labels', data_dir.replace('data', 'res'), method, 
                'numAtoms_%d'%numAtoms, '%s%d'%(train_set, ifold))
    

    lst_seed = set()
    lst_foi = []
    for ff in listdir(res_path):
        if 'a-%g'%alpha not in ff:
            continue
        opt_f = ff
        if len(lst_seed)==0:
            lst_seed = set([int(f.split('-')[-1]) for f in listdir(join(res_path, ff)) if 'seed' in f])
        else:
            lst_seed = lst_seed & set([int(f.split('-')[-1]) for f in listdir(join(res_path, ff)) if 'seed' in f])
            print(set([int(f.split('-')[-1]) for f in listdir(join(res_path, ff)) if 'seed' in f]))
        lst_foi.append(ff)

    lst_seed = np.sort(list(lst_seed))

    lst_av_mae = []
    lst_f = []
    for ff in lst_foi:
        dictionary = []
        oxog_sig = []
        acc = []
        gacc = []
        mae = []
        for seed in lst_seed:
            f = 'seed-%d'%seed
            try:
                results = np.load(join(res_path, ff, f, 'results.npz'))
            except:
                mae.append(float('NaN'))
                continue
            MAE = np.nanmean(np.abs(X - np.round(np.dot(results['dictionary'], results['z'])))/X_nan)
#             mae.append(results['mae'])
            mae.append(MAE)

        lst_av_mae.append(mae)

        lst_f.append(ff)

    lst_av_mae = np.array(lst_av_mae)


    dictionary = []
    mae = []
    for i, seed in enumerate(lst_seed):
        f = 'seed-%d'%seed
        results = np.load(join(res_path, opt_f, f, 'results.npz'))
        if np.isnan(results['dictionary']).sum() ==  results['dictionary'].size:
            continue
        dictionary.append(results['dictionary'])    
        MAE = np.nanmean(np.abs(X - np.round(np.dot(results['dictionary'], results['z'])))/X_nan)
#             mae.append(results['mae'])
        mae.append(MAE)
    if len(dictionary)==0:
        continue
    print(lst_seed)
    dictionary = np.hstack(tuple(dictionary))


    model = AgglomerativeClustering(n_clusters=numAtoms,  
                                    affinity='cosine', 
                                    linkage='complete')

    cluster_indices = model.fit_predict(dictionary.T)
    sil_score = silhouette_score(dictionary.T, cluster_indices)

    av_dictionary = np.zeros((len(dictionary), cluster_indices.max()+1))
    av_z = np.zeros((cluster_indices.max()+1, ))

    for i in range(cluster_indices.max()+1):
        av_dictionary[:,i] = np.mean(dictionary[:,cluster_indices==i], axis=1)

    print('# Atoms', numAtoms, '# run', len(mae), 
          'Silhouette score', '%3.3f'%sil_score, 
          'MAE', '%3.3f (%3.3f)'%(np.mean(mae),np.std(mae)))
    all_mae.append(np.mean(mae))
    all_sil.append(sil_score)
    
    best_res_path = join(res_path, opt_f)
    np.savez(join(out_path, 'numAtoms_%d_%s_a-%g.npz'%(numAtoms,train_set, alpha)), 
             dictionary=av_dictionary, 
             sil_score=sil_score, 
             mae=mae,
             path=best_res_path,
             lst_seed=lst_seed)


plt.figure()
plt.plot(lst_numAtoms, all_mae, '.-', color='C0')
plt.ylabel('Mean Absolute Error', color='C0', fontweight='bold')
ax2 = plt.gca().twinx()
plt.plot(lst_numAtoms, all_sil, '.-', color='C1')
plt.ylabel('Silhouette score', color='C1', fontweight='bold')
plt.xlabel('# Atoms')
plt.show()
plt.close()




