import os
import sys
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["OPENBLAS_NUM_THREADS"] = "16" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "32" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "16" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "32" # export NUMEXPR_NUM_THREADS=6
# os.environ["NUMBA_CACHE_DIR"]='/tmp/numba_cache'
import numpy as np
import pandas as pd
import scipy as sp
import scipy.sparse
import h5py

import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras.layers import Layer, Dense, BatchNormalization, LeakyReLU
from tensorflow.keras.utils import Progbar

tfd = tfp.distributions

import matplotlib.pyplot as plt
from types import SimpleNamespace
from sklearn.model_selection import train_test_split


with h5py.File('data/pbmc_count.h5', 'r') as f:
    print(f.keys())
    ADT_names = np.array(f['ADT_names'], dtype='S32').astype(str)
    gene_names = np.array(f['gene_names'], dtype='S32').astype(str)
    X = sp.sparse.csc_matrix(
        (np.array(f['RNA.data'], dtype=np.float32), 
         np.array(f['RNA.indices'], dtype=np.int32),
         np.array(f['RNA.indptr'], dtype=np.int32)
        ), 
        shape = np.array(f['RNA.shape'], dtype=np.int32)).tocsc().astype(np.float32).T.toarray()
    Y = np.array(f['ADT'], dtype=np.float32)
    cell_types = np.array(f['celltype.l1'], dtype='S32').astype(str)    
    


cell_type_list =  ['Mono', 'CD4 T']
cell_type_test = cell_type_list[int(sys.argv[1])]
p_feat = np.linspace(0,1,6)[int(sys.argv[2])]
path_root = 'result/ex5/{}/{:.01f}/'.format(cell_type_test, p_feat)

dim_input_arr = [len(gene_names), len(ADT_names)]
config = {
    'dim_input_arr': dim_input_arr,
    'dimensions':[256], 
    'dim_latent':32,
    'dim_block': np.array(dim_input_arr),
    'dist_block':['NB','NB'], 
    'dim_block_enc':np.array([256, 128]),
    'dim_block_dec':np.array([256, 128]),
    'dim_block_embed':np.array([32, 16]),
    
    'block_names':np.array(['rna', 'adt']),
    'uni_block_names':np.array(['rna','adt']),
    
    'beta_kl':1.,
    'beta_unobs':2./3.,
    'beta_modal':np.array([0.15,0.85]),
    'beta_reverse':0.5,

    "p_feat" : p_feat,
    "p_modal" : np.ones(2)/2,
    
}
config = SimpleNamespace(**config)
n_samples = 50


# preprocess
X = np.log(X/np.sum(X, axis=1, keepdims=True)*1e4+1.)
Y = np.log(Y/np.sum(Y, axis=1, keepdims=True)*1e4+1.)

# data spliting
data = np.c_[X, Y]
data_norm = np.exp(data)-1
data_train_norm = data_norm[cell_types!=cell_type_test]
data_test_norm = data_norm[cell_types==cell_type_test]

from functools import partial
from scVAEIT.VAEIT import scVAEIT
model = scVAEIT(config, data[cell_types!=cell_type_test])


dataset_test = tf.data.Dataset.from_tensor_slices((
    data[cell_types==cell_type_test],
    model.cat_enc.transform(np.zeros((np.sum(cell_types==cell_type_test),1))).toarray().astype(np.float32),
    np.zeros(np.sum(cell_types==cell_type_test)).astype(np.int32)
)).batch(512).prefetch(tf.data.experimental.AUTOTUNE)
X_test = X[cell_types==cell_type_test].copy()
Y_test = Y[cell_types==cell_type_test].copy()
del X, Y, data



############################################################
#
# Test on the hold-out cell type
#
############################################################
def comp_cor_flatten(x, y):
    pearson_r, pearson_p = scipy.stats.pearsonr(x, y)
    print(f"Found pearson's correlation/p of {pearson_r:.4f}/{pearson_p:.4g}")
    spearman_corr, spearman_p = scipy.stats.spearmanr(x, y)
    print(f"Found spearman's collelation/p of {spearman_corr:.4f}/{spearman_p:.4g}")
    return pearson_r, spearman_corr

def evaluate(vae, epoch, dataset_test, X_test, Y_test):
    mask_rna = np.zeros((1,np.sum(vae.config.dim_input_arr)), dtype=np.float32)
    mask_rna[:,:vae.config.dim_input_arr[0]] = -1.
    recon = vae.get_recon(dataset_test, mask_rna)
    X_hat = recon[:, :vae.config.dim_input_arr[0]]
    Y_hat = recon[:, vae.config.dim_input_arr[0]:]
    
    res = []
    pr, sr = comp_cor_flatten(Y_test.flatten(), Y_hat.flatten())
    mse = np.mean((Y_test.flatten()-Y_hat.flatten())**2)
    res.append(['ADT', 'ADT', pr, sr, mse])
    pr, sr = comp_cor_flatten(X_test.flatten(), X_hat.flatten())
    mse = np.mean((X_test.flatten()-X_hat.flatten())**2)
    res.append(['ADT', 'RNA', pr, sr, mse])
    
    _df = pd.DataFrame({
        'RNA':gene_names,
        'Pearson r':[np.corrcoef(X_hat[:,i], X_test[:,i])[0,1] for i in np.arange(X_hat.shape[1])],
        'Spearman r':[scipy.stats.spearmanr(X_hat[:,i], X_test[:,i])[0] for i in np.arange(X_hat.shape[1])],
        'MSE':np.mean((X_hat-X_test)**2, axis=0)
    })
    print(np.quantile(_df['MSE'].values, [0.,0.5,1.0]))
    _df.to_csv(path_root+'res_scVAEIT_RNA_{:d}.csv'.format(epoch))
    
    
    mask_adt = np.zeros((1,np.sum(vae.config.dim_input_arr)), dtype=np.float32)
    mask_adt[:,vae.config.dim_input_arr[0]:] = -1.
    recon = vae.get_recon(dataset_test, mask_adt)
    X_hat = recon[:, :vae.config.dim_input_arr[0]]
    Y_hat = recon[:, vae.config.dim_input_arr[0]:]
    
    pr, sr = comp_cor_flatten(Y_test.flatten(), Y_hat.flatten())
    mse = np.mean((Y_test.flatten()-Y_hat.flatten())**2)
    res.append(['RNA', 'ADT', pr, sr, mse])
    pr, sr = comp_cor_flatten(X_test.flatten(), X_hat.flatten())
    mse = np.mean((X_test.flatten()-X_hat.flatten())**2)
    res.append(['RNA', 'RNA', pr, sr, mse])

    _df = pd.DataFrame({
        'ADT':ADT_names,
        'Pearson r':[np.corrcoef(Y_hat[:,i], Y_test[:,i])[0,1] for i in np.arange(Y_hat.shape[1])],
        'Spearman r':[scipy.stats.spearmanr(Y_hat[:,i], Y_test[:,i])[0] for i in np.arange(Y_hat.shape[1])],
        'MSE':np.mean((Y_hat-Y_test)**2, axis=0)
    })
    print(np.quantile(_df['Pearson r'].values, [0.,0.5,1.0]))
    _df.to_csv(path_root+'res_scVAEIT_ADT_{:d}.csv'.format(epoch))


    pd.DataFrame(res, 
        columns=['Source', 'Target', 'Pearson r', 'Spearman r', 'MSE']).to_csv(
        path_root+'res_scVAEIT_overall_{:d}.csv'.format(epoch))


model.train(
    valid=False, num_epoch=500, batch_size=512, save_every_epoch=50,
    verbose=True, checkpoint_dir=path_root+'checkpoint/',
    eval_func=partial(evaluate, dataset_test=dataset_test, 
                      X_test=X_test, Y_test=Y_test)
)



'''
Evaluate
'''
from tqdm import tqdm
mask_list = pd.read_csv('data/mask_pbmc.csv', index_col=0).values
for epoch in range(1,11):
    status = checkpoint.restore(path_root+'checkpoint/ckpt-{}'.format(epoch))
    model.vae(tf.zeros((1,np.sum(model.vae.config.dim_input_arr))),
              tf.zeros((1,np.sum(model.vae.config.dim_input_arr))),
              tf.zeros((1,np.sum(model.batches.shape[1]))), 
              pre_train=True, L=1, training=False)
    print(status)
    
    res = []
    for ip in np.arange(2,9,2):
        p = ip/10.
        for i in tqdm(range(10)):

            mask = mask_list[(10*(ip-1)+i):(10*(ip-1)+i+1),:] 

            id_rna = np.where(mask[0,:config.dim_input_arr[0]]==0)[0]
            id_adt = np.where(mask[0,config.dim_input_arr[0]:]==0)[0] + config.dim_input_arr[0]
            id_rna_test = np.where(mask[0,:config.dim_input_arr[0]]==1)[0]
            id_adt_test = np.where(mask[0,config.dim_input_arr[0]:]==1)[0]
            l_rna = np.median(np.sum(data_train_norm[:, id_rna], axis=-1))
            l_adt = np.median(np.sum(data_train_norm[:, id_adt], axis=-1))

            data_test_sub = data_raw_test.copy()
            data_test_sub[:, id_rna] = np.log1p(
                data_test_sub[:, id_rna]/np.sum(data_test_sub[:, id_rna], axis=1, keepdims=True)*l_rna)
            data_test_sub[:, id_adt] = np.log1p(
                data_test_sub[:, id_adt]/np.sum(data_test_sub[:, id_adt], axis=1, keepdims=True)*l_adt)

            dataset_test_sub = tf.data.Dataset.from_tensor_slices((
                data_test_sub.astype(tf.keras.backend.floatx()),
                model.cat_enc.transform(np.zeros((np.sum(cell_types==cell_type_test),1))).toarray().astype(np.float32),
                np.zeros(np.sum(cell_types==cell_type_test)).astype(np.int32)
            )).batch(512).prefetch(tf.data.experimental.AUTOTUNE)

            res += evaluate(model.vae, dataset_test_sub, X_test, Y_test, mask, 
                            gene_names, ADT_names, id_rna_test, id_adt_test, p, i)

    pd.DataFrame(
            res, columns=['i', 'Target', 'Pearson r', 'Spearman r', 'MSE']
        ).to_csv(path_root+'res_scVAEIT_masked_{:d}.csv'.format(int(epoch*50)))