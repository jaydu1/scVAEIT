import os
import sys
os.environ["OMP_NUM_THREADS"] = "11"
os.environ["OPENBLAS_NUM_THREADS"] = "8" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "11" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "8" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "11" # export NUMEXPR_NUM_THREADS=6
os.environ["NUMBA_CACHE_DIR"]='/tmp/numba_cache'
import numpy as np
import pandas as pd
import scipy as sp
import scipy.sparse
import h5py

import tensorflow as tf
import tensorflow_probability as tfp

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

from tensorflow.keras.layers import Layer, Dense, BatchNormalization, LeakyReLU
from tensorflow.keras.utils import Progbar

tfd = tfp.distributions

import matplotlib.pyplot as plt
from types import SimpleNamespace
from sklearn.model_selection import train_test_split

from types import SimpleNamespace

with h5py.File('data/dogma_cite_asap.h5', 'r') as f:
    print(f.keys())
    peak_names = np.array(f['peak_names'], dtype='S32').astype(str)
    ADT_names = np.array(f['ADT_names'], dtype='S32').astype(str)
    gene_names = np.array(f['gene_names'], dtype='S32').astype(str)
    X = sp.sparse.csc_matrix(
        (np.array(f['RNA.data'], dtype=np.float32), 
         np.array(f['RNA.indices'], dtype=np.int32),
         np.array(f['RNA.indptr'], dtype=np.int32)
        ), 
        shape = np.array(f['RNA.shape'], dtype=np.int32)).tocsc().astype(np.float32)
    Y = np.array(f['ADT'], dtype=np.float32)
    Z = sp.sparse.csc_matrix(
        (np.array(f['peaks.data'], dtype=np.float32),
         np.array(f['peaks.indices'], dtype=np.int32),
         np.array(f['peaks.indptr'], dtype=np.int32)
        ), 
        shape = np.array(f['peaks.shape'], dtype=np.int32)).tocsc().astype(np.float32)
    cell_types = np.array(f['cell_types'], dtype='S32').astype(str)
    batches = np.array(f['batches'], dtype=np.float32)    
    id_X_dogma = np.array(f['id_X_dogma'], dtype=np.int32)
    id_Y_dogma = np.array(f['id_Y_dogma'], dtype=np.int32)
    id_Z_dogma = np.array(f['id_Z_dogma'], dtype=np.int32)    
    id_X_cite = np.array(f['id_X_cite'], dtype=np.int32)
    id_Y_cite = np.array(f['id_Y_cite'], dtype=np.int32)    
    id_Y_asap = np.array(f['id_Y_asap'], dtype=np.int32)
    id_Z_asap = np.array(f['id_Z_asap'], dtype=np.int32)    
    sample_sizes = np.array(f['sample_sizes'], dtype=np.int32)
    
    
chunk_atac = np.array([
    np.sum(np.char.startswith(peak_names, 'chr%d-'%i)) for i in range(1,23)
    ], dtype=np.int32)
dim_input_arr = np.array([len(gene_names),len(ADT_names)])
print(dim_input_arr)
    

X = X.toarray()
X[batches[:,-1]!=2,:] = np.log(X[batches[:,-1]!=2,:]/np.sum(X[batches[:,-1]!=2,:], axis=1, keepdims=True)*1e4+1.)
Y = np.log(Y/np.sum(Y, axis=1, keepdims=True)*1e4+1.)
Z[Z>0.] = 1.
Z = Z.toarray()
data = np.c_[X, Y, Z]

masks = - np.ones((3, np.sum(dim_input_arr)), dtype=np.float32)
masks[0,id_X_dogma] = 0.
masks[0,id_Y_dogma+dim_input_arr[0]] = 0.
masks[0,id_Z_dogma+np.sum(dim_input_arr[:2])] = 0.
masks[1,id_X_cite] = 0.
masks[1,id_Y_cite+dim_input_arr[0]] = 0.
masks[2,id_Y_asap+dim_input_arr[0]] = 0.
masks[2,id_Z_asap+np.sum(dim_input_arr[:2])] = 0.
masks = tf.convert_to_tensor(masks, dtype=tf.float32)

single_list = ['single','multi']
single = single_list[int(sys.argv[1])]
cell_type_list =  ['CD4 T', 'CD8 T', 'Mono', 'B', 'DC', 'NK']
cell_type_test = cell_type_list[int(sys.argv[2])]
print(single, cell_type_test)
path_root = 'result/ex5/{}/{}/'.format(single, cell_type_test)

config = {
    'dim_input_arr': dim_input_arr,
    'dimensions':[256], 
    'dim_latent':32,
    'dim_block': np.array([len(gene_names),len(ADT_names)]),
    'dist_block':np.array(['NB','NB']),
    'dim_block_enc':np.array([256, 128]),
    'dim_block_dec':np.array([256, 128]),
    'block_names':np.array(['rna', 'adt']),
    'uni_block_names':np.array(['rna','adt']),
    'dim_block_embed':np.array([32, 16]),

    'beta_kl':1.,
    'beta_unobs':2./3.,
    'beta_modal':np.array([0.15,0.85]),
    'beta_reverse':0.5,

    "p_feat" : 0.2,
    "p_modal" : np.ones(2)/2,
    
}
config = SimpleNamespace(**config)

with open('data/dogma_cite_asap.npz', 'rb') as f:
    tmp = np.load(f)
    data = tmp['data']
    masks = tmp['masks']
    

from scVAEIT.VAEIT import scVAEIT
if single=='single':
    cell_types = cell_types[:sample_sizes[0]]
    data = data[:sample_sizes[0],:]
    masks = masks[:sample_sizes[0],:]
    batches = batches[:sample_sizes[0],:]

model = scVAEIT(config, data[cell_types!=cell_type_test][:,:np.sum(dim_input_arr)], 
                masks[:,:np.sum(dim_input_arr)], batches[cell_types!=cell_type_test])
del data, masks
model.train(
        valid=False, num_epoch=300, batch_size=512, save_every_epoch=50,
        verbose=True, checkpoint_dir=path_root+'checkpoint/')



'''
Evaluate
'''
def comp_cor_flatten(x, y):
    pearson_r, pearson_p = scipy.stats.pearsonr(x, y)
    print(f"Found pearson's correlation/p of {pearson_r:.4f}/{pearson_p:.4g}")
    spearman_corr, spearman_p = scipy.stats.spearmanr(x, y)
    print(f"Found spearman's collelation/p of {spearman_corr:.4f}/{spearman_p:.4g}")
    return pearson_r, spearman_corr


def evaluate(vae, dataset_test, Y_test):
    mask_adt = vae.masks.copy()
    mask_adt[:,vae.config.dim_input_arr[0]:] = -1.
    recon = vae.get_recon(dataset_test, mask_adt)
    Y_hat = recon[:, vae.config.dim_input_arr[0]:]
    res = []
    name_list = ['dogma','cite','asap']
    for i in range(3):
        name = name_list[i]
        for batch in [0,1]:
            id_data = (batch_test[:,0]==batch)&(batch_test[:,-1]==i)
            id_adt = vae.masks[i,vae.config.dim_input_arr[0]:]!=-1
            _Y_hat = Y_hat[id_data,:][:,id_adt]
            _Y_test = Y_test[id_data,:][:,id_adt]
            _df = pd.DataFrame({
                'ADT':ADT_names[id_adt],
                'Pearson r':[np.corrcoef(_Y_hat[:,i], _Y_test[:,i])[0,1] for i in np.arange(_Y_hat.shape[1])],
                'Spearman r':[scipy.stats.spearmanr(_Y_hat[:,i], _Y_test[:,i])[0] for i in np.arange(_Y_hat.shape[1])],
                'MSE':np.mean((_Y_hat-_Y_test)**2, axis=0)
            })
            print(np.quantile(_df['Pearson r'].values, [0.,0.5,1.0]))
            _df.to_csv(path_root+'res_scVAEIT_%s_ADT_%d.csv'%(name, int(batch)))

            pr, sr = comp_cor_flatten(_Y_test.flatten(), _Y_hat.flatten())
            mse = np.mean((_Y_test.flatten()-_Y_hat.flatten())**2)
            res.append([cell_type_test, name, 'ADT', batch, pr, sr, mse])

    pd.DataFrame(res, 
        columns=['Celltype', 'Name', 'Target', 'Batch', 'Metric_1', 'Metric_2', 'MSE']).to_csv(path_root+'res_scVAEIT_overall.csv')

evaluate(model.vae, dataset_test, Y_test)