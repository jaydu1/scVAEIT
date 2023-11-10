import warnings
from typing import Optional, Union
from types import SimpleNamespace

import tensorflow as tf

import scVAEIT.model as model 
import scVAEIT.train as train
from scVAEIT.utils import check_arr_type

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import OrdinalEncoder

from sklearn.model_selection import train_test_split
import numpy as np

import scanpy as sc


class scVAEIT():
    """
    Variational Inference for Trajectory by AutoEncoder.
    """
    def __init__(self, config: SimpleNamespace, data, masks, id_dataset=None, batches_cate=None, batches_cont=None
        ):
        '''
        Get input data for model.


        Parameters
        ----------
        config : SimpleNamespace
            Dict of config.
        data : np.array
            The cell-by-feature matrix.
        masks : np.array
            Masks that indicate missingness. 1 is missing and 0 is observed.
            It can be the full mask matrix with the same shape as `data`, or a condensed matrix that can be indexed 
            using values in `id_dataset`.
        id_dataset : np.array, optional
            The dataset integer id for each cell. If masks is a condensed matrix, it is required.
        batches_cate : np.array, optional
            The categorical batch information for each cell.
        batches_cont : np.array, optional
            The continuous batch information for each cell.

        Returns
        -------
        None.
        '''
        self.dict_method_scname = {
            'PCA' : 'X_pca',
            'UMAP' : 'X_umap',
            'TSNE' : 'X_tsne',
            'diffmap' : 'X_diffmap',
            'draw_graph' : 'X_draw_graph_fa'
        }

        self.data = tf.convert_to_tensor(data, dtype=tf.keras.backend.floatx())
        
        # prprocessing config
        config = {**{'beta_kl':1., # weight for beta-VAE
                     'beta_reverse':0., # weight for reverse prediction (use masked out features to predict the observed features)
                    }, **config}
        
        if isinstance(config, dict):            
            config = SimpleNamespace(**config)

        if np.isscalar(config.dim_input_arr):
            config.dim_input_arr = np.array([config.dim_input_arr], dtype=np.int32)
        n_modal = len(config.dim_input_arr)
        if config.uni_block_names is None:
            config.uni_block_names = np.char.add(
                np.repeat('NB-', n_modal), np.arange(n_modal).astype(str)
            )
        
        if config.dim_block is None:
            config.dim_block = np.array(config.dim_input_arr, dtype=np.int32)
        
        config.dimensions = check_arr_type(config.dimensions, np.int32)
        
        n_block = len(config.dim_block)
        if config.dist_block is None:
            config.dist_block = np.repeat('NB', n_block)
        if config.block_names is None:
            config.block_names = np.char.add(
                np.repeat('NB-', n_block), np.arange(n_block).astype(str)
            )
        if np.isscalar(config.dim_block_embed):
            config.dim_block_embed = np.full(n_block, config.dim_block_embed, dtype=np.int32)
        if config.dim_block_enc is None:
            config.dim_block_enc = np.zeros(n_block, dtype=np.int32)
        else:
            config.dim_block_enc = check_arr_type(config.dim_block_enc, np.int32)
            
        if config.dim_block_dec is None:
            config.dim_block_dec = np.zeros(n_block, dtype=np.int32)
        else:
            config.dim_block_dec = check_arr_type(config.dim_block_dec, np.int32)
        if config.beta_modal is None:
            config.beta_modal = np.ones(n_modal, dtype=np.float32)

        self.config = config

        # preprocess batch
        self.batches = np.array([], dtype=np.float32).reshape((data.shape[0],0))
        if batches_cate is not None:
            batches_cate = np.array(batches_cate)
            self.cat_enc = OneHotEncoder().fit(batches_cate)
            self.batches = self.cat_enc.transform(batches_cate).toarray()
        if batches_cont is not None:
            self.cont_enc = StandardScaler()            
            batches_cont = self.cont_enc.fit_transform(batches_cont)
            batches_cont = np.nan_to_num(batches_cont)
            batches_cont = np.array(batches_cont, dtype=np.float32)            
            self.batches = np.c_[self.batches, batches_cont]
        self.batches = tf.convert_to_tensor(self.batches, dtype=tf.keras.backend.floatx())
            
        # [num_cells, num_features]
        self.masks = tf.convert_to_tensor(masks, dtype=tf.keras.backend.floatx())
                
        if self.masks.shape==self.data.shape:
            self.id_dataset = self.masks
            self.full_masks = True
        else:
            self.id_dataset = tf.convert_to_tensor(id_dataset, dtype=tf.int32)
            self.full_masks = False
        
        self.reset()
        
        
    def reset(self):
        train.clear_session()
        if hasattr(self, 'vae'):
            del self.vae
            import gc
            gc.collect()
        self.vae = model.VariationalAutoEncoder(self.config, self.masks)
        

    def train(self, valid = False, stratify = False, test_size = 0.1, random_state: int = 0,
            learning_rate: float = 1e-3, batch_size: Optional[int] = None, batch_size_inference: Optional[int] = None, 
            L: int = 1, num_epoch: int = 200, num_step_per_epoch: Optional[int] = None, save_every_epoch: Optional[int] = 25,
            early_stopping_patience: int = 10, early_stopping_tolerance: float = 1e-4, 
            early_stopping_relative: bool = True, verbose: bool = False,
            checkpoint_dir: Optional[str] = None, delete_existing: Optional[str] = True, eval_func=None): 
        '''Pretrain the model with specified learning rate.

        Parameters
        ----------
        test_size : float or int, optional
            The proportion or size of the test set.
        random_state : int, optional
            The random state for data splitting.
        learning_rate : float, optional
            The initial learning rate for the Adam optimizer.
        batch_size : int, optional 
            The batch size for training. Default is 256 when using full mask matrices, or 64 otherwise.
        batch_size_inference : int, optional
            The batch size for inference. Default is 256 when using full mask matrices, or 64 otherwise.
        L : int, optional 
            The number of MC samples.
        num_epoch : int, optional 
            The maximum number of epochs.
        num_step_per_epoch : int, optional 
            The number of step per epoch, it will be inferred from number of cells and batch size if it is None.            
        early_stopping_patience : int, optional 
            The maximum number of epochs if there is no improvement.
        early_stopping_tolerance : float, optional 
            The minimum change of loss to be considered as an improvement.
        early_stopping_relative : bool, optional
            Whether monitor the relative change of loss as stopping criteria or not.
        path_to_weights : str, optional 
            The path of weight file to be saved; not saving weight if None.

        Returns
        -------
        hist : dict
            The history of loss.
        '''

        if batch_size is None:
            batch_size = 256 if self.full_masks else 64
        if batch_size_inference is None:
            batch_size_inference = batch_size

        if valid:
            if stratify is False:
                stratify = None    

            id_train, id_valid = train_test_split(
                                    np.arange(self.data.shape[0]),
                                    test_size=test_size,
                                    stratify=stratify,
                                    random_state=random_state)
            
            self.dataset_train = tf.data.Dataset.from_tensor_slices((
                self.data[id_train], self.batches[id_train], self.id_dataset[id_train]
                )).shuffle(buffer_size = len(id_train), seed=0,
                           reshuffle_each_iteration=True).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

            self.dataset_valid = tf.data.Dataset.from_tensor_slices((
                    self.data[id_valid], self.batches[id_valid], self.id_dataset[id_valid]
                )).batch(batch_size_inference).prefetch(tf.data.experimental.AUTOTUNE)
        else:
            id_train = np.arange(self.data.shape[0])
            self.dataset_train = tf.data.Dataset.from_tensor_slices((
                self.data, self.batches, self.id_dataset
                )).shuffle(buffer_size = len(id_train), seed=0,
                           reshuffle_each_iteration=True).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
            self.dataset_valid = None
            
        if num_step_per_epoch is None:
            num_step_per_epoch = len(id_train)//batch_size+1
            
        checkpoint_dir = 'checkpoint/pretrain/' if checkpoint_dir is None else checkpoint_dir
        if delete_existing and tf.io.gfile.exists(checkpoint_dir):
            print("Deleting old log directory at {}".format(checkpoint_dir))
            tf.io.gfile.rmtree(checkpoint_dir)
        tf.io.gfile.makedirs(checkpoint_dir)
        
        self.vae, hist = train.train(
            self.dataset_train,
            self.dataset_valid,
            self.vae,
            checkpoint_dir,
            learning_rate,                        
            L,
            num_epoch,
            num_step_per_epoch,
            save_every_epoch,
            early_stopping_patience,
            early_stopping_tolerance,
            early_stopping_relative,
            self.full_masks,
            verbose,
            eval_func
        )
        return hist
        
            
    def save_model(self, path_to_weights):
        checkpoint = tf.train.Checkpoint(net=self.vae)
        manager = tf.train.CheckpointManager(
            checkpoint, path_to_weights, max_to_keep=None
        )
        save_path = manager.save()        
        print("Saved checkpoint: {}".format(save_path))
        

    def load_model(self, path_to_weights):
        checkpoint = tf.train.Checkpoint(net=self.vae)
        status = checkpoint.restore(path_to_weights)
        print("Loaded checkpoint: {}".format(status))
    
        
    def update_z(self, masks=None, batch_size_inference=512):
        self.z = self.get_latent_z(masks, batch_size_inference)
        self.adata = sc.AnnData(self.z)
        sc.pp.neighbors(self.adata)

            
    def get_latent_z(self, masks=None, batch_size_inference=512):
        ''' get the posterier mean of current latent space z (encoder output)

        Returns
        ----------
        z : np.array
            \([N,d]\) The latent means.
        ''' 
        if not hasattr(self, 'dataset_full'):
            self.dataset_full = tf.data.Dataset.from_tensor_slices((
                    self.data, self.batches, self.id_dataset
                )).batch(batch_size_inference).prefetch(tf.data.experimental.AUTOTUNE)

        return self.vae.get_z(self.dataset_full, masks)


    def get_denoised_data(self, masks=None, zero_out=True, batch_size_inference=512, L=50):
        if not hasattr(self, 'dataset_full'):
            self.dataset_full = tf.data.Dataset.from_tensor_slices((
                    self.data, self.batches, self.id_dataset
                )).batch(batch_size_inference).prefetch(tf.data.experimental.AUTOTUNE)

        return self.vae.get_recon(self.dataset_full, masks, zero_out, L)

    
    def visualize_latent(self, method: str = "UMAP", 
                         color = None, **kwargs):
        '''
        visualize the current latent space z using the scanpy visualization tools

        Parameters
        ----------
        method : str, optional
            Visualization method to use. The default is "draw_graph" (the FA plot). Possible choices include "PCA", "UMAP", 
            "diffmap", "TSNE" and "draw_graph"
        color : TYPE, optional
            Keys for annotations of observations/cells or variables/genes, e.g., 'ann1' or ['ann1', 'ann2'].
            The default is None. Same as scanpy.
        **kwargs :  
            Extra key-value arguments that can be passed to scanpy plotting functions (scanpy.pl.XX).   

        Returns
        -------
        None.

        '''
          
        if method not in ['PCA', 'UMAP', 'TSNE', 'diffmap', 'draw_graph']:
            raise ValueError("visualization method should be one of 'PCA', 'UMAP', 'TSNE', 'diffmap' and 'draw_graph'")
        
        temp = list(self.adata.obsm.keys())
        if method == 'PCA' and not 'X_pca' in temp:
            print("Calculate PCs ...")
            sc.tl.pca(self.adata)
        elif method == 'UMAP' and not 'X_umap' in temp:  
            print("Calculate UMAP ...")
            sc.tl.umap(self.adata)
        elif method == 'TSNE' and not 'X_tsne' in temp:
            print("Calculate TSNE ...")
            sc.tl.tsne(self.adata)
        elif method == 'diffmap' and not 'X_diffmap' in temp:
            print("Calculate diffusion map ...")
            sc.tl.diffmap(self.adata)
        elif method == 'draw_graph' and not 'X_draw_graph_fa' in temp:
            print("Calculate FA ...")
            sc.tl.draw_graph(self.adata)
    
        if method == 'PCA':
            axes = sc.pl.pca(self.adata, color = color, **kwargs)
        elif method == 'UMAP':            
            axes = sc.pl.umap(self.adata, color = color, **kwargs)
        elif method == 'TSNE':
            axes = sc.pl.tsne(self.adata, color = color, **kwargs)
        elif method == 'diffmap':
            axes = sc.pl.diffmap(self.adata, color = color, **kwargs)
        elif method == 'draw_graph':
            axes = sc.pl.draw_graph(self.adata, color = color, **kwargs)
            
        return axes








 

    
