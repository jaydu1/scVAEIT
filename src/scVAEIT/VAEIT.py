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



class VAEIT():
    """
    Variational Inference for integration and transfer learning.
    """
    def __init__(self, config: SimpleNamespace, data, masks, id_dataset=None, batches_cate=None, batches_cont=None, conditions=None
        ):
        '''
        Initialize the VAEIT model with the provided configuration and data.

        Parameters
        ----------
        config : SimpleNamespace
            Configuration object containing model parameters.
        data : np.array
            The cell-by-feature matrix representing the input data.
        masks : np.array
            Masks indicating missingness, where 1 represents missing and 0 represents observed.
            This can be a full mask matrix with the same shape as `data`, or a condensed matrix indexed by `id_dataset`.
        id_dataset : np.array, optional
            Integer IDs for each cell in the dataset. Required if `masks` is a condensed matrix.
        batches_cate : np.array, optional
            Categorical batch information for each cell.
        batches_cont : np.array, optional
            Continuous batch information for each cell.
        conditions : np.array, optional
            Conditions for each cell, used for calculating MMD loss if `config.gamma` is not 0.

        Returns
        -------
        None
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
        if 'dim_input_arr' not in config:
            config['dim_input_arr'] = data.shape[1]
        if np.isscalar(config['dim_input_arr']):
            config['dim_input_arr'] = np.array([config['dim_input_arr']], dtype=np.int32)
        n_modal = len(config['dim_input_arr'])
        n_block = len(config['dim_block']) if 'dim_block' in config else n_modal
        
        
        config = dict(filter(lambda item: item[1] is not None, config.items()))


        config = {**{
            'beta_kl':2., # weight for beta-VAE
            'beta_reverse':0., # weight for reverse prediction (use masked out features to predict the observed features)
            'beta_modal':np.ones(n_modal, dtype=np.float32), # weight for each modality
            'p_modal':None,

            'uni_block_names':np.char.add(np.repeat('M-', n_modal), np.arange(n_modal).astype(str)),
            'block_names':np.char.add(np.repeat('M-', n_block), np.arange(n_block).astype(str)),
            'dist_block':np.repeat('NB', n_block),
            'dim_block':np.array(config['dim_input_arr'], dtype=np.int32),                     
            'dim_block_enc':np.zeros(n_block, dtype=np.int32),
            'dim_block_dec':np.zeros(n_block, dtype=np.int32),


            'skip_conn':False, # whether to use skip connection in decoder                     
            'max_vals':None,
            
            'gamma':0.
        }, **config}

        if isinstance(config, dict):
            config = SimpleNamespace(**config)

        
        config.dimensions = check_arr_type(config.dimensions, np.int32)        
        config.dist_block = check_arr_type(config.dist_block, str)

        if np.isscalar(config.dim_block_embed):
            config.dim_block_embed = np.full(n_block, config.dim_block_embed, dtype=np.int32)
        config.dim_block_enc = check_arr_type(config.dim_block_enc, np.int32)
        config.dim_block_dec = check_arr_type(config.dim_block_dec, np.int32)
        

        # preprocess batch
        self.batches = np.array([], dtype=np.float32).reshape((data.shape[0],0))
        if batches_cate is not None:
            batches_cate = np.array(batches_cate)
            self.cat_enc = OneHotEncoder(drop='first').fit(batches_cate)
            self.batches = self.cat_enc.transform(batches_cate).toarray()
        if batches_cont is not None:
            self.cont_enc = StandardScaler()            
            batches_cont = self.cont_enc.fit_transform(batches_cont)
            batches_cont = np.nan_to_num(batches_cont)
            batches_cont = np.array(batches_cont, dtype=np.float32)            
            self.batches = np.c_[self.batches, batches_cont]
        self.batches = tf.convert_to_tensor(self.batches, dtype=tf.keras.backend.floatx())
        
        if conditions is not None and config.gamma != 0.:
            ## observations with np.nan will not participant in calculating mmd_loss
            conditions = np.array(conditions)
            if len(conditions.shape)<2:
                conditions = conditions[:,None]
            self.conditions = OrdinalEncoder(dtype=int, encoded_missing_value=-1).fit_transform(conditions) + int(1)
            
        else:
            config.gamma = 0.
            self.conditions = np.zeros((data.shape[0],1), dtype=np.int32)
        self.conditions = tf.cast(self.conditions, tf.int32)

        

        # [num_cells, num_features]
        self.masks = tf.convert_to_tensor(masks, dtype=tf.keras.backend.floatx())
                
        if self.masks.shape==self.data.shape:
            self.id_dataset = self.masks
            self.full_masks = True
        else:
            self.id_dataset = tf.convert_to_tensor(id_dataset, dtype=tf.int32)
            self.full_masks = False

        if config.max_vals is None:
            segment_ids = np.repeat(np.arange(n_block), config.dim_block)
            config.max_vals = np.zeros(segment_ids.max()+1)
            np.maximum.at(config.max_vals, segment_ids, np.max(data,axis=0))
            for i, dist in enumerate(config.dist_block):
                if dist != 'NB':
                    config.max_vals[i] = tf.constant(float('nan'))
        elif np.isscalar(config.max_vals):
            config.max_vals = tf.constant(config.max_vals, shape=n_block, dtype=tf.keras.backend.floatx())
        config.max_vals = tf.convert_to_tensor(config.max_vals, dtype=tf.keras.backend.floatx())
        
        self.batch_size_inference = 512
        self.config = config
        self.reset()
        print(self.config, self.masks.shape, self.data.shape, self.batches.shape, flush = True)
        

    def reset(self):
        train.clear_session()
        if hasattr(self, 'vae'):
            del self.vae
            import gc
            gc.collect()
        self.vae = model.VariationalAutoEncoder(self.config, self.masks)
        

    def train(self, valid = False, stratify = False, test_size = 0.1, random_state: int = 0,
            learning_rate: float = 1e-3, batch_size: Optional[int] = None, batch_size_inference: Optional[int] = None,
            L: int = 1, num_epoch: int = 200, num_step_per_epoch: Optional[int] = None, save_every_epoch: Optional[int] = 25, init_epoch: Optional[int] = 1,
            early_stopping_patience: int = 10, early_stopping_tolerance: float = 1e-4, 
            early_stopping_relative: bool = True, verbose: bool = False,
            checkpoint_dir: Optional[str] = None, delete_existing: Optional[str] = True, eval_func = None): 
        '''
        Train the VAEIT model with the specified parameters.

        Parameters
        ----------
        valid : bool, optional
            Whether to use a validation set during training. Default is False.
        stratify : bool, optional
            Whether to stratify the split when creating the validation set. Default is False.
        test_size : float or int, optional
            The proportion or size of the validation set. Default is 0.1.
        random_state : int, optional
            The random state for data splitting. Default is 0.
        learning_rate : float, optional
            The initial learning rate for the Adam optimizer. Default is 1e-3.
        batch_size : int, optional 
            The batch size for training. Default is 256 when using full mask matrices, or 64 otherwise.
        batch_size_inference : int, optional
            The batch size for inference. Default is 256 when using full mask matrices, or 64 otherwise.
        L : int, optional 
            The number of Monte Carlo samples. Default is 1.
        num_epoch : int, optional 
            The maximum number of epochs. Default is 200.
        num_step_per_epoch : int, optional 
            The number of steps per epoch. If None, it will be inferred from the number of cells and batch size. Default is None.
        save_every_epoch : int, optional 
            Frequency (in epochs) to save model checkpoints. Default is 25.
        init_epoch : int, optional 
            The initial epoch number. Default is 1.
        early_stopping_patience : int, optional 
            The number of epochs to wait for improvement before early stopping. Default is 10.
        early_stopping_tolerance : float, optional 
            The minimum change in loss to be considered as an improvement. Default is 1e-4.
        early_stopping_relative : bool, optional
            Whether to monitor the relative change in loss for early stopping. Default is True.
        verbose : bool, optional
            Whether to print the training process. Default is False.
        checkpoint_dir : str, optional 
            Directory to save model checkpoints. Default is None.
        delete_existing : bool, optional 
            Whether to delete existing checkpoints in the directory. Default is True.
        eval_func : function, optional
            A function to evaluate the model, which takes the VAE as an input. Default is None.

        Returns
        -------
        hist : dict
            A dictionary containing the history of training and validation losses.
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
                self.data[id_train], self.batches[id_train], self.id_dataset[id_train], self.conditions[id_train]
                )).shuffle(buffer_size = len(id_train), seed=0,
                           reshuffle_each_iteration=True).batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)

            self.dataset_valid = tf.data.Dataset.from_tensor_slices((
                    self.data[id_valid], self.batches[id_valid], self.id_dataset[id_valid], self.conditions[id_valid]
                )).batch(batch_size_inference).prefetch(tf.data.experimental.AUTOTUNE)
        else:
            id_train = np.arange(self.data.shape[0])
            self.dataset_train = tf.data.Dataset.from_tensor_slices((
                self.data, self.batches, self.id_dataset, self.conditions
                )).shuffle(buffer_size = len(id_train), seed=0,
                           reshuffle_each_iteration=True).batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
            self.dataset_valid = None
            
        if num_step_per_epoch is None:
            num_step_per_epoch = len(id_train)//batch_size
            
        if checkpoint_dir is not None:        
            if init_epoch==1 and delete_existing and tf.io.gfile.exists(checkpoint_dir):
                print("Deleting old log directory at {}".format(checkpoint_dir))
                tf.io.gfile.rmtree(checkpoint_dir)
            tf.io.gfile.makedirs(checkpoint_dir)
        
        self.vae, hist = train.train(
            self.dataset_train,
            self.dataset_valid,
            self.vae,
            checkpoint_dir,
            learning_rate,                        
            int(L),
            int(num_epoch),
            int(num_step_per_epoch),
            int(save_every_epoch),
            init_epoch,
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
        print("Saved checkpoint: {}".format(save_path), flush = True)
        

    def load_model(self, path_to_weights):
        checkpoint = tf.train.Checkpoint(net=self.vae)
        status = checkpoint.restore(path_to_weights)
        print("Loaded checkpoint: {}".format(status), flush = True)


    def set_dataset(self, batch_size_inference=512):
        if not hasattr(self, 'dataset_full') or batch_size_inference != self.batch_size_inference:
            id_samples = np.arange(self.data.shape[0])
            self.dataset_full = tf.data.Dataset.from_tensor_slices((
                    self.data, self.batches, self.id_dataset, tf.constant(id_samples)
                )).shuffle(
                    buffer_size = len(id_samples), seed=0,
                    reshuffle_each_iteration=True
                ).batch(batch_size_inference).prefetch(tf.data.experimental.AUTOTUNE)
            self.batch_size_inference = batch_size_inference

            
    def get_latent_z(self, masks=None, zero_out=True, batch_size_inference=512):
        ''' get the posterier mean of current latent space z (encoder output)

        Returns
        ----------
        z : np.array
            \([N,d]\) The latent means.
        ''' 
        self.set_dataset(batch_size_inference)
        return self.vae.get_z(self.dataset_full, self.full_masks, masks, zero_out)


    def get_denoised_data(self, masks=None, zero_out=True, batch_size_inference=256, return_mean=True, L=50):
        self.set_dataset(batch_size_inference)        

        return self.vae.get_recon(self.dataset_full, self.full_masks, masks, zero_out, return_mean, L)
    
        
    def update_z(self, masks=None, zero_out=True, batch_size_inference=512, **kwargs):
        self.z = self.get_latent_z(masks, zero_out, batch_size_inference)
        self.adata = sc.AnnData(self.z)
        sc.pp.neighbors(self.adata, **kwargs)

    
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
