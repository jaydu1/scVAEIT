import os
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
            'PCA': 'X_pca', 'UMAP': 'X_umap', 'TSNE': 'X_tsne',
            'diffmap': 'X_diffmap', 'draw_graph': 'X_draw_graph_fa'
        }
        
        self.data = tf.convert_to_tensor(data, dtype=tf.keras.backend.floatx())
        
        # Process configuration
        if 'dim_input_arr' not in config or config['dim_input_arr'] is None:
            config['dim_input_arr'] = data.shape[1]
        if np.isscalar(config['dim_input_arr']):
            config['dim_input_arr'] = np.array([config['dim_input_arr']], dtype=np.int32)
        
        n_modal = len(config['dim_input_arr'])
        n_block = len(config['dim_block']) if 'dim_block' in config and config['dim_block'] is not None else n_modal
        
        self.config = self._process_config(config, n_modal, n_block)
        
        # Process batches, conditions, and masks
        self.batches = self._process_batches(batches_cate, batches_cont)
        self.conditions = self._process_conditions(conditions)
        self.masks, self.id_dataset, self.full_masks = self._process_masks(masks, id_dataset)
        
        # Compute Gaussian statistics
        self._compute_stats(data, n_block)

        self.batch_size_inference = 512
        self.reset()
        
        print(self.config, self.masks.shape, self.data.shape, self.batches.shape, flush=True)


    
    def _process_config(self, config, n_modal, n_block):
        """Process and validate configuration parameters."""
        config = dict(filter(lambda item: item[1] is not None, config.items()))
        
        # Default configuration
        defaults = {
            'beta_kl': 2.,
            'beta_unobs': 0.5,
            'beta_reverse': 0.,
            'beta_modal': np.ones(n_modal, dtype=np.float32),
            'p_modal': None,
            'p_feat': 0.2,
            'uni_block_names': np.char.add(np.repeat('M-', n_modal), np.arange(n_modal).astype(str)),
            'block_names': np.char.add(np.repeat('M-', n_block), np.arange(n_block).astype(str)),
            'dist_block': np.repeat('Gaussian', n_block),
            'dim_block': np.array(config['dim_input_arr'], dtype=np.int32),
            'dim_block_enc': np.zeros(n_block, dtype=np.int32),
            'dim_block_dec': np.zeros(n_block, dtype=np.int32),
            'skip_conn': False,
            'mean_vals': None,
            'min_vals': None,
            'max_vals': None,
            'max_disp': 6.,
            'max_zi_prob': None,
            'gamma': 0.
        }
        
        config = {**defaults, **config}
        
        if isinstance(config, dict):
            config = SimpleNamespace(**config)
        
        # Process arrays
        config.dimensions = check_arr_type(config.dimensions, np.int32)
        config.dist_block = check_arr_type(config.dist_block, str)
        
        if np.isscalar(config.dim_block_embed):
            config.dim_block_embed = np.full(n_block, config.dim_block_embed, dtype=np.int32)
        
        config.dim_block_enc = check_arr_type(config.dim_block_enc, np.int32)
        config.dim_block_dec = check_arr_type(config.dim_block_dec, np.int32)
        
        return config


    def _process_batches(self, batches_cate, batches_cont):
        """Process categorical and continuous batch information."""
        batches = np.array([], dtype=np.float32).reshape((self.data.shape[0], 0))

        if batches_cate is not None:
            # self.cat_enc = OneHotEncoder(drop='first').fit(batches_cate)
            # This affects a lot!!!
            self.cat_enc = OneHotEncoder().fit(batches_cate)
            batches = self.cat_enc.transform(batches_cate).toarray()
        
        if batches_cont is not None:
            self.cont_enc = StandardScaler()
            batches_cont = self.cont_enc.fit_transform(batches_cont)
            batches_cont = np.nan_to_num(batches_cont).astype(np.float32)
            batches = np.c_[batches, batches_cont]
        
        return tf.convert_to_tensor(batches, dtype=tf.keras.backend.floatx())

    def _process_conditions(self, conditions):
        """Process conditions for MMD loss calculation."""
        if conditions is not None:
            conditions = np.array(conditions)
            if len(conditions.shape) < 2:
                conditions = conditions[:, None]
            conditions = OrdinalEncoder(dtype=int, encoded_missing_value=-1).fit_transform(conditions) + int(1)
        else:
            self.config.gamma = 0.
            conditions = np.zeros((self.data.shape[0], 1), dtype=np.int32)

        return tf.cast(conditions, tf.int32)


    def _process_masks(self, masks, id_dataset):
        """Process mask information and determine if using full masks."""
        # Print missingness info: overall, per row, and per column
        missing_overall = np.mean(masks==-1)
        missing_per_row = np.mean(masks==-1, axis=1)
        missing_per_col = np.mean(masks==-1, axis=0)
        print(f"Missingness (overall): {missing_overall:.4f}")
        print(f"Missingness (per row): mean={np.mean(missing_per_row):.4f}, min={np.min(missing_per_row):.4f}, max={np.max(missing_per_row):.4f}")
        print(f"Missingness (per column): mean={np.mean(missing_per_col):.4f}, min={np.min(missing_per_col):.4f}, max={np.max(missing_per_col):.4f}")

        if missing_overall == 0:
            warnings.warn(
                "No missing values detected in the mask (missing_overall == 0). "
                "Please check if the mask is correct. If you intended to supply a mask, ensure missing values are marked as -1."
            )

        masks_tensor = tf.convert_to_tensor(masks, dtype=tf.keras.backend.floatx())

        if masks_tensor.shape == self.data.shape:
            return masks_tensor, masks_tensor, True
        else:
            id_tensor = tf.convert_to_tensor(id_dataset, dtype=tf.int32)
            return masks_tensor, id_tensor, False
        

    def _compute_stats(self, data, n_block):
        """Compute mean, min, and max values for Gaussian blocks."""
        if not self.full_masks:
            masks = tf.gather(self.masks, self.id_dataset, axis=0).numpy()
        else:
            masks = self.masks.numpy()
        masked_data = np.ma.array(data, mask=(masks == -1))
        
        if self.config.mean_vals is None:
            mean_vals = np.zeros(data.shape[1])
        else:
            mean_vals = self.config.mean_vals
            assert len(mean_vals) == data.shape[1], f"mean_vals should have {data.shape[1]} features, got {len(mean_vals)}"
    
        # Initialize max_vals and min_vals at feature level
        if self.config.max_vals is None:
            max_vals = np.max(data, axis=0)
        elif np.isscalar(self.config.max_vals):
            max_vals = np.full(data.shape[1], self.config.max_vals, dtype=np.float32)
        else:
            max_vals = self.config.max_vals
            # If block-level values provided, expand to feature level
            if len(max_vals) == n_block:
                max_vals = np.repeat(max_vals, self.config.dim_block)
            assert len(max_vals) == data.shape[1], f"max_vals should have {data.shape[1]} features or {n_block} blocks, got {len(max_vals)}"
    
        if self.config.min_vals is None:
            min_vals = np.zeros(data.shape[1])
        elif np.isscalar(self.config.min_vals):
            min_vals = np.full(data.shape[1], self.config.min_vals, dtype=np.float32)
        else:
            min_vals = self.config.min_vals
            # If block-level values provided, expand to feature level
            if len(min_vals) == n_block:
                min_vals = np.repeat(min_vals, self.config.dim_block)
            assert len(min_vals) == data.shape[1], f"min_vals should have {data.shape[1]} features or {n_block} blocks, got {len(min_vals)}"
    
        # Process distribution-related blocks
        feature_start = 0
        for (block_size, dist) in zip(self.config.dim_block, self.config.dist_block):
            feature_end = feature_start + block_size
    
            if dist == 'Gaussian':
                if self.config.mean_vals is None:
                    mean_vals[feature_start:feature_end] = np.ma.mean(
                        masked_data[:, feature_start:feature_end], axis=0).filled(0)
    
                if self.config.max_vals is None:
                    max_vals[feature_start:feature_end] = np.max(np.maximum(
                        mean_vals[feature_start:feature_end] + 3 * np.ma.std(masked_data[:, feature_start:feature_end], axis=0).filled(self.config.max_disp),
                        np.ma.max(masked_data[:, feature_start:feature_end], axis=0)
                    ))
                if self.config.min_vals is None:
                    min_vals[feature_start:feature_end] = np.min(np.minimum(
                        mean_vals[feature_start:feature_end] - 3 * np.ma.std(masked_data[:, feature_start:feature_end], axis=0).filled(self.config.max_disp),
                        np.ma.min(masked_data[:, feature_start:feature_end], axis=0)
                    ))
    
            elif dist == 'Bernoulli':
                if self.config.min_vals is None:
                    min_vals[feature_start:feature_end] = 1e-5
                if self.config.max_vals is None:
                    max_vals[feature_start:feature_end] = 1.0 - 1e-5
            elif dist == 'NB' or dist == 'Poisson':
                if self.config.max_vals is None:
                    max_vals[feature_start:feature_end] = np.max(max_vals[feature_start:feature_end])
            else:
                # Ensure min_vals are nonnegative for (zero-inflated) Negative Binomial and Poisson blocks
                min_vals[feature_start:feature_end] = np.maximum(min_vals[feature_start:feature_end], 0)
    
            feature_start += block_size
    
        # Convert to tensors
        self.config.mean_vals = tf.convert_to_tensor(mean_vals, dtype=tf.keras.backend.floatx())
        self.config.max_vals = tf.convert_to_tensor(max_vals, dtype=tf.keras.backend.floatx())
        self.config.min_vals = tf.convert_to_tensor(min_vals, dtype=tf.keras.backend.floatx())


    def reset(self):
        train.clear_session()
        if hasattr(self, 'vae'):
            del self.vae
            import gc

            gc.collect()
        self.vae = model.VariationalAutoEncoder(self.config, self.masks)
        

    def train(self, valid = False, stratify = False, test_size = 0.1, random_state: int = 0,
            learning_rate: float = 3e-4, num_repeat: Optional[int] = 1, batch_size: Optional[int] = None, batch_size_inference: Optional[int] = None,
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
            Frequency (in epochs) to save model checkpoints. Default is num_epoch.
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
        num_repeat = int(num_repeat)
        batch_size = np.minimum(batch_size, self.data.shape[0]*num_repeat)

        if batch_size_inference is None:
            batch_size_inference = batch_size
            
        def make_tf_dataset(index, batch_size, repeat=1, shuffle=False, drop_remainder=False):
            id_tensor = tf.convert_to_tensor(index, dtype=tf.int32)
            ds = tf.data.Dataset.from_tensor_slices((
                tf.gather(self.data, id_tensor),
                tf.gather(self.batches, id_tensor),
                tf.gather(self.id_dataset, id_tensor),
                tf.gather(self.conditions, id_tensor)
            ))
            if shuffle:
                ds = ds.shuffle(buffer_size=len(index), seed=0, reshuffle_each_iteration=True)
            if repeat > 1: ds = ds.repeat(repeat)
            ds = ds.batch(batch_size, drop_remainder=drop_remainder).prefetch(tf.data.experimental.AUTOTUNE)
            return ds

        if valid:
            if stratify is False:
                stratify = None    

            id_train, id_valid = train_test_split(
                np.arange(self.data.shape[0]),
                test_size=test_size,
                stratify=stratify,
                random_state=random_state
            )
            
            self.dataset_train = make_tf_dataset(id_train, batch_size, repeat=num_repeat, shuffle=True, drop_remainder=True)
            self.dataset_valid = make_tf_dataset(id_valid, batch_size_inference, repeat=1, shuffle=False)
        else:
            id_train = np.arange(self.data.shape[0])
            self.dataset_train = make_tf_dataset(id_train, batch_size, repeat=num_repeat, shuffle=True, drop_remainder=True)
            self.dataset_valid = None
            
        if num_step_per_epoch is None:
            num_step_per_epoch = (len(id_train) * num_repeat) // batch_size

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
            int(num_epoch) if save_every_epoch is None else int(save_every_epoch),
            init_epoch,
            early_stopping_patience,
            early_stopping_tolerance,
            early_stopping_relative,
            self.full_masks,
            verbose,
            eval_func
        )
        return hist
        
            
    def save_model(self, path_to_model):
        """
        Save the model weights to the specified path.

        Parameters
        ----------
        path_to_model : str
            Path to the directory where the model weights will be saved.
        """

        checkpoint = tf.train.Checkpoint(net=self.vae)
        manager = tf.train.CheckpointManager(
            checkpoint, path_to_model, max_to_keep=None
        )
        save_path = manager.save()
        print("Saved checkpoint: {}".format(save_path), flush=True)


    def load_model(self, path_to_model):
        """
        Load the model weights from the specified path.
        
        Parameters
        ----------
        path_to_model : str
            Path to the directory or specific checkpoint file containing the model weights.
            If a directory is provided, the latest checkpoint will be loaded.
        """
        checkpoint = tf.train.Checkpoint(net=self.vae)

        if os.path.isdir(path_to_model):
            # Load latest checkpoint from directory
            ckpt_path = tf.train.latest_checkpoint(path_to_model)
            if ckpt_path is None:
                raise FileNotFoundError(f"No checkpoint found in directory: {path_to_model}")
        else:
            # Assume path_to_model is a checkpoint file
            ckpt_path = path_to_model

        checkpoint.restore(ckpt_path)
        print(f"Loaded checkpoint: {ckpt_path}", flush=True)


    def set_dataset(self, num_repeat=5, batch_size_inference=512):
        batch_size_inference = int(batch_size_inference)
        num_repeat = 1 if self.data.shape[0] <= batch_size_inference else num_repeat
            
        if not hasattr(self, 'dataset_full') or batch_size_inference != self.batch_size_inference:
            id_samples = np.arange(self.data.shape[0])
            self.dataset_full = tf.data.Dataset.from_tensor_slices((
                    self.data, self.batches, self.id_dataset, tf.constant(id_samples)
                )).shuffle(
                    buffer_size = len(id_samples), seed=0,
                    reshuffle_each_iteration=True
                ).repeat(num_repeat).batch(batch_size_inference).prefetch(tf.data.experimental.AUTOTUNE)
            self.batch_size_inference = batch_size_inference

            
    def get_latent_z(self, masks=None, zero_out=True, num_repeat=5, batch_size_inference=512, training=True):
        ''' Get the posterior mean of latent space z.

        Parameters
        ----------
        masks : np.array, optional
            Masks indicating missingness, where 1 represents missing and 0 represents observed.
            If None, the full masks will be used. Default is None.
        zero_out : bool, optional
            Whether to zero out the missing values in the output. Default is True.
        num_repeat : int, optional
            The number of times to repeat the dataset to remove effects of batch shuffling for inference.
            Default is 5; if sample size is smaller than batch_size_inference, it is set to 1.
        batch_size_inference : int, optional
            The batch size for inference. Default is 512.
        training : bool, optional
            Whether to use the model in training mode when batch normalization is performed
            based on batch mean and variance; otherwise moving average mean and variance are used. 
            When using small datasets or small a number of epochs for training, it is recommended 
            to set this to True. Default is True.
        
        Returns
        ----------
        z : np.array
            \([N,d]\) The latent means.
        ''' 
        self.set_dataset(num_repeat, batch_size_inference)
        return self.vae.get_z(self.dataset_full, self.data.shape[0], self.full_masks, masks, zero_out, training)


    def get_denoised_data(
            self, masks=None, zero_out=True, return_mean=True, 
            num_repeat=5, L=50, batch_size_inference=256, training=True):
        ''' Get the denoised data (decoder output) from the current latent space z

        Parameters
        ----------
        masks : np.array, optional
            Masks indicating missingness, where 1 represents missing and 0 represents observed.
            If None, the full masks will be used. Default is None.
        zero_out : bool, optional
            Whether to zero out the missing values in the output. Default is True.
        return_mean : bool, optional
            Whether to return the mean of the denoised data. Default is True.
        num_repeat : int, optional
            The number of times to repeat the dataset to remove effects of batch shuffling for inference. 
            Default is 5; if sample size is smaller than batch_size_inference, it is set to 1.
        L : int, optional
            The number of Monte Carlo samples for denoising. Default is 50.
        batch_size_inference : int, optional
            The batch size for inference. Default is 512.
        training : bool, optional
            Whether to use the model in training mode when batch normalization is performed
            based on batch mean and variance; otherwise moving average mean and variance are used. 
            When using small datasets or small a number of epochs for training, it is recommended 
            to set this to True. Default is True.
        
        Returns
        -------
        denoised_data : np.array
            The denoised data with shape \([N, d]\) where \(N\) is the number of cells and \(d\) is the number of features.
        '''
        self.set_dataset(num_repeat, batch_size_inference)
        return self.vae.get_recon(self.dataset_full, self.data.shape[0], self.full_masks, masks, zero_out, return_mean, int(L), training)
    
        
    def update_z(self, masks=None, zero_out=True, batch_size_inference=512, **kwargs):
        '''
        Update the latent representation z based on the current input data and masks.

        Parameters
        ----------
        masks : np.array, optional
            Masks indicating missingness, where 1 represents missing and 0 represents observed.
            If None, the full masks will be used. Default is None.
        zero_out : bool, optional
            Whether to zero out the missing values in the output. Default is True.
        batch_size_inference : int, optional
            The batch size for inference. Default is 512.
        **kwargs :
            Additional keyword arguments to be passed to the scanpy neighbors function.
        '''
        self.z = self.get_latent_z(masks, zero_out, batch_size_inference)
        self.adata = sc.AnnData(self.z)
        sc.pp.neighbors(self.adata, **kwargs)

    
    def visualize_latent(self, method: str = "UMAP", 
                         color = None, **kwargs):
        '''
        Visualize the current latent space z using the scanpy visualization tools

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
        axes : matplotlib.axes.Axes
            Axes object containing the visualization.
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
