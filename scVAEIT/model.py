import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from scVAEIT.utils import ModalMaskGenerator
from scVAEIT.nn_utils import Encoder, Decoder
from tensorflow.keras.layers import Layer, Dense, BatchNormalization
from tensorflow.keras.utils import Progbar

from tensorflow.keras import backend as K
            
            
class VariationalAutoEncoder(tf.keras.Model):
    """
    Combines the encoder, decoder into an end-to-end model for training and inference.
    """
    def __init__(self, config, masks, name = 'autoencoder', **kwargs):
        '''
        Parameters
        ----------
        dimensions : np.array
            The dimensions of hidden layers of the encoder.
        dim_latent : int
            The latent dimension of the encoder.
        
        dim_block : list of int
            (num_block,) The dimension of each input block.        
        dist_block : list of str
            (num_block,) `'NB'`, `'ZINB'`, `'Bernoulli'` or `'Gaussian'`.
        dim_block_enc : list of int
            (num_block,) The dimension of output of first layer of the encoder for each block.
        dim_block_dec : list of int
            (num_block,) The dimension of output of last layer of the decoder for each block.        
        block_names : list of str, optional
            (num_block,) The name of first layer for each block.
        name : str, optional
            The name of the layer.
        **kwargs : 
            Extra keyword arguments.
        '''
        super(VariationalAutoEncoder, self).__init__(name = name, **kwargs)
        self.config = config
        self.masks = masks
        self.embed_layer = Dense(np.sum(self.config.dim_block_embed), 
                                 activation = tf.nn.tanh, name = 'embed')
        self.encoder = Encoder(self.config.dimensions, self.config.dim_latent,
            self.config.dim_block, self.config.dim_block_enc, self.config.dim_block_embed, self.config.mean_vals, self.config.block_names)
        self.decoder = Decoder(self.config.dimensions[::-1], self.config.dim_block,
            self.config.dist_block, self.config.dim_block_dec, self.config.dim_block_embed, 
            self.config.mean_vals, self.config.min_vals, self.config.max_vals, self.config.max_disp, self.config.block_names)
        
        self.mask_generator = ModalMaskGenerator(
            config.dim_input_arr, config.p_feat, config.p_modal)
        
        
    def generate_mask(self, inputs, mask=None, p=None):
        '''
        Generate mask for inputs.

        Parameters
        ----------
        inputs : tf.Tensor of type tf.float32
            The input data.
        mask : tf.Tensor of type tf.float32
            The mask of input data: -1,0,1 indicate missing, observed, and masked.
        '''
        return self.mask_generator(inputs, mask, p)


    def call(self, x, masks, batches, conditions = None, gamma = 0., L=1, training=True):
        '''Feed forward through encoder and decoder.

        Parameters
        ----------
        x : np.array, optional
            The input data.
        masks : np.array, optional
            The mask of input data: -1,0,1 indicate missing, observed, and masked.
        batches : np.array, optional
            The covariates of each cell.
        L : int, optional
            The number of MC samples.
        training : boolean, optional
            Whether in the training phase or not.

        Returns
        ----------
        losses : float
            the loss.
        '''
                            
        z_mean_obs, z_log_var_obs, log_probs_obs = self._get_reconstruction_loss(
            x, masks!=-1., masks!=-1., batches, L, training=training)
        z_mean_unobs_1, z_log_var_unobs_1, log_probs_unobs = self._get_reconstruction_loss(
            x, masks==0., masks==1., batches, L, training=training)
        if self.config.beta_reverse>0:
            z_mean_unobs_2, z_log_var_unobs_2, log_probs_ = self._get_reconstruction_loss(
                x, masks==1., masks==0., batches, L, training=training)            
            log_probs_unobs = (1-self.config.beta_reverse) * log_probs_unobs + self.config.beta_reverse*log_probs_
        
        self.add_loss(
            [- (1-self.config.beta_unobs) * 
             tf.reduce_sum(tf.where(self.config.block_names==name, log_probs_obs, 0.)) * 
             self.config.beta_modal[i] for i,name in enumerate(self.config.uni_block_names)]
        )
        self.add_loss(
            [- self.config.beta_unobs * 
             tf.reduce_sum(tf.where(self.config.block_names==name, log_probs_unobs, 0.)) * 
             self.config.beta_modal[i] for i,name in enumerate(self.config.uni_block_names)]
        )
        
        kl = self._get_kl_normal(
                z_mean_unobs_1, z_log_var_unobs_1, z_mean_obs, z_log_var_obs)
        if self.config.beta_reverse > 0:
            kl = (1-self.config.beta_reverse) * kl + \
                self.config.beta_reverse * self._get_kl_normal(
            z_mean_unobs_2, z_log_var_unobs_2, z_mean_obs, z_log_var_obs)
        self.add_loss(self.config.beta_kl * kl)

        
        if gamma == 0.:
            mmd_loss = tf.constant(0.0, dtype=tf.keras.backend.floatx())
        else:
            mmd_loss = self._get_total_mmd_loss(conditions, z_mean_obs, tf.constant(gamma, dtype=tf.keras.backend.floatx()))
        self.add_loss(mmd_loss)
        return self.losses
    

    @tf.function(reduce_retracing=True)
    def _get_reconstruction_loss(self, x, bool_mask_in, bool_mask_out, batches, L, training=True):
        '''
        Parameters
        ----------
        x : tf.Tensor of type tf.float32
            The input data \(Y_i\).
        bool_mask_in : tf.Tensor of type tf.bool
            False indicates missing.
        bool_mask_out : tf.Tensor of type tf.bool
            Compute likelihood for entries with value True.
        batches : tf.Tensor of type tf.int32
            The batch index of each cell.
        L : int
            The number of MC samples.
        training : boolean, optional
            Whether in the training phase or not.
        '''
        _masks = tf.where(bool_mask_in, 0., 1.)        
        embed = self.embed_layer(_masks, training=training)
        z_mean, z_log_var, z, x_embed = self.encoder(x, bool_mask_in, embed, batches, L=L, training=training)
        if not self.config.skip_conn:
            x_embed = tf.zeros_like(x_embed)
        log_probs = tf.reduce_mean(
            self.decoder(x, embed, bool_mask_out, batches, z, x_embed, training=training), axis=0)
        return z_mean, z_log_var, log_probs
    
    
    @tf.function(reduce_retracing=True)
    def _get_kl_normal(self, mu_0, log_var_0, mu_1, log_var_1):
        kl = 0.5 * (
            tf.exp(tf.clip_by_value(log_var_0-log_var_1, -6., 6.)) + 
            (mu_1 - mu_0)**2 / tf.exp(tf.clip_by_value(log_var_1, -6., 6.)) - 1.
             + log_var_1 - log_var_0)
        return tf.reduce_mean(tf.reduce_sum(kl, axis=-1))
    
    
    def get_recon(self, dataset_test, full_masks, masks=None, zero_out=True, 
                  return_mean=True, L=50):
        '''
        Compute the reconstruction of the input data.

        Parameters
        ----------
        dataset_test : tf.Dataset
            Dataset containing (x, batches).
        full_masks : boolean
            Whether to use full masks.
        masks : np.array, optional
            The mask of input data: -1,0,1 indicate missing, observed, and masked.
        zero_out : boolean, optional
            Whether to zero out the missing values.
        return_mean : boolean, optional
            Whether to return the mean of posterior samples.
        L : int, optional
            The number of MC samples.

        Returns
        ----------
        x_hat : np.array
            The reconstruction.
        '''
        if masks is None:
            masks = self.masks
        x_hat = []
        idx = []
        for x,b,m,_idx in dataset_test:
            if not full_masks:
                m = tf.gather(masks, m)
            _m = tf.where(m==0., 0., 1.)
            embed = self.embed_layer(_m, training=False)
            if not zero_out:
                _m *= 0.
            _, _, z, x_embed = self.encoder(x, _m==0, embed, b, L=L, training=False)
            if not self.config.skip_conn:
                x_embed = tf.zeros_like(x_embed)
            _x_hat = self.decoder(x, embed, tf.ones_like(m, dtype=tf.bool), 
                    b, z, x_embed, training=False, return_prob=False)
            if return_mean:
                _x_hat = tf.reduce_mean(_x_hat, axis=1)
            x_hat.append(_x_hat.numpy())
            idx.append(_idx.numpy())
        idx = np.concatenate(idx)
        x_hat = np.concatenate(x_hat)[np.argsort(idx)]     

        return x_hat
    
    
    def get_z(self, dataset_test, full_masks=False, masks=None, zero_out=True):
        '''Compute latent variables.

        Parameters
        ----------
        dataset_test : tf.Dataset
            Dataset containing (x, batches).
        full_masks : boolean
            Whether to use full masks.
        masks : np.array, optional
            The mask of input data: -1,0,1 indicate missing, observed, and masked.
        zero_out : boolean, optional
            Whether to zero out the missing values.

        Returns
        ----------
        z_mean : np.array
            The latent mean.
        '''
        if masks is None:
            masks = self.masks
        z_mean = []
        idx = []
        for x,b,m,_idx in dataset_test:
            if not full_masks:
                m = tf.gather(masks, m)
            m = tf.where(m==0., 0., 1.)
            if not zero_out:
                m *= 0.
            embed = self.embed_layer(m)
            _z_mean, _, _, _ = self.encoder(x, m==0, embed, b, L=1, training=False)
            z_mean.append(_z_mean.numpy())
            idx.append(_idx.numpy())
        idx = np.concatenate(idx)
        z_mean = np.concatenate(z_mean)[np.argsort(idx)]

        return z_mean


    
    def _get_total_mmd_loss(self, conditions, z, gamma):
        mmd_loss = tf.constant(0.0, dtype=tf.keras.backend.floatx())
        
        n_group = conditions.shape[1]

        for i in range(n_group):
            sub_conditions = conditions[:, i]
            n_sub_group = tf.unique(sub_conditions)[0].shape[0]
            cond = (sub_conditions != tf.constant(0, dtype=tf.int32))
            z_cond = tf.boolean_mask(z, cond)
            sub_conditions = tf.boolean_mask(sub_conditions, cond) - 1

            if (n_sub_group == 1) | (n_sub_group == 0):
                _loss = tf.constant(0.0, dtype=tf.keras.backend.floatx())
            else:
                _loss = self._mmd_loss(y_true=sub_conditions, y_pred=z_cond, gamma=gamma,
                                       n_conditions=n_sub_group, kernel_method='multi-scale-rbf')
            mmd_loss = mmd_loss + _loss
        return mmd_loss

    # each loop the input shape is changed. Can not use @tf.function
    # tf graph requires static shape and tensor dtype
    def _mmd_loss(self, y_true, y_pred, gamma, n_conditions, kernel_method='multi-scale-rbf'):
        conditions_mmd = tf.dynamic_partition(y_pred, y_true, num_partitions=n_conditions)
        loss = tf.constant(0.0, dtype=tf.keras.backend.floatx())
        for i in range(len(conditions_mmd)):
            for j in range(i):
                if conditions_mmd[i].shape[0]>0 and conditions_mmd[j].shape[0]>0:
                    loss += _nan2zero(compute_mmd(conditions_mmd[i], conditions_mmd[j], kernel_method))

        return gamma * loss
    
    
    
    
    
# Below are some functions used in calculating MMD loss

def compute_kernel(x, y, kernel='rbf', **kwargs):
    """Computes RBF kernel between x and y.

    Parameters
    ----------
        x: Tensor
            Tensor with shape [batch_size, z_dim]
        y: Tensor
            Tensor with shape [batch_size, z_dim]

    Returns
    ----------
        The computed RBF kernel between x and y
    """
    scales = kwargs.get("scales", [])
    if kernel == "rbf":
        x_size = K.shape(x)[0]
        y_size = K.shape(y)[0]
        dim = K.shape(x)[1]
        tiled_x = K.tile(K.reshape(x, K.stack([x_size, 1, dim])), K.stack([1, y_size, 1]))
        tiled_y = K.tile(K.reshape(y, K.stack([1, y_size, dim])), K.stack([x_size, 1, 1]))
        return K.exp(-K.mean(K.square(tiled_x - tiled_y), axis=2) / K.cast(dim, tf.float32))
    elif kernel == 'raphy':
        scales = K.variable(value=np.asarray(scales))
        squared_dist = K.expand_dims(squared_distance(x, y), 0)
        scales = K.expand_dims(K.expand_dims(scales, -1), -1)
        weights = K.eval(K.shape(scales)[0])
        weights = K.variable(value=np.asarray(weights))
        weights = K.expand_dims(K.expand_dims(weights, -1), -1)
        return K.sum(weights * K.exp(-squared_dist / (K.pow(scales, 2))), 0)
    elif kernel == "multi-scale-rbf":
        sigmas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6]

        beta = 1. / (2. * (K.expand_dims(sigmas, 1)))
        distances = squared_distance(x, y)
        s = K.dot(beta, K.reshape(distances, (1, -1)))

        return K.reshape(tf.reduce_sum(input_tensor=tf.exp(-s), axis=0), K.shape(distances)) / len(sigmas)


def squared_distance(x, y):
    '''Compute the pairwise euclidean distance.

    Parameters
    ----------
    x: Tensor
        Tensor with shape [batch_size, z_dim]
    y: Tensor
        Tensor with shape [batch_size, z_dim]

    Returns
    ----------
    The pairwise euclidean distance between x and y.
    '''
    r = K.expand_dims(x, axis=1)
    return K.sum(K.square(r - y), axis=-1)


def compute_mmd(x, y, kernel, **kwargs):
    """Computes Maximum Mean Discrepancy(MMD) between x and y.
    
    Parameters
    ----------
    x: Tensor
        Tensor with shape [batch_size, z_dim]
    y: Tensor
        Tensor with shape [batch_size, z_dim]
    kernel: str
        The kernel type used in MMD. It can be 'rbf', 'multi-scale-rbf' or 'raphy'.
    **kwargs: dict
        The parameters used in kernel function.
    
    Returns
    ----------
    The computed MMD between x and y
    """
    x_kernel = compute_kernel(x, x, kernel=kernel, **kwargs)
    y_kernel = compute_kernel(y, y, kernel=kernel, **kwargs)
    xy_kernel = compute_kernel(x, y, kernel=kernel, **kwargs)
    return K.mean(x_kernel) + K.mean(y_kernel) - 2 * K.mean(xy_kernel)



def _nan2zero(x):
    return tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)

def _nan2inf(x):
    return tf.where(tf.math.is_nan(x), tf.zeros_like(x) + np.inf, x)

def _nelem(x):
    nelem = tf.reduce_sum(input_tensor=tf.cast(~tf.math.is_nan(x), tf.float32))
    return tf.cast(tf.compat.v1.where(tf.equal(nelem, 0.), 1., nelem), x.dtype)

def _reduce_mean(x):
    nelem = _nelem(x)
    x = _nan2zero(x)
    return tf.divide(tf.reduce_sum(input_tensor=x), nelem)