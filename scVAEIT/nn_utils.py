import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from tensorflow.keras.layers import Layer, Dense, BatchNormalization, LeakyReLU, Lambda
use_bias = True


###########################################################################
#
# Input and output blocks for multimodal datasets
#
###########################################################################
class InputBlock(tf.keras.layers.Layer):
    def __init__(self, dim_inputs, dim_latents, dim_embed, names=None, bn=False, **kwargs):
        '''
        Parameters
        ----------
        dim_inputs : list of int
            (B+1,) The dimension of each input block, where the last block 
            is assumed to be the batch effects.
        dim_latent : list of int
            (B,) The dimension of output of first layer for each block.
        names : list of str, optional
            (B,) The name of first layer for each block.
        **kwargs : 
            Extra keyword arguments.
        '''
        super(InputBlock, self).__init__()
                
        self.dim_inputs = dim_inputs
        self.dim_embed = dim_embed
        if names is None:
            names = ['Block_{}'.format(i) for i in range(len(dim_latents))]
        self.names = names
        self.dim_latents = dim_latents
        
        self.linear_layers = [
            Dense(d, use_bias=False, activation = LeakyReLU(), name=names[i]) if d>0 else 
            tf.keras.layers.Lambda(lambda x,training: tf.identity(x))
                for i,d in enumerate(self.dim_latents)
        ]
        if bn:
            self.bn = BatchNormalization(center=False)
        else:
            self.bn = Lambda(lambda x,training: tf.identity(x))
        self.concat = tf.keras.layers.Concatenate()
        

    @tf.function
    def call(self, x, embed, batches, training=True):
        x_list = tf.split(x, self.dim_inputs, axis=1)
        embed_list = tf.split(embed, self.dim_embed, axis=1)
        outputs = self.concat([
            self.linear_layers[i](
                tf.concat([x_list[i], embed_list[i], batches], axis=1), training=training
                ) for i in range(len(self.dim_latents))])
        outputs = self.bn(outputs, training=training)
        
        return outputs



def get_dist(dist, x_hat, mask, disp, max_val=10.):    
    if dist=='NB':
        x_hat = tfp.math.clip_by_value_preserve_gradient(x_hat, -max_val, max_val)
        generative_dist = tfd.Independent(tfd.Masked(
                tfd.NegativeBinomial.experimental_from_mean_dispersion(
                    mean = x_hat,# * tf.math.log(10**4+1.), 
                    dispersion = disp, name='NB_rv'
                ), mask), reinterpreted_batch_ndims=1)

    elif dist=='ZINB':        
        # Not tested in graph mode yet
        dim = tf.cast(tf.shape(x_hat)[-1]/2, tf.int32)
        phi_rna = tf.clip_by_value(x_hat[..., dim:], 1e-5, 1.-1e-5)
        x_hat = x_hat[..., :dim]
        x_hat = tfp.math.clip_by_value_preserve_gradient(x_hat, -max_val, max_val)
        generative_dist = tfd.Independent(tfd.Masked(
            tfd.Mixture(
                cat=tfd.Categorical(
                    probs=tf.stack([phi_rna, 1.0 - phi_rna], axis=-1)),
                components=[tfd.Deterministic(loc=tf.zeros_like(phi_rna)), 
                            tfd.NegativeBinomial.experimental_from_mean_dispersion(
                                mean = x_hat,# * tf.math.log(10**4+1.),
                                dispersion = disp)],
                name='ZINB_rv'
            ), mask), reinterpreted_batch_ndims=1)

    elif dist=='Bernoulli':
        x_hat = tfp.math.clip_by_value_preserve_gradient(x_hat, 1e-5, 1.-1e-5)
        generative_dist = tfd.Independent(tfd.Masked(
            tfd.Bernoulli(
                probs = x_hat,
                dtype=tf.float32, name='Bernoulli_rv'
            ), mask), reinterpreted_batch_ndims=1)  

    elif dist=='Gaussian':
        x_hat = tfp.math.clip_by_value_preserve_gradient(x_hat, -max_val, max_val)
        generative_dist = tfd.Independent(tfd.Masked(
            tfd.Normal(
                loc = x_hat, scale = disp, name='Gaussian_rv'
            ), mask), reinterpreted_batch_ndims=1)

    elif dist=='Poisson':
        x_hat = tfp.math.clip_by_value_preserve_gradient(x_hat, -tf.inf, tf.math.log(max_val))
        generative_dist = tfd.Independent(tfd.Masked(
            tfd.Poisson(
                log_rate = x_hat, name='Poisson_rv'
            ), mask), reinterpreted_batch_ndims=1)
    return generative_dist



class OutputBlock(tf.keras.layers.Layer):
    def __init__(self, dim_outputs, dist_outputs, dim_latents, dim_embed, names=None, bn=True, **kwargs):
        '''
        Parameters
        ----------
        dim_outputs : list of int
            (B,) The dimension of each output block.
        dist_outputs : list of str
            (B,) The distribution of each output block.
        dim_latents : list of int
            (B,) The dimension of output of last layer for each block.
        names : list of str, optional
            (B,) The name of last layer for each block.
        bn : boolean
            Whether use batch normalization or not.
        **kwargs : 
            Extra keyword arguments.
        '''        
        super(OutputBlock, self).__init__()
        self.dim_inputs = dim_outputs
        self.dim_embed = dim_embed
        self.dim_outputs = [d*2 if dist_outputs[i]=='ZINB' else d for i,d in enumerate(dim_outputs)]
        self.dist_outputs = dist_outputs
        self.dim_latents = dim_latents
        if names is None:
            names = ['Block_{}'.format(i) for i in range(len(dim_latents))]
        self.names = names        
        
        self.linear_layers = [
            Dense(d, use_bias=use_bias, activation = LeakyReLU(), name=names[i]) if d>0 else 
            Lambda(lambda x,training: tf.identity(x))
                for i,d in enumerate(self.dim_latents)
        ]
        if bn:
            self.bn = [BatchNormalization(center=False) for _ in range(len(dim_latents))]
        else:
            self.bn = [Lambda(lambda x,training: tf.identity(x)) for _ in range(len(dim_latents))]
        # out_act = [None if dist=='Gaussian' else tf.nn.sigmoid for dist in self.dist_outputs]
        self.output_layers = [
            Dense(d, use_bias=use_bias, name=names[i]#, activation = out_act[i]
            ) 
            for i,d in enumerate(self.dim_outputs)
        ]
        self.out_act = [tf.identity if dist in ['Gaussian', 'Poisson'] else tf.nn.softplus for dist in self.dist_outputs]

        self.disp = [
            Dense(d, use_bias=False, activation = tf.nn.softplus, name="disp".format(names[i])) 
            if self.dist_outputs[i]!='Bernoulli' else 
            Lambda(lambda x,training: tf.zeros((1,d), dtype=tf.float32))
                for i,d in enumerate(self.dim_inputs)
        ]        
        
        self.dists = [Lambda(lambda x: get_dist(x[0], x[1], x[2], x[3], x[4])) 
                      for dist in self.dist_outputs]
        self.concat = tf.keras.layers.Concatenate()
        
        
    @tf.function
    def call(self, x, embed, masks, batches, z, x_embed, max_vals, training=True):
        '''
        Parameters
        ----------
        x : tf.Tensor
            \([B, D]\) the observed \(x\).
        z : tf.Tensor
            \([B, L, d]\) the sampled \(z\).
        batches : tf.Tensor
            \([B, b]\) the sampled \(z\).
        masks : tf.Tensor
            \([B, D]\) the mask indicating feature missing.
        training : boolean, optional
            whether in the training or inference mode.
        '''

        m_list = tf.split(tf.expand_dims(masks,1), self.dim_inputs, axis=-1)
        x_list = tf.split(tf.expand_dims(x,1), self.dim_inputs, axis=-1)
        x_emded_list = tf.split(tf.expand_dims(x_embed,1), self.dim_latents, axis=-1)
        max_vals = tf.split(tf.expand_dims(tf.expand_dims(max_vals,0),0), self.dim_inputs, axis=-1)

        L = tf.shape(z)[1]
        probs = self.concat([
            self.dists[i]([
                self.dist_outputs[i],
                self.out_act[i](
                    self.output_layers[i](self.bn[i](
                        self.linear_layers[i](z,training=training), training=training)  
                        + x_emded_list[i], training=training)
                ),
                m_list[i], 
                tf.expand_dims(
                    tfp.math.clip_by_value_preserve_gradient(
                    self.disp[i](batches, training=training), 0., 6.), 1),
                max_vals[i]
            ]).log_prob(x_list[i]) for i in range(len(self.dim_latents))
        ])

        return probs

    
    @tf.function
    def get_recon(self, embed, masks, batches, z, x_embed, max_vals, training=True):
        m_list = tf.split(tf.expand_dims(masks,1), self.dim_inputs, axis=-1)
        x_emded_list = tf.split(tf.expand_dims(x_embed,1), self.dim_latents, axis=-1)
        max_vals = tf.split(tf.expand_dims(tf.expand_dims(max_vals,0),0), self.dim_inputs, axis=-1)

        L = tf.shape(z)[1]
        x_hat = self.concat([
            self.dists[i]([
                self.dist_outputs[i],
                self.out_act[i](
                    self.output_layers[i](self.bn[i](
                        self.linear_layers[i](z, training=training), training=training)
                        + x_emded_list[i], training=training)
                ), 
                m_list[i], 
                tf.expand_dims(
                    tfp.math.clip_by_value_preserve_gradient(
                    self.disp[i](batches, training=training), 0., 6.), 1),
                max_vals[i]
            ]).mean() for i in range(len(self.dim_latents))
        ])

        return x_hat


###########################################################################
#
# Sampling layers in the latent space
#
###########################################################################
class cdf_layer(Layer):
    '''
    The Normal cdf layer with custom gradients.
    '''
    def __init__(self):
        '''
        '''
        super(cdf_layer, self).__init__()
        
    @tf.function
    def call(self, x):
        return self.func(x)
        
    @tf.custom_gradient
    def func(self, x):
        '''Return cdf(x) and pdf(x).

        Parameters
        ----------
        x : tf.Tensor
            The input tensor.
        
        Returns
        ----------
        f : tf.Tensor
            cdf(x).
        grad : tf.Tensor
            pdf(x).
        '''   
        dist = tfp.distributions.Normal(
            loc = tf.constant(0.0, tf.keras.backend.floatx()), 
            scale = tf.constant(1.0, tf.keras.backend.floatx()), 
            allow_nan_stats=False)
        f = dist.cdf(x)
        def grad(dy):
            gradient = dist.prob(x)
            return dy * gradient
        return f, grad
    

class Sampling(Layer):
    """Sampling latent variable \(z\) from \(N(\\mu_z, \\log \\sigma_z^2\)).    
    Used in Encoder.
    """
    def __init__(self, seed=0, **kwargs):
        super(Sampling, self).__init__(**kwargs)
        self.seed = seed

    @tf.function
    def call(self, z_mean, z_log_var):
        '''Return cdf(x) and pdf(x).

        Parameters
        ----------
        z_mean : tf.Tensor
            \([B, L, d]\) The mean of \(z\).
        z_log_var : tf.Tensor
            \([B, L, d]\) The log-variance of \(z\).

        Returns
        ----------
        z : tf.Tensor
            \([B, L, d]\) The sampled \(z\).
        '''   
   #     seed = tfp.util.SeedStream(self.seed, salt="random_normal")
   #     epsilon = tf.random.normal(shape = tf.shape(z_mean), seed=seed(), dtype=tf.keras.backend.floatx())
        epsilon = tf.random.normal(shape = tf.shape(z_mean), dtype=tf.keras.backend.floatx())
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        z = tf.clip_by_value(z, -1e6, 1e6)
        return z



###########################################################################
#
# Encoder
# 
###########################################################################
class Encoder(Layer):
    '''
    Encoder, model \(p(Z_i|Y_i,X_i)\).
    '''
    def __init__(self, dimensions, dim_latent, 
        dim_block_inputs, dim_block_latents, dim_embed, block_names=None, name='encoder', **kwargs):
        '''
        Parameters
        ----------
        dimensions : np.array
            The dimensions of hidden layers of the encoder.
        dim_latent : int
            The latent dimension of the encoder.
        dim_block_inputs : list of int
            (num_block,) The dimension of each input block, where the last block 
            is assumed to be the batch effects.
        dim_block_latents : list of int
            (num_block,) The dimension of output of first layer for each block.
        block_names : list of str, optional
            (num_block,) The name of first layer for each block.  
        name : str, optional
            The name of the layer.
        **kwargs : 
            Extra keyword arguments.
        ''' 
        super(Encoder, self).__init__(name = name, **kwargs)
        self.input_layer = InputBlock(dim_block_inputs, dim_block_latents, dim_embed, block_names, bn=False)
        self.dense_layers = [Dense(dim, activation = LeakyReLU(),
                                          name = 'encoder_%i'%(i+1)) \
                             for (i, dim) in enumerate(dimensions)]
        self.batch_norm_layers = [BatchNormalization(center=False) \
                                    for _ in range(len(dimensions))]
        self.batch_norm_layers.append(BatchNormalization(center=False))
        self.latent_mean = Dense(dim_latent, name = 'latent_mean')
        self.latent_log_var = Dense(dim_latent, name = 'latent_log_var')
        self.sampling = Sampling()
    
    
    @tf.function
    def call(self, x, embed, batches, L=1, training=True):
        '''Encode the inputs and get the latent variables.

        Parameters
        ----------
        x : tf.Tensor
            \([B, L, d]\) The input.
        L : int, optional
            The number of MC samples.
        training : boolean, optional
            Whether in the training or inference mode.
        
        Returns
        ----------
        z_mean : tf.Tensor
            \([B, L, d]\) The mean of \(z\).
        z_log_var : tf.Tensor
            \([B, L, d]\) The log-variance of \(z\).
        z : tf.Tensor
            \([B, L, d]\) The sampled \(z\).
        '''
        tmp = self.input_layer(x, embed, batches, training=training)
        _z = tmp
        for dense, bn in zip(self.dense_layers, self.batch_norm_layers):
            _z = dense(_z)
            _z = bn(_z, training=training)
        z_mean = self.batch_norm_layers[-1](self.latent_mean(_z), training=training)
        z_log_var = self.latent_log_var(_z)
        _z_mean = tf.tile(tf.expand_dims(z_mean, 1), (1,L,1))
        _z_log_var = tf.tile(tf.expand_dims(z_log_var, 1), (1,L,1))
        z = self.sampling(_z_mean, _z_log_var)
        return z_mean, z_log_var, z, tmp



###########################################################################
#
# Decoder
# 
###########################################################################
class Decoder(Layer):
    '''
    Decoder, model \(p(Y_i|Z_i,X_i)\).
    '''
    def __init__(self, dimensions, dim_block_outputs, 
        dist_block_outputs, dim_block_latents, dim_embed, max_vals,
        block_names=None, name = 'decoder', **kwargs):
        '''
        Parameters
        ----------
        dimensions : np.array
            The dimensions of hidden layers of the encoder.
        dim_block_outputs : list of int
            (B,) The dimension of each output block.
        dist_block_outputs : list of str
            (B,) `'NB'`, `'ZINB'`, `'Bernoulli'` or `'Gaussian'`.
        dim_block_latents : list of int
            (B,) The dimension of output of last layer for each block.
        dim_embed : int
            The dimension of the embedding layer.
        max_vals : np.array
            The maximum values of the outputs.
        block_names : list of str, optional
            (B,) The name of last layer for each block.
        name : str, optional
            The name of the layer.
        '''
        super(Decoder, self).__init__(name = name, **kwargs)
        self.output_layer = OutputBlock(
            dim_block_outputs, dist_block_outputs, dim_block_latents, dim_embed, block_names, bn=False)

        self.dense_layers = [Dense(dim, activation = LeakyReLU(),
                                          name = 'decoder_%i'%(i+1)) \
                             for (i,dim) in enumerate(dimensions)]
        self.batch_norm_layers = [BatchNormalization(center=False) \
                                    for _ in range(len((dimensions)))]
        self.max_vals = max_vals

       
    @tf.function
    def call(self, x, embed, masks, batches, z, x_embed, training=True, return_prob=True):
        '''Decode the latent variables and get the reconstructions.

        Parameters
        ----------
        z : tf.Tensor
            \([B, L, d]\) the sampled \(z\).
        training : boolean, optional
            whether in the training or inference mode.

        Returns
        ----------
        log_probs : tf.Tensor
            \([B, block]\) The log probability.
        '''
        L = tf.shape(z)[1]
        _z = tf.concat([
            z, 
            # tf.tile(tf.expand_dims(tf.concat([embed,batches], axis=-1), 1), (1,L,1))
            tf.tile(tf.expand_dims(batches, 1), (1,L,1))
        ], axis=-1)
        for dense, bn in zip(self.dense_layers, self.batch_norm_layers):
            _z = dense(_z)
            _z = bn(_z, training=training)

        if return_prob:
            log_probs = self.output_layer(x, embed, masks, batches, _z, x_embed, self.max_vals, training=training)
            return log_probs
        else:
            x_hat = self.output_layer.get_recon(embed, masks, batches, _z, x_embed, self.max_vals, training=training)
            return x_hat

