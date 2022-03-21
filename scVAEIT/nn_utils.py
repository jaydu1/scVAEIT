import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, BatchNormalization, LeakyReLU

use_bias = True


# def softplus_std(x):
#     std = tf.math.reduce_std(x, axis=-1, keepdims=True)
#     std = tf.where(std==0, 1., std)
#     return tf.nn.softplus(x/std)*std

class MemoryLayer(Layer):
    storage = {}
    def __init__(self, store_id, output=False, add=True):
        super(MemoryLayer, self).__init__()
        self.store_id = store_id
        self.out = output
        self.add = add

    def call(self, input_tensor):
        if not self.out:
            self.storage[self.store_id] = input_tensor
            return input_tensor
        else:
            if self.store_id not in self.storage:
                raise ValueError('MemoryLayer: {} is not initialized.'.format(self.store_id))
            stored = self.storage[self.store_id]
            if not self.add:
                data = tf.concat([input_tensor, stored], axis=-1)
            else:
                data = input_tensor + stored
            return data


class SkipConnection(Layer):
    def __init__(self, layers):
        super(SkipConnection, self).__init__()
        self.inner_net = tf.keras.Sequential(layers)

    def call(self, input_tensor):
        return input_tensor + self.inner_net(input_tensor)


def mlp_block(dim):
    return tf.keras.Sequential([
        SkipConnection([
            BatchNormalization(center=use_bias),
            LeakyReLU(),
            Dense(dim, use_bias=use_bias)            
        ])
    ])


def MLPBlocks(dim, n):
    return tf.keras.Sequential([mlp_block(dim) for _ in range(n)])



def proposal_network(dim_latent):
    return tf.keras.Sequential([
        Dense(256, use_bias=use_bias),
        BatchNormalization(center=use_bias),
        LeakyReLU(),
#         Dense(32, use_bias=use_bias),
#         MemoryLayer('#1'),
#         BatchNormalization(center=use_bias),
#         LeakyReLU(),
        Dense(dim_latent, use_bias=use_bias),
#         MLPBlocks(dim_latent, 1)
    ])


def generative_network(dim_output):
    return tf.keras.Sequential([        
#         Dense(32, use_bias=use_bias),
#         MemoryLayer('#1', True),
#         BatchNormalization(center=use_bias),
#         LeakyReLU(),
        Dense(256, use_bias=use_bias),        
        MemoryLayer('#0', True),
        BatchNormalization(center=use_bias),
        LeakyReLU(),
        Dense(dim_output, use_bias=use_bias)
    ])


def prior_network(dim_latent):
    return tf.keras.Sequential([        
        Dense(256, use_bias=use_bias),
        MemoryLayer('#0'),        
#         BatchNormalization(center=use_bias),
#         LeakyReLU(),
#         Dense(32, use_bias=use_bias),
#         MemoryLayer('#1'),
        BatchNormalization(center=use_bias),
        LeakyReLU(),
        Dense(dim_latent, use_bias=use_bias),
    ])      
