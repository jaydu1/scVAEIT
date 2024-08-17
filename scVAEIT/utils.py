import os
import random
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp



def reset_random_seeds(seed):
    os.environ['PYTHONHASHSEED']=str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def check_arr_type(arr, dtype):    
    if np.isscalar(arr):
        arr = np.array([arr], dtype=dtype)
    else:
        arr = np.array(arr, dtype=dtype)
    return arr


class MaskGenerator(object):
    def __init__(self, p=0.95):
        self.p = p

    def __call__(self, inputs):
        # (batch_size, num_features)
        mask = np.random.choice(2, size=inputs.shape,
                                p=[1 - self.p, self.p]).astype(tf.keras.backend.floatx())
        return mask


class MixtureMaskGenerator(object):

    def __init__(self, generators, weights):
        self.generators = generators
        self.weights = np.array(weights/np.sum(weights)).astype(tf.keras.backend.floatx())

    def __call__(self, inputs, p=None):
        c_ids = np.random.choice(len(self.weights), inputs.shape[0], True, self.weights)
        mask = np.zeros_like(inputs)

        for i, gen in enumerate(self.generators):
            ids = np.where(c_ids == i)[0]
            if len(ids) == 0:
                continue
            samples = gen(tf.gather(inputs, ids, axis=0), p)
            mask[ids] = samples
        return mask    
    

class FixedGenerator(object):
    def __init__(self, masks, p=None):
        self.masks = masks
        self.n = self.masks.shape[0]
        if p is None:
            self.p = np.ones(self.n)/self.n
        else:
            self.p = p

    def __call__(self, inputs, mask=None, p=None):
        i_masks = np.random.choice(self.n, size=inputs.shape[0],
                                p=self.p).astype(int)
        
        return self.masks[i_masks,:].copy()
    
    
class ModalMaskGenerator(object):
    def __init__(self, dim_arr, p_feat=0.05, p_modal=None):
        self.p_feat = tf.constant(p_feat, dtype=tf.keras.backend.floatx())
        self.dim_arr = tf.constant(dim_arr)
        self.segment_ids = tf.constant(np.repeat(np.arange(len(dim_arr)), dim_arr), dtype=tf.int32)
        if p_modal is None:
            p_modal = np.array(dim_arr, dtype=float)
            p_modal /= np.sum(p_modal)
        self.p_modal = tf.constant(p_modal, dtype=tf.keras.backend.floatx())

    @tf.function(reduce_retracing=True)
    def __call__(self, inputs, missing_mask=None, p=None):
        if p is None:
            p = self.p_feat

        # (batch_size, dim_features)
        mask = tfp.distributions.Bernoulli(probs=p, dtype=tf.keras.backend.floatx()).sample(sample_shape=tf.shape(inputs))
        
        # No random missing
        mask_modal = tfp.distributions.Bernoulli(probs=0.5).sample(sample_shape=(tf.shape(inputs)[0],1))
        mask = tf.where(mask_modal==int(0), tf.constant(0., dtype=tf.keras.backend.floatx()), mask)
        
        # Modality missing
        if missing_mask is None:
            missing_mask = tf.zeros_like(inputs)

        if len(self.dim_arr)>1:
            mask_modal = tfp.distributions.Categorical(probs=self.p_modal, dtype=tf.int32).sample(sample_shape=(tf.shape(inputs)[0]))
            has_modal = tf.transpose(
                tf.math.segment_sum(tf.transpose(missing_mask+1), self.segment_ids))
            for i in tf.range(len(self.dim_arr), dtype=tf.int32):
                condition = tf.logical_and(
                  tf.equal(mask_modal, i),
                  tf.reduce_any(
                      tf.where(tf.expand_dims(tf.range(tf.shape(has_modal)[1])!=i, 0), has_modal, 0)>0, axis=-1)
                )

                mask = tf.where(tf.logical_and(
                    tf.expand_dims(condition,-1), tf.expand_dims(tf.equal(self.segment_ids, i),0)),
                    1., mask)
            
        mask = tf.where(missing_mask==-1., -1., mask)
        
        return mask
    
    
class Early_Stopping():
    '''
    The early-stopping monitor.
    '''
    def __init__(self, warmup=0, patience=10, tolerance=1e-3, 
            relative=False, is_minimize=True):
        self.warmup = warmup
        self.patience = patience
        self.tolerance = tolerance
        self.is_minimize = is_minimize
        self.relative = relative

        self.step = 0
        self.best_step = 0
        self.best_metric = np.inf

        if not self.is_minimize:
            self.factor = -1.0
        else:
            self.factor = 1.0

    def __call__(self, metric):
        self.step += 1
        
        if self.step < self.warmup:
            return False
        elif (self.best_metric==np.inf) or \
                (self.relative and (self.best_metric-metric)/self.best_metric > self.tolerance) or \
                ((not self.relative) and self.factor*metric<self.factor*self.best_metric-self.tolerance):
            self.best_metric = metric
            self.best_step = self.step
            return False
        elif self.step - self.best_step>self.patience:
            print('Best Epoch: %d. Best Metric: %f.'%(self.best_step, self.best_metric))
            return True
        else:
            return False    
