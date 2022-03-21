import numpy as np
import tensorflow as tf

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

    def __call__(self, inputs, p=None):
        i_masks = np.random.choice(self.n, size=inputs.shape[0],
                                p=self.p).astype(int)
        
        return self.masks[i_masks,:].copy()
    
    
class BiModalMaskGenerator(object):
    def __init__(self, dim_rna, dim_adt, p_feat=0.05, p_modal=None):
        self.p_feat = p_feat
        self.dim_rna = dim_rna
        self.dim_adt = dim_adt 
        if p_modal is None:
            p_modal = np.array([dim_rna, dim_adt], dtype=float)
            p_modal /= np.sum(p_modal)
        self.p_modal = p_modal

    def __call__(self, inputs, p=None):
        if p is None:
            p = self.p_feat
        # (batch_size, dim_rna + dim_adt)
        mask = np.random.choice(2, size=inputs.shape,
                                p=[1 - p, p]).astype(tf.keras.backend.floatx())
        
        # No random missing
        mask_modal = np.random.choice(2, size=(inputs.shape[0], ), p=self.p_modal)
        mask[mask_modal==0, :] = 0.

        # Modality missing
        mask_modal = np.random.choice(2, size=(inputs.shape[0], ), p=self.p_modal)
        mask[mask_modal==0, :self.dim_rna] = 1.
        mask[mask_modal==1, self.dim_rna:] = 1.
        
        return mask
    
    
class BiModalandFixMaskGenerator(object):
    def __init__(self, masks, dim_rna, dim_adt, p_feat=0.05, p_modal=0.5):
        p_modal = np.array([p_modal,1-p_modal])
        self.bm_gen = BiModalMaskGenerator(dim_rna, dim_adt, p_feat, p_modal)
        if masks is None:
            self.generator = self.bm_gen
        else:
            self.f_gen = FixedGenerator(masks)
            self.generator = MixtureMaskGenerator([self.f_gen, self.bm_gen], [0.1, 0.9])
        
    def __call__(self, inputs, p=None):
        return self.generator(inputs, p)
    
#     def __init__(self, fixed_masks, dim_rna, dim_adt, 
#                  p_feat=0.1, p_modal=np.ones(4)/4, p_fixed=0.1):
#         self.p_feat = p_feat
#         self.p_modal = p_modal
#         self.dim_rna = dim_rna
#         self.dim_adt = dim_adt 
#         self.fixed_mask = fixed_mask
#         self.p_fixed = p_fixed
#         self.n_fixed = self.fixed_mask.shape[0]

#     def __call__(self, inputs):       
#         # (batch_size, dim_rna + dim_adt)
#         mask = np.random.choice(2, size=inputs.shape,
#                                 p=[1 - self.p_feat, self.p_feat]).astype(tf.keras.backend.floatx())

#         mask_modal = np.random.choice(4, size=(inputs.shape[0], ), p=self.p_modal)
#         mask[mask_modal==0, :] = 0.
#         mask[mask_modal==2, :self.dim_rna] = 1.
#         mask[mask_modal==3, self.dim_rna:] = 1.
        
#         return mask    
    
    
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
