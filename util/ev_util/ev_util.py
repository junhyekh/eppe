"""
Modified from https://github.com/FlyingGiraffe/vnn-pc
"""

import numpy as np
# import torch
import einops
import jax.numpy as jnp
import jax

# from ev_utils.rotm_util import Y_func_V2

EPS = 1e-6

def knn(x:jnp.ndarray, k):
    '''
    (... P F)
    '''

    # pairwise_distance = jnp.sum((x[...,None,:] - x[...,None,:,:])**2, axis=-1)

    inner = -2*jnp.matmul(x, x.swapaxes(-1, -2)) # (... P P)
    xx:jnp.ndarray = jnp.sum(x**2, axis=-1, keepdims=True) # (... P 1)
    pairwise_distance:jnp.ndarray = xx + inner + xx.swapaxes(-1, -2)
    
    return jnp.argsort(pairwise_distance, axis=-1)[...,:k]


def get_graph_feature(x, k=20, cross=False, idx_out=False):
    nf = x.shape[-2]
    x = einops.rearrange(x, '... f d -> ... (f d)')
    idx = knn(x, k=k)   # (batch_size, num_points, k)
    feature = jnp.take_along_axis(x[...,None,:], idx[...,None], axis=-3) # (B, P, K, FD)
    x = einops.repeat(x, '... j -> ... r j', r=k)
    if cross:
        cross = jnp.cross(feature, x)
        feature = jnp.stack((feature-x, x, cross), axis=-1)
    else:
        feature = jnp.stack((feature-x, x), axis=-1)
    feature = einops.rearrange(feature, '... (f d) k -> ... f (d k)', f=nf)
    if idx_out:
        return feature, idx
    else:
        return feature


def max_norm_pooling(x:jnp.ndarray, axis=-3, keepdims=False):
    '''
    (... V F D)
    
    '''
    x_norm = jnp.linalg.norm(x, axis=-2, keepdims=True) # feature dim
    maxv_idx = jnp.argmax(x_norm, axis=axis, keepdims=True) # max axis
    if keepdims:
        return jnp.take_along_axis(x, maxv_idx, axis)
    else:
        return jnp.take_along_axis(x, maxv_idx, axis).squeeze(axis)


def reduce_top_k_emb(x:jnp.ndarray, ratio):
    '''
    (... F D)
    
    '''
    x_norm = jnp.linalg.norm(x, axis=-2) # feature dim
    # maxv_idx = jnp.argsort(-x_norm, axis=-1, keepdims=True) # max axis
    embs_norm_topidx = -jnp.sort(-x_norm, axis=-1)
    return jnp.where((x_norm>embs_norm_topidx[..., int(x.shape[-1]*ratio)][...,None])[...,None,:], x, 0)



if __name__ == '__main__':
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    jkey = jax.random.PRNGKey(0)
    x = jax.random.normal(jkey, shape=(10,3,1))
    
    feat, idx = get_graph_feature(x, k=4, cross=True, idx_out=True)
    import util.transform_util as trutil
    randquat = trutil.qrand((1,))
    randR = trutil.q2R(randquat)
    
    featrot2 = jnp.einsum('...ij,...kjd->...kid', randR, feat)
    rotx = jnp.einsum('...ij,...jd->...id', randR, x)
    featrot, idx2 = get_graph_feature(rotx, k=4, cross=True, idx_out=True)

    print(1)