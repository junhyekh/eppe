from PCN.models.pcn import PCN
import torch

device = "cuda:0"

import einops
import os
import numpy as np
import jax
import jax.numpy as jnp
import glob
import pickle
from torch.utils.data import DataLoader, Dataset

class OccDataDirLoader(Dataset):
    def __init__(self, dataset_dir=os.path.join('/home/user/cs479/eppe/data/partial_occ'), eval_type='test'):
        ds_dir_list = glob.glob(os.path.join(dataset_dir, '*.pkl'))
        def replace_np_ele(x, y, ns, i):
            x[i] = y
            return x

        print('start ds loading ' + eval_type)
        for i, dsdir in enumerate(ds_dir_list):
            with open(dsdir, "rb") as f:
                loaded = pickle.load(f)
            if i ==0:
                ns = 1
                self.entire_ds = jax.tree_map(lambda x: np.concatenate([x[None], np.zeros_like(einops.repeat(x, 'i ... -> r i ...', r=len(ds_dir_list)-1))], 0), loaded)
            else:
                self.entire_ds = jax.tree_map(lambda x,y: replace_np_ele(x, y, ns, i), self.entire_ds,loaded)
        print('end ds loading ' + eval_type)

        # viewpoint devision for test (not shape)
        nvp = self.entire_ds[0].shape[1]
        if eval_type=='test':
            # self.entire_ds = jax.tree_map(lambda x: x[:,:nvp//8], self.entire_ds)
            pass
        else:
            self.entire_ds = jax.tree_map(lambda x: x[:,nvp//8:], self.entire_ds)

    def __len__(self):
        return jax.tree_util.tree_flatten(self.entire_ds)[0][0].shape[0]

    def __getitem__(self, index):

        dpnts = jax.tree_map(lambda x: x[index], self.entire_ds)
        partial_spnts, seg, qps, occ = dpnts

        idx_vp = np.random.randint(0, partial_spnts.shape[0], size=(16)) # nvp = 16
        idx_qps = np.random.randint(0, qps.shape[1], size=(1,)) # batch_size = 10

        qps_res, occ_res = qps[idx_vp][:,idx_qps], occ[idx_vp][:,idx_qps]
        pspnts_res, seg_res = partial_spnts, seg
        return (pspnts_res.astype(np.float32), seg_res.astype(np.int32), qps_res.astype(np.float32), occ_res.astype(np.float32))
    
    from torch.utils.data import DataLoader, Dataset
import jax
import jax.numpy as jnp

eval_dataset = DataLoader(OccDataDirLoader(eval_type='test'), batch_size=1, shuffle=False, num_workers=1, drop_last=True)

for i, ds in enumerate(eval_dataset):
    partial = ds[0]

    full = partial.reshape(*partial.shape[:2],-1,3)
    rand_idx = torch.randint(0, 10000, (1,16,20))
    break

""" sinc(t) := sin(t) / t """
import torch
from torch import sin, cos

def sinc1(t):
    """ sinc1: t -> sin(t)/t """
    e = 0.01
    r = torch.zeros_like(t)
    a = torch.abs(t)

    s = a < e
    c = (s == 0)
    t2 = t[s] ** 2
    r[s] = 1 - t2/6*(1 - t2/20*(1 - t2/42))  # Taylor series O(t^8)
    r[c] = sin(t[c]) / t[c]

    return r

def sinc1_dt(t):
    """ d/dt(sinc1) """
    e = 0.01
    r = torch.zeros_like(t)
    a = torch.abs(t)

    s = a < e
    c = (s == 0)
    t2 = t ** 2
    r[s] = -t[s]/3*(1 - t2[s]/10*(1 - t2[s]/28*(1 - t2[s]/54)))  # Taylor series O(t^8)
    r[c] = cos(t[c])/t[c] - sin(t[c])/t2[c]

    return r

def sinc1_dt_rt(t):
    """ d/dt(sinc1) / t """
    e = 0.01
    r = torch.zeros_like(t)
    a = torch.abs(t)

    s = a < e
    c = (s == 0)
    t2 = t ** 2
    r[s] = -1/3*(1 - t2[s]/10*(1 - t2[s]/28*(1 - t2[s]/54)))  # Taylor series O(t^8)
    r[c] = (cos(t[c]) / t[c] - sin(t[c]) / t2[c]) / t[c]

    return r


def rsinc1(t):
    """ rsinc1: t -> t/sinc1(t) """
    e = 0.01
    r = torch.zeros_like(t)
    a = torch.abs(t)

    s = a < e
    c = (s == 0)
    t2 = t[s] ** 2
    r[s] = (((31*t2)/42 + 7)*t2/60 + 1)*t2/6 + 1  # Taylor series O(t^8)
    r[c] = t[c] / sin(t[c])

    return r

def rsinc1_dt(t):
    """ d/dt(rsinc1) """
    e = 0.01
    r = torch.zeros_like(t)
    a = torch.abs(t)

    s = a < e
    c = (s == 0)
    t2 = t[s] ** 2
    r[s] = ((((127*t2)/30 + 31)*t2/28 + 7)*t2/30 + 1)*t[s]/3  # Taylor series O(t^8)
    r[c] = 1/sin(t[c]) - (t[c]*cos(t[c]))/(sin(t[c])*sin(t[c]))

    return r

def rsinc1_dt_csc(t):
    """ d/dt(rsinc1) / sin(t) """
    e = 0.01
    r = torch.zeros_like(t)
    a = torch.abs(t)

    s = a < e
    c = (s == 0)
    t2 = t[s] ** 2
    r[s] = t2*(t2*((4*t2)/675 + 2/63) + 2/15) + 1/3  # Taylor series O(t^8)
    r[c] = (1/sin(t[c]) - (t[c]*cos(t[c]))/(sin(t[c])*sin(t[c]))) / sin(t[c])

    return r


def sinc2(t):
    """ sinc2: t -> (1 - cos(t)) / (t**2) """
    e = 0.01
    r = torch.zeros_like(t)
    a = torch.abs(t)

    s = a < e
    c = (s == 0)
    t2 = t ** 2
    r[s] = 1/2*(1-t2[s]/12*(1-t2[s]/30*(1-t2[s]/56)))  # Taylor series O(t^8)
    r[c] = (1-cos(t[c]))/t2[c]

    return r

def sinc2_dt(t):
    """ d/dt(sinc2) """
    e = 0.01
    r = torch.zeros_like(t)
    a = torch.abs(t)

    s = a < e
    c = (s == 0)
    t2 = t ** 2
    r[s] = -t[s]/12*(1 - t2[s]/5*(1.0/3 - t2[s]/56*(1.0/2 - t2[s]/135)))  # Taylor series O(t^8)
    r[c] = sin(t[c])/t2[c] - 2*(1-cos(t[c]))/(t2[c]*t[c])

    return r


def sinc3(t):
    """ sinc3: t -> (t - sin(t)) / (t**3) """
    e = 0.01
    r = torch.zeros_like(t)
    a = torch.abs(t)

    s = a < e
    c = (s == 0)
    t2 = t[s] ** 2
    r[s] = 1/6*(1-t2/20*(1-t2/42*(1-t2/72)))  # Taylor series O(t^8)
    r[c] = (t[c]-sin(t[c]))/(t[c]**3)

    return r

def sinc3_dt(t):
    """ d/dt(sinc3) """
    e = 0.01
    r = torch.zeros_like(t)
    a = torch.abs(t)

    s = a < e
    c = (s == 0)
    t2 = t[s] ** 2
    r[s] = -t[s]/60*(1 - t2/21*(1 - t2/24*(1.0/2 - t2/165)))  # Taylor series O(t^8)
    r[c] = (3*sin(t[c]) - t[c]*(cos(t[c]) + 2))/(t[c]**4)

    return r


def sinc4(t):
    """ sinc4: t -> 1/t^2 * (1/2 - sinc2(t))
                  = 1/t^2 * (1/2 - (1 - cos(t))/t^2)
    """
    e = 0.01
    r = torch.zeros_like(t)
    a = torch.abs(t)

    s = a < e
    c = (s == 0)
    t2 = t ** 2
    r[s] = 1/24*(1-t2/30*(1-t2/56*(1-t2/90)))  # Taylor series O(t^8)
    r[c] = (0.5 - (1 - cos(t))/t2) / t2


class Sinc1_autograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta):
        ctx.save_for_backward(theta)
        return sinc1(theta)

    @staticmethod
    def backward(ctx, grad_output):
        theta, = ctx.saved_tensors
        grad_theta = None
        if ctx.needs_input_grad[0]:
            grad_theta = grad_output * sinc1_dt(theta).to(grad_output)
        return grad_theta

Sinc1 = Sinc1_autograd.apply

class RSinc1_autograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta):
        ctx.save_for_backward(theta)
        return rsinc1(theta)

    @staticmethod
    def backward(ctx, grad_output):
        theta, = ctx.saved_tensors
        grad_theta = None
        if ctx.needs_input_grad[0]:
            grad_theta = grad_output * rsinc1_dt(theta).to(grad_output)
        return grad_theta

RSinc1 = RSinc1_autograd.apply

class Sinc2_autograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta):
        ctx.save_for_backward(theta)
        return sinc2(theta)

    @staticmethod
    def backward(ctx, grad_output):
        theta, = ctx.saved_tensors
        grad_theta = None
        if ctx.needs_input_grad[0]:
            grad_theta = grad_output * sinc2_dt(theta).to(grad_output)
        return grad_theta

Sinc2 = Sinc2_autograd.apply

class Sinc3_autograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta):
        ctx.save_for_backward(theta)
        return sinc3(theta)

    @staticmethod
    def backward(ctx, grad_output):
        theta, = ctx.saved_tensors
        grad_theta = None
        if ctx.needs_input_grad[0]:
            grad_theta = grad_output * sinc3_dt(theta).to(grad_output)
        return grad_theta

Sinc3 = Sinc3_autograd.apply




def cross_prod(x, y):
    z = torch.cross(x.view(-1, 3), y.view(-1, 3), dim=1).view_as(x)
    return z

def liebracket(x, y):
    return cross_prod(x, y)

def mat(x):
    # size: [*, 3] -> [*, 3, 3]
    x_ = x.view(-1, 3)
    x1, x2, x3 = x_[:, 0], x_[:, 1], x_[:, 2]
    O = torch.zeros_like(x1)

    X = torch.stack((
        torch.stack((O, -x3, x2), dim=1),
        torch.stack((x3, O, -x1), dim=1),
        torch.stack((-x2, x1, O), dim=1)), dim=1)
    return X.view(*(x.size()[0:-1]), 3, 3)

def vec(X):
    X_ = X.view(-1, 3, 3)
    x1, x2, x3 = X_[:, 2, 1], X_[:, 0, 2], X_[:, 1, 0]
    x = torch.stack((x1, x2, x3), dim=1)
    return x.view(*X.size()[0:-2], 3)

def genvec():
    return torch.eye(3)

def genmat():
    return mat(genvec())

def RodriguesRotation(x):
    # for autograd
    w = x.view(-1, 3)
    t = w.norm(p=2, dim=1).view(-1, 1, 1)
    W = mat(w)
    S = W.bmm(W)
    I = torch.eye(3).to(w)

    # Rodrigues' rotation formula.
    #R = cos(t)*eye(3) + sinc1(t)*W + sinc2(t)*(w*w');
    #R = eye(3) + sinc1(t)*W + sinc2(t)*S

    R = I + sinc.Sinc1(t)*W + sinc.Sinc2(t)*S

    return R.view(*(x.size()[0:-1]), 3, 3)

def exp(x):
    w = x.view(-1, 3)
    t = w.norm(p=2, dim=1).view(-1, 1, 1)
    W = mat(w)
    S = W.bmm(W)
    I = torch.eye(3).to(w)

    # Rodrigues' rotation formula.
    #R = cos(t)*eye(3) + sinc1(t)*W + sinc2(t)*(w*w');
    #R = eye(3) + sinc1(t)*W + sinc2(t)*S

    R = I + sinc1(t)*W + sinc2(t)*S

    return R.view(*(x.size()[0:-1]), 3, 3)

def inverse(g):
    R = g.view(-1, 3, 3)
    Rt = R.transpose(1, 2)
    return Rt.view_as(g)

def btrace(X):
    # batch-trace: [B, N, N] -> [B]
    n = X.size(-1)
    X_ = X.view(-1, n, n)
    tr = torch.zeros(X_.size(0)).to(X)
    for i in range(tr.size(0)):
        m = X_[i, :, :]
        tr[i] = torch.trace(m)
    return tr.view(*(X.size()[0:-2]))

def log(g):
    eps = 1.0e-7
    R = g.view(-1, 3, 3)
    tr = btrace(R)
    c = (tr - 1) / 2
    t = torch.acos(c)
    sc = sinc1(t)
    idx0 = (torch.abs(sc) <= eps)
    idx1 = (torch.abs(sc) > eps)
    sc = sc.view(-1, 1, 1)

    X = torch.zeros_like(R)
    if idx1.any():
        X[idx1] = (R[idx1] - R[idx1].transpose(1, 2)) / (2*sc[idx1])

    if idx0.any():
        # t[idx0] == math.pi
        t2 = t[idx0] ** 2
        A = (R[idx0] + torch.eye(3).type_as(R).unsqueeze(0)) * t2.view(-1, 1, 1) / 2
        aw1 = torch.sqrt(A[:, 0, 0])
        aw2 = torch.sqrt(A[:, 1, 1])
        aw3 = torch.sqrt(A[:, 2, 2])
        sgn_3 = torch.sign(A[:, 0, 2])
        sgn_3[sgn_3 == 0] = 1
        sgn_23 = torch.sign(A[:, 1, 2])
        sgn_23[sgn_23 == 0] = 1
        sgn_2 = sgn_23 * sgn_3
        w1 = aw1
        w2 = aw2 * sgn_2
        w3 = aw3 * sgn_3
        w = torch.stack((w1, w2, w3), dim=-1)
        W = mat(w)
        X[idx0] = W

    x = vec(X.view_as(g))
    return x

def transform(g, a):
    # g in SO(3):  * x 3 x 3
    # a in R^3:    * x 3[x N]
    if len(g.size()) == len(a.size()):
        b = g.matmul(a)
    else:
        b = g.matmul(a.unsqueeze(-1)).squeeze(-1)
    return b

def group_prod(g, h):
    # g, h : SO(3)
    g1 = g.matmul(h)
    return g1



def vecs_Xg_ig(x):
    """ Vi = vec(dg/dxi * inv(g)), where g = exp(x)
        (== [Ad(exp(x))] * vecs_ig_Xg(x))
    """
    t = x.view(-1, 3).norm(p=2, dim=1).view(-1, 1, 1)
    X = mat(x)
    S = X.bmm(X)
    #B = x.view(-1,3,1).bmm(x.view(-1,1,3))  # B = x*x'
    I = torch.eye(3).to(X)

    #V = sinc1(t)*eye(3) + sinc2(t)*X + sinc3(t)*B
    #V = eye(3) + sinc2(t)*X + sinc3(t)*S

    V = I + sinc2(t)*X + sinc3(t)*S

    return V.view(*(x.size()[0:-1]), 3, 3)

def inv_vecs_Xg_ig(x):
    """ H = inv(vecs_Xg_ig(x)) """
    t = x.view(-1, 3).norm(p=2, dim=1).view(-1, 1, 1)
    X = mat(x)
    S = X.bmm(X)
    I = torch.eye(3).to(x)

    e = 0.01
    eta = torch.zeros_like(t)
    s = (t < e)
    c = (s == 0)
    t2 = t[s] ** 2
    eta[s] = ((t2/40 + 1)*t2/42 + 1)*t2/720 + 1/12 # O(t**8)
    eta[c] = (1 - (t[c]/2) / torch.tan(t[c]/2)) / (t[c]**2)

    H = I - 1/2*X + eta*S
    return H.view(*(x.size()[0:-1]), 3, 3)


class ExpMap(torch.autograd.Function):
    """ Exp: so(3) -> SO(3)
    """
    @staticmethod
    def forward(ctx, x):
        """ Exp: R^3 -> M(3),
            size: [B, 3] -> [B, 3, 3],
              or  [B, 1, 3] -> [B, 1, 3, 3]
        """
        ctx.save_for_backward(x)
        g = exp(x)
        return g

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        g = exp(x)
        gen_k = genmat().to(x)
        #gen_1 = gen_k[0, :, :]
        #gen_2 = gen_k[1, :, :]
        #gen_3 = gen_k[2, :, :]

        # Let z = f(g) = f(exp(x))
        # dz = df/dgij * dgij/dxk * dxk
        #    = df/dgij * (d/dxk)[exp(x)]_ij * dxk
        #    = df/dgij * [gen_k*g]_ij * dxk

        dg = gen_k.matmul(g.view(-1, 1, 3, 3))
        # (k, i, j)
        dg = dg.to(grad_output)

        go = grad_output.contiguous().view(-1, 1, 3, 3)
        dd = go * dg
        grad_input = dd.sum(-1).sum(-1)

        return grad_input

Exp = ExpMap.apply


#EOF

def exp(x):
    w = x.view(-1, 3)
    t = w.norm(p=2, dim=1).view(-1, 1, 1)
    W = mat(w)
    S = W.bmm(W)
    I = torch.eye(3).to(w)

    # Rodrigues' rotation formula.
    #R = cos(t)*eye(3) + sinc1(t)*W + sinc2(t)*(w*w');
    #R = eye(3) + sinc1(t)*W + sinc2(t)*S

    R = I + sinc1(t)*W + sinc2(t)*S

    return R.view(*(x.size()[0:-1]), 3, 3)

def gen_randrot(mag_max=None, mag_random=True):
    # tensor: [N, 3]
    mag_max = 180 if mag_max is None else mag_max
    amp = torch.rand(1) if mag_random else 1.0
    deg = amp * mag_max
    w = torch.randn(1, 3)
    w = w / w.norm(p=2, dim=1, keepdim=True) * deg * np.pi / 180

    g = exp(w)      # [1, 3, 3]
    g = g.squeeze(0)    # [3, 3]
    return g, deg


import argparse
import os, sys
import pandas as pd
import torch
import torch.utils.data
import numpy as np
from tqdm import tqdm

import sys
sys.path.append('/home/user/cs479/eppe/baeline')

import im2mesh.config as bae_conifg
import im2mesh.data
import im2mesh.common
from im2mesh.checkpoints import CheckpointIO
import random
import torch.backends.cudnn as cudnn
from registration.register_utils import *



cfg = im2mesh.config.load_config(
    path = "/home/user/cs479/eppe/baeline/configs/registration/vnn_pointnet_resnet_registration.yaml", 
    default_path = '/home/user/cs479/eppe/baeline/configs/registration/default_registration.yaml')
device = torch.device(device)

# Seed control
SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(SEED)
g = torch.Generator()
g.manual_seed(SEED)

# Shorthands
out_dir = cfg['training']['out_dir']
out_file = os.path.join(out_dir, 'eval_full.pkl')
out_file_class = os.path.join(out_dir, 'eval_180.csv')

model = bae_conifg.get_model(cfg, device=device, dataset=None)

# Checkpoint module init + load checkpoint
checkpoint_io = CheckpointIO(out_dir, model=model)
try:
    checkpoint_io.load(cfg['test']['model_file'])
except FileExistsError:
    print('Model file does not exist. Exiting.')
    exit()


# Evaluate
model.eval()
eval_dicts = []  
eval = 0

# Input to numpy for visualization
for i, ds in enumerate(eval_dataset):
    R, _ = gen_randrot()

    partial = ds[0]

    full = partial.reshape(*partial.shape[:2],-1,3)
    rand_idx = torch.randint(0, 10000, (1,16,20))
    sampled_full = torch.take_along_dim(full, rand_idx[...,None], dim=2)
    sampled_full = sampled_full.reshape(sampled_full.shape[0], -1, 3)

    rand_idx = torch.randint(0, 10000, (1,16,1024))
    sampled_partial = torch.take_along_dim(full, rand_idx[...,None], dim=2)
    rand_idx = torch.randint(0, 16, (1,2))
    sampled_partial = torch.take_along_dim(sampled_partial, rand_idx[...,None,None], dim=1)
    sampled_partial = sampled_partial.reshape(sampled_partial.shape[0], -1, 3)

    R, _ = gen_randrot()
    rotated_sampled_partial = sampled_partial @ (R.T)

    pcn = PCN(16384, 1024, 4).to("cuda:0")
    pcn.load_state_dict(torch.load("/home/user/cs479/eppe/PCN/checkpoint/best_l1_cd.pth"))
    pcn.eval()

    estimated_full = pcn(rotated_sampled_partial.cuda())[0]

    estimated_full.shape # B, N, 3
    rand_idx = torch.randint(0, estimated_full.shape[1], (320,), device=device)
    estimated_sampled = torch.take_along_dim(estimated_full, rand_idx[None,...,None], dim=1)

    data_partial = estimated_sampled
    data_full = sampled_full

    np_inputs1 = data_full.cpu().detach().squeeze(0).numpy()
    np_inputs2 = data_partial.cpu().detach().squeeze(0).numpy()
    # Encode
    out_1 = model.encode_inputs(data_full.cuda())
    out_2 = model.encode_inputs(data_partial.cuda())

    # Predict R

    R_gt   = R
    R_pred = solve_R(out_1, out_2)

    # Numpy
    np_R_gt   = R_gt.cpu().squeeze(0).detach().numpy()
    np_R_pred = R_pred.cpu().squeeze(0).detach().numpy()

    # Rotate back
    np_inputs2_registered = np_inputs2@np_R_pred

    # Metric
    angle_diff_degree = angle_diff_func(R_pred.cuda(), R_gt.cuda())
    angle_diff_degree = angle_diff_degree.item()
    eval += im2mesh.common.chamfer_distance(torch.Tensor(np.expand_dims(np_inputs1, axis=0)).to(device), 
                                                    torch.Tensor(np.expand_dims(np_inputs2_registered, axis=0)).to(device)).item()

print(eval/(i+1))