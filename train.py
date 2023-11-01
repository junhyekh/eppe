import jax.numpy as jnp
import jax
import einops
from flax import linen as nn
import glob
import optax
from torch.utils.tensorboard import SummaryWriter
import datetime
import os, sys
import pickle
import numpy as np
import typing
import shutil

from torch.utils.data import DataLoader, Dataset

import argparse

try:
    import vessl
    vessl.init()
    vessl_on = True
except:
    vessl_on = False

BASEDIR = os.path.dirname(__file__)
if BASEDIR not in sys.path:
    sys.path.insert(0, BASEDIR)

import util.ev_util.ev_layers as evl
import util.ev_util.ev_util as eutil
import util.ev_util.rotm_util as rmutil

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=10)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--nvp", type=int, default=4)
parser.add_argument("--batch_query_size", type=int, default=256)
parser.add_argument("--enc_base_dim", type=int, default=256)
parser.add_argument("--dec_base_dim", type=int, default=128)
parser.add_argument("--subpnts_no", type=int, default=256)

args = parser.parse_args()

class OccDataDirLoader(Dataset):
    def __init__(self, dataset_dir=os.path.join(BASEDIR, 'data/partial_occ'), eval_type='test'):
        

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
            self.entire_ds = jax.tree_map(lambda x: x[:,:nvp//8], self.entire_ds)
        else:
            self.entire_ds = jax.tree_map(lambda x: x[:,nvp//8:], self.entire_ds)

    def __len__(self):
        return jax.tree_util.tree_flatten(self.entire_ds)[0][0].shape[0]

    def __getitem__(self, index):

        dpnts = jax.tree_map(lambda x: x[index], self.entire_ds)
        partial_spnts, seg, qps, occ = dpnts

        idx_vp = np.random.randint(0, partial_spnts.shape[0], size=(args.nvp,))
        idx_qps = np.random.randint(0, qps.shape[1], size=(args.batch_query_size,))

        qps_res, occ_res = qps[idx_vp][:,idx_qps], occ[idx_vp][:,idx_qps]
        pspnts_res, seg_res = partial_spnts[idx_vp], seg[idx_vp]
        return (pspnts_res.astype(np.float32), seg_res.astype(np.int32), qps_res.astype(np.float32), occ_res.astype(np.float32))

train_dataset = DataLoader(OccDataDirLoader(eval_type='train'), batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=True)
eval_dataset = DataLoader(OccDataDirLoader(eval_type='test'), batch_size=args.batch_size, shuffle=False, num_workers=1, drop_last=True)


class Encoder(nn.Module):
    args:typing.NamedTuple
    rot_configs:typing.Sequence

    @nn.compact
    def __call__(self, x, seg, jkey):
        '''
        '''
        base_dim = args.enc_base_dim

        # segmentation sample
        x = einops.rearrange(x, '... i j k -> ... (i j) k')
        seg = einops.rearrange(seg, '... i j -> ... (i j)')
        pm_idx = jnp.arange(x.shape[-2])
        pm_idx = jax.random.permutation(jkey, pm_idx)
        x = x[...,pm_idx,:]
        seg = seg[...,pm_idx]

        seg_flat = seg.reshape(-1, seg.shape[-1])
        flat_idx = jax.vmap(lambda x: jnp.where(x>=0, size=args.subpnts_no, fill_value=-1))(seg_flat)[0]
        flat_idx = flat_idx.reshape(seg.shape[:-1] + (args.subpnts_no,))
        x = jnp.take_along_axis(x, flat_idx[...,None], axis=-2)
        x = jnp.where(flat_idx[...,None]>=0, x, 0)

        x = jnp.expand_dims(x, -1)
        x = eutil.get_graph_feature(x, k=10, cross=True) # (B, P, K, F, D)
        x = evl.MakeHDFeature(self.args, self.rot_configs)(x)
        x = evl.EVLinearLeakyReLU(self.args, base_dim)(x)
        x = jnp.mean(x, -3) # (B, P, F, D)

        np = x.shape[-3]
        x = nn.Dense(base_dim, use_bias=False)(x)
        for i in range(4):
            x = evl.EVNResnetBlockFC(self.args, base_dim, base_dim)(x)
            pooled = einops.repeat(jnp.mean(x, -3), '... f d -> ... r f d', r=np)
            x = jnp.concatenate([x, pooled], -1)
        x = evl.EVNResnetBlockFC(self.args, base_dim, base_dim)(x)
        x = jnp.mean(x, -3)
        x = evl.EVNNonLinearity(self.args)(x)
        x = nn.Dense(base_dim, use_bias=False)(x)

        return x


class Decoder(nn.Module):
    args:typing.NamedTuple
    rot_configs:typing.Sequence

    @nn.compact
    def __call__(self, emb, p):
        '''
        emb : (... V F D)
        '''
        base_dim = args.dec_base_dim
        np_ = p.shape[-2]
        emb_merge = eutil.max_norm_pooling(emb) # (... F D)

        p_ext = evl.MakeHDFeature(self.args, self.rot_configs)(p[...,None]).squeeze(-1)

        net = (p_ext * p_ext).sum(-1, keepdims=True)
        net_z = jnp.einsum('...mf,...fd->...md', p_ext, emb_merge)
        z_dir = nn.Dense(emb_merge.shape[-1], use_bias=False)(emb_merge)
        z_inv = (emb_merge * z_dir).sum(-2)
        z_inv = einops.repeat(z_inv, '... b -> ... r b', r=np_)
        net = jnp.concatenate([net, net_z, z_inv], axis=-1)

        net = nn.Dense(base_dim)(net)
        activation = nn.relu
        for i in range(5):
            x_s = net
            net = nn.Dense(base_dim)(activation(net))
            net = nn.Dense(base_dim)(activation(net)) + x_s
        out = nn.Dense(1)(activation(net))
        out = out.squeeze(-1)

        return out


for ds in train_dataset:
    ds_sample = ds
    break
ds_sample = jax.tree_map(lambda x: jnp.array(x), ds_sample)
jkey = jax.random.PRNGKey(args.seed)
rot_configs = rmutil.init_rot_config(args.seed, [1,2], 'custom')
enc_model = Encoder(args, rot_configs)
dec_model = Decoder(args, rot_configs)

emb, enc_params = enc_model.init_with_output(jkey, ds_sample[0], ds_sample[1], jkey)
_, jkey = jax.random.split(jkey)

qps = einops.rearrange(ds_sample[2], '... i j k -> ... (i j) k')
dec_params = dec_model.init(jkey, emb, qps)
params = (enc_params, dec_params)

# %%
# start training
def BCE_loss(yp_logit, yt):
    assert yp_logit.shape == yt.shape
    yp = nn.sigmoid(yp_logit).clip(1e-5, 1-1e-5)
    loss = - yt*jnp.log(yp) - (1-yt)*jnp.log(1-yp)
    return loss

def cal_loss(params, ds, jkey):
    vp_dropout_idx = jax.random.uniform(jkey, shape=(args.batch_size, args.nvp)) < 0.5
    _, jkey = jax.random.split(jkey)
    partial_points = jnp.where(vp_dropout_idx[...,None,None,None], ds[0], 0)
    seg = jnp.where(vp_dropout_idx[...,None,None], ds[1], 0)
    
    embs = enc_model.apply(params[0], partial_points, seg, jkey)
    _, jkey = jax.random.split(jkey)
    
    embs_dropout_idx = jax.random.uniform(jkey, shape=embs[...,:1,:].shape) < 0.3
    _, jkey = jax.random.split(jkey)
    embs = jnp.where(embs_dropout_idx, embs, 0)

    qps = einops.rearrange(ds[2], '... i j k -> ... (i j) k')
    occ_pred = dec_model.apply(params[1], embs, qps)

    occ_pred = occ_pred.reshape(ds[3].shape)

    occ_loss = BCE_loss(occ_pred, ds[3])
    occ_loss = jnp.where(vp_dropout_idx, jnp.mean(occ_loss, axis=-1), 0)
    occ_loss = jnp.mean(occ_loss)

    loss = occ_loss
    return loss, {'train_occ_loss': occ_loss}

cal_loss_grad = jax.grad(cal_loss, has_aux=True)

cal_loss_jit = jax.jit(cal_loss)

# %%
# define train func
def l2_norm(tree):
    """Compute the l2 norm of a pytree of arrays. Useful for weight decay."""
    leaves, _ = jax.tree_util.tree_flatten(tree)
    return jnp.sqrt(sum(jnp.vdot(x, x) for x in leaves))

def clip_grads(grad_tree, max_norm):
    """Clip gradients stored as a pytree of arrays to maximum norm `max_norm`."""
    norm = l2_norm(grad_tree)
    normalize = lambda g: jnp.where(norm < max_norm, g, g * (max_norm / norm))
    return jax.tree_map(normalize, grad_tree)

optimizer = optax.adam(3e-4)
opt_state = optimizer.init(params)
def train_func(params, opt_state, ds, jkey):
    _, jkey = jax.random.split(jkey)
    grad, train_loss_dict = cal_loss_grad(params, ds, jkey)
    grad = jax.tree_map(lambda x: jnp.where(jnp.isnan(x), 0, x), grad)
    grad = clip_grads(grad, 1.0)
    _, jkey = jax.random.split(jkey)
    updates, opt_state = optimizer.update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, jkey, train_loss_dict, grad

train_func_jit = jax.jit(train_func)
# train_func_jit = train_func

@jax.jit
def occ_inf_test(params, ds, jkey, rot_aug=True):
    embs = enc_model.apply(params[0], ds[0], ds[1], jkey)
    qps = einops.rearrange(ds[2], '... i j k -> ... (i j) k')
    occ_pred = dec_model.apply(params[1], embs, qps)
    occ_pred = occ_pred.reshape(ds[3].shape)
    occ_loss = BCE_loss(occ_pred, ds[3])

    return occ_pred, occ_loss


# %%
# start training
now = datetime.datetime.now() # current date and time
date_time = now.strftime("%m%d%Y-%H%M%S")
if vessl_on:
    logs_dir = os.path.join('/output', date_time)
else:
    logs_dir = os.path.join('logs', date_time)
writer = SummaryWriter(logs_dir)
shutil.copy(__file__, logs_dir)
shutil.copytree(os.path.join(BASEDIR, 'util'), os.path.join(logs_dir, 'util'), dirs_exist_ok=True)
shutil.rmtree(os.path.join(logs_dir, 'util/__pycache__'))
shutil.rmtree(os.path.join(logs_dir, 'util/ev_util/__pycache__'))

writer.add_text('args', args.__str__(), 0)

best_occ_acc = 0
for itr in range(10000):
    train_loss = 0
    tr_cnt = 0
    for i, ds in enumerate(train_dataset):
        ds = jax.tree_map(lambda x: jnp.asarray(x), ds) 
        params, opt_state, jkey, train_loss_dict_, grad_ = train_func_jit(params, opt_state, ds, jkey)
        _, jkey = jax.random.split(jkey)
        if i == 0:
            train_loss_dict = train_loss_dict_
        else:
            train_loss_dict = jax.tree_map(lambda x,y: x+y, train_loss_dict, train_loss_dict_)
        tr_cnt += 1
        # if i%100==0:
        #     print(i, train_loss_dict_)
    train_loss_dict = jax.tree_map(lambda x: x/tr_cnt, train_loss_dict)

    # evaluations
    total_occ = 0
    occ_acc = 0
    eval_occ_loss = 0
    for i, ds in enumerate(eval_dataset):
        _, jkey = jax.random.split(jkey)
        ds = jax.tree_map(lambda x: jnp.asarray(x), ds)
        occ_pred, occ_loss_ = occ_inf_test(params, ds, jkey)
        eval_occ_loss += occ_loss_
        occ_acc += jnp.sum(jnp.logical_and(occ_pred>0, ds[3]>0.5)) + jnp.sum(jnp.logical_and(occ_pred<0, ds[3]<0.5))
        total_occ += ds[3].shape[0] * ds[3].shape[1] * ds[3].shape[2]
    cur_occ_acc = occ_acc/total_occ
    eval_occ_loss = jnp.sum(eval_occ_loss)/total_occ
    if best_occ_acc < cur_occ_acc:
        best_occ_acc = cur_occ_acc
        with open(os.path.join(logs_dir, 'saved.pkl'), 'wb') as f:
            pickle.dump({'params':params, 'args':args, 'rot_configs':rot_configs}, f)
        # if not vessl_on:
        #     import open3d as o3d
        #     from examples.visualize_occ import create_mesh
        #     mesh_path = os.path.join(BASEDIR, 'data/DexGraspNet/32_64_1_v4/sem-Bowl-8eab5598b81afd7bab5b523beb03efcd.obj')
        #     mesh_0, mesh_basename = create_mesh(args, jkey, None, mesh_path, input_type='cvx', models=models.set_params(params))
        #     o3d.io.write_triangle_mesh(os.path.join(logs_dir, mesh_basename), mesh_0)

    log_dict = {'occ_acc':cur_occ_acc,
                "best_occ_acc":best_occ_acc,
                    "eval_occ_loss":eval_occ_loss,
                    **train_loss_dict}
    log_dict = {k:np.array(log_dict[k]) for k in log_dict}
    print(f'itr: {itr}, {log_dict}')
    
    for k in log_dict:
        writer.add_scalar(k, log_dict[k], itr)

    if vessl_on:
        base_name = ""
        log_dict = {base_name+k: log_dict[k] for k in log_dict}
        vessl.log(step=itr, payload=log_dict)



