import os, sys
import pickle
from torch.utils.data import DataLoader
from data.data_util import OccDataDirLoader
import jax
import jax.numpy as jnp
import numpy as np
import einops
import matplotlib.pyplot as plt
import open3d as o3d
from scipy.spatial.transform import Rotation as R

BASEDIR = os.path.dirname(__file__)
if BASEDIR not in sys.path:
    sys.path.insert(0, BASEDIR)

from train import Encoder, Decoder
import util.transform_util as tutil
import util.ev_util.ev_util as evutil
import util.ev_util.rotm_util as rmutil


def subsample(pnts_, seg_, subpnts_no, jkey):
    pm_idx = jnp.arange(pnts_.shape[-2])
    pm_idx = jax.random.permutation(jkey, pm_idx)
    _, jkey = jax.random.split(jkey)
    pnts_ = pnts_[...,pm_idx,:]
    seg_ = seg_[...,pm_idx]
    seg_flat = seg_.reshape(-1, seg_.shape[-1])
    flat_idx = jax.vmap(lambda x: jnp.where(x>=0, size=subpnts_no, fill_value=-1))(seg_flat)[0]
    flat_idx = flat_idx.reshape(seg_.shape[:-1] + (subpnts_no,))
    pick_pnts = jnp.take_along_axis(pnts_, flat_idx[...,None], axis=-2)
    pick_pnts = jnp.where(flat_idx[...,None]>=0, pick_pnts, 0)
    return pick_pnts, jkey

# ckpt_dir = 'logs/11202023-170107/saved.pkl'
ckpt_dir = 'logs/11212023-153229/saved.pkl' # local representation
# ckpt_dir = 'logs/11212023-183559/saved.pkl' # global representation


with open(ckpt_dir, 'rb') as f:
    raw_loaded = pickle.load(f)
params = raw_loaded['params']
args = raw_loaded['args']
rot_configs = raw_loaded['rot_configs']

args.nvp = 8
eval_dataset = DataLoader(OccDataDirLoader(eval_type='train', args=args), batch_size=4, shuffle=False, num_workers=1, drop_last=True)

for ds in eval_dataset:
    ds_sample = ds
    break

# for i in range(ds_sample[0].shape[0]):
#     plt.figure()
#     for j in range(4):
#         plt.subplot(2,2,1+j)
#         plt.imshow(ds_sample[0][i,j]%1)
#     plt.show()

ds_sample = jax.tree_map(lambda x: jnp.array(x), ds_sample)
jkey = jax.random.PRNGKey(args.seed)
enc_model = Encoder(args, rot_configs)
dec_model = Decoder(args, rot_configs)


# calculate relative rotations
cem_nitr = 10
cem_nb = 20000
cem_top_ratio = 0.001

def loss_func(w, emb1, emb2):
    R1 = tutil.q2R(tutil.aa2q(w))
    emb1_rot = rmutil.apply_rot(emb1[...,None,:,:], R1, rot_configs, feature_axis=-2, expand_R_no=None)
    return jnp.sum((emb2[...,None,:,:] - emb1_rot)**2, axis=(-1,-2))

def cem(emb1, emb2, jkey, w_gt=None):
    if w_gt is not None:
        w = jax.random.normal(jkey, shape=emb1.shape[:-2] + (cem_nb,3)) + w_gt[:,None]
    else:
        w = tutil.q2aa(tutil.qrand(emb1.shape[:-2] + (cem_nb,)))
    _, jkey = jax.random.split(jkey)

    for itr in range(cem_nitr):
        loss = loss_func(w, emb1, emb2)
        top_idx = jnp.argsort(loss, axis=-1)[...,:int(cem_nb*cem_top_ratio)]
        w_top = jnp.take_along_axis(w, top_idx[...,None], -2)
        w_mean = jnp.mean(w_top, axis=-2, keepdims=True)
        w_std = jnp.std(w_top, axis=-2, keepdims=True)
        w = w_mean + w_std*jax.random.normal(jkey, w.shape)
        _, jkey = jax.random.split(jkey)
        # print(itr, jnp.min(loss, -1))
    
    loss = loss_func(w, emb1, emb2)
    top_idx = jnp.argmin(loss, axis=-1, keepdims=True)
    w_top = jnp.take_along_axis(w, top_idx[...,None], -2).squeeze(-2)

    return w_top, jkey

def chamfer_dist(ref_pnts, query_pnts, ref_seg, query_seg, w, jkey, visualize=False):

    ref_pnts = einops.rearrange(ref_pnts, '... v i j k -> ... (v i j) k')
    query_pnts = einops.rearrange(query_pnts, '... v i j k -> ... (v i j) k')
    ref_seg = einops.rearrange(ref_seg, '... v i j -> ... (v i j)')
    query_seg = einops.rearrange(query_seg, '... v i j -> ... (v i j)')

    ref_pnts_ss, jkey = subsample(ref_pnts, ref_seg, 2000, jkey)
    query_pnts_ss, jkey = subsample(query_pnts, query_seg, 500, jkey)

    q_query = tutil.aa2q(w)
    query_pnts_rot = tutil.qaction(q_query[...,None,:], query_pnts_ss)

    if visualize:
        for i in range(ref_pnts.shape[0]):
            # print((loss_opt - loss_gt)[i])
            pcd_ref = o3d.geometry.PointCloud()
            pcd_ref.points = o3d.utility.Vector3dVector(np.array(ref_pnts_ss[i]))
            pcd_ref.paint_uniform_color(np.array([0.1,0.2,1]))
            pcd_q = o3d.geometry.PointCloud()
            pcd_q.points = o3d.utility.Vector3dVector(np.array(query_pnts_ss[i]))
            pcd_q.paint_uniform_color(np.array([1,0.1,0.2]))
            pcd_qr = o3d.geometry.PointCloud()
            pcd_qr.points = o3d.utility.Vector3dVector(np.array(query_pnts_rot[i]))
            pcd_qr.paint_uniform_color(np.array([0.2,1.0,0.1]))
            o3d.visualization.draw_geometries([pcd_ref, pcd_q, pcd_qr])

    sq_dif = (ref_pnts_ss[...,None,:] - query_pnts_rot[...,None,:,:])**2
    sq_dif = jnp.sum(sq_dif, axis=-1)
    return 0.5*jnp.mean(jnp.sqrt(jnp.min(sq_dif,axis=-1)),axis=-1) + 0.5*jnp.mean(jnp.sqrt(jnp.min(sq_dif,axis=-2)),axis=-1)
    # return 0.5*jnp.mean(jnp.sort(jnp.sqrt(jnp.min(sq_dif,axis=-1)),axis=-1)[...,-5:], axis=-1) + \
    #     0.5*jnp.mean(jnp.sort(jnp.sqrt(jnp.min(sq_dif,axis=-2)),axis=-1)[...,-5:], axis=-1)

USE_ICP = True

# start calculate rotation
nv_query = 1
for ds in eval_dataset:
    pnts, seg, _, _ = jax.tree_map(lambda x : jnp.array(x), ds)

    pm_idx = jax.random.permutation(jkey, jnp.arange(pnts.shape[1]), axis=-1)[None]
    _, jkey = jax.random.split(jkey)
    pnts_ref = jnp.take_along_axis(pnts, pm_idx[...,nv_query:,None,None,None], -4)
    seg_ref = jnp.take_along_axis(seg, pm_idx[...,nv_query:,None,None], -3)

    rot_pnts_query = jnp.take_along_axis(pnts, pm_idx[...,:nv_query,None,None,None], -4)
    seg_query = jnp.take_along_axis(seg, pm_idx[...,:nv_query,None,None], -3)
    random_quat = tutil.qrand(pnts.shape[0:1], jkey)
    _, jkey = jax.random.split(jkey)
    rot_pnts_query = tutil.qaction(random_quat[...,None,None,None,:], rot_pnts_query)

    emb = enc_model.apply(params[0], pnts_ref, seg_ref, jkey)
    emb_ref = evutil.max_norm_pooling(emb)
    _, jkey = jax.random.split(jkey)

    emb_query = enc_model.apply(params[0], rot_pnts_query, seg_query, jkey)
    if args.model_type == 0:
        emb_query = evutil.max_norm_pooling(emb_query)
        emb_query = evutil.reduce_top_k_emb(emb_query, 0.7)
    _, jkey = jax.random.split(jkey)

    if USE_ICP:
        # ICP between pnts_ref, seg_ref & rot_pnts_query, seg_query
        pnts_ref_o3d = np.array(pnts_ref) # (B, 7, 100, 100, 3)
        seg_ref_o3d = np.array(seg_ref) # (B, 7, 100, 100)
        rot_pnts_query_o3d = np.array(rot_pnts_query) # (B, 1, 100, 100, 3)
        seg_query_o3d = np.array(seg_query) # (B, 1, 100, 100)

        random_q = tutil.qrand((len(pnts_ref_o3d),), jkey)
        _, jkey = jax.random.split(jkey)
        random_rotmat = tutil.q2R(random_q)
        random_rotmat = np.array(random_rotmat)
        est_aa_list = []

        for cur_idx in range(len(pnts_ref_o3d)):
            pnts_ref_cur = pnts_ref_o3d[cur_idx].reshape(-1, 3)
            seg_ref_cur  = seg_ref_o3d[cur_idx].reshape(-1)   
            pnts_ref_cur_pure = np.take_along_axis(pnts_ref_cur, np.argwhere(seg_ref_cur==0), axis=0)

            pnts_rot_cur = rot_pnts_query_o3d[cur_idx].reshape(-1, 3)
            seg_rot_cur  = seg_query_o3d[cur_idx].reshape(-1)
            pnts_rot_cur_pure = np.take_along_axis(pnts_rot_cur, np.argwhere(seg_rot_cur==0), axis=0)

            print(1)

            source = o3d.geometry.PointCloud()
            target = o3d.geometry.PointCloud()
            source.points = o3d.utility.Vector3dVector(pnts_ref_cur_pure)
            target.points = o3d.utility.Vector3dVector(pnts_rot_cur_pure)

            # Apply ICP
            threshold = 0.05  # Set this to an appropriate value depending on your data
            trans_init = np.eye(4)
            trans_init[:3, :3] = random_rotmat[cur_idx]

            relative_fitness = 0.000001
            relative_rmse = 0.000001
            max_iteration = 1000

            # Estimate normal
            target.estimate_normals()

            reg_p2p = o3d.pipelines.registration.registration_generalized_icp(
                    source, target, threshold, trans_init,
                    # o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    # o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness,
                                            relative_rmse,
                                            max_iteration))

            # Return the transformation matrix
            transf_matrix = reg_p2p.transformation
            fitness = reg_p2p.fitness
            inlier_rmse = reg_p2p.inlier_rmse
            estimated_rot = transf_matrix[:3, :3].copy()
            estimated_quat = R.from_matrix(estimated_rot).as_quat()
            est_aa = tutil.aa2q(jnp.array(estimated_quat))

            est_aa_list.append(est_aa)

        est_aa_list = jnp.stack(est_aa_list, axis=0)

        cd_icp = chamfer_dist(pnts_ref, rot_pnts_query, seg_ref, seg_query,\
                            est_aa_list, jkey, visualize=True)
        _, jkey = jax.random.split(jkey)

        print('chamfer_distances of ICP', cd_icp)

    # for i in range(emb_query.shape[0]):
    #     plt.figure()
    #     plt.plot(jnp.linalg.norm(emb_ref[i], axis=-2))
    #     plt.plot(jnp.linalg.norm(emb_query[i], axis=-2))
    #     plt.show()

    w_gt = tutil.q2aa(tutil.qinv(random_quat))
    w_res, jkey = cem(emb_query, emb_ref, jkey, w_gt=None)

    loss_opt = loss_func(w_res[...,None,:], emb_query, emb_ref).squeeze(-1)
    loss_gt = loss_func(w_gt[...,None,:], emb_query, emb_ref).squeeze(-1)
    print('loss from gt')
    print(loss_opt - loss_gt)

    cd = chamfer_dist(pnts_ref, rot_pnts_query, seg_ref, seg_query, w_res, jkey, visualize=True)
    _, jkey = jax.random.split(jkey)

    print('chamfer_distances', cd)

print(1)