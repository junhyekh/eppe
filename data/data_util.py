import glob
import os, sys
import pickle
import jax
import numpy as np
from torch.utils.data import DataLoader, Dataset
import einops

BASEDIR = os.path.dirname(os.path.dirname(__file__))
if BASEDIR not in sys.path:
    sys.path.insert(0, BASEDIR)


class OccDataDirLoader(Dataset):
    def __init__(self, dataset_dir=os.path.join(BASEDIR, 'data/partial_occ'), eval_type='test', args=None):
        self.args = args
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

        # idx_vp = np.random.randint(0, partial_spnts.shape[0], size=(self.args.nvp,))
        # idx_qps = np.random.randint(0, qps.shape[1], size=(self.args.batch_query_size,))
        idx_vp = np.random.permutation(partial_spnts.shape[0])[:self.args.nvp]
        idx_qps = np.random.permutation(qps.shape[1])[:self.args.batch_query_size]

        qps_res, occ_res = qps[idx_vp][:,idx_qps], occ[idx_vp][:,idx_qps]
        pspnts_res, seg_res = partial_spnts[idx_vp], seg[idx_vp]
        return (pspnts_res.astype(np.float32), seg_res.astype(np.int32), qps_res.astype(np.float32), occ_res.astype(np.float32))