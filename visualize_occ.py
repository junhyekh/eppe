import jax
import jax.numpy as jnp
import numpy as np
import mcubes
import open3d as o3d
import os, sys
import pickle
import pickle
from torch.utils.data import DataLoader
from data.data_util import OccDataDirLoader

BASEDIR = os.path.dirname(os.path.dirname(__file__))
if BASEDIR not in sys.path:
    sys.path.insert(0, BASEDIR)
from train import Encoder, Decoder
import util.ev_util.ev_util as evutil

def create_mesh(jkey, checkpoint_dir, visualize=False):
    
    # load model
    with open(checkpoint_dir, 'rb') as f:
        raw_loaded = pickle.load(f)
    params = raw_loaded['params']
    args = raw_loaded['args']
    rot_configs = raw_loaded['rot_configs']

    jkey = jax.random.PRNGKey(args.seed)
    enc_model = Encoder(args, rot_configs)
    dec_model = Decoder(args, rot_configs)

    args.nvp = 8
    eval_dataset = DataLoader(OccDataDirLoader(eval_type='train', args=args), batch_size=1, shuffle=False, num_workers=1, drop_last=True)
    for ds in eval_dataset:
        ds_sample = ds # (point_cloud: (NB, NV, I, J, 3), seg: (NB, NV, I, J) .....)
        pnts, segs, _, _ = jax.tree_map(lambda x : jnp.array(x), ds)
        
        for nview in range(1, 4):
            
            print(f"number of camera viewpoint: {nview}")
            index = np.random.choice(pnts.shape[1], nview, replace=False) 

            pnts_nview = pnts[:, index]
            segs_nview = segs[:, index]

            emb = enc_model.apply(params[0], pnts_nview, segs_nview, jkey)

            # marching cube
            density = 202
            # density = 128
            qp_bound = 0.2
            gap = 2*qp_bound / density
            x = np.linspace(-qp_bound, qp_bound, density+1)
            y = np.linspace(-qp_bound, qp_bound, density+1)
            z = np.linspace(-qp_bound, qp_bound, density+1)
            xv, yv, zv = np.meshgrid(x, y, z)
            grid = np.stack([xv, yv, zv]).astype(np.float32).reshape(3, -1).transpose()[None]
            grid = jnp.array(grid)
            ndiv = 50
            output = None
            dif = grid.shape[1]//ndiv
            for i in range(ndiv+1):
                _, jkey = jax.random.split(jkey)
                grid_ = grid[:,dif*i:dif*(i+1)]
                output_ = dec_model.apply(params[1], emb, grid_)[0]
                if output is None:
                    output = output_
                else:
                    output = jnp.concatenate([output, output_], 0)
            volume = output.reshape(density+1, density+1, density+1).transpose(1, 0, 2)
            volume = np.array(volume)
            
            verts, faces = mcubes.marching_cubes(volume, 0.0)
            verts *= gap
            verts -= qp_bound

            mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(verts), triangles=o3d.utility.Vector3iVector(faces))
            mesh.compute_vertex_normals()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pnts_nview[segs_nview>=0])
            o3d.visualization.draw_geometries([mesh, pcd])


if __name__ == '__main__':
    save_dir = 'logs/11212023-153229/saved.pkl'
    # mesh_path = 'data/DexGraspNet/32_64_1_v4/core-mug-6c379385bf0a23ffdec712af445786fe.obj'
    
    jkey = jax.random.PRNGKey(0)
    mesh_0, mesh_basename = create_mesh(jkey, save_dir, visualize=True)

    # o3d.io.write_triangle_mesh(os.path.join(os.path.dirname(save_dir), mesh_basename), mesh_0)