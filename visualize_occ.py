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
    eval_dataset = DataLoader(OccDataDirLoader(eval_type='train', args=args), batch_size=2, shuffle=False, num_workers=1, drop_last=True)
    for ds in eval_dataset:
        ds_sample = ds # (point_cloud: (NB, NV, I, J, 3), seg: (NB, NV, I, J) .....)
        break

    emb = enc_model(params[0], input_pnts[None])

    # marching cube
    density = 202
    # density = 128
    qp_bound = 1.0
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
        print(i)
        grid_ = grid[:,dif*i:dif*(i+1)]
        output_ = dec_model(params[1], emb, grid_, jkey)[0]
        if output is None:
            output = output_
        else:
            output = jnp.concatenate([output, output_], 0)
    volume = output.reshape(density+1, density+1, density+1).transpose(1, 0, 2)
    volume = np.array(volume)
    # print("start smoothing")
    # volume = mcubes.smooth(volume)
    # print("end smoothing")
    verts, faces = mcubes.marching_cubes(volume, 0.0)
    verts *= gap
    verts -= qp_bound

    mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(verts), triangles=o3d.utility.Vector3iVector(faces))
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh])

    # same mesh
    # o3d.io.write_triangle_mesh(os.path.join(os.path.dirname(save_dir), os.path.basename(mesh_fn)), mesh)

    # print("Cluster connected triangles")
    # with o3d.utility.VerbosityContextManager(
    #         o3d.utility.VerbosityLevel.Debug) as cm:
    #     triangle_clusters, cluster_n_triangles, cluster_area = (
    #         mesh.cluster_connected_triangles())
    # triangle_clusters = np.asarray(triangle_clusters)
    # cluster_n_triangles = np.asarray(cluster_n_triangles)
    # cluster_area = np.asarray(cluster_area)

    # import copy
    # print("Show mesh with small clusters removed")
    # mesh_0 = copy.deepcopy(mesh)
    # triangles_to_remove = cluster_n_triangles[triangle_clusters] < 10000
    # mesh_0.remove_triangles_by_mask(triangles_to_remove)

    # if visualize:
    #     mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=np.array([0., 0., 0.]))
    #     o3d.visualization.draw_geometries([mesh, input_pcd, mesh_frame])

    # return mesh, os.path.basename(mesh_path)


if __name__ == '__main__':
    save_dir = 'logs/11012023-211835/saved.pkl'
    # mesh_path = 'data/DexGraspNet/32_64_1_v4/core-mug-6c379385bf0a23ffdec712af445786fe.obj'
    
    jkey = jax.random.PRNGKey(0)
    mesh_0, mesh_basename = create_mesh(None, jkey, save_dir, visualize=True)

    # o3d.io.write_triangle_mesh(os.path.join(os.path.dirname(save_dir), mesh_basename), mesh_0)