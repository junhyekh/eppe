import pybullet as p
import os, sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import pickle

BASEDIR = os.path.dirname(os.path.dirname(__file__))
if BASEDIR not in sys.path:
    sys.path.insert(0, BASEDIR)

import util.camera_util as cutil

ds_dir = 'data/partial_occ'
os.makedirs(ds_dir, exist_ok=True)

p.connect(p.DIRECT)

mesh_dir = glob.glob('data/32_64_1_v4/*.obj')[:100]
pixel_size = [100,100]
far = 2.0
near = 0.005
nvp = 16
nqpnt = 10000
intrinsic = cutil.default_intrinsic(pixel_size=pixel_size)
lrbt = cutil.intrinsic_to_pb_lrbt(intrinsic, near)

for md in mesh_dir:
    rgb = np.random.uniform(size=(3,))
    viss = p.createVisualShape(p.GEOM_MESH, fileName=md, rgbaColor=list(rgb) + [1,])
    pos = np.zeros(3)
    bid = p.createMultiBody(baseVisualShapeIndex=viss, basePosition=pos)

    seg_list = []
    depth_list = []
    partial_spts_list = []
    cam_dir = np.random.normal(size=(nvp,3,))
    cam_dir = cam_dir/np.linalg.norm(cam_dir, axis=-1, keepdims=True)
    cam_dist = np.random.uniform(0.15, 0.6, size=(nvp,1,)) 
    cam_pos = cam_dir*cam_dist
    cam_up = np.random.normal(size=(nvp,3,))
    cam_up = cam_up/np.linalg.norm(cam_up, axis=-1, keepdims=True)
    for vp_idx in range(nvp):
        view_m = p.computeViewMatrix(cam_pos[vp_idx], np.zeros(3), cam_up[vp_idx])
        proj_m = p.computeProjectionMatrix(*lrbt, near, far)
        cam_img = p.getCameraImage(width=pixel_size[1], height=pixel_size[0], viewMatrix=view_m, projectionMatrix=proj_m)

        seg = np.array(cam_img[4]).reshape(*pixel_size)
        depthimg = np.array(cam_img[3]).reshape(*pixel_size)
        depth = far*near/(far-(far-near)*depthimg)
        seg_list.append(seg)
        depth_list.append(depth)

        partial_spts = cutil.partial_pcd_from_depth(depth, intrinsic, pixel_size, coordinate='opengl', visualize=False)
        partial_spts_bar = np.concatenate([partial_spts, np.ones_like(partial_spts[...,:1])],-1)
        partial_spts = np.einsum('...k,kj->...j', partial_spts_bar, np.linalg.inv(np.array(view_m).reshape(4,4)))[...,:3]
        partial_spts_list.append(partial_spts)

        # import open3d as o3d
        # pcd_o3d = o3d.geometry.PointCloud()
        # pcd_o3d.points = o3d.utility.Vector3dVector(partial_spts.reshape(-1,3))
        # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        #     size=0.1, origin=[0, 0, 0])
        # o3d.visualization.draw_geometries([pcd_o3d, mesh_frame])

    p.removeBody(bid)

    # plt.figure()
    # for i in range(nvp):
    #     plt.subplot(4,4,i+1)
    #     plt.imshow(depth_list[i])
    # plt.show()

    # occupancy query
    mesh = o3d.io.read_triangle_mesh(md)
    mesh.compute_vertex_normals()
    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    tmesh_id = scene.add_triangles(tmesh)

    qpnts_list = []
    occ_label_list = []
    for vp_idx in range(nvp):
        qps = partial_spts_list[vp_idx][np.where(seg_list[vp_idx]>=0)]
        nq = int(nqpnt/qps.shape[0]) + 1
        qps = qps[...,None,:] + np.random.normal(size=(qps.shape[0], nq, 3)) * 0.01
        qps = qps.reshape(-1,3)[:nqpnt]
        qps = o3d.core.Tensor(qps,
                            dtype=o3d.core.Dtype.Float32)
        ans = scene.compute_occupancy(qps)
        qps = qps.numpy().astype(np.float16)
        occ_label = ans.numpy().astype(bool)
        qpnts_list.append(qps)
        occ_label_list.append(occ_label)

    dpnts = (np.array(partial_spts_list).astype(np.float16), np.array(seg_list).astype(np.int32), np.array(qpnts_list).astype(np.float16), np.array(occ_label_list).astype(bool))

    save_path = os.path.join(ds_dir, os.path.basename(md)[:-4] + '.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(dpnts, f)

    print('save ' + save_path)

print(1)

