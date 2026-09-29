"""真値の外板点の、再構成表面に対する符号付き距離（負=再構成の内側）
一様なずれ（中央値）と、それを除いたばらつき（細部の誤差）を部位別に出す
使い方: python 04b_signed_offset.py <car_id>
"""
import sys, os, json
import numpy as np, trimesh, open3d as o3d

ROOT = os.path.expanduser("~/Desktop/ikeda/detail_edit")
car = sys.argv[1]; d = f"{ROOT}/data/cars/{car}"
recon = trimesh.load(f"{d}/recon/slat.ply", force="mesh")
gt = trimesh.load(f"{d}/recon/gt_normalized.obj", force="mesh")
norm = json.load(open(f"{d}/norm.json"))
obj = trimesh.load(f"{d}/unmodified.obj", force="mesh")
MM = (obj.bounds[1] - obj.bounds[0]).max() / norm["scale"] * 1000
print("再構成は閉じたメッシュか:", recon.is_watertight)

def scene_of(mesh):
    s = o3d.t.geometry.RaycastingScene()
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),
                                  o3d.utility.Vector3iVector(mesh.faces))
    s.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))
    return s

def visible(scene, pts, n_dirs=64, R=2.0, eps=3.0 / MM):
    i = np.arange(n_dirs) + 0.5
    phi, th = np.arccos(1 - 2 * i / n_dirs), np.pi * (1 + 5 ** 0.5) * i
    dirs = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)
    vis = np.zeros(len(pts), bool)
    for v in dirs:
        rays = np.hstack([pts + v * R, np.repeat(-v[None], len(pts), 0)]).astype(np.float32)
        vis |= scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy() > R - eps
    return vis

pg, _ = trimesh.sample.sample_surface(gt, 200_000)
pg = pg[visible(scene_of(gt), pg)]
sd = scene_of(recon).compute_signed_distance(
        o3d.core.Tensor(pg.astype(np.float32))).numpy() * MM   # 負=再構成の内側

def report(name, s):
    off = np.median(s)
    dev = np.abs(s - off)                       # 一様なずれを除いたばらつき
    print(f"{name:10s} | 内側の割合 {(s<0).mean()*100:5.1f}% | ずれ(中央値) {off:6.1f} | "
          f"ばらつき 中央値 {np.median(dev):4.1f} 90% {np.percentile(dev,90):5.1f} "
          f"95% {np.percentile(dev,95):5.1f}  [mm]")

report("全体", sd)
z0, z1 = gt.bounds[0][2], gt.bounds[1][2]; L = z1 - z0
for name, (lo, hi) in {"z-端(20%)": (z0, z0 + .2 * L), "中央(40%)": (z0 + .3 * L, z0 + .7 * L),
                       "z+端(20%)": (z1 - .2 * L, z1)}.items():
    m = (pg[:, 2] >= lo) & (pg[:, 2] <= hi)
    report(name, sd[m])
