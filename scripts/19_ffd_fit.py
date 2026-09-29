"""ドナーのフロントを粗い格子（FFD）で変形し、大きな形をターゲットに合わせる
使い方: python 19_ffd_fit.py <target_id> <donor_id> [--deg 2 3 3] [--region 1.0] [--fit 0.8] [--iters 8] [--lam 1e-3]
  --deg:    格子の次数（幅, 高さ, 前後）。節点の数は各方向で次数+1
  --region: 格子の範囲（前端からの距離 m）
  --fit:    合わせに使うドナーの点の範囲（前端からの距離 m）
出力: data/cars/<target_id>/donors/<donor_id>ffd_aligned.obj
"""
import os, json, argparse
import numpy as np, trimesh, open3d as o3d
from scipy.spatial import cKDTree
from scipy.special import comb

ap = argparse.ArgumentParser()
ap.add_argument("target"); ap.add_argument("donor")
ap.add_argument("--deg", type=int, nargs=3, default=[2, 3, 3])
ap.add_argument("--region", type=float, default=1.0)
ap.add_argument("--fit", type=float, default=0.8)
ap.add_argument("--iters", type=int, default=8)
ap.add_argument("--lam", type=float, default=1e-3)
ap.add_argument("--smooth", type=float, default=1.0)   # 隣り合う節点の移動量の差を抑える強さ
a = ap.parse_args()
ROOT = os.path.expanduser("~/Desktop/ikeda/detail_edit")
cd = f"{ROOT}/data/cars/{a.target}"
S = json.load(open(f"{cd}/norm.json"))["scale"]                # 1 m = S [obj単位]
T = trimesh.load(f"{cd}/unmodified.obj", force="mesh", process=False)
D = trimesh.load(f"{cd}/donors/{a.donor}_aligned.obj", force="mesh", process=False)
zf = T.bounds[0][2]                                            # 前端（z- 側）

def exterior(mesh, n=300000, n_dirs=48):
    """外から見える点と、外向きの法線"""
    pts, fid = trimesh.sample.sample_surface(mesh, n)
    nrm = mesh.face_normals[fid].copy()
    s = o3d.t.geometry.RaycastingScene()
    lm = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices), o3d.utility.Vector3iVector(mesh.faces))
    s.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(lm))
    ext = np.ptp(mesh.vertices, axis=0).max(); R, eps = 2 * ext, 1e-3 * ext
    i = np.arange(n_dirs) + 0.5
    ph, th = np.arccos(1 - 2 * i / n_dirs), np.pi * (1 + 5 ** 0.5) * i
    dirs = np.stack([np.cos(th) * np.sin(ph), np.sin(th) * np.sin(ph), np.cos(ph)], 1)
    vis = np.zeros(len(pts), bool); seen = np.zeros((len(pts), 3))
    for v in dirs:
        r = np.hstack([pts + v * R, np.repeat(-v[None], len(pts), 0)]).astype(np.float32)
        hit = s.cast_rays(o3d.core.Tensor(r))["t_hit"].numpy() > R - eps
        seen[hit & ~vis] = v; vis |= hit
    flip = np.einsum("ij,ij->i", nrm, seen) < 0                # 見えた方向を向くように法線を揃える
    nrm[flip] *= -1
    return pts[vis], nrm[vis]

Pt, Nt = exterior(T)
Pd, _ = exterior(D)
tree = cKDTree(Pt)

# 格子の範囲: フロントのドナー頂点をすべて含む箱（後端 = 前端から region m）
V = D.vertices
reg = V[:, 2] <= zf + a.region * S
lo = V[reg].min(0) - 0.05 * S; hi = V[reg].max(0) + 0.05 * S
lo[2] = min(lo[2], zf - 0.05 * S); hi[2] = zf + a.region * S
l, m, n = a.deg
K = (l + 1) * (m + 1) * (n + 1)

def basis(P):
    u = np.clip((P - lo) / (hi - lo), 0, 1)
    def bern(t, d):
        return np.stack([comb(d, i) * t ** i * (1 - t) ** (d - i) for i in range(d + 1)], 1)
    Bx, By, Bz = bern(u[:, 0], l), bern(u[:, 1], m), bern(u[:, 2], n)
    return (Bx[:, :, None, None] * By[:, None, :, None] * Bz[:, None, None, :]).reshape(len(P), -1)

kk = np.arange(K) % (n + 1)                                    # 前後方向の層番号
free = kk < n                                                  # 後端の層は固定

F = Pd[Pd[:, 2] <= zf + a.fit * S]
F = np.vstack([F, F * np.array([-1, 1, 1])])                   # 左右に鏡映して対称な変形にする
WF = basis(F)[:, free]
Kf = free.sum()
delta = np.zeros((Kf, 3))
# なめらかさの制約: 格子の隣り合う節点の移動量の差（グラフラプラシアン）を小さくする
gid = lambda i, j, k: (i * (m + 1) + j) * (n + 1) + k
Lg = np.zeros((K, K))
for i in range(l + 1):
    for j in range(m + 1):
        for k in range(n + 1):
            for di, dj, dk in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
                i2, j2, k2 = i + di, j + dj, k + dk
                if 0 <= i2 <= l and 0 <= j2 <= m and 0 <= k2 <= n:
                    Lg[gid(i, j, k), gid(i, j, k)] += 1; Lg[gid(i, j, k), gid(i2, j2, k2)] -= 1
Lf = Lg[:, free]                                   # 固定した節点（移動0）との差も含める
R3 = np.kron(np.eye(3), Lf.T @ Lf)
MM = 1000 / S
d0 = tree.query(F)[0]
print(f"格子: 節点 {l+1}x{m+1}x{n+1}（動かす節点 {Kf}） | 合わせに使う点 {len(F)}")
for it in range(a.iters):
    cur = F + WF @ delta
    dist, idx = tree.query(cur)
    q, nn = Pt[idx], Nt[idx]
    w = np.where(dist < 0.05 * S, 1.0, 0.05 * S / np.maximum(dist, 1e-12))   # 大きく離れた点は影響を弱める
    A = np.hstack([WF * nn[:, [0]], WF * nn[:, [1]], WF * nn[:, [2]]])       # 点と面の距離（法線方向）
    b = -np.einsum("ij,ij->i", nn, F - q)
    sw = np.sqrt(w)
    Aw, bw = A * sw[:, None], b * sw
    H = Aw.T @ Aw
    H += a.lam * np.trace(H) / H.shape[0] * np.eye(H.shape[0])
    H += a.smooth * np.trace(H) / np.trace(R3) * R3
    delta = np.linalg.solve(H, Aw.T @ bw).reshape(3, Kf).T
    print(f"  反復 {it+1}: 距離 中央値 {np.median(dist)*MM:5.1f} mm / 平均 {dist.mean()*MM:5.1f} mm")

dist = tree.query(F + WF @ delta)[0]
print(f"合わせる前: 中央値 {np.median(d0)*MM:.1f} mm, 平均 {d0.mean()*MM:.1f} mm")
print(f"合わせた後: 中央値 {np.median(dist)*MM:.1f} mm, 平均 {dist.mean()*MM:.1f} mm")
print(f"節点の移動量: 最大 {np.linalg.norm(delta, axis=1).max()*MM:.0f} mm")

full = np.zeros((K, 3)); full[free] = delta
Vn = V.copy(); Vn[reg] = V[reg] + basis(V[reg]) @ full
out = trimesh.Trimesh(Vn, D.faces, process=False)
out.export(f"{cd}/donors/{a.donor}ffd_aligned.obj")
json.dump({"deg": a.deg, "lo": lo.tolist(), "hi": hi.tolist(), "delta": full.tolist()},
          open(f"{cd}/donors/{a.donor}ffd_params.json", "w"))
print("保存:", f"{cd}/donors/{a.donor}ffd_aligned.obj")
