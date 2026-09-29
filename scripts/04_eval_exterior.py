"""外から見える面のみで再構成誤差を評価（全体 + 前端/中央/後端）
使い方: python 04_eval_exterior.py <car_id>
"""
import sys, os, json
import numpy as np, trimesh, open3d as o3d
from scipy.spatial import cKDTree

ROOT = os.path.expanduser("~/Desktop/ikeda/detail_edit")
car = sys.argv[1]; d = f"{ROOT}/data/cars/{car}"
recon = trimesh.load(f"{d}/recon/slat.ply", force="mesh")
gt = trimesh.load(f"{d}/recon/gt_normalized.obj", force="mesh")
norm = json.load(open(f"{d}/norm.json"))
obj = trimesh.load(f"{d}/unmodified.obj", force="mesh")
MM = (obj.bounds[1] - obj.bounds[0]).max() / norm["scale"] * 1000   # 正規化1単位 -> mm
print(f"1単位 = {MM:.0f} mm | 64^3ボクセル: {MM/64:.1f} mm | 256^3デコーダ: {MM/256:.1f} mm\n")

def visible(mesh, pts, n_dirs=64, R=2.0, eps_mm=3.0):
    """周囲n_dirs方向から光線を飛ばし、どれかで遮られずに届く点を「外から見える」とする"""
    scene = o3d.t.geometry.RaycastingScene()
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),
                                  o3d.utility.Vector3iVector(mesh.faces))
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))
    i = np.arange(n_dirs) + 0.5
    phi, th = np.arccos(1 - 2 * i / n_dirs), np.pi * (1 + 5 ** 0.5) * i
    dirs = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)
    vis = np.zeros(len(pts), bool)
    eps = eps_mm / MM
    for v in dirs:
        rays = np.hstack([pts + v * R, np.repeat(-v[None], len(pts), 0)]).astype(np.float32)
        t = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
        vis |= t > R - eps
    return vis

N = 300_000
pr, _ = trimesh.sample.sample_surface(recon, N)
pg, _ = trimesh.sample.sample_surface(gt, N)
vr, vg = visible(recon, pr), visible(gt, pg)
print(f"外から見える点の割合: 真値 {vg.mean()*100:.1f}% | 再構成 {vr.mean()*100:.1f}%\n")
pr_e, pg_e = pr[vr], pg[vg]
acc = cKDTree(pg_e).query(pr_e)[0] * MM      # 正確さ: 再構成(外板) -> 真値(外板)
comp = cKDTree(pr_e).query(pg_e)[0] * MM     # 網羅性: 真値(外板) -> 再構成(外板)

def report(name, a, c):
    line = f"{name:10s} | 正確さ 中央値 {np.median(a):5.1f} 90% {np.percentile(a,90):5.1f} | " \
           f"網羅性 中央値 {np.median(c):5.1f} 90% {np.percentile(c,90):5.1f} |"
    for t in [5, 10, 20]:
        P, R = (a < t).mean(), (c < t).mean()
        line += f" F@{t} {2*P*R/(P+R+1e-12):.2f}"
    print(line + "  [mm]")

report("全体", acc, comp)
# z軸=車長方向。両端20%と中央40%に分ける（どちらが前かはレンダリング画像で確認）
z0, z1 = gt.bounds[0][2], gt.bounds[1][2]; L = z1 - z0
regions = {"z-端(20%)": (z0, z0 + 0.2 * L), "中央(40%)": (z0 + 0.3 * L, z0 + 0.7 * L),
           "z+端(20%)": (z1 - 0.2 * L, z1)}
for name, (lo, hi) in regions.items():
    ma = (pr_e[:, 2] >= lo) & (pr_e[:, 2] <= hi)
    mc = (pg_e[:, 2] >= lo) & (pg_e[:, 2] <= hi)
    report(name, acc[ma], comp[mc])

def colorize(pts, err, path, vmax=30.0):
    t = np.clip(err / vmax, 0, 1)[:, None]
    col = (np.hstack([t, np.zeros_like(t), 1 - t]) * 255).astype(np.uint8)
    trimesh.PointCloud(pts, colors=np.hstack([col, np.full((len(pts), 1), 255, np.uint8)])).export(path)
colorize(pg_e, comp, f"{d}/recon/gt_ext_err.ply")
print("\n書き出し: recon/gt_ext_err.ply（外板の真値を網羅性の誤差で色付け、青0mm〜赤30mm以上）")
