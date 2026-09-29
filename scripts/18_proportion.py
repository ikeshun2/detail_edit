"""フロントのプロポーション（設計変数 7,8,13,14,18,19,25 に対応する輪郭）を測って比べる
使い方: python 18_proportion.py <target_id> <ラベル>=<objのパス> [<ラベル>=<objのパス> ...]
  すべて unmodified.obj と同じ座標系（フロントは z- 側、y が上）のメッシュを渡す
出力: results/phase1/proportion_<target_id>.png と、ターミナルに差の表
"""
import sys, os, json
import numpy as np, trimesh, open3d as o3d
from scipy.ndimage import uniform_filter1d
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = os.path.expanduser("~/Desktop/ikeda/detail_edit")
car = sys.argv[1]; cd = f"{ROOT}/data/cars/{car}"
scale = json.load(open(f"{cd}/norm.json"))["scale"]           # obj座標1単位 = 1/scale [m]
target_path = f"{cd}/unmodified.obj"
items = [("CX-5", target_path)] + [tuple(a.split("=", 1)) for a in sys.argv[2:]]
T = trimesh.load(target_path, force="mesh")
z0, y0 = T.bounds[0][2], T.bounds[0][1]                       # 前端、地面（CX-5基準）
REGION, STEP, SMOOTH = 1.5, 0.02, 7                            # 前端から1.5m、2cm刻み、約14cmでならす

def exterior_points(mesh, n=400000, n_dirs=48):
    pts, _ = trimesh.sample.sample_surface(mesh, n)
    s = o3d.t.geometry.RaycastingScene()
    lm = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices), o3d.utility.Vector3iVector(mesh.faces))
    s.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(lm))
    ext = np.ptp(mesh.vertices, axis=0).max(); R, eps = 2 * ext, 1e-3 * ext
    i = np.arange(n_dirs) + 0.5
    ph, th = np.arccos(1 - 2 * i / n_dirs), np.pi * (1 + 5 ** 0.5) * i
    dirs = np.stack([np.cos(th) * np.sin(ph), np.sin(th) * np.sin(ph), np.cos(ph)], 1)
    vis = np.zeros(len(pts), bool)
    for v in dirs:
        r = np.hstack([pts + v * R, np.repeat(-v[None], len(pts), 0)]).astype(np.float32)
        vis |= s.cast_rays(o3d.core.Tensor(r))["t_hit"].numpy() > R - eps
    return pts[vis]

def to_m(P):
    return np.stack([P[:, 0] / scale, (P[:, 1] - y0) / scale, (P[:, 2] - z0) / scale], 1)  # 幅, 高さ, 前端からの距離

def binned(key, val, edges, fn):
    idx = np.digitize(key, edges) - 1
    out = np.full(len(edges) - 1, np.nan)
    for b in range(len(edges) - 1):
        v = val[idx == b]
        if len(v): out[b] = fn(v)
    ok = ~np.isnan(out)
    if ok.sum() > 1:                                           # 空のビンを補間してからならす
        out = np.interp(np.arange(len(out)), np.where(ok)[0], out[ok])
        out = uniform_filter1d(out, SMOOTH)
    return out

zb = np.arange(0, REGION + STEP, STEP); hb = np.arange(0.1, 1.8, STEP)
curves = {}
for name, path in items:
    P = to_m(exterior_points(trimesh.load(path, force="mesh")))
    P = P[P[:, 2] <= REGION + 0.2]
    cl = P[np.abs(P[:, 0]) < 0.10]                             # 中心線付近（幅±10cm）
    side = P[np.abs(P[:, 0]) > 0.70]                           # 側面付近
    curves[name] = {
        "前端の輪郭 (中心線)": binned(cl[:, 1], cl[:, 2], hb, np.min),     # 高さごとの最前端
        "上面の輪郭 (中心線)": binned(cl[:, 2], cl[:, 1], zb, np.max),     # 位置ごとの最高点
        "下面の輪郭 (中心線)": binned(cl[:, 2], cl[:, 1], zb, np.min),
        "側面の上端 (フェンダー)": binned(side[:, 2], side[:, 1], zb, np.max),
        "上から見た幅": binned(P[:, 2], np.abs(P[:, 0]), zb, np.max),
    }
    print("計測完了:", name)

# --- 差の表（CX-5との差、mm） ---
print(f"\n{'曲線':22s}" + "".join(f"{n:>22s}" for n, _ in items[1:]))
for key in curves["CX-5"]:
    row = f"{key:22s}"
    for name, _ in items[1:]:
        d = np.abs(curves[name][key] - curves["CX-5"][key]) * 1000
        row += f"{'平均 %5.0f / 最大 %5.0f' % (np.nanmean(d), np.nanmax(d)):>22s}"
    print(row)

# --- 図 ---
fig, ax = plt.subplots(5, 1, figsize=(10, 17))
for a, key in zip(ax, curves["CX-5"]):
    for name, _ in items:
        c = curves[name][key]
        if key.startswith("前端"):
            a.plot(c, (hb[:-1] + hb[1:]) / 2, label=name); a.set_xlabel("distance from front [m]"); a.set_ylabel("height [m]")
        else:
            a.plot((zb[:-1] + zb[1:]) / 2, c, label=name); a.set_xlabel("distance from front [m]")
    a.set_title(key, fontname="DejaVu Sans"); a.legend(); a.grid(alpha=0.3)
ax[0].set_title("front profile (centerline)"); ax[1].set_title("top profile (centerline)")
ax[2].set_title("bottom profile (centerline)"); ax[3].set_title("side upper outline (fender)"); ax[4].set_title("half width (top view)")
out = f"{ROOT}/results/phase1/proportion_{car}.png"
plt.tight_layout(); plt.savefig(out, dpi=100)
print("\n保存:", out)
