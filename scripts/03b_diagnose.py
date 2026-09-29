"""再構成誤差の分解と可視化
使い方: python 03b_diagnose.py <car_id>
出力: recon/gt_err.ply（真値の点を網羅性の誤差で色付け）
      recon/recon_err.ply（再構成の点を正確さの誤差で色付け）
"""
import sys, os, json
import numpy as np, trimesh
from scipy.spatial import cKDTree

ROOT = os.path.expanduser("~/Desktop/ikeda/detail_edit")
car = sys.argv[1]; d = f"{ROOT}/data/cars/{car}"
recon = trimesh.load(f"{d}/recon/slat.ply", force="mesh")
gt = trimesh.load(f"{d}/recon/gt_normalized.obj", force="mesh")
norm = json.load(open(f"{d}/norm.json"))
obj = trimesh.load(f"{d}/unmodified.obj", force="mesh")
mm = (obj.bounds[1] - obj.bounds[0]).max() / norm["scale"] * 1000   # 正規化1単位 -> mm

N = 300_000
pr, _ = trimesh.sample.sample_surface(recon, N)
pg, _ = trimesh.sample.sample_surface(gt, N)
acc = cKDTree(pg).query(pr)[0] * mm     # 正確さ: 再構成 -> 真値
comp = cKDTree(pr).query(pg)[0] * mm    # 網羅性: 真値 -> 再構成

def stats(name, x):
    p = np.percentile(x, [50, 90, 95, 99])
    print(f"{name}: 平均 {x.mean():6.1f} | 中央値 {p[0]:5.1f} | 90% {p[1]:6.1f} | "
          f"95% {p[2]:6.1f} | 99% {p[3]:6.1f} | 最大 {x.max():6.1f}  [mm]")
stats("正確さ (recon->gt)", acc)
stats("網羅性 (gt->recon)", comp)

print("\nF-score（距離τ以内に入る点の割合）")
for t in [2, 5, 10, 20]:
    P, R = (acc < t).mean(), (comp < t).mean()
    F = 2 * P * R / (P + R + 1e-12)
    print(f"  τ={t:2d}mm: 適合率 {P:.3f}  再現率 {R:.3f}  F {F:.3f}")

# 網羅性の誤差が大きい真値の点はどこにあるか
far = pg[comp > 50]
print(f"\n網羅性の誤差 >50mm の真値の点: {len(far)/N*100:.1f}%")
if len(far):
    print("  その点の範囲 min:", np.round(far.min(0), 3).tolist(),
          " max:", np.round(far.max(0), 3).tolist())
    # 車体の外形（真値のバウンディングボックス）に対して、どれだけ内側にあるか
    inner = np.minimum(far - gt.bounds[0], gt.bounds[1] - far).min(1) * mm
    print(f"  外形からの距離（中央値）: {np.median(inner):.0f} mm  ※大きいほど車の内部")

# 色付き点群の書き出し（青=小さい誤差, 赤=大きい誤差、0〜50mmで色分け）
def colorize(pts, err, path, vmax=50.0):
    t = np.clip(err / vmax, 0, 1)[:, None]
    col = (np.hstack([t, np.zeros_like(t), 1 - t]) * 255).astype(np.uint8)
    trimesh.PointCloud(pts, colors=np.hstack([col, 255 * np.ones((len(pts), 1), np.uint8)])).export(path)
colorize(pg, comp, f"{d}/recon/gt_err.ply")
colorize(pr, acc, f"{d}/recon/recon_err.ply")
print("\n書き出し: recon/gt_err.ply, recon/recon_err.ply")
