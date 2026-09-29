"""編集結果の比較（画像と数値）
使い方: python 09_compare.py <car_id> <donor_id> <result_tag> [ratio=0.2]
例:     python 09_compare.py cx5_001 donor_001 donor_001_r0.2_seed42
出力:   results/<result_tag>/compare_renders.png, change_map.png
"""
import sys, os, shutil
ROOT = os.path.expanduser("~/Desktop/ikeda/detail_edit")
REPO = f"{ROOT}/repos/3dmorph"
sys.path.insert(0, REPO); os.chdir(REPO)
os.system("pgrep -f 'Xvfb :99' >/dev/null || Xvfb :99 -screen 0 1024x768x16 &")

import numpy as np, trimesh, open3d as o3d
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
from _3DMorph.renderer.render_simple import BlenderRenderer

car, did, tag = sys.argv[1], sys.argv[2], sys.argv[3]
ratio = float(sys.argv[4]) if len(sys.argv) > 4 else 0.2
cd = f"{ROOT}/data/cars/{car}"; rd = f"{cd}/results/{tag}"
target_obj = f"{cd}/unmodified.obj"

# --- 座標系を揃える: SLATの出力(最長辺=1) -> unmodified.obj の座標系 ---
t = trimesh.load(target_obj, force="mesh")
c_t = (t.bounds[0] + t.bounds[1]) / 2; ext_t = np.ptp(t.vertices, axis=0).max()
MM = ext_t / (0.98 / t.bounds[1][2] * t.bounds[1][2]) if False else None
import json
MM = ext_t / json.load(open(f"{cd}/norm.json"))["scale"] * 1000     # obj座標1単位 -> mm
def to_obj_frame(path):
    m = trimesh.load(path, force="mesh"); m.vertices = m.vertices * ext_t + c_t; return m
before = to_obj_frame(f"{cd}/recon/slat.ply")
after = to_obj_frame(f"{rd}/edited_raw.ply")
donor = trimesh.load(f"{cd}/donors/{did}_aligned.obj", force="mesh")
before.export(f"{rd}/_before.obj"); after.export(f"{rd}/_after.obj")

z_front, L = t.bounds[0][2], np.ptp(t.vertices[:, 2])
z_cut = z_front + ratio * L

# --- 距離計算の道具 ---
def scene_of(m):
    s = o3d.t.geometry.RaycastingScene()
    lm = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(m.vertices), o3d.utility.Vector3iVector(m.faces))
    s.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(lm)); return s
def visible(s, pts, n=64, R=3.0, eps_mm=3.0):
    i = np.arange(n) + .5; ph, th = np.arccos(1 - 2*i/n), np.pi*(1 + 5**.5)*i
    dirs = np.stack([np.cos(th)*np.sin(ph), np.sin(th)*np.sin(ph), np.cos(ph)], 1)
    vis = np.zeros(len(pts), bool)
    for v in dirs:
        r = np.hstack([pts + v*R, np.repeat(-v[None], len(pts), 0)]).astype(np.float32)
        vis |= s.cast_rays(o3d.core.Tensor(r))["t_hit"].numpy() > R - eps_mm/MM
    return vis
def dist(s, pts): return s.compute_distance(o3d.core.Tensor(pts.astype(np.float32))).numpy() * MM
S_b, S_a, S_d = scene_of(before), scene_of(after), scene_of(donor)

pa, _ = trimesh.sample.sample_surface(after, 200000); pa = pa[visible(S_a, pa)]
change = dist(S_b, pa)                                   # 編集後の外板点 -> 編集前の表面

# --- 数値 ---
print(f"フロントの範囲（切り貼りした範囲）: 前端から {ratio*L*MM/1000:.2f} m")
ch = pa[change > 10]
if len(ch):
    reach = (np.percentile(ch[:, 2], 99) - z_front) * MM / 1000
    print(f"10mm以上変化した点: 外板の {len(ch)/len(pa)*100:.1f}% / 前端から {reach:.2f} m まで（99%点）")
rest = pa[:, 2] > z_cut + 0.05 * L
print(f"フロントより後ろの変化: 中央値 {np.median(change[rest]):.1f} mm, 90% {np.percentile(change[rest],90):.1f} mm, "
      f"10mm超 {np.mean(change[rest] > 10)*100:.1f}%")
pd, _ = trimesh.sample.sample_surface(donor, 200000); pd = pd[visible(S_d, pd)]
pd = pd[pd[:, 2] < z_cut]
db, da = dist(S_b, pd), dist(S_a, pd)
print(f"ドナーのフロントまでの距離（中央値）: 編集前 {np.median(db):.1f} mm -> 編集後 {np.median(da):.1f} mm")
print(f"  ドナーのフロント表面が20mm以内に再現された割合: 編集前 {np.mean(db<20)*100:.0f}% -> 編集後 {np.mean(da<20)*100:.0f}%")

# --- 変化の分布図 ---
fig, ax = plt.subplots(2, 1, figsize=(12, 8))
for a, (i, j, name) in zip(ax, [(2, 1, "side (z-y)"), (2, 0, "top (z-x)")]):
    sc = a.scatter(pa[:, i], pa[:, j], c=np.clip(change, 0, 50), s=0.3, cmap="jet", vmin=0, vmax=50)
    a.axvline(z_cut, color="k", ls="--", lw=1); a.text(z_cut, a.get_ylim()[1], " cut", va="top")
    a.set_aspect("equal"); a.set_title(f"change after editing [mm]  {name}  (front = left)")
fig.colorbar(sc, ax=ax, shrink=0.8)
plt.savefig(f"{rd}/change_map.png", dpi=110, bbox_inches="tight")

# --- 比較画像（3モデル x 4方向） ---
renderer = BlenderRenderer(512)
picks = [1, 3, 0, 5]          # 45°, 135°, 0°, 225°（pitch 20°）
rows = []
for name, obj in [("before", f"{rd}/_before.obj"), ("after", f"{rd}/_after.obj"),
                  ("donor", f"{cd}/donors/{did}_aligned.obj")]:
    d = f"{rd}/_tmp_{name}"; shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    renderer.render(obj, target_obj, save_dir=d, n_views=8, mode="linear",
                    lin_view_args={"offset": (0, 20), "step": 45, "direction": "right", "set_fov": 30})
    ims = []
    for k in picks:
        im = Image.open(f"{d}/render_{k:03d}.png").convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255)); bg.alpha_composite(im)
        ims.append(np.array(bg.convert("RGB")))
    rows.append(np.hstack(ims)); shutil.rmtree(d)
Image.fromarray(np.vstack(rows)).save(f"{rd}/compare_renders.png")
print("保存:", f"{rd}/compare_renders.png", f"{rd}/change_map.png")
print("比較画像: 上から before / after / donor、左から 45° / 135° / 0°(真横) / 225°(斜め後ろ)")
