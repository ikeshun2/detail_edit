"""Screened Poisson再構成で、フロントの外板を1枚のつながった高解像度の面として作り直す
使い方: python 22_poisson_skin.py <target_id> <donor_id> [--zone 0.7] [--keep 0.6] [--depth 10]
                                  [--spacing 2.0] [--tau 15] [--trim 0.03]
  --zone:    作り直す範囲（前端からの距離 m）
  --keep:    作り直した面を使う範囲（前端からの距離 m）。これより後ろは元のCX-5
  --spacing: 部品の面から点を取る間隔 [mm]
  --trim:    点の密度が低い部分（すき間に張られた膜など）を取り除く割合
入力: results/partsfill_<donor_id>/parts_only.obj, slat_blend.obj（21_parts_fill.py の出力）
出力: results/partsfill_<donor_id>/poisson/
"""
import os, sys, json, shutil, argparse
ROOT = os.path.expanduser("~/Desktop/ikeda/detail_edit")
REPO = f"{ROOT}/repos/3dmorph"
sys.path.insert(0, REPO); os.chdir(REPO)
os.system("pgrep -f 'Xvfb :99' >/dev/null || Xvfb :99 -screen 0 1024x768x16 &")
import numpy as np, trimesh, open3d as o3d
from PIL import Image
from _3DMorph.renderer.render_simple import BlenderRenderer

ap = argparse.ArgumentParser()
ap.add_argument("target"); ap.add_argument("donor")
ap.add_argument("--zone", type=float, default=0.7); ap.add_argument("--keep", type=float, default=0.6)
ap.add_argument("--depth", type=int, default=10); ap.add_argument("--spacing", type=float, default=2.0)
ap.add_argument("--tau", type=float, default=15); ap.add_argument("--trim", type=float, default=0.03)
a = ap.parse_args()
cd = f"{ROOT}/data/cars/{a.target}"; pd = f"{cd}/results/partsfill_{a.donor}"
od = f"{pd}/poisson"; os.makedirs(od, exist_ok=True)
target_obj = f"{cd}/unmodified.obj"
S = json.load(open(f"{cd}/norm.json"))["scale"]; mm = S / 1000       # 1 mm [obj単位]
def load(p):
    m = trimesh.load(p, force="mesh", process=False); m.merge_vertices(); return m
T, P, SL = load(target_obj), load(f"{pd}/parts_only.obj"), load(f"{pd}/slat_blend.obj")
zf = T.bounds[0][2]
dz = lambda pts: (pts[:, 2] - zf) / S                                   # 前端からの距離 [m]
if SL.volume < 0: SL.invert()

# --- ホイール（タイヤ・ホイール・ブレーキ）は作り直しの対象から外し、元のまま戻す ---
comps = P.split(only_watertight=False)
def is_wheel(c):
    ctr = c.bounds.mean(0); z = dz(c.triangles_center)
    return abs(ctr[0]) / S > 0.5 and (ctr[1] - T.bounds[0][1]) / S < 0.55 and z.min() > 0.3 and z.max() < 1.4
wheels = [c for c in comps if is_wheel(c)]
body = trimesh.util.concatenate([c for c in comps if not is_wheel(c)])
print(f"部品 {len(comps)} 個のうち、ホイールとして除外: {len(wheels)} 個")

def scene_of(m):
    s = o3d.t.geometry.RaycastingScene()
    lm = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(m.vertices), o3d.utility.Vector3iVector(m.faces))
    s.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(lm)); return s
S_body = scene_of(body)
dirs = (lambda n: (lambda i: np.stack([np.cos(np.pi*(1+5**.5)*i)*np.sin(np.arccos(1-2*i/n)),
                                        np.sin(np.pi*(1+5**.5)*i)*np.sin(np.arccos(1-2*i/n)),
                                        np.cos(np.arccos(1-2*i/n))], 1))(np.arange(n) + 0.5))(48)

def exterior(src, n):
    """src の面から点を取り、body に遮られずに外から見える点と外向きの法線を返す"""
    pts, fid = trimesh.sample.sample_surface(src, n); nrm = src.face_normals[fid].copy()
    R = 1.0; vis = np.zeros(len(pts), bool); seen = np.zeros_like(pts)
    for v in dirs:
        r = np.hstack([pts + v * R, np.repeat(-v[None], len(pts), 0)]).astype(np.float32)
        hit = S_body.cast_rays(o3d.core.Tensor(r))["t_hit"].numpy() > R - mm
        seen[hit & ~vis] = v; vis |= hit
    nrm[np.einsum("ij,ij->i", nrm, seen) < 0] *= -1
    return pts[vis], nrm[vis]

# --- 1. 部品の外板の点（フロントの範囲） ---
zone_faces = np.where(dz(body.triangles_center) < a.zone)[0]
src = body.submesh([zone_faces], append=True)
n = int(min(4e6, src.area / (a.spacing * mm) ** 2))
Pp, Np = exterior(src, n)
print(f"部品の外板の点: {len(Pp)}（取った点 {n} のうち外から見えるもの）")

# --- 2. 大きなすき間の形の手がかり（SLATの補間結果のうち、部品から離れた外から見える点） ---
sl_zone = SL.submesh([np.where(dz(SL.triangles_center) < a.zone)[0]], append=True)
qs, qf = trimesh.sample.sample_surface(sl_zone, 300000); qn = sl_zone.face_normals[qf]
far = S_body.compute_distance(o3d.core.Tensor(qs.astype(np.float32))).numpy() > a.tau * mm
r = np.hstack([qs + qn * 2 * mm, qn]).astype(np.float32)
exposed = ~(S_body.cast_rays(o3d.core.Tensor(r))["t_hit"].numpy() < 0.3 * S)
Pg, Ng = qs[far & exposed], qn[far & exposed]
print(f"すき間の手がかりの点（SLAT）: {len(Pg)}")

# --- 3. Screened Poisson 再構成 ---
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.vstack([Pp, Pg]))
pcd.normals = o3d.utility.Vector3dVector(np.vstack([Np, Ng]))
mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=a.depth, scale=1.1)
dens = np.asarray(dens)
mesh.remove_vertices_by_mask(dens < np.quantile(dens, a.trim))      # 点の少ない場所の膜を除く
skin = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles), process=False)
cell = np.ptp(np.vstack([Pp, Pg]), axis=0).max() * 1.1 / 2 ** a.depth / mm
print(f"Poisson: 格子の細かさ 約 {cell:.1f} mm, 面 {len(skin.faces)}")

# --- 4. 合成: 前端から keep m までは作り直した面、それより後ろは元のCX-5、ホイールは元のまま ---
ov = 0.02                                                              # 2cm 重ねる
front = skin.submesh([np.where(dz(skin.triangles_center) < a.keep + ov)[0]], append=True)
tail = T.submesh([np.where(dz(T.triangles_center) >= a.keep - ov)[0]], append=True)
final = trimesh.util.concatenate([front, tail] + wheels)
front.export(f"{od}/front_skin.obj"); final.export(f"{od}/final_poisson.obj")
print(f"合成: 作り直した面 {len(front.faces)} + CX-5 {len(tail.faces)} + ホイール {sum(len(w.faces) for w in wheels)}")

# --- 5. レンダリング ---
def render(obj, res, n, step, yaw0=0):
    tmp = f"{od}/_tmp"; shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    BlenderRenderer(res).render(obj, target_obj, save_dir=tmp, n_views=n, mode="linear",
        lin_view_args={"offset": (yaw0, 20), "step": step, "direction": "right", "set_fov": 30})
    ims = []
    for k in range(n):
        im = Image.open(f"{tmp}/render_{k:03d}.png").convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255)); bg.alpha_composite(im)
        ims.append(np.array(bg.convert("RGB")))
    shutil.rmtree(tmp); return ims
rows = []
for obj in [target_obj, f"{pd}/final.obj", f"{od}/final_poisson.obj"]:
    ims = render(obj, 512, 8, 45)
    rows.append(np.hstack([ims[k] for k in [1, 3, 0, 2]]))
Image.fromarray(np.vstack(rows)).save(f"{od}/compare_poisson.png")
crops = []
for yaw in [45, 135]:
    im = render(f"{od}/final_poisson.obj", 2048, 1, 0, yaw0=yaw)[0]
    H, W = im.shape[:2]
    crops.append(im[int(0.35*H):int(0.85*H), int(0.45*W):W] if yaw == 45 else im[int(0.35*H):int(0.85*H), 0:int(0.55*W)])
Image.fromarray(np.hstack(crops)).save(f"{od}/closeup_poisson.png")
print("保存:", od, "（final_poisson.obj, front_skin.obj, compare_poisson.png, closeup_poisson.png）")
