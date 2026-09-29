"""部品単位での置換のプレビュー（SLATでの補完はまだしない）
使い方: python 20_parts_preview.py <target_id> <donor_id> [--zmin 0.3] [--zmax 0.65]
  フロントの部品 = 前端から zmin m 以内に面があり、かつ面の99%が前端から zmax m 以内にある部品
出力: data/cars/<target_id>/results/parts_<donor_id>/
"""
import os, sys, json, shutil, argparse
ROOT = os.path.expanduser("~/Desktop/ikeda/detail_edit")
REPO = f"{ROOT}/repos/3dmorph"
sys.path.insert(0, REPO); os.chdir(REPO)
os.system("pgrep -f 'Xvfb :99' >/dev/null || Xvfb :99 -screen 0 1024x768x16 &")
import numpy as np, trimesh
from PIL import Image
from _3DMorph.renderer.render_simple import BlenderRenderer

ap = argparse.ArgumentParser()
ap.add_argument("target"); ap.add_argument("donor")
ap.add_argument("--zmin", type=float, default=0.3)
ap.add_argument("--zmax", type=float, default=0.65)
a = ap.parse_args()
cd = f"{ROOT}/data/cars/{a.target}"
od = f"{cd}/results/parts_{a.donor}"; os.makedirs(od, exist_ok=True)
target_obj = f"{cd}/unmodified.obj"
S = json.load(open(f"{cd}/norm.json"))["scale"]

def load(path):
    m = trimesh.load(path, force="mesh", process=False); m.merge_vertices(); return m
T, D = load(target_obj), load(f"{cd}/donors/{a.donor}_aligned.obj")
zf = T.bounds[0][2]

def classify(mesh, name):
    comps = mesh.split(only_watertight=False)
    front, rest = [], []
    for c in comps:
        z = (c.triangles_center[:, 2] - zf) / S
        (front if (z.min() < a.zmin and np.percentile(z, 99) <= a.zmax) else rest).append(c)
    print(f"\n[{name}] 部品 {len(comps)} 個 → フロントの部品 {len(front)} 個"
          f"（面 {sum(len(c.faces) for c in front)}）/ それ以外 {len(rest)} 個")
    print("  フロントの部品（面の数が多い順）: 面の数, 前端からの範囲 [m]")
    for c in sorted(front, key=lambda c: -len(c.faces))[:15]:
        z = (c.triangles_center[:, 2] - zf) / S
        print(f"    {len(c.faces):7d}  {z.min():.2f} 〜 {z.max():.2f}")
    return trimesh.util.concatenate(front), trimesh.util.concatenate(rest)

d_front, _ = classify(D, "ドナー")
t_front, t_rest = classify(T, "CX-5")
d_front.export(f"{od}/donor_front.obj"); t_front.export(f"{od}/cx5_front.obj"); t_rest.export(f"{od}/cx5_rest.obj")
comp = trimesh.util.concatenate([d_front, t_rest]); comp.export(f"{od}/composite.obj")
d_front.visual.vertex_colors = np.tile([220, 80, 80, 255], (len(d_front.vertices), 1))
t_rest.visual.vertex_colors = np.tile([170, 170, 170, 255], (len(t_rest.vertices), 1))
trimesh.util.concatenate([d_front, t_rest]).export(f"{od}/composite_colored.ply")

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
for obj in [target_obj, f"{od}/composite.obj"]:
    ims = render(obj, 512, 8, 45)
    rows.append(np.hstack([ims[k] for k in [1, 3, 0, 2]]))      # 45°, 135°, 真横, 正面
Image.fromarray(np.vstack(rows)).save(f"{od}/compare_parts.png")
crops = []
for yaw in [45, 135]:
    im = render(f"{od}/composite.obj", 2048, 1, 0, yaw0=yaw)[0]
    H, W = im.shape[:2]
    crops.append(im[int(0.35*H):int(0.85*H), int(0.45*W):W] if yaw == 45 else im[int(0.35*H):int(0.85*H), 0:int(0.55*W)])
Image.fromarray(np.hstack(crops)).save(f"{od}/closeup_parts.png")
print("\n保存:", od, "（composite.obj, composite_colored.ply, compare_parts.png, closeup_parts.png）")
