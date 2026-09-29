"""ドナーの形を、帯の中で徐々にCX-5の面へ溶け込ませる（高解像度・生成なし）
使い方: python 23_fade_blend.py <target_id> <donor_id> [--z1 0.30] [--z2 0.50] [--absent 15]
  z1〜z2（前端からの距離 m）: ドナーの形が 100% → 0% に変わる帯
  --absent: 帯の中で、ドナーの面からこの距離[mm]以上離れたCX-5の面は残す（ドナーの面がない場所の穴を防ぐ）
出力: data/cars/<target_id>/results/fade_<donor_id>_<z1>_<z2>/
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
ap.add_argument("--z1", type=float, default=0.30); ap.add_argument("--z2", type=float, default=0.50)
ap.add_argument("--absent", type=float, default=15)
a = ap.parse_args()
cd = f"{ROOT}/data/cars/{a.target}"
od = f"{cd}/results/fade_{a.donor}_{a.z1}_{a.z2}"; os.makedirs(od, exist_ok=True)
target_obj = f"{cd}/unmodified.obj"
S = json.load(open(f"{cd}/norm.json"))["scale"]; mm = S / 1000
def load(p):
    m = trimesh.load(p, force="mesh", process=False); m.merge_vertices(); return m
T, D = load(target_obj), load(f"{cd}/donors/{a.donor}_aligned.obj")
zf = T.bounds[0][2]
dz = lambda p: (p[:, 2] - zf) / S                                  # 前端からの距離 [m]

def scene_of(m):
    s = o3d.t.geometry.RaycastingScene()
    lm = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(m.vertices), o3d.utility.Vector3iVector(m.faces))
    s.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(lm)); return s
i = np.arange(48) + 0.5
ph, th = np.arccos(1 - 2 * i / 48), np.pi * (1 + 5 ** 0.5) * i
DIRS = np.stack([np.cos(th) * np.sin(ph), np.sin(th) * np.sin(ph), np.cos(ph)], 1)
def exterior_faces(mesh, mask=None):
    """外から見える面（車内などの隠れた面を除く）"""
    idx = np.arange(len(mesh.faces)) if mask is None else np.where(mask)[0]
    pts, sc = mesh.triangles_center[idx], scene_of(mesh)
    vis = np.zeros(len(idx), bool)
    for v in DIRS:
        r = np.hstack([pts + v, np.repeat(-v[None], len(pts), 0)]).astype(np.float32)
        vis |= sc.cast_rays(o3d.core.Tensor(r))["t_hit"].numpy() > 1 - mm
    out = np.zeros(len(mesh.faces), bool); out[idx[vis]] = True
    return out

# --- ドナー: z1より前は全部、帯の中は外から見える面だけ ---
zD = dz(D.triangles_center)
exD = exterior_faces(D, zD < a.z2 + 0.02)
Dk = D.submesh([np.where((zD < a.z1) | ((zD < a.z2 + 0.02) & exD))[0]], append=True)

# --- CX-5の外板 ---
zT = dz(T.triangles_center)
exT = exterior_faces(T, zT < a.z2 + 0.3)
T_ext_scene = scene_of(T.submesh([np.where(exT)[0]], append=True))

# --- 帯の中で、ドナーの頂点をCX-5の外板へ徐々に寄せる ---
V = Dk.vertices.copy()
t = np.clip((dz(V) - a.z1) / (a.z2 - a.z1), 0, 1)
w = t * t * (3 - 2 * t)                                            # 0(ドナー) → 1(CX-5) をなめらかに
mv = w > 0
q = T_ext_scene.compute_closest_points(o3d.core.Tensor(V[mv].astype(np.float32)))["points"].numpy()
shift = np.linalg.norm(q - V[mv], axis=1) / mm
V[mv] += w[mv, None] * (q - V[mv])
Dk.vertices = V
print(f"帯の中で動かした頂点: {mv.sum()} | CX-5の面までの距離: 中央値 {np.median(shift):.1f} mm / 90% {np.percentile(shift, 90):.1f} mm")

# --- CX-5: z2より後ろは全部。帯の中は、隠れた面と、ドナーの面がない場所の外板だけ残す ---
d_to_D = scene_of(Dk).compute_distance(o3d.core.Tensor(T.triangles_center.astype(np.float32))).numpy() / mm
keepT = (zT >= a.z2) | ((zT >= a.z1) & (~exT | (d_to_D > a.absent)))
Tk = T.submesh([np.where(keepT)[0]], append=True)
print(f"面の数: ドナー {len(Dk.faces)} / CX-5 {len(Tk.faces)}")

final = trimesh.util.concatenate([Dk, Tk])
final.export(f"{od}/final.obj")
col = (np.array([220, 80, 80]) * (1 - w[:, None]) + np.array([170, 170, 170]) * w[:, None]).astype(np.uint8)
Dk.visual.vertex_colors = np.hstack([col, np.full((len(col), 1), 255, np.uint8)])
Tk.visual.vertex_colors = np.tile([170, 170, 170, 255], (len(Tk.vertices), 1))
trimesh.util.concatenate([Dk, Tk]).export(f"{od}/final_colored.ply")

# --- レンダリング（上から: CX-5 / compare_hybrid の結果 / 今回） ---
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
prev = f"{cd}/results/softsplice_donor_001ffd_r0.1_w6/hybrid/hybrid.obj"
rows = []
for obj in [target_obj] + ([prev] if os.path.exists(prev) else []) + [f"{od}/final.obj"]:
    ims = render(obj, 512, 8, 45)
    rows.append(np.hstack([ims[k] for k in [1, 3, 0, 2]]))
Image.fromarray(np.vstack(rows)).save(f"{od}/compare_fade.png")
crops = []
for yaw in [45, 135]:
    im = render(f"{od}/final.obj", 2048, 1, 0, yaw0=yaw)[0]
    H, W = im.shape[:2]
    crops.append(im[int(0.35*H):int(0.85*H), int(0.45*W):W] if yaw == 45 else im[int(0.35*H):int(0.85*H), 0:int(0.55*W)])
Image.fromarray(np.hstack(crops)).save(f"{od}/closeup_fade.png")
print("保存:", od, "（final.obj, final_colored.ply, compare_fade.png, closeup_fade.png）")
