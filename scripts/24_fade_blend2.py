"""ドナーの形を帯の中で徐々にCX-5の面へ溶け込ませる（改良版）
  - 帯の中のドナーの面はすべて残す（すき間の防止）
  - 頂点の移動量を隣り合う頂点の間でならす（面の裂けの防止）
  - ドナーの外から見えない面（ダクト・裏打ち）は動かさずに残す（吸気口の透けの防止）
  - 帯の中のCX-5の外板は、光線で前後を調べ、ドナーの面がない場所だけ残す
使い方: python 24_fade_blend2.py <target_id> <donor_id> [--z1 0.20] [--z2 0.45] [--dcover 30] [--smooth_iters 10]
出力: data/cars/<target_id>/results/fade2_<donor_id>_<z1>_<z2>/
"""
import os, sys, json, shutil, argparse
ROOT = os.path.expanduser("~/Desktop/ikeda/detail_edit")
REPO = f"{ROOT}/repos/3dmorph"
sys.path.insert(0, REPO); os.chdir(REPO)
os.system("pgrep -f 'Xvfb :99' >/dev/null || Xvfb :99 -screen 0 1024x768x16 &")
import numpy as np, trimesh, open3d as o3d
from scipy.sparse import coo_matrix
from PIL import Image
from _3DMorph.renderer.render_simple import BlenderRenderer

ap = argparse.ArgumentParser()
ap.add_argument("target"); ap.add_argument("donor")
ap.add_argument("--z1", type=float, default=0.20); ap.add_argument("--z2", type=float, default=0.45)
ap.add_argument("--dcover", type=float, default=30); ap.add_argument("--smooth_iters", type=int, default=10)
a = ap.parse_args()
cd = f"{ROOT}/data/cars/{a.target}"
od = f"{cd}/results/fade2_{a.donor}_{a.z1}_{a.z2}"; os.makedirs(od, exist_ok=True)
target_obj = f"{cd}/unmodified.obj"
S = json.load(open(f"{cd}/norm.json"))["scale"]; mm = S / 1000
def load(p):
    m = trimesh.load(p, force="mesh", process=False); m.merge_vertices(); return m
T, D = load(target_obj), load(f"{cd}/donors/{a.donor}_aligned.obj")
zf = T.bounds[0][2]
dz = lambda p: (p[:, 2] - zf) / S

def scene_of(m):
    s = o3d.t.geometry.RaycastingScene()
    lm = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(m.vertices), o3d.utility.Vector3iVector(m.faces))
    s.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(lm)); return s
def cast(sc, orig, d):
    r = np.hstack([orig, d]).astype(np.float32)
    return sc.cast_rays(o3d.core.Tensor(r))["t_hit"].numpy()
i = np.arange(48) + 0.5
ph, th = np.arccos(1 - 2 * i / 48), np.pi * (1 + 5 ** 0.5) * i
DIRS = np.stack([np.cos(th) * np.sin(ph), np.sin(th) * np.sin(ph), np.cos(ph)], 1)
def exterior_faces(mesh, mask):
    """外から見える面と、その外向きの法線（見えた方向に揃える）"""
    idx = np.where(mask)[0]
    pts, sc = mesh.triangles_center[idx], scene_of(mesh)
    nrm = mesh.face_normals[idx].copy()
    vis = np.zeros(len(idx), bool); seen = np.zeros_like(pts)
    for v in DIRS:
        hit = cast(sc, pts + v, np.repeat(-v[None], len(pts), 0)) > 1 - mm
        seen[hit & ~vis] = v; vis |= hit
    nrm[np.einsum("ij,ij->i", nrm, seen) < 0] *= -1
    ex = np.zeros(len(mesh.faces), bool); ex[idx[vis]] = True
    N = np.zeros((len(mesh.faces), 3)); N[idx] = nrm
    return ex, N

# --- ドナー: 前端から z2(+4cm) までの面をすべて残す ---
zD = dz(D.triangles_center)
keep = np.where(zD < a.z2 + 0.04)[0]
exD, _ = exterior_faces(D, zD < a.z2 + 0.04)
Dk = trimesh.Trimesh(D.vertices.copy(), D.faces[keep], process=False)
Dk.remove_unreferenced_vertices()
exDk = exD[keep]

# --- CX-5の外板 ---
zT = dz(T.triangles_center)
exT, nT = exterior_faces(T, zT < a.z2 + 0.3)
T_ext_scene = scene_of(trimesh.Trimesh(T.vertices, T.faces[exT], process=False))

# --- ドナーの外板の頂点だけを、CX-5の外板へ徐々に寄せる（移動量はならす） ---
V = Dk.vertices.copy()
t = np.clip((dz(V) - a.z1) / (a.z2 - a.z1), 0, 1)
w = t * t * (3 - 2 * t)
mv = np.zeros(len(V), bool); mv[Dk.faces[exDk].ravel()] = True
mv &= w > 0
q = T_ext_scene.compute_closest_points(o3d.core.Tensor(V[mv].astype(np.float32)))["points"].numpy()
disp = np.zeros_like(V); disp[mv] = w[mv, None] * (q - V[mv])
exact = disp.copy(); pin = mv & (w > 0.95)                         # 帯の終わりは正確にCX-5に合わせる
e = Dk.edges_unique
A = coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(len(V), len(V))).tocsr()
deg = np.asarray(A.sum(1)).ravel(); deg[deg == 0] = 1
for _ in range(a.smooth_iters):
    avg = (A @ disp) / deg[:, None]
    disp[mv] = 0.5 * disp[mv] + 0.5 * avg[mv]
    disp[pin] = exact[pin]
V += disp
Dk.vertices = V
print(f"ドナー: 面 {len(Dk.faces)}（外板 {exDk.sum()} / 外から見えない面 {(~exDk).sum()}）| 動かした頂点 {mv.sum()}"
      f" | 移動量 中央値 {np.median(np.linalg.norm(disp[mv], axis=1))/mm:.1f} mm")

# --- CX-5: z2より後ろは全部。帯の中は、内側の面と、ドナーの面がない場所の外板だけ残す ---
SDk = scene_of(Dk)
band = (zT >= a.z1) & (zT < a.z2)
bi = np.where(band & exT)[0]
c, n = T.triangles_center[bi], nT[bi]
covered = cast(SDk, c + n * mm, n) < 0.3 * S                       # 外側がドナーに覆われている
hides = cast(SDk, c - n * mm, -n) < a.dcover * mm                  # すぐ内側にドナーの面がある
coinc = SDk.compute_distance(o3d.core.Tensor(c.astype(np.float32))).numpy() < 2 * mm
keep_ext = np.zeros(len(T.faces), bool); keep_ext[bi[~covered & ~hides & ~coinc]] = True
keepT = (zT >= a.z2) | (band & ~exT) | keep_ext
Tk = T.submesh([np.where(keepT)[0]], append=True)
print(f"CX-5: 帯の中の外板 {len(bi)} 面のうち、残した面（ドナーの面がない場所） {keep_ext.sum()} | 合計 {len(Tk.faces)} 面")

final = trimesh.util.concatenate([Dk, Tk])
final.export(f"{od}/final.obj")
col = (np.array([220, 80, 80]) * (1 - w[:, None]) + np.array([170, 170, 170]) * w[:, None]).astype(np.uint8)
Dk.visual.vertex_colors = np.hstack([col, np.full((len(col), 1), 255, np.uint8)])
Tk.visual.vertex_colors = np.tile([120, 160, 220, 255], (len(Tk.vertices), 1))    # CX-5は青で区別
trimesh.util.concatenate([Dk, Tk]).export(f"{od}/final_colored.ply")

# --- レンダリング（上から: CX-5 / 前回の結果 / 今回） ---
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
prev = f"{cd}/results/fade_{a.donor}_{a.z1}_{a.z2}/final.obj"
rows = []
for obj in [target_obj] + ([prev] if os.path.exists(prev) else []) + [f"{od}/final.obj"]:
    ims = render(obj, 512, 8, 45)
    rows.append(np.hstack([ims[k] for k in [1, 3, 0, 2]]))
Image.fromarray(np.vstack(rows)).save(f"{od}/compare_fade2.png")
crops = []
for yaw in [45, 135, 90]:
    im = render(f"{od}/final.obj", 2048, 1, 0, yaw0=yaw)[0]
    H, W = im.shape[:2]
    if yaw == 45:   crops.append(im[int(0.35*H):int(0.85*H), int(0.45*W):W])
    elif yaw == 135: crops.append(im[int(0.35*H):int(0.85*H), 0:int(0.55*W)])
    else:           crops.append(im[int(0.35*H):int(0.85*H), int(0.2*W):int(0.8*W)])
h = min(c.shape[0] for c in crops)
Image.fromarray(np.hstack([c[:h] for c in crops])).save(f"{od}/closeup_fade2.png")
print("保存:", od, "（final.obj, final_colored.ply, compare_fade2.png, closeup_fade2.png）")
