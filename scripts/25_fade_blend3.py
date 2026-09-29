"""ドナーの形を帯の中で徐々にCX-5の面へ溶け込ませる（第3版）
  ① CX-5の面の判定にはドナーの外板だけを使い、床下の面もすき間を防ぐために残す
  ② ドナーの外から見えない面を z_int まで残す（車の内側にあるものだけ）
  ③ 帯の終わりより前に収まる小さな部品（ランプなど）は溶け込ませない
使い方: python 25_fade_blend3.py <target_id> <donor_id> [--z1 0.20] [--z2 0.45] [--z_int 0.7]
                                 [--dcover 30] [--smooth_iters 10] [--protect_max 20000]
出力: data/cars/<target_id>/results/fade3_<donor_id>_<z1>_<z2>/
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
ap.add_argument("--z_int", type=float, default=0.7)
ap.add_argument("--dcover", type=float, default=30); ap.add_argument("--smooth_iters", type=int, default=10)
ap.add_argument("--protect_max", type=int, default=20000)   # この面の数より小さい部品を守る対象にする
a = ap.parse_args()
cd = f"{ROOT}/data/cars/{a.target}"
od = f"{cd}/results/fade3_{a.donor}_{a.z1}_{a.z2}"; os.makedirs(od, exist_ok=True)
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
    d = np.broadcast_to(d, orig.shape)
    r = np.hstack([orig, d]).astype(np.float32)
    return sc.cast_rays(o3d.core.Tensor(r))["t_hit"].numpy()
i = np.arange(48) + 0.5
ph, th = np.arccos(1 - 2 * i / 48), np.pi * (1 + 5 ** 0.5) * i
DIRS = np.stack([np.cos(th) * np.sin(ph), np.sin(th) * np.sin(ph), np.cos(ph)], 1)
def exterior_faces(mesh, mask):
    idx = np.where(mask)[0]
    pts, sc = mesh.triangles_center[idx], scene_of(mesh)
    nrm = mesh.face_normals[idx].copy()
    vis = np.zeros(len(idx), bool); seen = np.zeros_like(pts)
    for v in DIRS:
        hit = cast(sc, pts + v, -v) > 1 - mm
        seen[hit & ~vis] = v; vis |= hit
    nrm[np.einsum("ij,ij->i", nrm, seen) < 0] *= -1
    ex = np.zeros(len(mesh.faces), bool); ex[idx[vis]] = True
    N = np.zeros((len(mesh.faces), 3)); N[idx] = nrm
    return ex, N
sub = lambda m, mask: trimesh.Trimesh(m.vertices, m.faces[mask], process=False)

# --- ドナー: 外板は z2(+4cm) まで、外から見えない面は z_int まで ---
zD = dz(D.triangles_center)
exD, _ = exterior_faces(D, zD < a.z_int)
keep = np.where(((zD < a.z2 + 0.04) & exD) | ((zD < a.z_int) & ~exD))[0]
Dk = trimesh.Trimesh(D.vertices.copy(), D.faces[keep], process=False)
Dk.remove_unreferenced_vertices()
exDk, zDk = exD[keep], zD[keep]

# --- ③ 守る部品: 小さく、全体が帯の終わりより前にある部品 ---
lab = trimesh.graph.connected_component_labels(Dk.face_adjacency, node_count=len(Dk.faces))
n_lab = np.bincount(lab); zmax_lab = np.full(len(n_lab), -np.inf)
np.maximum.at(zmax_lab, lab, zDk)
prot_lab = (n_lab < a.protect_max) & (zmax_lab < a.z2) & (np.bincount(lab, weights=exDk.astype(float)) > 0)
prot_face = prot_lab[lab]
print(f"守る部品: {prot_lab.sum()} 個（面 {prot_face.sum()}）")

# --- CX-5の外板 ---
zT = dz(T.triangles_center)
exT, nT = exterior_faces(T, zT < a.z_int + 0.1)
T_ext_scene = scene_of(sub(T, exT))

# --- ドナーの外板の頂点を、CX-5の外板へ徐々に寄せる（守る部品は動かさない、移動量はならす） ---
V = Dk.vertices.copy()
t = np.clip((dz(V) - a.z1) / (a.z2 - a.z1), 0, 1)
w = t * t * (3 - 2 * t)
mv = np.zeros(len(V), bool); mv[Dk.faces[exDk & ~prot_face].ravel()] = True
mv[Dk.faces[prot_face].ravel()] = False
mv &= w > 0
q = T_ext_scene.compute_closest_points(o3d.core.Tensor(V[mv].astype(np.float32)))["points"].numpy()
disp = np.zeros_like(V); disp[mv] = w[mv, None] * (q - V[mv])
exact = disp.copy(); pin = mv & (w > 0.95)
e = Dk.edges_unique
A = coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(len(V), len(V))).tocsr()
deg = np.asarray(A.sum(1)).ravel(); deg[deg == 0] = 1
for _ in range(a.smooth_iters):
    avg = (A @ disp) / deg[:, None]
    disp[mv] = 0.5 * disp[mv] + 0.5 * avg[mv]
    disp[pin] = exact[pin]
V += disp; Dk.vertices = V

# --- ② ドナーの内部の面のうち、帯より後ろのものは「車の内側にある」ものだけ残す ---
D_ext_scene = scene_of(sub(Dk, exDk))
body_ext = trimesh.util.concatenate([sub(T, exT), sub(Dk, exDk)])
B_scene = scene_of(body_ext)
chk = np.where(~exDk & (zDk >= a.z1))[0]
c = Dk.triangles_center[chk]
inside = np.ones(len(chk), bool)
for d in [np.array([0, 1, 0]), np.array([1, 0, 0]), np.array([-1, 0, 0])]:  # 上・左・右
    inside &= np.isfinite(cast(B_scene, c + d * mm, d))
drop = np.zeros(len(Dk.faces), bool); drop[chk[~inside]] = True
Dk = trimesh.Trimesh(Dk.vertices, Dk.faces[~drop], process=False)
w_face_ok = ~drop
print(f"ドナー: 外板 {exDk.sum()} / 内部の面 {(~exDk).sum()}（うち外に突き出るため除外 {drop.sum()}）")

# --- ① CX-5: z2より後ろは全部。帯の中と、前側の床下は、ドナーの外板がない場所だけ残す ---
band = (zT >= a.z1) & (zT < a.z2)
under = (zT < a.z1) & (nT[:, 1] < -0.5)                            # 下向きの面（床下）
bi = np.where((band | under) & exT)[0]
c, n = T.triangles_center[bi], nT[bi]
covered = cast(D_ext_scene, c + n * mm, n) < 0.3 * S
hides = cast(D_ext_scene, c - n * mm, -n) < a.dcover * mm
coinc = D_ext_scene.compute_distance(o3d.core.Tensor(c.astype(np.float32))).numpy() < 2 * mm
keep_ext = np.zeros(len(T.faces), bool); keep_ext[bi[~covered & ~hides & ~coinc]] = True
keepT = (zT >= a.z2) | (band & ~exT) | keep_ext
Tk = T.submesh([np.where(keepT)[0]], append=True)
print(f"CX-5: 帯と床下の外板 {len(bi)} 面のうち残した面 {keep_ext.sum()}（うち床下 {np.sum(keep_ext & under)}）")

final = trimesh.util.concatenate([Dk, Tk])
final.export(f"{od}/final.obj")
Dk.visual.vertex_colors = np.hstack([(np.array([220, 80, 80]) * (1 - w[:, None]) + np.array([170, 170, 170]) * w[:, None]).astype(np.uint8),
                                     np.full((len(w), 1), 255, np.uint8)])
Tk.visual.vertex_colors = np.tile([120, 160, 220, 255], (len(Tk.vertices), 1))
trimesh.util.concatenate([Dk, Tk]).export(f"{od}/final_colored.ply")

# --- レンダリング（上から: CX-5 / 前回(fade2) / 今回） ---
def render(obj, res, n, step, yaw0=0, pitch=20):
    tmp = f"{od}/_tmp"; shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    BlenderRenderer(res).render(obj, target_obj, save_dir=tmp, n_views=n, mode="linear",
        lin_view_args={"offset": (yaw0, pitch), "step": step, "direction": "right", "set_fov": 30})
    ims = []
    for k in range(n):
        im = Image.open(f"{tmp}/render_{k:03d}.png").convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255)); bg.alpha_composite(im)
        ims.append(np.array(bg.convert("RGB")))
    shutil.rmtree(tmp); return ims
prev = f"{cd}/results/fade2_{a.donor}_{a.z1}_{a.z2}/final.obj"
rows = []
for obj in [target_obj] + ([prev] if os.path.exists(prev) else []) + [f"{od}/final.obj"]:
    ims = render(obj, 512, 8, 45)
    rows.append(np.hstack([ims[k] for k in [1, 3, 0, 2]]))
Image.fromarray(np.vstack(rows)).save(f"{od}/compare_fade3.png")
crops = []
for yaw, pitch in [(45, 20), (90, 20), (60, -40)]:                  # 斜め前、正面、斜め下から
    im = render(f"{od}/final.obj", 2048, 1, 0, yaw0=yaw, pitch=pitch)[0]
    H, W = im.shape[:2]
    crops.append(im[int(0.35*H):int(0.85*H), int(0.45*W):W] if yaw == 45 else im[int(0.3*H):int(0.8*H), int(0.2*W):int(0.8*W)])
h = min(c.shape[0] for c in crops)
Image.fromarray(np.hstack([c[:h] for c in crops])).save(f"{od}/closeup_fade3.png")
print("保存:", od, "（final.obj, final_colored.ply, compare_fade3.png, closeup_fade3.png）")
