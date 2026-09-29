"""ドナーの形を帯の中で徐々にCX-5の面へ溶け込ませる（第4版）
  - ドナーの面を「表皮（CX-5の外板の近く）」と「奥の面」に分け、表皮だけを溶け込ませる（吸気口の奥を保つ）
  - CX-5の床下の面は、地面近くの下向きの面だけ残す
  - 結合部分の少し内側に、SLATで作った裏打ちの面を置く（すき間の奥を空洞にしない）
使い方: python 26_fade_blend4.py <target_id> <donor_id> [--z1 0.20] [--z2 0.45] [--z_int 0.7] [--dskin 40]
                                 [--liner 25] [--slat_dir <ドナーのSLATのフォルダ>]
出力: data/cars/<target_id>/results/fade4_<donor_id>_<z1>_<z2>/
"""
import os, sys, glob, json, shutil, argparse
ROOT = os.path.expanduser("~/Desktop/ikeda/detail_edit")
REPO = f"{ROOT}/repos/3dmorph"
sys.path.insert(0, REPO); os.chdir(REPO)
import xformers_patch
os.system("pgrep -f 'Xvfb :99' >/dev/null || Xvfb :99 -screen 0 1024x768x16 &")
os.environ["ATTN_BACKEND"] = "xformers"; os.environ["SPCONV_ALGO"] = "native"
import numpy as np, torch, trimesh, open3d as o3d
from scipy.sparse import coo_matrix
from PIL import Image
from trellis.pipelines import _3DMorphInpaint
from trellis.modules import sparse as sp
from _3DMorph.renderer.render_simple import BlenderRenderer

ap = argparse.ArgumentParser()
ap.add_argument("target"); ap.add_argument("donor")
ap.add_argument("--z1", type=float, default=0.20); ap.add_argument("--z2", type=float, default=0.45)
ap.add_argument("--z_int", type=float, default=0.7)
ap.add_argument("--dskin", type=float, default=40)        # CX-5の外板からこの距離[mm]以内のドナーの面を「表皮」とする
ap.add_argument("--dcover", type=float, default=30); ap.add_argument("--smooth_iters", type=int, default=10)
ap.add_argument("--protect_max", type=int, default=20000)
ap.add_argument("--liner", type=float, default=25)        # 裏打ちの面を表面からこの距離[mm]内側に置く
ap.add_argument("--slat_dir", default=None)
a = ap.parse_args()
cd = f"{ROOT}/data/cars/{a.target}"
slat_dir = a.slat_dir or f"{cd}/donors/donor_001ffd"
od = f"{cd}/results/fade4_{a.donor}_{a.z1}_{a.z2}_s{int(a.dskin)}"; os.makedirs(od, exist_ok=True)
target_obj = f"{cd}/unmodified.obj"
S = json.load(open(f"{cd}/norm.json"))["scale"]; mm = S / 1000
def load(p):
    m = trimesh.load(p, force="mesh", process=False); m.merge_vertices(); return m
T, D = load(target_obj), load(f"{cd}/donors/{a.donor}_aligned.obj")
zf, yg = T.bounds[0][2], T.bounds[0][1]
c0, ext = (T.bounds[0] + T.bounds[1]) / 2, np.ptp(T.vertices, axis=0).max()
dz = lambda p: (p[:, 2] - zf) / S
smooth = lambda t: t * t * (3 - 2 * t)

def scene_of(m):
    s = o3d.t.geometry.RaycastingScene()
    lm = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(m.vertices), o3d.utility.Vector3iVector(m.faces))
    s.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(lm)); return s
def cast(sc, orig, d):
    d = np.broadcast_to(d, orig.shape)
    return sc.cast_rays(o3d.core.Tensor(np.hstack([orig, d]).astype(np.float32)))["t_hit"].numpy()
dist = lambda sc, p: sc.compute_distance(o3d.core.Tensor(p.astype(np.float32))).numpy()
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

# --- CX-5の外板 ---
zT = dz(T.triangles_center)
exT, nT = exterior_faces(T, zT < a.z_int + 0.1)
S_Text = scene_of(sub(T, exT))

# --- ドナー: 表皮（CX-5の外板の近く）と奥の面に分ける ---
zD = dz(D.triangles_center)
dD = np.full(len(zD), np.inf); m_ = zD < a.z_int
dD[m_] = dist(S_Text, D.triangles_center[m_]) / mm
skin = dD < a.dskin
keep = (zD < a.z1) | ((zD < a.z2 + 0.04)) | ((zD < a.z_int) & ~skin)
keep_idx = np.where(keep)[0]
Dk = trimesh.Trimesh(D.vertices.copy(), D.faces[keep_idx], process=False); Dk.remove_unreferenced_vertices()
skinK, zDk = skin[keep_idx], zD[keep_idx]

# 守る部品（ランプなど）: 小さく、全体が帯の終わりより前にあり、表皮を含む部品
lab = trimesh.graph.connected_component_labels(Dk.face_adjacency, node_count=len(Dk.faces))
n_lab = np.bincount(lab); zmax_lab = np.full(len(n_lab), -np.inf); np.maximum.at(zmax_lab, lab, zDk)
prot = ((n_lab < a.protect_max) & (zmax_lab < a.z2) & (np.bincount(lab, weights=skinK.astype(float)) > 0))[lab]

# 表皮の頂点だけを、帯の中でCX-5の外板へ徐々に寄せる（移動量はならす）
V = Dk.vertices.copy()
w = smooth(np.clip((dz(V) - a.z1) / (a.z2 - a.z1), 0, 1))
mv = np.zeros(len(V), bool); mv[Dk.faces[skinK & ~prot].ravel()] = True
mv[Dk.faces[prot | ~skinK].ravel()] = False
mv &= w > 0
q = S_Text.compute_closest_points(o3d.core.Tensor(V[mv].astype(np.float32)))["points"].numpy()
disp = np.zeros_like(V); disp[mv] = w[mv, None] * (q - V[mv])
exact = disp.copy(); pin = mv & (w > 0.95)
e = Dk.edges_unique
A = coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(len(V), len(V))).tocsr()
deg = np.asarray(A.sum(1)).ravel(); deg[deg == 0] = 1
for _ in range(a.smooth_iters):
    avg = (A @ disp) / deg[:, None]
    disp[mv] = 0.5 * disp[mv] + 0.5 * avg[mv]; disp[pin] = exact[pin]
Dk.vertices = V + disp

# 奥の面のうち、帯より後ろにあるものは、車の内側にあるものだけ残す（上・左・右で外板に当たる）
B_scene = scene_of(trimesh.util.concatenate([sub(T, exT), sub(Dk, skinK)]))
chk = np.where(~skinK & (zDk >= a.z1))[0]
inside = np.ones(len(chk), bool)
for d in [np.array([0, 1, 0]), np.array([1, 0, 0]), np.array([-1, 0, 0])]:
    inside &= np.isfinite(cast(B_scene, Dk.triangles_center[chk] + d * mm, d))
okD = np.ones(len(Dk.faces), bool); okD[chk[~inside]] = False
okD &= ~(skinK & (zDk >= a.z2 + 0.04))                              # 帯より後ろの表皮は使わない
wv = w.copy()
Dk = trimesh.Trimesh(Dk.vertices, Dk.faces[okD], process=False); skinK = skinK[okD]
print(f"ドナー: 表皮 {skinK.sum()} 面 / 奥の面 {(~skinK).sum()} 面 | 守る部品の面 {prot.sum()} | 外に突き出るため除外 {(~inside).sum()}")

# --- CX-5: z2より後ろは全部。帯の中と床下は、ドナーの表皮がない場所だけ残す ---
S_skin = scene_of(sub(Dk, skinK))
band = (zT >= a.z1) & (zT < a.z2)
height = (T.triangles_center[:, 1] - yg) / S
under = (zT < a.z1) & (nT[:, 1] < -0.8) & (height < 0.25)          # 地面近くの真下向きの面（床下）
bi = np.where((band | under) & exT)[0]
c, n = T.triangles_center[bi], nT[bi]
covered = cast(S_skin, c + n * mm, n) < 0.3 * S
hides = cast(S_skin, c - n * mm, -n) < a.dcover * mm
coinc = dist(S_skin, c) < 2 * mm
keep_ext = np.zeros(len(T.faces), bool); keep_ext[bi[~covered & ~hides & ~coinc]] = True
Tk = T.submesh([np.where((zT >= a.z2) | (band & ~exT) | keep_ext)[0]], append=True)
print(f"CX-5: 帯と床下の外板 {len(bi)} 面のうち残した面 {keep_ext.sum()}")
body = trimesh.util.concatenate([Dk, Tk])

# --- SLATの裏打ちの面: CX-5とドナーの潜在表現を帯と同じ重みで補間 → デコード → 内側にずらす ---
TT = np.load(f"{cd}/features/unmodified_slat.npz")
DD = np.load(glob.glob(f"{slat_dir}/features/*_slat.npz")[0])
def vox_z(cc):
    pz = (cc.astype(float) + 0.5) / 64 - 0.5
    return ((-pz[:, 1]) * ext + c0[2] - zf) / S                     # 前端からの距離 [m]
wdon = lambda cc: 1 - smooth(np.clip((vox_z(cc) - a.z1) / (a.z2 - a.z1), 0, 1))
key = lambda cc: cc[:, 0].astype(np.int64) * 4096 + cc[:, 1].astype(np.int64) * 64 + cc[:, 2]
ct, cdn = TT["coords"].astype(np.int64), DD["coords"].astype(np.int64)
_, it, idd = np.intersect1d(key(ct), key(cdn), return_indices=True)
ot = np.setdiff1d(np.arange(len(ct)), it); odn = np.setdiff1d(np.arange(len(cdn)), idd)
wb, wt, wd_ = wdon(ct[it]), wdon(ct[ot]), wdon(cdn[odn])
coords = np.vstack([ct[it], ct[ot][wt < 0.5], cdn[odn][wd_ >= 0.5]]).astype(np.int32)
feats = np.vstack([(1 - wb)[:, None] * TT["feats"][it] + wb[:, None] * DD["feats"][idd],
                   TT["feats"][ot][wt < 0.5], DD["feats"][odn][wd_ >= 0.5]]).astype(np.float32)
cc_ = torch.cat([torch.zeros(len(coords), 1).int(), torch.from_numpy(coords).int()], 1)
pipeline = _3DMorphInpaint.from_pretrained("pretrained_weights/TRELLIS-image-large"); pipeline.cuda()
with torch.no_grad():
    mm_ = pipeline.decode_slat(sp.SparseTensor(feats=torch.from_numpy(feats), coords=cc_).cuda(), ["mesh"])["mesh"][0]
L = trimesh.Trimesh(mm_.vertices.cpu().numpy(), mm_.faces.cpu().numpy(), process=False)
L.apply_transform(trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0]))
L.vertices = L.vertices * ext + c0
if L.volume < 0: L.invert()
L.vertices = L.vertices - L.vertex_normals * a.liner * mm              # 内側へずらす
zl = dz(L.triangles_center)
L = sub(L, (zl > a.z1 - 0.05) & (zl < a.z2 + 0.10))                  # 結合部分の周りだけ
F_scene = scene_of(body)
c, n = L.triangles_center, L.face_normals
outer = ~(cast(F_scene, c + n * mm, n) < 0.3 * S)                    # 外から直接見える
behind = cast(F_scene, c - n * mm, -n) < 50 * mm                     # そのすぐ内側に表面がある = 表面より外に飛び出している
L = sub(L, ~(outer & behind)); L.remove_unreferenced_vertices()
print(f"裏打ちの面: {len(L.faces)} 面（表面より外に飛び出すため除外 {(outer & behind).sum()}）")

final = trimesh.util.concatenate([Dk, Tk, L])
final.export(f"{od}/final.obj")
Dk.visual.vertex_colors = np.hstack([(np.array([220, 80, 80]) * (1 - wv[:len(Dk.vertices), None]) + np.array([170, 170, 170]) * wv[:len(Dk.vertices), None]).astype(np.uint8),
                                     np.full((len(Dk.vertices), 1), 255, np.uint8)])
Tk.visual.vertex_colors = np.tile([120, 160, 220, 255], (len(Tk.vertices), 1))
L.visual.vertex_colors = np.tile([240, 200, 60, 255], (len(L.vertices), 1))
trimesh.util.concatenate([Dk, Tk, L]).export(f"{od}/final_colored.ply")

# --- レンダリング（上から: CX-5 / 前回(fade3) / 今回） ---
def render(obj, res, nv, step, yaw0=0, pitch=20):
    tmp = f"{od}/_tmp"; shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    BlenderRenderer(res).render(obj, target_obj, save_dir=tmp, n_views=nv, mode="linear",
        lin_view_args={"offset": (yaw0, pitch), "step": step, "direction": "right", "set_fov": 30})
    ims = []
    for k in range(nv):
        im = Image.open(f"{tmp}/render_{k:03d}.png").convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255)); bg.alpha_composite(im)
        ims.append(np.array(bg.convert("RGB")))
    shutil.rmtree(tmp); return ims
prev = f"{cd}/results/fade3_{a.donor}_{a.z1}_{a.z2}/final.obj"
rows = []
for obj in [target_obj] + ([prev] if os.path.exists(prev) else []) + [f"{od}/final.obj"]:
    ims = render(obj, 512, 8, 45)
    rows.append(np.hstack([ims[k] for k in [1, 3, 0, 2]]))
Image.fromarray(np.vstack(rows)).save(f"{od}/compare_fade4.png")
crops = []
for yaw, pitch in [(45, 20), (90, 20), (60, -40)]:
    im = render(f"{od}/final.obj", 2048, 1, 0, yaw0=yaw, pitch=pitch)[0]
    H, W = im.shape[:2]
    crops.append(im[int(0.35*H):int(0.85*H), int(0.45*W):W] if yaw == 45 else im[int(0.3*H):int(0.8*H), int(0.2*W):int(0.8*W)])
h = min(x.shape[0] for x in crops)
Image.fromarray(np.hstack([x[:h] for x in crops])).save(f"{od}/closeup_fade4.png")
print("保存:", od, "（final.obj, final_colored.ply, compare_fade4.png, closeup_fade4.png）")
