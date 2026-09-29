"""部品単位の置換 + SLATによるすき間の補完
使い方: python 21_parts_fill.py <target_id> <donor_id> [--zmin 0.3] [--zmax 0.65] [--trans 1.0] [--tau 15] [--zone 0.8]
  --trans: 重みが切り替わる幅（ボクセル数、1ボクセル≒71mm）
  --tau:   部品からこの距離[mm]以上離れたSLATの面だけを、すき間を埋める面として使う
  --zone:  補完を行う範囲（前端からの距離 m）
出力: data/cars/<target_id>/results/partsfill_<donor_id>/
"""
import os, sys, glob, json, shutil, argparse
ROOT = os.path.expanduser("~/Desktop/ikeda/detail_edit")
REPO = f"{ROOT}/repos/3dmorph"
sys.path.insert(0, REPO); os.chdir(REPO)
import xformers_patch
os.system("pgrep -f 'Xvfb :99' >/dev/null || Xvfb :99 -screen 0 1024x768x16 &")
os.environ["ATTN_BACKEND"] = "xformers"; os.environ["SPCONV_ALGO"] = "native"
import numpy as np, torch, trimesh, open3d as o3d
from PIL import Image
from trellis.pipelines import _3DMorphInpaint
from trellis.modules import sparse as sp
from _3DMorph.renderer.render_simple import BlenderRenderer

ap = argparse.ArgumentParser()
ap.add_argument("target"); ap.add_argument("donor")
ap.add_argument("--zmin", type=float, default=0.3); ap.add_argument("--zmax", type=float, default=0.65)
ap.add_argument("--trans", type=float, default=1.0); ap.add_argument("--tau", type=float, default=15)
ap.add_argument("--zone", type=float, default=0.8)
ap.add_argument("--dmax", type=float, default=50)   # 内側この距離[mm]以内に部品があれば「かぶり」とみなす
ap.add_argument("--minfaces", type=int, default=100)   # 大きなすき間の面で、これより小さい破片は捨てる
ap.add_argument("--junc", type=float, default=30)   # 両方の部品からこの距離[mm]以内なら継ぎ目とみなす
ap.add_argument("--rings", type=int, default=3)   # 大きなすき間の面の縁を広げる段数
a = ap.parse_args()
cd = f"{ROOT}/data/cars/{a.target}"
od = f"{cd}/results/partsfill_{a.donor}"; os.makedirs(od, exist_ok=True)
target_obj = f"{cd}/unmodified.obj"
S = json.load(open(f"{cd}/norm.json"))["scale"]                 # 1 m = S [obj単位]

def load(p):
    m = trimesh.load(p, force="mesh", process=False); m.merge_vertices(); return m
T, D = load(target_obj), load(f"{cd}/donors/{a.donor}_aligned.obj")
zf = T.bounds[0][2]
c0, ext = (T.bounds[0] + T.bounds[1]) / 2, np.ptp(T.vertices, axis=0).max()

# --- 部品の分類（左右で対になる部品は、どちらかが選ばれたら両方を選ぶ） ---
def classify(mesh, name):
    comps = mesh.split(only_watertight=False)
    zs = [(c.triangles_center[:, 2] - zf) / S for c in comps]
    flag = np.array([z.min() < a.zmin and np.percentile(z, 99) <= a.zmax for z in zs])
    cen = np.array([np.average(c.triangles_center, axis=0, weights=c.area_faces + 1e-12) for c in comps])
    nf = np.array([len(c.faces) for c in comps])
    n_fixed = 0
    for i in range(len(comps)):
        mir = cen[i] * np.array([-1, 1, 1])
        for j in np.where((nf == nf[i]) & (np.linalg.norm(cen - mir, axis=1) < 0.02 * S))[0]:
            if flag[i] != flag[j]:
                flag[i] = flag[j] = True; n_fixed += 1
    front = [c for c, f in zip(comps, flag) if f]; rest = [c for c, f in zip(comps, flag) if not f]
    print(f"[{name}] 部品 {len(comps)} 個 → フロント {flag.sum()} 個（左右の対の修正 {n_fixed} 件）")
    return trimesh.util.concatenate(front), trimesh.util.concatenate(rest)
d_front, _ = classify(D, "ドナー")
_, t_rest = classify(T, "CX-5")
comp = trimesh.util.concatenate([d_front, t_rest])

def scene_of(m):
    s = o3d.t.geometry.RaycastingScene()
    lm = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(m.vertices), o3d.utility.Vector3iVector(m.faces))
    s.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(lm)); return s
dist_to = lambda s, p: s.compute_distance(o3d.core.Tensor(p.astype(np.float32))).numpy()
SD, ST, SC = scene_of(d_front), scene_of(t_rest), scene_of(comp)

# --- ボクセルごとの重み（ドナーの部品に近いほど1） ---
TT = np.load(f"{cd}/features/unmodified_slat.npz")
DD = np.load(glob.glob(f"{cd}/donors/{a.donor}/features/*_slat.npz")[0])
def vox_to_obj(c):
    pz = (c.astype(float) + 0.5) / 64 - 0.5                        # TRELLISの座標（Z-up）
    p = np.stack([pz[:, 0], pz[:, 2], -pz[:, 1]], 1)               # Y-up（デコード後の -90°回転と同じ）
    return p * ext + c0
def weight(c):
    p = vox_to_obj(c)
    s = a.trans * ext / 64
    w = np.clip(0.5 + (dist_to(ST, p) - dist_to(SD, p)) / (2 * s), 0, 1)
    w[(p[:, 2] - zf) / S > a.zone] = 0.0
    return w
key = lambda c: c[:, 0].astype(np.int64) * 4096 + c[:, 1].astype(np.int64) * 64 + c[:, 2]
ct, cdn = TT["coords"].astype(np.int64), DD["coords"].astype(np.int64)
kt, kd = key(ct), key(cdn)
_, it, idd = np.intersect1d(kt, kd, return_indices=True)
ot = np.setdiff1d(np.arange(len(kt)), it); odn = np.setdiff1d(np.arange(len(kd)), idd)
wb, wt, wd = weight(ct[it]), weight(ct[ot]), weight(cdn[odn])
coords = np.vstack([ct[it], ct[ot][wt < 0.5], cdn[odn][wd >= 0.5]]).astype(np.int32)
feats = np.vstack([(1 - wb)[:, None] * TT["feats"][it] + wb[:, None] * DD["feats"][idd],
                   TT["feats"][ot][wt < 0.5], DD["feats"][odn][wd >= 0.5]]).astype(np.float32)
print(f"ボクセル {len(coords)} | 重み0.05〜0.95（混ぜ合わせ）のボクセル {np.sum((wb > 0.05) & (wb < 0.95))}")

# --- デコードして、部品から離れた部分だけをすき間の補完に使う ---
c = torch.cat([torch.zeros(len(coords), 1).int(), torch.from_numpy(coords).int()], 1)
slat = sp.SparseTensor(feats=torch.from_numpy(feats), coords=c).cuda()
pipeline = _3DMorphInpaint.from_pretrained("pretrained_weights/TRELLIS-image-large"); pipeline.cuda()
with torch.no_grad():
    mm = pipeline.decode_slat(slat, ["mesh"])["mesh"][0]
sl = trimesh.Trimesh(mm.vertices.cpu().numpy(), mm.faces.cpu().numpy(), process=False)
sl.apply_transform(trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0]))
sl.vertices = sl.vertices * ext + c0
sl.export(f"{od}/slat_blend.obj")
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
if sl.volume < 0:
    sl.invert()                                                    # 法線を外向きに揃える
tc, nrm = sl.triangles_center, sl.face_normals
dC, dD, dT = dist_to(SC, tc), dist_to(SD, tc), dist_to(ST, tc)
inzone = (tc[:, 2] - zf) / S < a.zone
mm, eps = S / 1000, 2 * S / 1000
def cast(orig, d):
    r = np.hstack([orig, d]).astype(np.float32)
    return SC.cast_rays(o3d.core.Tensor(r))["t_hit"].numpy()
exposed = ~(cast(tc + nrm * eps, nrm) < 0.3 * S)                  # 外側を部品に覆われていない
covering = cast(tc - nrm * eps, -nrm) < a.dmax * mm                # すぐ内側に部品がある

nF, fa = len(sl.faces), sl.face_adjacency
Adj = coo_matrix((np.ones(2 * len(fa)), (np.r_[fa[:, 0], fa[:, 1]], np.r_[fa[:, 1], fa[:, 0]])),
                 shape=(nF, nF)).tocsr()
def drop_small(mask, minf):
    idx = np.where(mask)[0]
    out = np.zeros(nF, bool)
    if len(idx):
        _, lab = connected_components(Adj[idx][:, idx], directed=False)
        out[idx[np.bincount(lab)[lab] >= minf]] = True
    return out

A = drop_small(inzone & exposed & ~covering & (dC > a.tau * mm), a.minfaces)          # 大きなすき間
B = inzone & exposed & (dC > 2 * mm) & (dD < a.junc * mm) & (dT < a.junc * mm)          # 継ぎ目の細いすき間
sel = A | B
front, band, grown = A.copy(), inzone & exposed & (dC <= a.tau * mm), 0
for _ in range(int(a.rings)):                                      # Aの縁を部品の縁まで広げる
    add = ((Adj @ front.astype(np.float32)) > 0) & band & ~sel
    sel |= add; front = add; grown += add.sum()
    if not add.any():
        break
print(f"すき間を埋める面: A(大きなすき間) {A.sum()} + B(継ぎ目) {np.sum(B & ~A)} + C(縁の延長) {grown} = {sel.sum()} 面")
filler = sl.submesh([np.where(sel)[0]], append=True)
print(f"すき間を埋める面: {len(filler.faces)} 面（面積 {filler.area / S**2:.3f} m²）")

final = trimesh.util.concatenate([d_front, t_rest, filler])
comp.export(f"{od}/parts_only.obj"); final.export(f"{od}/final.obj")
for m, col in [(d_front, [220, 80, 80, 255]), (t_rest, [170, 170, 170, 255]), (filler, [240, 150, 60, 255])]:
    m.visual.vertex_colors = np.tile(col, (len(m.vertices), 1))
trimesh.util.concatenate([d_front, t_rest, filler]).export(f"{od}/final_colored.ply")

# --- レンダリング ---
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
for obj in [target_obj, f"{od}/parts_only.obj", f"{od}/final.obj"]:
    ims = render(obj, 512, 8, 45)
    rows.append(np.hstack([ims[k] for k in [1, 3, 0, 2]]))
Image.fromarray(np.vstack(rows)).save(f"{od}/compare_fill.png")
crops = []
for yaw in [45, 135]:
    im = render(f"{od}/final.obj", 2048, 1, 0, yaw0=yaw)[0]
    H, W = im.shape[:2]
    crops.append(im[int(0.35*H):int(0.85*H), int(0.45*W):W] if yaw == 45 else im[int(0.35*H):int(0.85*H), 0:int(0.55*W)])
Image.fromarray(np.hstack(crops)).save(f"{od}/closeup_fill.png")
print("保存:", od, "（final.obj, final_colored.ply, compare_fill.png, closeup_fill.png）")
