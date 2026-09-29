"""高解像度の合成: フロント=ドナーのSTL、継ぎ目の帯=SLATの補間結果、ボディ=CX-5のSTL
使い方: python 17_hybrid_mesh.py <target_id> <donor_id> <softsplice_tag> [ratio=0.1] [width=6] [overlap=1]
  overlap: 帯を前後に延ばして隣と重ねる量（ボクセル数、1ボクセル≒71mm）
出力: data/cars/<target_id>/results/<softsplice_tag>/hybrid/
"""
import sys, os, shutil
ROOT = os.path.expanduser("~/Desktop/ikeda/detail_edit")
REPO = f"{ROOT}/repos/3dmorph"
sys.path.insert(0, REPO); os.chdir(REPO)
os.system("pgrep -f 'Xvfb :99' >/dev/null || Xvfb :99 -screen 0 1024x768x16 &")
import numpy as np, trimesh
from PIL import Image
from _3DMorph.renderer.render_simple import BlenderRenderer

tid, did, stag = sys.argv[1], sys.argv[2], sys.argv[3]
ratio = float(sys.argv[4]) if len(sys.argv) > 4 else 0.1
width = int(sys.argv[5]) if len(sys.argv) > 5 else 6
overlap = float(sys.argv[6]) if len(sys.argv) > 6 else 1
cd = f"{ROOT}/data/cars/{tid}"; rd = f"{cd}/results/{stag}"
od = f"{rd}/hybrid"; os.makedirs(od, exist_ok=True)

# すべて unmodified.obj の座標系（フロントは z- 側）で扱う
target_obj = f"{cd}/unmodified.obj"
t = trimesh.load(target_obj, force="mesh", process=False)
c, ext = (t.bounds[0] + t.bounds[1]) / 2, np.ptp(t.vertices, axis=0).max()
d = trimesh.load(f"{cd}/donors/{did}_aligned.obj", force="mesh", process=False)
s = trimesh.load(f"{rd}/edited_raw.ply", force="mesh", process=False)
s.vertices = s.vertices * ext + c                      # SLATの出力座標 -> obj座標

cut = 64 - round(ratio * 64)
z_of = lambda i: (-0.5 + (64 - i) / 64) * ext + c[2]   # ボクセル軸1の番号 -> obj座標のz
z_a, z_b = z_of(cut + width / 2), z_of(cut - width / 2)  # 帯の前端（ドナー側）、後端（CX-5側）
ov = overlap / 64 * ext
MM = ext / __import__("json").load(open(f"{cd}/norm.json"))["scale"] * 1000   # obj座標1単位 -> mm

def crop(m, lo, hi):
    zc = m.triangles_center[:, 2]
    return m.submesh([np.where((zc >= lo) & (zc < hi))[0]], append=True)

front = crop(d, -np.inf, z_a)
band = crop(s, z_a - ov, z_b + ov)
body = crop(t, z_b, np.inf)
z0 = t.bounds[0][2]
print(f"帯: 前端から {(z_a - z0)*MM:.0f} mm 〜 {(z_b - z0)*MM:.0f} mm（重ね {ov*MM:.0f} mm）")
print(f"面の数: フロント(ドナーSTL) {len(front.faces)} / 帯(SLAT) {len(band.faces)} / ボディ(CX-5 STL) {len(body.faces)}")

hybrid = trimesh.util.concatenate([front, band, body])
hybrid.export(f"{od}/hybrid.obj")
for m, col in [(front, [90, 140, 230, 255]), (band, [240, 150, 60, 255]), (body, [170, 170, 170, 255])]:
    m.visual.vertex_colors = np.tile(col, (len(m.vertices), 1))
trimesh.util.concatenate([front, band, body]).export(f"{od}/hybrid_colored.ply")
s.export(f"{od}/_softsplice_objframe.obj")

# --- 比較画像（4モデル x 4方向） ---
def render(obj, res, n, step, yaw0=0):
    tmp = f"{od}/_tmp"; shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    BlenderRenderer(res).render(obj, target_obj, save_dir=tmp, n_views=n, mode="linear",
        lin_view_args={"offset": (yaw0, 20), "step": step, "direction": "right", "set_fov": 30})
    ims = []
    for k in range(n):
        im = Image.open(f"{tmp}/render_{k:03d}.png").convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255)); bg.alpha_composite(im)
        ims.append(np.array(bg.convert("RGB")))
    shutil.rmtree(tmp)
    return ims

models = [("CX-5 (STL)", target_obj), ("SLAT soft splice", f"{od}/_softsplice_objframe.obj"),
          ("hybrid", f"{od}/hybrid.obj"), ("donor (STL)", f"{cd}/donors/{did}_aligned.obj")]
rows = []
for name, obj in models:
    ims = render(obj, 512, 8, 45)
    rows.append(np.hstack([ims[k] for k in [1, 3, 0, 5]]))       # 45°, 135°, 0°, 225°
    print("レンダリング完了:", name)
Image.fromarray(np.vstack(rows)).save(f"{od}/compare_hybrid.png")

# --- フロントの拡大（斜め前 45°、高解像度で撮って切り出す） ---
crops = []
for name, obj in models[1:3]:
    im = render(obj, 2048, 1, 0, yaw0=45)[0]
    H, W = im.shape[:2]
    crops.append(im[int(0.35 * H):int(0.85 * H), int(0.45 * W):W])
Image.fromarray(np.hstack(crops)).save(f"{od}/closeup_front.png")
print("保存:", od, "（hybrid.obj, hybrid_colored.ply, compare_hybrid.png, closeup_front.png）")
