"""Phase 3b準備: 貼り合わせ結果を「元の物体」とする作業フォルダを作り、条件画像をレンダリングする
使い方: python 15_prepare_refine.py <car_id> <donor_id> <splice_tag>
作業フォルダ: data/cars/<car_id>__<splice_tag>/
"""
import sys, os, shutil, json
ROOT = os.path.expanduser("~/Desktop/ikeda/detail_edit")
REPO = f"{ROOT}/repos/3dmorph"
sys.path.insert(0, REPO); os.chdir(REPO)
os.system("pgrep -f 'Xvfb :99' >/dev/null || Xvfb :99 -screen 0 1024x768x16 &")
import numpy as np, trimesh
from _3DMorph.renderer.render_simple import BlenderRenderer

car, did, stag = sys.argv[1], sys.argv[2], sys.argv[3]
src = f"{ROOT}/data/cars/{car}"; rd = f"{src}/results/{stag}"
wd = f"{ROOT}/data/cars/{car}__{stag}"
shutil.rmtree(wd, ignore_errors=True)
for d in ["features", "recon", "explore_inpaint", "donors", "results"]:
    os.makedirs(f"{wd}/{d}")

# 貼り合わせ結果（SLAT出力の座標）を、unmodified.obj の座標系に戻して「元の物体」にする
t = trimesh.load(f"{src}/unmodified.obj", force="mesh")
c, ext = (t.bounds[0] + t.bounds[1]) / 2, np.ptp(t.vertices, axis=0).max()
m = trimesh.load(f"{rd}/edited_raw.ply", force="mesh", process=False)
m.vertices = m.vertices * ext + c
m.export(f"{wd}/unmodified.obj")
print("外形 CX-5:", np.round(t.bounds, 3).tolist(), "\n外形 貼り合わせ:", np.round(m.bounds, 3).tolist())

shutil.copy(f"{src}/norm.json", wd)
shutil.copy(f"{src}/recon/slat.ply", f"{wd}/recon/slat.ply")          # 比較の基準は元のCX-5
shutil.copy(f"{src}/donors/{did}_aligned.obj", f"{wd}/donors/")
S = np.load(f"{rd}/pred_slat.npz")
np.savez_compressed(f"{wd}/features/unmodified_slat.npz", coords=S["coords"], feats=S["feats"])

# 条件画像 = 貼り合わせ結果を同じカメラで撮ったもの
cam = json.load(open(f"{src}/explore_inpaint/transforms.json"))[0]
tmp = f"{wd}/_tmp"; os.makedirs(tmp)
BlenderRenderer(1024).render(
    f"{wd}/unmodified.obj", f"{wd}/unmodified.obj", save_dir=tmp, n_views=1, mode="linear",
    lin_view_args={"offset": (np.degrees(cam["yaw"]), np.degrees(cam["pitch"])), "step": 0,
                   "direction": "right", "set_fov": np.degrees(cam["camera_angle_x"])})
shutil.copy(f"{tmp}/render_000.png", f"{wd}/explore_inpaint/modified.png")
shutil.copy(f"{src}/explore_inpaint/unmodified.png", f"{wd}/explore_inpaint/unmodified.png")
shutil.copy(f"{src}/explore_inpaint/transforms.json", f"{wd}/explore_inpaint/transforms.json")
shutil.rmtree(tmp)
print("作業フォルダ:", wd)
