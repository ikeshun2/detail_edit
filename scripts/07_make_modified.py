"""ドナーのフロントをCX-5の画像に切り貼りして modified.png を作る
使い方: python 07_make_modified.py <target_id> <donor_id> [ratio=0.2]
前提: 06_render_views.py pair を実行済み（フロントは z- 側）
"""
import sys, os, json, shutil
ROOT = os.path.expanduser("~/Desktop/ikeda/detail_edit")
REPO = f"{ROOT}/repos/3dmorph"
sys.path.insert(0, REPO)
os.chdir(REPO)
os.system("pgrep -f 'Xvfb :99' >/dev/null || Xvfb :99 -screen 0 1024x768x16 &")

import numpy as np, trimesh
from PIL import Image
from _3DMorph.renderer.render_simple import BlenderRenderer

tid, did = sys.argv[1], sys.argv[2]
ratio = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2
car = f"{ROOT}/data/cars/{tid}"
ex = f"{car}/explore_inpaint"
target_obj = f"{car}/unmodified.obj"
donor_obj = f"{car}/donors/{did}_aligned.obj"
cam = json.load(open(f"{ex}/transforms.json"))[0]
yaw, pitch = np.degrees(cam["yaw"]), np.degrees(cam["pitch"])
fov = np.degrees(cam["camera_angle_x"])
renderer = BlenderRenderer(1024)

# フロントの範囲: 前端(zの最小値)から車長のratio倍まで
t = trimesh.load(target_obj, force="mesh")
z_front, L = t.bounds[0][2], np.ptp(t.vertices[:, 2])
z_cut = z_front + ratio * L
print(f"フロントの範囲: z < {z_cut:.4f}（前端 {z_front:.4f} から車長の {ratio*100:.0f}%）")

def front_part(path, out):
    m = trimesh.load(path, force="mesh")
    f = m.slice_plane(plane_origin=[0, 0, z_cut], plane_normal=[0, 0, -1])   # z < z_cut を残す
    f.export(out)
    return out

def render_alpha(obj, name):
    d = f"{ex}/_tmp_{name}"
    shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    renderer.render(obj, target_obj, save_dir=d, n_views=1, mode="linear",
                    lin_view_args={"offset": (yaw, pitch), "step": 0,
                                   "direction": "right", "set_fov": fov})
    a = np.array(Image.open(f"{d}/render_000.png").convert("RGBA"))[..., 3] > 0
    shutil.rmtree(d)
    return a

m_donor = render_alpha(front_part(donor_obj, f"{ex}/_donor_front.obj"), "df")
m_target = render_alpha(front_part(target_obj, f"{ex}/_target_front.obj"), "tf")

T = np.array(Image.open(f"{ex}/unmodified.png").convert("RGBA"))
D = np.array(Image.open(f"{ex}/donor.png").convert("RGBA"))
M = m_donor | m_target                 # 置き換える範囲
out = T.copy()
out[M] = D[M]
Image.fromarray(out).save(f"{ex}/modified.png")

# 確認用: 左=元, 中=マスク(赤=ドナーのフロント, 青=CX-5のフロントのみ), 右=modified
vis = np.full(T.shape, 255, np.uint8)
vis[m_target & ~m_donor] = [60, 60, 255, 255]
vis[m_donor] = [255, 60, 60, 255]
Image.fromarray(np.hstack([T, vis, out])).save(f"{ex}/_preview.png")
print(f"置き換えた画素数: {M.sum()}（画像全体の {M.mean()*100:.1f}%）")
print("保存:", f"{ex}/modified.png", "と", f"{ex}/_preview.png")
