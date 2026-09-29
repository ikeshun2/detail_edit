"""3DMorph用のレンダリング
使い方:
  python 06_render_views.py sweep <car_id> [pitch=20]
      -> results/phase1/sweep_<car_id>/ に8方向（45°刻み）の画像
  python 06_render_views.py pair <target_id> <donor_id> <yaw> <pitch> <front: z+|z->
      -> data/cars/<target_id>/explore_inpaint/ に unmodified.png, donor.png, transforms.json
"""
import sys, os, json, shutil, glob
ROOT = os.path.expanduser("~/Desktop/ikeda/detail_edit")
REPO = f"{ROOT}/repos/3dmorph"
sys.path.insert(0, REPO)
os.chdir(REPO)
os.system("pgrep -f 'Xvfb :99' >/dev/null || Xvfb :99 -screen 0 1024x768x16 &")

import numpy as np, trimesh
from _3DMorph.renderer.render_simple import BlenderRenderer

FOV, RES = 30, 1024
renderer = BlenderRenderer(RES)
car_obj = lambda cid: f"{ROOT}/data/cars/{cid}/unmodified.obj"


def render(obj, ref, out_dir, yaw, pitch, n=1, step=0):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    renderer.render(obj, ref, save_dir=out_dir, n_views=n, mode="linear",
                    lin_view_args={"offset": (yaw, pitch), "step": step,
                                   "direction": "right", "set_fov": FOV},
                    save_transforms=True)


def align_donor(target_id, donor_id, front):
    """倍率は共通のまま、接地面(yの最小値)と前端(zの端)をターゲットに揃える"""
    t = trimesh.load(car_obj(target_id), force="mesh")
    d = trimesh.load(car_obj(donor_id), force="mesh")
    dy = t.bounds[0][1] - d.bounds[0][1]
    dz = (t.bounds[1][2] - d.bounds[1][2]) if front == "z+" else (t.bounds[0][2] - d.bounds[0][2])
    d.apply_translation([0.0, dy, dz])          # xは両方とも中心が0
    out_dir = f"{ROOT}/data/cars/{target_id}/donors"
    os.makedirs(out_dir, exist_ok=True)
    out = f"{out_dir}/{donor_id}_aligned.obj"
    d.export(out)
    print(f"{donor_id} を移動: dy={dy:+.4f}, dz={dz:+.4f} -> {out}")
    print(f"  車幅 target {np.ptp(t.vertices[:,0]):.3f} / donor {np.ptp(d.vertices[:,0]):.3f}")
    print(f"  車高 target {np.ptp(t.vertices[:,1]):.3f} / donor {np.ptp(d.vertices[:,1]):.3f}")
    return out


if sys.argv[1] == "sweep":
    cid = sys.argv[2]
    pitch = float(sys.argv[3]) if len(sys.argv) > 3 else 20
    out = f"{ROOT}/results/phase1/sweep_{cid}"
    render(car_obj(cid), car_obj(cid), out, 0, pitch, n=8, step=45)
    print("保存:", out, "（render_000=yaw 0°, render_001=45°, ... render_007=315°）")

elif sys.argv[1] == "pair":
    tid, did, yaw, pitch, front = sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5]), sys.argv[6]
    donor_aligned = align_donor(tid, did, front)
    ex = f"{ROOT}/data/cars/{tid}/explore_inpaint"
    tmp_t, tmp_d = f"{ex}/_tmp_target", f"{ex}/_tmp_donor"
    render(car_obj(tid), car_obj(tid), tmp_t, yaw, pitch)
    render(donor_aligned, car_obj(tid), tmp_d, yaw, pitch)      # 参照=ターゲット（同じ座標系）
    shutil.copy(f"{tmp_t}/render_000.png", f"{ex}/unmodified.png")
    shutil.copy(f"{tmp_d}/render_000.png", f"{ex}/donor.png")
    tr = json.load(open(f"{tmp_t}/transforms.json"))
    tr = tr if isinstance(tr, list) else tr.get("frames", tr)
    json.dump(tr[:1], open(f"{ex}/transforms.json", "w"), indent=4)
    shutil.rmtree(tmp_t); shutil.rmtree(tmp_d)
    print("保存:", ex, "（unmodified.png, donor.png, transforms.json）")
    print("カメラ:", {k: tr[0].get(k) for k in ["yaw", "pitch", "camera_angle_x"]})
