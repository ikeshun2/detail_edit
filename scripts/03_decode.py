"""SLAT -> メッシュ（簡略化なし）。真値との簡易Chamfer距離も出す。
使い方: python 03_decode.py <car_id> [slat_npzのパス]
"""
import sys, os, glob, json
ROOT = os.path.expanduser("~/Desktop/ikeda/detail_edit")
REPO = f"{ROOT}/repos/3dmorph"
sys.path.insert(0, REPO)
os.chdir(REPO)

import xformers_patch
os.environ["ATTN_BACKEND"] = "xformers"
os.environ["SPCONV_ALGO"] = "native"

import numpy as np, torch, trimesh, shutil
from scipy.spatial import cKDTree
from trellis.pipelines import _3DMorphInpaint
from trellis.modules import sparse as sp

car_id = sys.argv[1]
car_dir = f"{ROOT}/data/cars/{car_id}"
slat_path = sys.argv[2] if len(sys.argv) > 2 else \
    sorted(glob.glob(f"{car_dir}/**/*_slat.npz", recursive=True))[0]
print("slat:", slat_path)

# SLATの読み込み（example_inpaint.pyと同じ形式）
d = np.load(slat_path)
coords = torch.from_numpy(d["coords"]).int()
feats = torch.from_numpy(d["feats"]).float()
coords = torch.cat([torch.zeros(len(coords), 1).int(), coords], 1)
slat = sp.SparseTensor(feats=feats, coords=coords).cuda()
print("voxels:", len(coords))

pipeline = _3DMorphInpaint.from_pretrained("pretrained_weights/TRELLIS-image-large")
pipeline.cuda()
with torch.no_grad():
    out = pipeline.decode_slat(slat, ["mesh"])
m = out["mesh"][0]
recon = trimesh.Trimesh(m.vertices.cpu().numpy(), m.faces.cpu().numpy(), process=False)
# Z-up(TRELLIS) -> Y-up(入力OBJと同じ向き)
recon.apply_transform(trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0]))

os.makedirs(f"{car_dir}/recon", exist_ok=True)
recon.export(f"{car_dir}/recon/slat.ply")
gt_src = f"{car_dir}/slat_preparation/unmodified_normalized_for_slat.obj"
shutil.copy(gt_src, f"{car_dir}/recon/gt_normalized.obj")
gt = trimesh.load(gt_src, force="mesh")

print("recon bounds:", np.round(recon.bounds, 3).tolist(), "faces:", len(recon.faces))
print("gt    bounds:", np.round(gt.bounds, 3).tolist())

# 簡易Chamfer距離（全体、正規化座標とmm）
N = 200_000
pr, pg = recon.sample(N), gt.sample(N)
d1 = cKDTree(pg).query(pr)[0]; d2 = cKDTree(pr).query(pg)[0]
cd = 0.5 * (d1.mean() + d2.mean())
norm = json.load(open(f"{car_dir}/norm.json"))
obj = trimesh.load(f"{car_dir}/unmodified.obj", force="mesh")
meters_per_unit = (obj.bounds[1] - obj.bounds[0]).max() / norm["scale"]   # 正規化1単位=元の最長辺[m]
print(f"Chamfer(全体): {cd:.5f} (正規化)  = {cd*meters_per_unit*1000:.2f} mm")
print(f"Hausdorff    : {max(d1.max(), d2.max())*meters_per_unit*1000:.1f} mm")
