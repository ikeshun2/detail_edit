"""TRELLIS標準の穴埋めだけを行う（面の数は減らさない）
使い方: python 12_fill_holes.py <car_id> <result_tag>
出力: results/<result_tag>/edited_filled.ply
"""
import sys, os
ROOT = os.path.expanduser("~/Desktop/ikeda/detail_edit")
REPO = f"{ROOT}/repos/3dmorph"
sys.path.insert(0, REPO); os.chdir(REPO)
import numpy as np, trimesh
from trellis.utils import postprocessing_utils as pu

car, tag = sys.argv[1], sys.argv[2]
rd = f"{ROOT}/data/cars/{car}/results/{tag}"
m = trimesh.load(f"{rd}/edited_raw.ply", force="mesh", process=False)
R = trimesh.transformations.rotation_matrix(np.radians(90), [1, 0, 0])
m.apply_transform(R)                                     # Y-up -> TRELLISの座標(Z-up)に戻す
v, f = pu.postprocess_mesh(m.vertices.astype(np.float32), m.faces.astype(np.int32),
                           simplify=False, fill_holes=True, verbose=True)
out = trimesh.Trimesh(v, f, process=False)
out.apply_transform(np.linalg.inv(R))                    # Y-upに戻す
out.export(f"{rd}/edited_filled.ply")
print(f"面の数: {len(m.faces)} -> {len(out.faces)}  保存: {rd}/edited_filled.ply")
