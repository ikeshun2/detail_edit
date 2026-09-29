"""Phase 3a: BBの外=ターゲットのSLAT、BBの中=ドナーのSLAT を貼り合わせてデコード（生成なし）
使い方: python 14_splice.py <target_id> <donor_id> [bb=0.25]
出力: data/cars/<target_id>/results/splice_<donor_id>_bb<bb>/edited_raw.ply, pred_slat.npz
"""
import sys, os, glob, math
ROOT = os.path.expanduser("~/Desktop/ikeda/detail_edit")
REPO = f"{ROOT}/repos/3dmorph"
sys.path.insert(0, REPO); os.chdir(REPO)
import xformers_patch
os.environ["ATTN_BACKEND"] = "xformers"; os.environ["SPCONV_ALGO"] = "native"
import numpy as np, torch, trimesh
from trellis.pipelines import _3DMorphInpaint
from trellis.modules import sparse as sp

tid, did = sys.argv[1], sys.argv[2]
bb = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
cd = f"{ROOT}/data/cars/{tid}"
T = np.load(f"{cd}/features/unmodified_slat.npz")
D = np.load(glob.glob(f"{cd}/donors/{did}/features/*_slat.npz")[0])
cut = 64 - round(bb * 64)                       # 64^3のボクセル単位で境界を決める（生成しないので16^3に揃える必要なし）
tk = T["coords"][:, 1] < cut                     # 軸1の小さい側 = フロント以外
dk = D["coords"][:, 1] >= cut                    # 軸1の大きい側 = フロント
coords = np.vstack([T["coords"][tk], D["coords"][dk]]).astype(np.int32)
feats = np.vstack([T["feats"][tk], D["feats"][dk]]).astype(np.float32)
print(f"境界: 軸1 = {cut} | ターゲットから {tk.sum()} / ドナーから {dk.sum()} ボクセル（合計 {len(coords)}）")

c = torch.from_numpy(coords).int()
c = torch.cat([torch.zeros(len(c), 1).int(), c], 1)
slat = sp.SparseTensor(feats=torch.from_numpy(feats), coords=c).cuda()
pipeline = _3DMorphInpaint.from_pretrained("pretrained_weights/TRELLIS-image-large"); pipeline.cuda()
with torch.no_grad():
    m = pipeline.decode_slat(slat, ["mesh"])["mesh"][0]
mesh = trimesh.Trimesh(m.vertices.cpu().numpy(), m.faces.cpu().numpy(), process=False)
mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0]))

out = f"{cd}/results/splice_{did}_bb{bb}"
os.makedirs(out, exist_ok=True)
mesh.export(f"{out}/edited_raw.ply")
np.savez_compressed(f"{out}/pred_slat.npz", feats=feats, coords=coords.astype(np.uint8))
print("保存:", out)
