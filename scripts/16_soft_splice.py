"""潜在表現の補間による、なめらかな貼り合わせ（生成なし・決定的）
使い方: python 16_soft_splice.py <target_id> <donor_id> [ratio=0.1] [width=6]
  ratio: 境界の位置（前端から車長の割合）
  width: 補間する帯の幅（ボクセル数、1ボクセル≒71mm）
出力: data/cars/<target_id>/results/softsplice_<donor_id>_r<ratio>_w<width>/
"""
import sys, os, glob
ROOT = os.path.expanduser("~/Desktop/ikeda/detail_edit")
REPO = f"{ROOT}/repos/3dmorph"
sys.path.insert(0, REPO); os.chdir(REPO)
import xformers_patch
os.environ["ATTN_BACKEND"] = "xformers"; os.environ["SPCONV_ALGO"] = "native"
import numpy as np, torch, trimesh
from trellis.pipelines import _3DMorphInpaint
from trellis.modules import sparse as sp

tid, did = sys.argv[1], sys.argv[2]
ratio = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
width = int(sys.argv[4]) if len(sys.argv) > 4 else 6
cd = f"{ROOT}/data/cars/{tid}"
T = np.load(f"{cd}/features/unmodified_slat.npz")
D = np.load(glob.glob(f"{cd}/donors/{did}/features/*_slat.npz")[0])
cut = 64 - round(ratio * 64)                        # 軸1がcut以上 = フロント（ドナー側）

def weight(i):
    """軸1の位置 i でのドナーの割合（帯の中で0->1に線形変化）"""
    return np.clip((i - (cut - width / 2)) / width, 0.0, 1.0)

key = lambda c: c[:, 0].astype(np.int64) * 4096 + c[:, 1] * 64 + c[:, 2]
kt, kd = key(T["coords"].astype(np.int64)), key(D["coords"].astype(np.int64))
both, it, idd = np.intersect1d(kt, kd, return_indices=True)
only_t = np.setdiff1d(np.arange(len(kt)), it)
only_d = np.setdiff1d(np.arange(len(kd)), idd)

coords, feats = [], []
# 両方にあるボクセル: 潜在表現を線形に混ぜる
w = weight(T["coords"][it, 1].astype(float))[:, None]
coords.append(T["coords"][it]); feats.append((1 - w) * T["feats"][it] + w * D["feats"][idd])
# 片方にしかないボクセル: 割合が半分を超える側だけ残す
wt = weight(T["coords"][only_t, 1].astype(float)); keep_t = only_t[wt < 0.5]
wd = weight(D["coords"][only_d, 1].astype(float)); keep_d = only_d[wd >= 0.5]
coords += [T["coords"][keep_t], D["coords"][keep_d]]
feats += [T["feats"][keep_t], D["feats"][keep_d]]
coords = np.vstack(coords).astype(np.int32); feats = np.vstack(feats).astype(np.float32)
print(f"境界: 軸1 = {cut}, 帯: 軸1 = {cut - width/2:.0f}〜{cut + width/2:.0f}（幅 {width} ボクセル ≒ {width*71} mm）")
print(f"ボクセル: 両方 {len(both)} / CX-5のみ {len(keep_t)} / ドナーのみ {len(keep_d)} / 合計 {len(coords)}")

c = torch.cat([torch.zeros(len(coords), 1).int(), torch.from_numpy(coords).int()], 1)
slat = sp.SparseTensor(feats=torch.from_numpy(feats), coords=c).cuda()
pipeline = _3DMorphInpaint.from_pretrained("pretrained_weights/TRELLIS-image-large"); pipeline.cuda()
with torch.no_grad():
    m = pipeline.decode_slat(slat, ["mesh"])["mesh"][0]
mesh = trimesh.Trimesh(m.vertices.cpu().numpy(), m.faces.cpu().numpy(), process=False)
mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0]))
out = f"{cd}/results/softsplice_{did}_r{ratio}_w{width}"
os.makedirs(out, exist_ok=True)
mesh.export(f"{out}/edited_raw.ply")
np.savez_compressed(f"{out}/pred_slat.npz", feats=feats, coords=coords.astype(np.uint8))
print("保存:", out)
