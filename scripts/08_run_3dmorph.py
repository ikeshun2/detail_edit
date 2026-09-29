"""3DMorphで編集を実行する
使い方: python 08_run_3dmorph.py <car_id> <tag> [seed=42] [front_ratio] [--steps 12] [--cfg_ss 7.5] [--cfg_slat 3.0]
  front_ratio を指定すると、BBを「前端から車長の front_ratio」に固定する
出力: data/cars/<car_id>/results/<tag>_seed<seed>/
"""
import sys, os, math, argparse
ROOT = os.path.expanduser("~/Desktop/ikeda/detail_edit")
REPO = f"{ROOT}/repos/3dmorph"
sys.path.insert(0, REPO); os.chdir(REPO)

ap = argparse.ArgumentParser()
ap.add_argument("car_id"); ap.add_argument("tag")
ap.add_argument("seed", type=int, nargs="?", default=42)
ap.add_argument("front_ratio", type=float, nargs="?", default=None)
ap.add_argument("--steps", type=int, default=12)
ap.add_argument("--cfg_ss", type=float, default=7.5)
ap.add_argument("--cfg_slat", type=float, default=3.0)
ap.add_argument("--band", type=int, nargs=2, default=None)   # 16^3での車長方向のマス範囲 [lo, hi]
args = ap.parse_args()

import xformers_patch
os.system("pgrep -f 'Xvfb :99' >/dev/null || Xvfb :99 -screen 0 1024x768x16 &")
os.environ["ATTN_BACKEND"] = "xformers"; os.environ["SPCONV_ALGO"] = "native"

import numpy as np, torch, trimesh
import trellis.pipelines._3DMorph_inpaint as P
from trellis.pipelines import _3DMorphInpaint
from trellis.utils import postprocessing_utils
from example_inpaint import prepare_assets

_orig_predict = P.MaskPredictor.predict_mask
def _predict_with_fixed_bb(self, *a, **k):
    m, extra = _orig_predict(self, *a, **k)
    if args.band is not None:
        lo, hi = args.band
        new = torch.zeros_like(m); idx = [slice(None)] * m.dim()
        idx[m.dim() - 2] = slice(lo, hi + 1); new[tuple(idx)] = 1
        print(f"[BB] 帯: 車長方向のマス {lo}〜{hi}（ボクセル {lo*4}〜{hi*4+3}）")
        return new, extra
    if args.front_ratio is None:
        return m, extra
    R = m.shape[-1]; n = max(1, math.ceil(args.front_ratio * R))
    new = torch.zeros_like(m); idx = [slice(None)] * m.dim()
    idx[m.dim() - 2] = slice(R - n, R)              # 車長方向の大きい番号側 = フロント
    new[tuple(idx)] = 1
    print(f"[BB] 固定: 車長方向のマス {R-n}〜{R-1}（前端から {n/R*100:.0f}%）")
    return new, extra
P.MaskPredictor.predict_mask = _predict_with_fixed_bb

work_dir = f"{ROOT}/data/cars/{args.car_id}"
out_dir = f"{work_dir}/results/{args.tag}_seed{args.seed}"
os.makedirs(out_dir, exist_ok=True)
print(f"設定: seed={args.seed} steps={args.steps} cfg_ss={args.cfg_ss} cfg_slat={args.cfg_slat}")

pipeline = _3DMorphInpaint.from_pretrained("pretrained_weights/TRELLIS-image-large"); pipeline.cuda()
assets = prepare_assets(work_dir)
outputs, new_slat = pipeline.run_inpaint(
    assets, seed=args.seed,
    sparse_structure_sampler_params={"steps": args.steps, "cfg_strength": args.cfg_ss},
    slat_sampler_params={"steps": args.steps, "cfg_strength": args.cfg_slat},
)
print("ボクセル数: 編集前", assets.slat.coords.shape[0], "/ 編集後", new_slat.coords.shape[0])

rot = trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0])
m = outputs["mesh"][0]; v, f = m.vertices.cpu().numpy(), m.faces.cpu().numpy()
raw = trimesh.Trimesh(v, f, process=False); raw.apply_transform(rot); raw.export(f"{out_dir}/edited_raw.ply")
v2, f2 = postprocessing_utils.postprocess_mesh(v, f, simplify_ratio=0.9)
simp = trimesh.Trimesh(v2, f2); simp.apply_transform(rot); simp.export(f"{out_dir}/edited.ply")
np.savez_compressed(f"{out_dir}/pred_slat.npz",
                    feats=new_slat.feats.cpu().numpy().astype(np.float32),
                    coords=new_slat.coords[:, 1:].cpu().numpy().astype(np.uint8))
print("保存:", out_dir)
