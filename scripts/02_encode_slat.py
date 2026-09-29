"""正規化OBJ -> 多視点レンダリング -> ボクセル化・特徴抽出 -> SLAT
使い方: python 02_encode_slat.py <car_id> [n_views=150]
"""
import sys, os, time
ROOT = os.path.expanduser("~/Desktop/ikeda/detail_edit")
REPO = f"{ROOT}/repos/3dmorph"
sys.path.insert(0, REPO)
os.chdir(REPO)                                  # 重みなどの相対パスを解決するため

import xformers_patch                           # Blackwell対応
os.system("pgrep -f 'Xvfb :99' >/dev/null || Xvfb :99 -screen 0 1024x768x16 &")
os.environ["ATTN_BACKEND"] = "xformers"
os.environ["SPCONV_ALGO"] = "native"

from _3DMorph.slat_encoder.feature_extractor import FeatureExtractor
from _3DMorph.slat_encoder.latent_encoder import LatentEncoder

car_id = sys.argv[1]
n_views = int(sys.argv[2]) if len(sys.argv) > 2 else 150
mesh_path = f"{ROOT}/data/cars/{car_id}/unmodified.obj"
assert os.path.exists(mesh_path), mesh_path

t0 = time.time()
extractor = FeatureExtractor(batch_size=n_views, n_views=n_views)
# 参照メッシュ=自分自身（Phase 0では車ごとに正規化）
feat_path = extractor.run_extractor(mesh_path, mesh_path, force_render=True)
slat_path = LatentEncoder().run_slat_encoder(feat_path)
print(f"features: {feat_path}\nslat    : {slat_path}\n所要時間: {time.time()-t0:.0f} s")
