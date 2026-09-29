"""位置合わせ済みのドナーを、ターゲットと同じ座標系でSLATにする
使い方: python 13_encode_donor.py <target_id> <donor_id>
前提: 06_render_views.py pair を実行済み（donors/<donor_id>_aligned.obj がある）
出力: data/cars/<target_id>/donors/<donor_id>/features/donor_slat.npz
"""
import sys, os, shutil, time
ROOT = os.path.expanduser("~/Desktop/ikeda/detail_edit")
REPO = f"{ROOT}/repos/3dmorph"
sys.path.insert(0, REPO); os.chdir(REPO)
import xformers_patch
os.system("pgrep -f 'Xvfb :99' >/dev/null || Xvfb :99 -screen 0 1024x768x16 &")
os.environ["ATTN_BACKEND"] = "xformers"; os.environ["SPCONV_ALGO"] = "native"
from _3DMorph.slat_encoder.feature_extractor import FeatureExtractor
from _3DMorph.slat_encoder.latent_encoder import LatentEncoder

tid, did = sys.argv[1], sys.argv[2]
src = f"{ROOT}/data/cars/{tid}/donors/{did}_aligned.obj"
d = f"{ROOT}/data/cars/{tid}/donors/{did}"
os.makedirs(d, exist_ok=True)
dst = f"{d}/donor.obj"
shutil.copy(src, dst)
t0 = time.time()
ext = FeatureExtractor(batch_size=150, n_views=150)
feat = ext.run_extractor(dst, f"{ROOT}/data/cars/{tid}/unmodified.obj", force_render=True)  # 参照=ターゲット
slat = LatentEncoder().run_slat_encoder(feat)
print(f"保存: {slat}  所要時間: {time.time()-t0:.0f} s")
