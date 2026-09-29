"""編集前後のSLATを比べて、実際に編集された範囲(BB)を調べる
使い方: python 10_inspect_edit.py <car_id> <result_tag>
"""
import sys, os, json
import numpy as np, trimesh

ROOT = os.path.expanduser("~/Desktop/ikeda/detail_edit")
car, tag = sys.argv[1], sys.argv[2]
cd = f"{ROOT}/data/cars/{car}"; rd = f"{cd}/results/{tag}"
A = np.load(f"{cd}/features/unmodified_slat.npz")      # 編集前
B = np.load(f"{rd}/pred_slat.npz")                      # 編集後
ca, fa = A["coords"].astype(int), A["feats"]
cb, fb = B["coords"].astype(int), B["feats"]
key = lambda c: c[:, 0] * 4096 + c[:, 1] * 64 + c[:, 2]
ka, kb = key(ca), key(cb)
common, ia, ib = np.intersect1d(ka, kb, return_indices=True)
diff = np.abs(fa[ia] - fb[ib]).max(1)
changed = cb[ib][diff > 1e-3]
added = cb[~np.isin(kb, ka)]
removed = ca[~np.isin(ka, kb)]
print(f"ボクセル数: 編集前 {len(ca)} / 編集後 {len(cb)} / 共通 {len(common)}")
print(f"  共通のうち潜在表現が変化: {len(changed)}  |  追加: {len(added)}  |  削除: {len(removed)}")

span = ca.max(0) - ca.min(0)
L_ax = int(np.argmax(span))                              # 車長方向のボクセル軸
m_per_vox = json.load(open(f"{cd}/norm.json"))
obj = trimesh.load(f"{cd}/unmodified.obj", force="mesh")
M = np.ptp(obj.vertices, axis=0).max() / m_per_vox["scale"] / 64   # 1ボクセル[m]
print(f"車長方向の軸: {L_ax}（範囲 {ca[:, L_ax].min()}〜{ca[:, L_ax].max()}）, 1ボクセル = {M*1000:.0f} mm")

edit = np.vstack([changed, added]) if len(added) else changed
if len(edit):
    lo, hi = edit.min(0), edit.max(0)
    print(f"編集されたボクセルの範囲（BBの推定）: 最小 {lo.tolist()} 最大 {hi.tolist()}")
    for end, ref in [("高い端", ca[:, L_ax].max()), ("低い端", ca[:, L_ax].min())]:
        d = np.abs(edit[:, L_ax] - ref)
        print(f"  車長方向で{end}から: {d.min()*M:.2f} m 〜 {d.max()*M:.2f} m")
    print(f"  編集範囲の車長に対する割合: {(hi[L_ax]-lo[L_ax]+1)/(span[L_ax]+1)*100:.0f}%")
if len(added):
    print(f"追加されたボクセルの範囲: 最小 {added.min(0).tolist()} 最大 {added.max(0).tolist()}")

# 比較の座標ずれの確認
b = trimesh.load(f"{cd}/recon/slat.ply", force="mesh")
a = trimesh.load(f"{rd}/edited_raw.ply", force="mesh")
print("メッシュの外形 編集前:", np.round(b.bounds, 3).tolist())
print("メッシュの外形 編集後:", np.round(a.bounds, 3).tolist())
