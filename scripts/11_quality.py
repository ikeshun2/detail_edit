"""編集結果のメッシュ品質の診断と後処理
使い方: python 11_quality.py <car_id> <result_tag> [--bb 0.25] [--fix] [--iters 10]
"""
import sys, os, argparse
import numpy as np, trimesh

ap = argparse.ArgumentParser()
ap.add_argument("car_id"); ap.add_argument("tag")
ap.add_argument("--bb", type=float, default=0.25)
ap.add_argument("--fix", action="store_true")
ap.add_argument("--iters", type=int, default=10)
a = ap.parse_args()
ROOT = os.path.expanduser("~/Desktop/ikeda/detail_edit")
rd = f"{ROOT}/data/cars/{a.car_id}/results/{a.tag}"
m = trimesh.load(f"{rd}/" + os.environ.get("EDITED", "edited_raw.ply"), force="mesh", process=False)
m.merge_vertices()

# SLATの出力座標: 最長辺=1、フロントは z = -0.5 側
z_bb = -0.5 + a.bb

def report(mesh, title):
    comps = mesh.split(only_watertight=False)
    sizes = sorted([len(c.faces) for c in comps], reverse=True)
    boundary = len(trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1))
    ang = np.degrees(mesh.face_adjacency_angles)
    zc = mesh.triangles_center[mesh.face_adjacency[:, 0], 2]
    ins, out = zc < z_bb - 0.02, zc > z_bb + 0.02
    band = np.abs(zc - z_bb) <= 0.02
    print(f"[{title}] 面 {len(mesh.faces)} | 破片 {len(comps)}（最大以外の面の合計 {sum(sizes[1:])}）"
          f" | 穴の縁の辺 {boundary} | 閉じたメッシュ {mesh.is_watertight}")
    print(f"   面の荒さ（隣接面の角度の平均）: BB内 {ang[ins].mean():.2f}° / BB外 {ang[out].mean():.2f}°"
          f" | 30°超の割合: BB内 {np.mean(ang[ins] > 30)*100:.2f}% / BB外 {np.mean(ang[out] > 30)*100:.2f}%")
    print(f"   継ぎ目付近: 荒さ {ang[band].mean():.2f}° | 30°超の割合 {np.mean(ang[band] > 30)*100:.2f}%")

report(m, "編集直後")
if a.fix:
    comps = m.split(only_watertight=False)
    big = max(len(c.faces) for c in comps)
    m = trimesh.util.concatenate([c for c in comps if len(c.faces) >= 0.01 * big])   # 小さな破片を除去
    m.merge_vertices()
    trimesh.repair.fill_holes(m)                                                       # 小さな穴を埋める
    v0 = m.vertices.copy()
    s = m.copy(); trimesh.smoothing.filter_taubin(s, lamb=0.5, nu=-0.53, iterations=a.iters)
    w = np.clip((z_bb - v0[:, 2]) / 0.03, 0, 1)[:, None]                              # BB内=1、境界で徐々に0へ
    m.vertices = w * s.vertices + (1 - w) * v0
    trimesh.repair.fix_normals(m)
    m.export(f"{rd}/edited_clean.ply")
    report(m, "後処理後")
    print("保存:", f"{rd}/edited_clean.ply")
