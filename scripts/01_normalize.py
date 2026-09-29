"""STL -> 正規化OBJ（全車共通スケール、Y-up、左右対称性による向きの補正）
使い方:
  python 01_normalize.py scan                          # 全STLから共通スケールを決定
  python 01_normalize.py check <car_id>                # 変換済みOBJの傾きを測るだけ
  python 01_normalize.py convert <stlのパス> <car_id>   # 1台を変換（傾きを自動補正）
"""
import sys, os, glob, json
import numpy as np, trimesh, open3d as o3d
from scipy.spatial import cKDTree

ROOT = os.path.expanduser("~/Desktop/ikeda/detail_edit")
GLOBAL = f"{ROOT}/data/meta/global_norm.json"
AX = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], float)  # Z-up -> Y-up


def load_clean(path):
    m = trimesh.load(path, force="mesh")
    m.merge_vertices()
    m.update_faces(m.nondegenerate_faces())
    m.remove_unreferenced_vertices()
    m.fix_normals()
    return m


def exterior_points(mesh, n=150000, n_dirs=48):
    """外から見える表面点（車内の非対称な形状を除くため）"""
    pts, _ = trimesh.sample.sample_surface(mesh, n)
    s = o3d.t.geometry.RaycastingScene()
    lm = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),
                                   o3d.utility.Vector3iVector(mesh.faces))
    s.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(lm))
    ext = np.ptp(mesh.vertices, axis=0).max(); R, eps = 2 * ext, 1e-3 * ext
    i = np.arange(n_dirs) + 0.5
    ph, th = np.arccos(1 - 2 * i / n_dirs), np.pi * (1 + 5 ** 0.5) * i
    dirs = np.stack([np.cos(th) * np.sin(ph), np.sin(th) * np.sin(ph), np.cos(ph)], 1)
    vis = np.zeros(len(pts), bool)
    for v in dirs:
        rays = np.hstack([pts + v * R, np.repeat(-v[None], len(pts), 0)]).astype(np.float32)
        vis |= s.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy() > R - eps
    return pts[vis]


def rot_y(d):   # 鉛直軸(y)まわり = ヨー
    c, s = np.cos(np.radians(d)), np.sin(np.radians(d))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(d):   # 前後軸(z)まわり = ロール
    c, s = np.cos(np.radians(d)), np.sin(np.radians(d))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def sym_cost(P):
    """x方向に鏡映したときの重なりの悪さ（小さいほど左右対称）"""
    c = np.median(P[:, 0])
    Q = P.copy(); Q[:, 0] = 2 * c - Q[:, 0]
    return cKDTree(P).query(Q[::3])[0].mean()


def search(P, rot, rng):
    coarse = np.arange(-rng, rng + 1e-9, 0.25)
    best = coarse[np.argmin([sym_cost(P @ rot(a).T) for a in coarse])]
    fine = np.arange(best - 0.3, best + 0.3 + 1e-9, 0.02)
    return float(fine[np.argmin([sym_cost(P @ rot(a).T) for a in fine])])


def estimate_pose(mesh):
    """Y-up・中心化済みのメッシュについて、ヨー・ロールを推定し、ピッチを測る"""
    P = exterior_points(mesh); P = P - P.mean(0)
    yaw = search(P, rot_y, 6.0)
    P = P @ rot_y(yaw).T
    roll = search(P, rot_z, 3.0)
    P = P @ rot_z(roll).T
    z0, z1 = P[:, 2].min(), P[:, 2].max(); L = z1 - z0
    fr, rr = P[P[:, 2] < z0 + 0.35 * L], P[P[:, 2] > z1 - 0.35 * L]
    pf, pr = fr[np.argmin(fr[:, 1])], rr[np.argmin(rr[:, 1])]      # 前輪・後輪の最下点
    pitch = float(np.degrees(np.arctan2(pf[1] - pr[1], pr[2] - pf[2])))
    return yaw, roll, pitch, sym_cost(P) / L


def scan():
    paths = sorted(glob.glob(f"{ROOT}/data/raw/*/*.stl"))
    if not paths:
        sys.exit(f"STLが見つかりません: {ROOT}/data/raw/*/*.stl")
    ext = {}
    for p in paths:
        b = trimesh.load(p, force="mesh").bounds
        ext[p] = float((b[1] - b[0]).max())
        print(f"{os.path.basename(p)}: 最長辺 {ext[p]:.3f}")
    max_ext, min_ext = max(ext.values()), min(ext.values())
    cfg = {"unit": "m", "n_cars": len(paths), "max_extent": max_ext,
           "min_extent": min_ext, "scale": 0.98 / max_ext}
    json.dump(cfg, open(GLOBAL, "w"), indent=2)
    print(json.dumps(cfg, indent=2))
    print(f"最小/最大の車長比: {min_ext / max_ext:.3f}")


def check(car_id):
    m = trimesh.load(f"{ROOT}/data/cars/{car_id}/unmodified.obj", force="mesh")
    yaw, roll, pitch, cost = estimate_pose(m)
    print(f"{car_id}: ヨー {yaw:+.2f}° | ロール {roll:+.2f}° | ピッチ(前輪-後輪) {pitch:+.2f}° "
          f"| 補正後の非対称度 {cost*1000:.2f}‰")


def convert(stl_path, car_id):
    cfg = json.load(open(GLOBAL))
    m = load_clean(stl_path)
    T = np.eye(4)
    def apply(M):
        nonlocal T
        m.apply_transform(M); T = M @ T
    apply(trimesh.transformations.translation_matrix(-(m.bounds[0] + m.bounds[1]) / 2))
    apply(AX)                                                    # Z-up -> Y-up
    yaw, roll, pitch, cost = estimate_pose(m)
    Rm = np.eye(4); Rm[:3, :3] = rot_z(roll) @ rot_y(yaw)
    apply(Rm)                                                    # 向きの補正
    apply(trimesh.transformations.translation_matrix(-(m.bounds[0] + m.bounds[1]) / 2))
    apply(np.diag([cfg["scale"]] * 3 + [1.0]))                   # 全車共通スケール
    out = f"{ROOT}/data/cars/{car_id}"
    for d in ["features", "recon", "explore_inpaint"]:
        os.makedirs(f"{out}/{d}", exist_ok=True)
    m.export(f"{out}/unmodified.obj")
    json.dump({"source": os.path.abspath(stl_path), "unit": "m", "scale": cfg["scale"],
               "yaw_deg": yaw, "roll_deg": roll, "pitch_deg_measured": pitch,
               "raw_to_normalized": T.tolist()},
              open(f"{out}/norm.json", "w"), indent=2)
    print(f"{car_id}: 補正 ヨー {yaw:+.2f}° ロール {roll:+.2f}° | ピッチ(参考) {pitch:+.2f}° | "
          f"bounds {np.round(m.bounds, 4).tolist()}")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "scan":
        scan()
    elif len(sys.argv) == 3 and sys.argv[1] == "check":
        check(sys.argv[2])
    elif len(sys.argv) == 4 and sys.argv[1] == "convert":
        convert(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)
