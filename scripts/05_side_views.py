"""全車の側面図（z-y）と上面図（z-x）を並べ、前後の向きを確認する
使い方: python 05_side_views.py
"""
import os, glob
import numpy as np, trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.expanduser("~/Desktop/ikeda/detail_edit")
cars = sorted(glob.glob(f"{ROOT}/data/cars/*/unmodified.obj"))
print("対象:", [c.split("/")[-2] for c in cars])
fig, axes = plt.subplots(len(cars), 2, figsize=(12, 2.6 * len(cars)))
axes = np.atleast_2d(axes)
for row, path in zip(axes, cars):
    name = path.split("/")[-2]
    m = trimesh.load(path, force="mesh")
    p, _ = trimesh.sample.sample_surface(m, 60000)
    row[0].scatter(p[:, 2], p[:, 1], s=0.2, c="k")
    row[0].set_title(f"{name}  side (horizontal: z, vertical: y)")
    row[1].scatter(p[:, 2], p[:, 0], s=0.2, c="k")
    row[1].set_title(f"{name}  top (horizontal: z, vertical: x)")
    for ax in row:
        ax.set_aspect("equal"); ax.set_xlim(-0.52, 0.52)
        ax.axvline(0.4, color="r", lw=0.5); ax.text(0.41, 0, "z+", color="r")
        ax.axvline(-0.4, color="b", lw=0.5); ax.text(-0.49, 0, "z-", color="b")
plt.tight_layout()
out = f"{ROOT}/results/phase1/side_views.png"
plt.savefig(out, dpi=120)
print("保存:", out)
