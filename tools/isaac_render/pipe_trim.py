# micro-skater tools/isaac_render: the author's Isaac Sim replay pipeline. Apache-2.0 (see LICENSE).
"""Shorter rendered half-pipe (Isaac only): trim pipe.json's y extent to the part the run actually uses.

    python pipe_trim.py [--dir chain_cine] [--margin 0.7]

MuJoCo's pipe runs y0..y1 (+-3 m); the chain only rides a short stretch of it near the entry end. From the run's own
kinematics (export_chain.py chain.kin.npz: every wheel centre, robot link and board body origin, every step) this takes
the y range of every point inside the MuJoCo pipe footprint (x_lo..x_hi, y0..y1), adds --margin at each end and clamps
to the MuJoCo pipe. An end the rider crosses (the carve entry through the open end) stays exactly where MuJoCo has it,
so the board always rolls from the pad onto rendered pipe. Writes <dir>/pipe_render.json = pipe.json with the trimmed
y0 / y1 plus y0_sim / y1_sim / trim info; build_park.py --pipe <that> builds the short pipe (end faces, steel edges and
capped copings at both ends, as before). check_chain.py then verifies that no wheel / link ever sits over a missing pipe.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=os.path.join(HERE, "chain_cine"))
ap.add_argument("--margin", type=float, default=0.7, help="m of pipe kept beyond the last used point")
a = ap.parse_args()

P = json.load(open(os.path.join(a.dir, "pipe.json")))
z = np.load(os.path.join(a.dir, "chain.kin.npz"))
pts = np.concatenate([z["wheel_center"].reshape(-1, 3), z["robot_body_pos"].reshape(-1, 3),
                      z["board_body_pos"].reshape(-1, 3)])
y0s, y1s = float(P.get("y0_sim", P["y0"])), float(P.get("y1_sim", P["y1"]))
inside = (pts[:, 0] >= P["x_lo"]) & (pts[:, 0] <= P["x_hi"]) & (pts[:, 1] >= y0s) & (pts[:, 1] <= y1s)
if not inside.any():
    raise SystemExit("pipe_trim: the run never enters the pipe footprint")
ylo, yhi = float(pts[inside, 1].min()), float(pts[inside, 1].max())
# an end crossed by the path (a point within 5 cm of it while inside the footprint, and points outside beyond it)
xin = (pts[:, 0] >= P["x_lo"]) & (pts[:, 0] <= P["x_hi"])
cross0 = bool(((pts[:, 1] < y0s) & xin).any() and ylo < y0s + 0.05)
cross1 = bool(((pts[:, 1] > y1s) & xin).any() and yhi > y1s - 0.05)
y0 = y0s if cross0 else max(y0s, ylo - a.margin)
y1 = y1s if cross1 else min(y1s, yhi + a.margin)
R = dict(P, y0=y0, y1=y1, y0_sim=y0s, y1_sim=y1s,
         trim=dict(margin=a.margin, used_y=[ylo, yhi], entry_end=("y0" if cross0 else "") + ("y1" if cross1 else ""),
                   length_sim=y1s - y0s, length_render=y1 - y0, n_points_inside=int(inside.sum())))
json.dump(R, open(os.path.join(a.dir, "pipe_render.json"), "w"), indent=1)
print(f"PIPE trim: used y {ylo:.3f}..{yhi:.3f} (entry end {R['trim']['entry_end'] or 'none'}); rendered y "
      f"{y0:.3f}..{y1:.3f} = {y1 - y0:.2f} m of MuJoCo's {y1s - y0s:.2f} m (margin {a.margin} m)")
