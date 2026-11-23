# micro-skater tools/isaac_render: the author's Isaac Sim replay pipeline. Apache-2.0 (see LICENSE).
"""Validate the baked chain replay against MuJoCo (plain usd-core: .usdvenv python; no renderer).

    python check_chain.py [--replay scenes\\skatepark_pipe\\replay.usda] [--timeline timeline_chain.npz] [--every 3]

For every --every-th output frame:
  - every robot link origin and every board body origin in the USD (composed, time-sampled) vs MuJoCo's xpos
    (timeline robot_body_pos / board_pos, interpolated exactly like the bake)
  - every wheel: the distance from the USD wheel centre to the USD half-pipe Surface mesh profile (the mesh actually
    rendered), minus the wheel radius = the visible gap (+) / sink (-) of the wheel on the pipe or the floor
Prints max / percentiles and writes check_chain.json next to the timeline.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from pxr import Usd, UsdGeom

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(HERE))  # the repo root: scenes/, output/ and assets/ live there
ap = argparse.ArgumentParser()
ap.add_argument("--replay", default=os.path.join(PROJECT, "scenes", "skatepark_pipe", "replay.usda"))
ap.add_argument("--timeline", default=os.path.join(HERE, "timeline_chain.npz"))
ap.add_argument("--every", type=int, default=3)
a = ap.parse_args()

class _TL(dict):          # cine: the timeline arrays decompressed once (NpzFile[key] re-reads the array per access)
    files = property(lambda self: list(self.keys()))


tl = _TL(np.load(a.timeline).items())
stage = Usd.Stage.Open(a.replay)
N = len(tl["shot"])
robot = stage.GetPrimAtPath("/World/Microduck/Robot")
trunk = stage.GetPrimAtPath("/World/Microduck/Robot/Geometry/trunk_base")
# link prims as the bake animates them: the parents of the robot's joint prims (isaac:physics:robotJoints)
links = {p.GetParentPath().name: stage.GetPrimAtPath(p.GetParentPath())
         for p in robot.GetRelationship("isaac:physics:robotJoints").GetTargets()}
links.update({stage.GetPrimAtPath(p.GetParentPath()).GetName(): stage.GetPrimAtPath(p.GetParentPath())
              for p in robot.GetRelationship("isaac:physics:robotJoints").GetTargets()})
rnames = [str(n) for n in tl["robot_body_names"]]
bnames = [str(n) for n in tl["board_body_names"]]
# the rendered half-pipe surface: profile (x, z) from the mesh points (constant in y)
surf = UsdGeom.Mesh(stage.GetPrimAtPath("/World/Park/HalfPipe/Surface"))
P = np.array(surf.GetPointsAttr().Get())
prof = np.unique(np.round(P[:, [0, 2]], 7), axis=0)
prof = prof[np.argsort(prof[:, 0])]
y0, y1 = P[:, 1].min(), P[:, 1].max()
gi = [i for i, n in enumerate(tl["board_geom_name"]) if str(n).endswith(("wheel_L_collision", "wheel_R_collision"))]
r_w = float(tl["board_geom_size"][gi[0]][0])


def surface_gap(c):
    x, y, z = c
    if not (y0 <= y <= y1 and prof[0, 0] < x < prof[-1, 0]):
        return z - r_w                                      # floor plane z = 0
    A, B = prof[:-1], prof[1:]
    ab = B - A
    p = np.array([x, z])
    t = np.clip(np.einsum("ij,ij->i", p - A, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-18), 0, 1)
    return float(np.min(np.linalg.norm(A + t[:, None] * ab - p, axis=1))) - r_w


errs_r, errs_b, gaps, gap_pipe = [], [], [], []
for k in range(0, N, a.every):
    cache = UsdGeom.XformCache(k)
    for bi, bn in enumerate(rnames):
        prim = trunk if bn == "trunk_base" else links.get(bn) or links.get(bn + "_1")
        if prim is None:
            continue
        p = np.array(cache.GetLocalToWorldTransform(prim).ExtractTranslation())
        errs_r.append(np.linalg.norm(p - tl["robot_body_pos"][k, bi]))
    for bi, bn in enumerate(bnames):
        prim = stage.GetPrimAtPath(f"/World/Skateboard/{bn}")
        p = np.array(cache.GetLocalToWorldTransform(prim).ExtractTranslation())
        errs_b.append(np.linalg.norm(p - tl["board_pos"][k, bi]))
    for g in gi:
        b, nm = str(tl["board_geom_body"][g]), "".join(ch if ch.isalnum() else "_" for ch in str(tl["board_geom_name"][g]))
        prim = stage.GetPrimAtPath(f"/World/Skateboard/{b}/{nm}")
        c = np.array(cache.GetLocalToWorldTransform(prim).ExtractTranslation())
        gp = surface_gap(c)
        gaps.append(gp)
        if y0 <= c[1] <= y1 and abs(c[0]) > 0.5:
            gap_pipe.append(gp)
# cine (2026-11-22, additive): a trimmed rendered pipe (pipe_trim.py; the timeline's pipe json has y0_sim / y1_sim).
# Every frame, every wheel centre, robot link and board body origin inside MuJoCo's pipe footprint must lie over the
# rendered pipe (its y range), else the robot would ride over a missing pipe: count and fail.
uncovered = None
PJ = json.loads(str(tl["pipe"])) if "pipe" in tl.files else {}
if "y0_sim" in PJ:
    uncovered, worst = 0, 0.0
    for k in range(N):
        wcs = []
        for g in gi:       # wheel centres: the timeline board body pose + the geom offset in the body frame
            bi_ = bnames.index(str(tl["board_geom_body"][g]))
            w_, x_, y_, z_ = tl["board_quat"][k, bi_]
            v_ = np.asarray(tl["board_geom_pos"][g], float)
            u_ = np.array([x_, y_, z_])
            wcs.append(tl["board_pos"][k, bi_] + v_ + 2 * np.cross(u_, np.cross(u_, v_) + w_ * v_))
        pts = np.concatenate([tl["robot_body_pos"][k], tl["board_pos"][k], np.array(wcs)])
        inn = ((pts[:, 0] >= PJ["x_lo"]) & (pts[:, 0] <= PJ["x_hi"]) & (pts[:, 1] >= PJ["y0_sim"])
               & (pts[:, 1] <= PJ["y1_sim"]))
        out_ = inn & ((pts[:, 1] < y0 - 1e-6) | (pts[:, 1] > y1 + 1e-6))
        if out_.any():
            uncovered += 1
            worst = max(worst, float(np.max(np.maximum(y0 - pts[out_, 1], pts[out_, 1] - y1))))
    print(f"trimmed pipe: rendered y {y0:.3f}..{y1:.3f} (MuJoCo {PJ['y0_sim']:.2f}..{PJ['y1_sim']:.2f}); frames with a "
          f"link / board body over missing pipe: {uncovered} of {N}" + (f" (worst {worst * 1000:.0f} mm)" if uncovered else ""))
errs_r, errs_b, gaps, gap_pipe = map(lambda v: np.array(v) * 1000, (errs_r, errs_b, gaps, gap_pipe))
q = lambda v: dict(min=float(v.min()), p5=float(np.percentile(v, 5)), median=float(np.median(v)),
                   p95=float(np.percentile(v, 95)), max=float(v.max()), n=int(len(v))) if len(v) else None
res = dict(robot_link_err_mm=q(errs_r), board_body_err_mm=q(errs_b), wheel_gap_mm_all=q(gaps),
           wheel_gap_mm_on_transitions=q(gap_pipe), frames=len(range(0, N, a.every)),
           uncovered_frames=uncovered)
json.dump(res, open(os.path.splitext(a.timeline)[0] + "_check.json", "w"), indent=1)
print(f"robot links: max {errs_r.max():.3f} mm, mean {errs_r.mean():.4f} mm ({len(errs_r)} samples)")
print(f"board bodies: max {errs_b.max():.3f} mm")
print("wheel gap to the rendered surface (mm; - = sunk into the surface, MuJoCo soft contact):")
print("  all      ", {k: round(v, 2) if isinstance(v, float) else v for k, v in q(gaps).items()})
if len(gap_pipe):
    print("  on walls ", {k: round(v, 2) if isinstance(v, float) else v for k, v in q(gap_pipe).items()})
if uncovered:
    raise SystemExit(f"check_chain: {uncovered} frames put the robot / board over missing (trimmed) pipe")
