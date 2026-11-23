# micro-skater tools/isaac_render: the author's Isaac Sim replay pipeline. Apache-2.0 (see LICENSE).
"""Export a Microduck skate-chain playback (chain_play.py -> demo_qpos.npz) to renderer-ready kinematics
(train venv: needs mujoco; CPU only).

    python export_chain.py out\\chain_v1\\demo_qpos.npz [--model model.mjb] [--out chain\\chain]

The model is the sequence task's compiled MuJoCo model (robot + sequence skateboard + half-pipe terrain). Default:
model.mjb next to demo_qpos.npz, else chain_v1's It is checked against demo_qpos (joint names and
qpos addresses) and against the current board XML (skateboard_sequence.xml geoms), so a stale model fails loudly
(chain_model.py rebuilds it from the chain env).

Writes, next to --out:
    <out>.kin.npz   the format (root, joints, every robot / board body, board geoms) + the chain's
                    stage / owner / apex logs; t at the 50 Hz policy rate
    pipe.json       the half-pipe exactly as MuJoCo collides with it: the upper envelope of the terrain box facets'
                    top faces (a polyline in x, constant in y over [y0, y1]), the platform boxes, the floor plane z
    meta.json       end-card numbers (picks.json / funnel.json of the chain run), stage times, apex times / rises
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
from micro_skater.paths import PACKAGE_DIR  # noqa: E402

SEQ = str(PACKAGE_DIR / "eval")   # chain_play.py (its camera constants are read from the source)
BOARD_XML = str(PACKAGE_DIR / "robot" / "assets" / "skateboard_sequence.xml")
FALLBACK_MODEL = os.environ.get("MICRO_SKATER_CHAIN_MJB", "model.mjb")   # chain_model.py writes one
TYPES = {int(mujoco.mjtGeom.mjGEOM_BOX): "box", int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
         int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere", int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule"}
STAGES = ["PUSH", "SETTLE", "MANUAL", "EXIT", "RAMP"]


def name(m, obj, i):
    return mujoco.mj_id2name(m, obj, i) or ""


def check_board(m):
    """Board geoms of the model vs the board XML (sizes / positions of every named geom): a stale mjb fails here."""
    if not os.path.exists(BOARD_XML):
        print("WARN board XML not found, skipped the board check")
        return
    xml = {g.get("name"): g for g in ET.parse(BOARD_XML).iter("geom") if g.get("name")}
    bad = []
    for k in range(m.ngeom):
        n = name(m, mujoco.mjtObj.mjOBJ_GEOM, k)
        if not n.startswith("board/"):
            continue
        g = xml.get(n[len("board/"):])
        if g is None:
            bad.append(f"{n}: not in XML")
            continue
        sz = [float(v) for v in (g.get("size") or "").split()]
        if sz and not np.allclose(m.geom_size[k][:len(sz)], sz, atol=2e-5):
            # fromto / defaults may set sizes differently: compare only explicit sizes
            bad.append(f"{n}: size {m.geom_size[k][:len(sz)]} vs XML {sz}")
    missing = set(xml) - {name(m, mujoco.mjtObj.mjOBJ_GEOM, k)[len("board/"):] for k in range(m.ngeom)}
    if missing:
        bad.append(f"XML geoms not in model: {sorted(missing)[:6]}")
    if bad:
        raise SystemExit("model.mjb does not match the current board XML (rebuild it: chain_model.py):\n  "
                         + "\n  ".join(bad[:12]))
    print(f"board check: {sum(name(m, mujoco.mjtObj.mjOBJ_GEOM, k).startswith('board/') for k in range(m.ngeom))} "
          "board geoms match skateboard_sequence.xml")


def pipe_profile(m):
    """Half-pipe of the terrain body: (x, z) polyline of the upper envelope of every box's top face (what wheels
    touch), the y range, the platforms' outer x. Boxes must be rotated about y only (the half-pipe builder)."""
    tb = next(b for b in range(m.nbody) if name(m, mujoco.mjtObj.mjOBJ_BODY, b) == "terrain")
    assert np.allclose(m.body_pos[tb], 0) and np.allclose(m.body_quat[tb], [1, 0, 0, 0]), "terrain body moved"
    segs, ys, floor = [], [], None
    for k in range(m.ngeom):
        if m.geom_bodyid[k] != tb:
            continue
        if m.geom_type[k] == mujoco.mjtGeom.mjGEOM_PLANE:
            floor = dict(z=float(m.geom_pos[k][2]), half=m.geom_size[k][:2].tolist())
            continue
        assert m.geom_type[k] == mujoco.mjtGeom.mjGEOM_BOX, name(m, mujoco.mjtObj.mjOBJ_GEOM, k)
        w, qx, qy, qz = m.geom_quat[k]
        assert abs(qx) < 1e-6 and abs(qz) < 1e-6, "pipe box rotated about x / z"
        th = 2 * math.atan2(qy, w)                      # rotation about +y: local x -> (cos, 0, -sin)
        hx, hy, hz = m.geom_size[k]
        c = m.geom_pos[k]
        ex, ez = np.array([math.cos(th), -math.sin(th)]), np.array([math.sin(th), math.cos(th)])
        top = c[[0, 2]] + hz * ez
        segs.append((top - hx * ex, top + hx * ex))
        ys.append((c[1] - hy, c[1] + hy))
    ys = np.array(ys)
    assert np.allclose(ys, ys[0], atol=1e-6), "pipe boxes with different y extents"
    xs_all = np.array([[a[0], b[0]] for a, b in segs])
    x_lo, x_hi = float(xs_all.min()), float(xs_all.max())
    # breakpoints: every segment end, refined; envelope z(x) = max over segments covering x and the floor
    def env(x):
        z = np.full_like(x, floor["z"] if floor else -1.0)
        for a, b in segs:
            lo, hi = min(a[0], b[0]), max(a[0], b[0])
            on = (x >= lo - 1e-12) & (x <= hi + 1e-12)
            if abs(b[0] - a[0]) < 1e-12:
                continue
            zz = a[1] + (x - a[0]) * (b[1] - a[1]) / (b[0] - a[0])
            z[on] = np.maximum(z[on], zz[on])
        return z
    # exact envelope vertices: segment ends + pairwise intersections of overlapping segments
    cand = set(np.round(xs_all.ravel(), 9).tolist())
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            (a1, b1), (a2, b2) = segs[i], segs[j]
            s1 = (b1[1] - a1[1]) / (b1[0] - a1[0])
            s2 = (b2[1] - a2[1]) / (b2[0] - a2[0])
            if abs(s1 - s2) < 1e-12:
                continue
            x = (a2[1] - a1[1] + s1 * a1[0] - s2 * a2[0]) / (s1 - s2)
            if max(min(a1[0], b1[0]), min(a2[0], b2[0])) - 1e-9 <= x <= min(max(a1[0], b1[0]), max(a2[0], b2[0])) + 1e-9:
                cand.add(round(float(x), 9))
    # floor crossings
    if floor:
        for a, b in segs:
            if (a[1] - floor["z"]) * (b[1] - floor["z"]) < 0:
                cand.add(round(float(a[0] + (floor["z"] - a[1]) * (b[0] - a[0]) / (b[1] - a[1])), 9))
    xs = np.array(sorted(cand))
    zs = env(xs)
    # also the envelope just inside / outside each vertex (steps, e.g. the platform's vertical inner face)
    eps = 1e-7
    P = []
    for x, z in zip(xs, zs):
        zl, zr = env(np.array([x - eps]))[0], env(np.array([x + eps]))[0]
        for zz in (zl, z, zr):
            if not P or abs(P[-1][0] - x) > 1e-9 or abs(P[-1][1] - zz) > 1e-9:
                P.append((float(x), float(zz)))
    # drop collinear points
    Q = [P[0]]
    for k in range(1, len(P) - 1):
        (x0, z0), (x1, z1), (x2, z2) = Q[-1], P[k], P[k + 1]
        if abs((x1 - x0) * (z2 - z0) - (z1 - z0) * (x2 - x0)) > 1e-12:
            Q.append(P[k])
    Q.append(P[-1])
    # verify on a dense grid
    xd = np.linspace(x_lo, x_hi, 20001)
    err = np.abs(np.interp(xd, [q[0] for q in Q], [q[1] for q in Q]) - env(xd))
    # (steps are vertical in the envelope: ignore the samples at them)
    print(f"pipe envelope: {len(Q)} vertices, x {x_lo:.4f}..{x_hi:.4f}, max |interp - envelope| "
          f"{np.percentile(err, 99.9) * 1000:.3f} mm (99.9 %)")
    return dict(profile_x=[q[0] for q in Q], profile_z=[q[1] for q in Q], y0=float(ys[0][0]), y1=float(ys[0][1]),
                x_lo=x_lo, x_hi=x_hi, h=float(max(q[1] for q in Q)), floor=floor, n_boxes=len(segs))


def export(qpos_path, model_path, out):
    z = np.load(qpos_path)
    m = mujoco.MjModel.from_binary_path(model_path)
    d = mujoco.MjData(m)
    q = z["qpos"].astype(np.float64)
    assert q.shape[1] == m.nq, (q.shape, m.nq)
    jn = [name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)]
    assert list(map(str, z["jnt_names"])) == jn, "demo_qpos joint names differ from the model"
    assert np.array_equal(z["jnt_qposadr"], m.jnt_qposadr), "demo_qpos qpos addresses differ from the model"
    assert np.allclose(z["origin"], 0), "env origin != 0: shift needed"
    check_board(m)

    bodies = [name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(m.nbody)]
    rb = [i for i, n in enumerate(bodies) if n.startswith("robot/")]
    bb = [i for i, n in enumerate(bodies) if n.startswith("board/")]
    hinge = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and jn[j].startswith("robot/")]
    jnames = [jn[j][len("robot/"):] for j in hinge]
    jadr = [int(m.jnt_qposadr[j]) for j in hinge]
    free = next(j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and jn[j].startswith("robot/"))
    fa = int(m.jnt_qposadr[free])
    N = len(q)
    rpos, rquat = np.zeros((N, len(rb), 3)), np.zeros((N, len(rb), 4))
    bpos, bquat = np.zeros((N, len(bb), 3)), np.zeros((N, len(bb), 4))
    wheel_g = [k for k in range(m.ngeom) if name(m, mujoco.mjtObj.mjOBJ_GEOM, k).endswith("wheel_L_collision")
               or name(m, mujoco.mjtObj.mjOBJ_GEOM, k).endswith("wheel_R_collision")]
    wheel_c = np.zeros((N, len(wheel_g), 3))
    # cine (2026-11-22, additive): per-step contacts from the logged qpos (mj_fwdPosition = kinematics + collision),
    # for the phase labels (MuJoCo's stage_labels: foot on the floor, both soles on the deck, wheels on the ground)
    gname = [name(m, mujoco.mjtObj.mjOBJ_GEOM, k) for k in range(m.ngeom)]
    tb_id = next(b for b in range(m.nbody) if name(m, mujoco.mjtObj.mjOBJ_BODY, b) == "terrain")
    terr = {k for k in range(m.ngeom) if m.geom_bodyid[k] == tb_id}
    feet = [gname.index(f"robot/{s}_foot_collision") if f"robot/{s}_foot_collision" in gname else -1
            for s in ("left", "right")]
    soles = [{k for k in range(m.ngeom) if gname[k].startswith(f"robot/{s}_sole_deck")} for s in ("left", "right")]
    deck = {k for k in range(m.ngeom) if gname[k].startswith("board/deck")}
    wheel_order = [gname.index(f"board/{w}_collision") if f"board/{w}_collision" in gname else -1
                   for w in ("front_wheel_L", "front_wheel_R", "rear_wheel_L", "rear_wheel_R")]
    foot_floor, sole_deck, wheel_ct = np.zeros((N, 2), bool), np.zeros((N, 2), bool), np.zeros((N, 4), bool)
    # every robot geom touching the terrain (floor / pipe), per robot body (the foot-push plant may use any geom)
    rb_index = {b: j for j, b in enumerate(rb)}
    robot_floor = np.zeros((N, len(rb)), bool)
    for i in range(N):
        d.qpos[:] = q[i]
        mujoco.mj_fwdPosition(m, d)
        rpos[i], rquat[i] = d.xpos[rb], d.xquat[rb]
        bpos[i], bquat[i] = d.xpos[bb], d.xquat[bb]
        wheel_c[i] = d.geom_xpos[wheel_g]
        for c in d.contact[:d.ncon]:
            if c.dist > 0.002:
                continue
            g1, g2 = int(c.geom1), int(c.geom2)
            for a_, b_ in ((g1, g2), (g2, g1)):
                if b_ in terr and m.geom_bodyid[a_] in rb_index:
                    robot_floor[i, rb_index[m.geom_bodyid[a_]]] = True
                if b_ in terr:
                    for s in range(2):
                        foot_floor[i, s] |= a_ == feet[s]
                    for w in range(4):
                        wheel_ct[i, w] |= a_ == wheel_order[w]
                if b_ in deck:
                    for s in range(2):
                        sole_deck[i, s] |= a_ in soles[s]
    g = [k for k in range(m.ngeom) if m.geom_bodyid[k] in bb]
    dt = float(z["dt"])
    stage = z["stage"].astype(int)
    out_d = dict(
        t=np.arange(N) * dt,
        root_pos=q[:, fa:fa + 3], root_quat_wxyz=q[:, fa + 3:fa + 7],
        joint_pos=q[:, jadr], joint_names=np.array(jnames),
        robot_body_names=np.array([bodies[i][len("robot/"):] for i in rb]), robot_body_pos=rpos, robot_body_quat=rquat,
        board_body_names=np.array([bodies[i][len("board/"):] for i in bb]), board_body_pos=bpos,
        board_body_quat_wxyz=bquat,
        board_geom_body=np.array([bodies[m.geom_bodyid[k]][len("board/"):] for k in g]),
        board_geom_name=np.array([name(m, mujoco.mjtObj.mjOBJ_GEOM, k).replace("board/", "") for k in g]),
        board_geom_type=np.array([TYPES.get(int(m.geom_type[k]), str(int(m.geom_type[k]))) for k in g]),
        board_geom_size=m.geom_size[g], board_geom_pos=m.geom_pos[g], board_geom_quat=m.geom_quat[g],
        board_geom_rgba=m.geom_rgba[g], board_geom_group=m.geom_group[g],
        wheel_center=wheel_c, wheel_radius=float(m.geom_size[wheel_g[0]][0]),
        stage=stage, owner=z["owner"].astype(int), apex=z["apex"].astype(bool),
        take=os.path.basename(os.path.dirname(qpos_path)),
        foot_floor=foot_floor, sole_deck=sole_deck, wheel_contact=wheel_ct,      # cine: contacts (additive keys)
        robot_floor=robot_floor,                                                # (N, robot bodies): body on terrain
    )
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez_compressed(out + ".kin.npz", **out_d)

    pipe = pipe_profile(m)
    # wheel contact check: every wheel's lowest point vs the pipe surface under it (cylinder axis is horizontal-ish;
    # approximate the contact by the wheel centre height minus the radius on the local slope)
    px, pz = np.array(pipe["profile_x"]), np.array(pipe["profile_z"])
    r = out_d["wheel_radius"]
    wc = wheel_c.reshape(-1, 3)
    inside = (wc[:, 1] > pipe["y0"]) & (wc[:, 1] < pipe["y1"])
    # distance in the xz plane from the wheel centre to the surface polyline (floor z=0 outside the pipe), minus r
    A = np.stack([px[:-1], pz[:-1]], 1)
    B = np.stack([px[1:], pz[1:]], 1)
    gap = np.empty(len(wc))
    for i, (x, y, zc) in enumerate(wc):
        if not inside[i] or not (px[0] < x < px[-1]):
            gap[i] = zc - r - (pipe["floor"]["z"] if pipe["floor"] else 0.0)
            continue
        p = np.array([x, zc])
        ab = B - A
        tt = np.clip(np.einsum("ij,ij->i", p - A, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-18), 0, 1)
        gap[i] = np.min(np.linalg.norm(A + tt[:, None] * ab - p, axis=1)) - r
    pipe["wheel_gap_mm"] = dict(min=float(gap.min() * 1000), p5=float(np.percentile(gap, 5) * 1000),
                                median=float(np.median(gap) * 1000))
    json.dump(pipe, open(os.path.join(os.path.dirname(os.path.abspath(out)), "pipe.json"), "w"), indent=1)

    # meta: end-card numbers from the chain run, stage times, apexes
    run = os.path.dirname(os.path.abspath(qpos_path))
    picks = json.load(open(os.path.join(run, "picks.json"))) if os.path.exists(os.path.join(run, "picks.json")) else {}
    funnel = json.load(open(os.path.join(run, "funnel.json"))) if os.path.exists(os.path.join(run, "funnel.json")) else {}
    # pipe_h (2026-11-21): chain runs record their true pipe (MICRODUCK_PIPE_H) in picks.json; the model must match it
    # (the fallback model.mjb is chain_v1's 12 cm pipe; chain_play.py now saves model.mjb next to demo_qpos.npz)
    pp = picks.get("pipe")
    # (the envelope max sits ~2-3 mm above the nominal h: the top facet box's overlap corner; 12 vs 8 cm differ by 40)
    if pp and abs(pipe["h"] - float(pp["h"])) > 0.006:
        raise SystemExit(f"model.mjb pipe h {pipe['h']:.3f} != the run's pipe h {float(pp['h']):.3f} (picks.json): "
                         "use the run's own model.mjb (chain_play.py / chain_model.py with MICRODUCK_PIPE_H set)")
    src = open(os.path.join(SEQ, "chain_play.py"), encoding="utf-8").read()
    const = lambda k, dflt: float(re.search(rf"^{k}\s*=\s*([0-9.]+)", src, re.M).group(1)) if re.search(
        rf"^{k}\s*=\s*([0-9.]+)", src, re.M) else dflt
    di = picks.get("picks", {}).get("demo")
    ep = funnel.get("episodes", [None] * 999)[di] if di is not None and funnel else None
    agg = funnel.get("agg", {})
    apex_i = np.nonzero(out_d["apex"])[0]
    # rise of the deck above its flat height at each apex (the chain's own pass measure is the deck rise)
    deck_z = bpos[:, 0, 2]
    flat = float(np.median(deck_z[stage == 0]))
    rises = [float((deck_z[max(0, i - 10):i + 1].max() - flat) * 100) for i in apex_i]
    meta = dict(
        run=run, n=picks.get("n"), demo_index=di, seed=picks.get("seed"),
        demo_score=picks.get("demo_score", {}).get("demo"),
        chain_complete=picks.get("chain_complete", agg.get("chain_complete")),
        rolling=picks.get("rolling"), rolling_trained=picks.get("rolling_trained"),
        pass_min_cm=picks.get("pass_min_cm", const("PASS_MIN_CM", None)),
        complete_passes=int(const("COMPLETE_PASSES", 2)),
        episode=({k: v for k, v in ep.items() if k != "handoffs"} if ep else None),
        dt=dt, steps=N, seconds=N * dt,
        stage_start={STAGES[s]: float(np.argmax(stage == s) * dt) for s in range(5) if (stage == s).any()},
        apex_t=[float(i * dt) for i in apex_i], apex_rise_cm=rises,
        checkpoints=picks.get("checkpoints"),
        pipe_h=float((picks.get("pipe") or {}).get("h", 0.12)), pipe_envelope_max_z=pipe["h"], pipe=picks.get("pipe"), obs_pipe=picks.get("obs_pipe"),   # pipe_h: the rendered pipe height
    )
    json.dump(meta, open(os.path.join(os.path.dirname(os.path.abspath(out)), "meta.json"), "w"), indent=1)
    print(f"{out}.kin.npz: {N} steps ({N * dt:.2f} s), robot bodies {len(rb)}, joints {len(jnames)}, "
          f"board bodies {len(bb)}, board geoms {len(g)}")
    print(f"pipe: y {pipe['y0']:.2f}..{pipe['y1']:.2f}, x {pipe['x_lo']:.3f}..{pipe['x_hi']:.3f}, h {pipe['h']:.3f}; "
          f"wheel gap to the surface (mm): {pipe['wheel_gap_mm']}")
    print("stages:", meta["stage_start"], " apexes:", [f"{t:.2f}s {r:.1f}cm" for t, r in zip(meta["apex_t"], rises)])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("qpos")
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default=os.path.join(HERE, "chain", "chain"))
    a = ap.parse_args()
    mp = a.model or os.path.join(os.path.dirname(os.path.abspath(a.qpos)), "model.mjb")
    if not os.path.exists(mp):
        print(f"no model.mjb next to {a.qpos}: using {FALLBACK_MODEL} (checked below)")
        mp = FALLBACK_MODEL
    export(a.qpos, mp, a.out)
