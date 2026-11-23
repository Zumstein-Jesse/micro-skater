# micro-skater tools/isaac_render: the author's Isaac Sim replay pipeline. Apache-2.0 (see LICENSE).
"""Native 9x16 / 4x5 cuts of the chain run (push -> manual -> half-pipe -> foot-push) for the Isaac Sim replay.

    python edit_cine.py --cut standard|cinematic --format 9x16|4x5 [--dir chain_cine] [--out timeline_cine_<cut>_<fmt>.npz]

Reads <dir>/chain.kin.npz (export_chain.py, incl. the contact logs), <dir>/meta.json and <dir>/pipe_render.json (the
trimmed rendered pipe, pipe_trim.py; falls back to pipe.json). Writes the timeline format (bake_replay.py /
check_chain.py read it unchanged) plus, for compose_cine.py:
    speed       per output frame: source seconds per output second (1 = real time; < 1 = slow motion, badged)
    phase       per output frame: index into phase_names (-1 on the held title / end frames)
    card_lines / title_sub / pipe / cut
Poses are never edited: every output frame is the MuJoCo state at its source time (lerp between the 50 Hz steps).

STANDARD   real time throughout (1x), cuts only change the camera; the phase label bottom-left.
CINEMATIC  the whole run in order (no source time skipped or repeated), close-ups on the board and the head, speed ramps
           into slow motion at the manual lift, the set-down, the best wall ride and the foot-push (each slowed frame
           carries a slow-motion badge in compose_cine.py), the held first frame under the title, the end card.

Phase labels (the MuJoCo clip's stage_labels rules, sequence_eval.py, rebuilt from stage + owner + contacts):
Push / Step on (from the last floor contact, >= 0.4 s or until both soles are on the deck) / Ride / Manual (front wheels
up >= 0.1 s .. all four wheels down) / Set down (0.5 s) / Carve in / Half-pipe / Foot push (owner 3).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

import cine_geom as edit  # helper functions (the old shot-list script is not shipped)

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--cut", choices=["standard", "cinematic"], required=True)
ap.add_argument("--format", choices=["9x16", "4x5"], required=True)
ap.add_argument("--dir", default=os.path.join(HERE, "chain_cine"))
ap.add_argument("--out", default=None)
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--intro-s", type=float, default=2.0)
ap.add_argument("--hold-s", type=float, default=6.5)
# bloopers (2026-11-22, additive, off by default): make_bloopers.py bloopers.json = failed runs of the same chain eval,
# each played up to the step the eval ended it, then frozen on that frame; appended before the end card
ap.add_argument("--bloopers", default=None)
ap.add_argument("--blooper-s", type=float, default=2.6)
ap.add_argument("--freeze-s", type=float, default=1.1)
a = ap.parse_args()
a.out = a.out or os.path.join(HERE, f"timeline_cine_{a.cut}_{a.format}.npz")
V = a.format == "9x16"
APERTURE = {"9x16": (20.25, 36.0), "4x5": (28.8, 36.0)}[a.format]      # both 36 mm tall: same vertical framing

KIN = {k: v for k, v in np.load(os.path.join(a.dir, "chain.kin.npz")).items()}
META = json.load(open(os.path.join(a.dir, "meta.json")))
_pr = os.path.join(a.dir, "pipe_render.json")
PIPE = json.load(open(_pr if os.path.exists(_pr) else os.path.join(a.dir, "pipe.json")))
PX, PZ = np.array(PIPE["profile_x"]), np.array(PIPE["profile_z"])
dt = float(KIN["t"][1] - KIN["t"][0])
T_END = float(KIN["t"][-1])
NSTEP = len(KIN["t"])


def ground_z(x, y):
    """The RENDERED park surface (trimmed pipe): camera clamp + line-of-sight checks."""
    x, y = np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(y, float))
    z = np.zeros_like(x)
    on = (x >= PIPE["x_lo"]) & (x <= PIPE["x_hi"]) & (y >= PIPE["y0"]) & (y <= PIPE["y1"])
    z[on] = np.interp(x[on], PX, PZ)
    return z


edit.ground_z = ground_z


# ── phase labels ─────────────────────────────────────────────────────────────
NAMES = ["Push", "Step on", "Ride", "Manual", "Set down", "Carve in", "Half-pipe", "Foot push"]


def phase_labels():
    st = KIN["stage"].astype(int)
    T = len(st)
    ff = KIN["foot_floor"].any(1) if "foot_floor" in KIN else np.zeros(T, bool)
    both = KIN["sole_deck"].all(1) if "sole_deck" in KIN else np.ones(T, bool)
    wh = KIN["wheel_contact"] if "wheel_contact" in KIN else np.ones((T, 4), bool)
    fdn, all4 = wh[:, :2].any(1), wh.all(1)
    hit = np.nonzero(ff & (st <= 1))[0]
    k_last = int(hit[-1]) if len(hit) else -1
    n_on = int(round(0.4 / dt))
    k2 = next((k for k in range(T) if st[k] >= 2), None)
    nl = max(1, int(round(0.1 / dt)))
    k_up = next((k for k in range(k2, T) if st[k] == 2 and not fdn[k:k + nl].any()), k2) if k2 is not None else None
    k3 = next((k for k in range(T) if st[k] >= 3), None)
    k_dn = next((k for k in range(k3, T) if st[k] >= 4 or all4[k]), None) if k3 is not None else None
    n_sd = int(round(0.5 / dt))
    out = []
    for k in range(T):
        s = st[k]
        if s <= 1:
            if k <= k_last:
                lab = "Push" if s == 0 else "Ride"
            elif k - k_last <= n_on or not both[k]:
                lab = "Step on"
            else:
                lab = "Ride"
        elif s == 2:
            lab = "Ride" if (k_up is not None and k < k_up) else "Manual"
        elif s == 3:
            if k_dn is None or k < k_dn:
                lab = "Manual"
            elif k < k_dn + n_sd:
                lab = "Set down"
            else:
                lab = "Carve in"
        else:
            lab = "Half-pipe"
        out.append(lab)
    n_min = max(1, int(round(0.2 / dt)))          # merge flicker (< 0.2 s) into the previous label, as MuJoCo does
    k = 0
    while k < T:
        j = k
        while j < T and out[j] == out[k]:
            j += 1
        if j - k < n_min and k > 0 and out[k] != "Half-pipe":
            out[k:j] = [out[k - 1]] * (j - k)
        k = j
    # Foot push (owner 3 = the foot-push policy): labelled only around a real floor contact of the robot inside the
    # owner-3 window (export_chain robot_floor: any robot geom on the terrain), from 0.5 s before the plant to 0.5 s
    # after it (>= 1.2 s on screen), clipped to the window; an owner-3 window without a contact keeps its label
    own = KIN["owner"].astype(int) if "owner" in KIN else np.zeros(T, int)
    rf = KIN["robot_floor"] if "robot_floor" in KIN else np.zeros((T, 1), bool)
    k = 0
    while k < T:
        if own[k] != 3:
            k += 1
            continue
        j = k
        while j + 1 < T and own[j + 1] == 3:
            j += 1
        hit = np.nonzero(rf[k:j + 1].any(1))[0]
        if len(hit):
            c0, c1 = k + int(hit[0]), k + int(hit[-1])
            pad = max(int(round(0.5 / dt)), int(round((1.2 / dt - (c1 - c0 + 1)) / 2)))
            body = int(np.argmax(rf[c0:c1 + 1].sum(0)))
            PLANTS.append(dict(t0=c0 * dt, t1=(c1 + 1) * dt, body=body))
            for q in range(max(k, c0 - pad), min(j, c1 + pad) + 1):
                out[q] = "Foot push"
        k = j + 1
    return np.array([NAMES.index(x) for x in out])


PLANTS = []          # foot-push floor contacts: source times + the robot body that touched
LAB = phase_labels()
print('foot plants:', PLANTS)
if PLANTS:
    pass


def runs(lab):
    r, k = [], 0
    while k < NSTEP:
        if LAB[k] == lab:
            j = k
            while j + 1 < NSTEP and LAB[j + 1] == lab:
                j += 1
            r.append((k * dt, (j + 1) * dt))
            k = j + 1
        else:
            k += 1
    return r


RUNS = {n: runs(i) for i, n in enumerate(NAMES)}
print("phase runs:", {n: [(round(x, 2), round(y, 2)) for x, y in r] for n, r in RUNS.items() if r})


def t_start(name, i=0, dflt=None):
    r = RUNS.get(name) or []
    return r[i][0] if len(r) > i else dflt


def t_stop(name, i=-1, dflt=None):
    r = RUNS.get(name) or []
    return r[i][1] if r else dflt


# ── subjects / headings ──────────────────────────────────────────────────────
RB, BB = KIN["robot_body_pos"], KIN["board_body_pos"]
rn, bn = [str(n) for n in KIN["robot_body_names"]], [str(n) for n in KIN["board_body_names"]]
RB_NAMES = rn
SUBJ = {
    "trunk": KIN["root_pos"],
    "head": RB[:, rn.index("jaw_soft")],
    "board": BB[:, 0],
    "front_truck": BB[:, bn.index("front_truck")],
    "rear_truck": BB[:, bn.index("rear_truck")],
    "feet": 0.5 * (RB[:, rn.index("ankle_left")] + RB[:, rn.index("ankle_right")]),
    "mix": 0.55 * KIN["root_pos"] + 0.45 * BB[:, 0],
}
SUBJ["plant"] = RB[:, PLANTS[0]["body"]] if PLANTS else SUBJ["feet"]          # the foot that plants on the floor
SUBJ["plant_mix"] = 0.5 * SUBJ["plant"] + 0.5 * BB[:, 0]
_q = KIN["board_body_quat_wxyz"][:, 0]
BYAW = np.unwrap(np.arctan2(2 * (_q[:, 0] * _q[:, 3] + _q[:, 1] * _q[:, 2]), 1 - 2 * (_q[:, 2] ** 2 + _q[:, 3] ** 2)))


def sm(x, sig_s):
    return edit.smooth(x, sig_s / dt) if sig_s > 0 else x.copy()


def sample(arr, f):
    return edit.sample(arr, f)


def ease(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


# ── time maps ────────────────────────────────────────────────────────────────
def time_map(sh):
    """Source times of the shot's output frames and the per-frame playback speed. hold: one source time held for
    `dur` s. slow=dict(t=(w0, w1), k, ease): real time, ramping (smoothstep over `ease` s of source time) down to k x
    over w0..w1 and back up. speed: constant."""
    fps = a.fps
    if sh.get("hold"):
        n = int(round(sh["dur"] * fps))
        return np.full(n, sh["t"][0]), np.zeros(n)
    t0, t1 = sh["t"]
    sl = sh.get("slow") or None
    sp0 = sh.get("speed", 1.0)
    if not sl:
        n = int(round((t1 - t0) / sp0 * fps))
        return t0 + np.arange(n) * sp0 / fps, np.full(n, sp0)
    w0, w1 = sl["t"]
    k, e = sl["k"], sl.get("ease", 0.35)

    def speed(s):
        if s < w0 - e or s > w1 + e:
            return 1.0
        if s < w0:
            u = float(ease((s - (w0 - e)) / e))
        elif s > w1:
            u = float(ease(((w1 + e) - s) / e))
        else:
            u = 1.0
        return 1.0 - (1.0 - k) * u
    ts, sp = [], []
    s = t0
    while s < t1 - 1e-9:
        v = speed(s)
        ts.append(s)
        sp.append(v)
        s += v / fps
    return np.array(ts), np.array(sp)


_time_map = time_map


def time_map(sh):
    """bloopers: the clip, then its last source time frozen for `freeze_s` (speed 0)"""
    ts, sp = _time_map(sh)
    fz = sh.get("freeze_s")
    if fz:
        n = int(round(fz * a.fps))
        ts, sp = np.r_[ts, np.full(n, sh["t"][1])], np.r_[sp, np.zeros(n)]
    return ts, sp


# ── cameras ──────────────────────────────────────────────────────────────────
def camera(cam, f, n_out, shot_t0):
    """eye, aim (n_out, 3) and focal (n_out,) for the step indices f. Kinds:
    follow  eye = smoothed subject + R(yaw) @ off (off -> off1 eased over the shot: a dolly); yaw = board heading
    fixed   eye fixed (world); aim follows the smoothed subject (per-axis weights `track`, rest from `target`)
    anchor  eye = subject at source time `at` + R(yaw at `at`) @ off, then fixed; aim follows the subject
    orbit   around the subject at radius r / height h, azimuth az0 + rate * t (rad, board frame)
    focal: `focal`, or `fill` = the rider's fraction of the frame height at the start distance (36 mm tall)."""
    subj_full = SUBJ[cam.get("subject", "trunk")]
    subj = sample(sm(subj_full, cam.get("smooth", 0.4)), f)
    yaw = sample(sm(BYAW, cam.get("yaw_smooth", 0.6)), f) + cam.get("az", 0.0)
    u = ease(np.arange(n_out) / max(1, n_out - 1)) if n_out > 1 else np.zeros(1)
    if "u_time" in cam:          # dolly over a sub-range of the shot (fractions)
        q0, q1 = cam["u_time"]
        u = ease((np.arange(n_out) / max(1, n_out - 1) - q0) / max(1e-6, q1 - q0))
    lead = cam.get("lead", 0.0)
    aim_s = sm(SUBJ[cam.get("aim_subject", cam.get("subject", "trunk"))], cam.get("aim_smooth", cam.get("smooth", 0.4)))
    aim = sample(aim_s, f)
    if lead:
        aim = 0.6 * aim + 0.4 * sample(aim_s, np.clip(f + lead / dt, 0, NSTEP - 1))
    aim = aim + np.array([0, 0, cam.get("look_z", 0.0)])
    if "aim_off" in cam:         # aim offset in the board frame (along, left, up)
        ox, oy, oz = cam["aim_off"]
        c, s = np.cos(yaw), np.sin(yaw)
        aim = aim + np.stack([c * ox - s * oy, s * ox + c * oy, np.full_like(c, oz)], 1)
    kind = cam["kind"]
    if kind == "follow":
        o0 = np.array(cam["off"], float)
        o1 = np.array(cam.get("off1", cam["off"]), float)
        off = o0[None] + (o1 - o0)[None] * u[:, None]
        az1 = cam.get("az1")
        if az1 is not None:
            yaw = yaw + (az1 - cam.get("az", 0.0)) * u
        c, s = np.cos(yaw), np.sin(yaw)
        base = sample(sm(subj_full, cam.get("eye_smooth", cam.get("smooth", 0.4))), f)
        eye = base + np.stack([c * off[:, 0] - s * off[:, 1], s * off[:, 0] + c * off[:, 1], off[:, 2]], 1)
    elif kind == "fixed":
        e0 = np.array(cam["eye"], float)
        e1 = np.array(cam.get("eye1", cam["eye"]), float)
        eye = e0[None] + (e1 - e0)[None] * u[:, None]
    elif kind == "anchor":
        fa = np.array([cam["at"] / dt])
        b0 = sample(sm(subj_full, cam.get("smooth", 0.4)), fa)[0]
        y0 = float(sample(sm(BYAW, 0.6), fa)[0]) + cam.get("az", 0.0)
        ox, oy, oz = cam["off"]
        e0 = b0 + np.array([math.cos(y0) * ox - math.sin(y0) * oy, math.sin(y0) * ox + math.cos(y0) * oy, oz])
        if "eye_abs_z" in cam:
            e0[2] = cam["eye_abs_z"]
        o1 = cam.get("off1")
        if o1 is not None:
            ox, oy, oz = o1
            e1 = b0 + np.array([math.cos(y0) * ox - math.sin(y0) * oy, math.sin(y0) * ox + math.cos(y0) * oy, oz])
            eye = e0[None] + (e1 - e0)[None] * u[:, None]
        else:
            eye = np.repeat(e0[None], n_out, 0)
    elif kind == "orbit":
        tt = np.arange(n_out) / a.fps
        az = yaw + cam["az0"] + cam["rate"] * tt
        r = cam["r"] + (cam.get("r1", cam["r"]) - cam["r"]) * u
        hh = cam["h"] + (cam.get("h1", cam["h"]) - cam["h"]) * u
        eye = subj + np.stack([r * np.cos(az), r * np.sin(az), hh], 1)
    else:
        raise ValueError(kind)
    if "target" in cam:          # fixed aim point, optionally tracking the subject on some axes
        tgt = np.array(cam["target"], float)
        w = np.array(cam.get("track", (0.0, 0.0, 0.0)), float)
        aim = tgt[None] * (1 - w)[None] + aim * w[None]
    eye[:, 2] = np.maximum(eye[:, 2], ground_z(eye[:, 0], eye[:, 1]) + cam.get("min_h", 0.02))
    if "shift" in cam:           # reframe: move the subject by (dx, dy) in NDC (right / up) by re-aiming
        aim = shift_aim(eye, aim, cam, n_out)
    if "fill" in cam:
        d0 = float(np.linalg.norm(eye[0] - aim[0]))
        f0 = cam["fill"] * 36.0 * d0 / 0.27
        f1 = f0 * cam.get("zoom", 1.0)
        focal = f0 + (f1 - f0) * u
    else:
        focal = np.full(n_out, float(cam.get("focal", 35.0))) * (1 + (cam.get("zoom", 1.0) - 1) * u)
    return eye, aim, focal


# ── shot lists ───────────────────────────────────────────────────────────────
S = META["stage_start"]
apex_t, rises = META.get("apex_t", []), META.get("apex_rise_cm", [])
best = int(np.argmax(rises)) if rises else None
t_step = t_start("Step on", 0, S.get("SETTLE", 3.0) - 0.4)
t_ride = t_start("Ride", 0, t_step + 0.6)
t_man = t_start("Manual", 0, S.get("MANUAL", t_ride + 1.0))
t_sd = t_start("Set down", 0, S.get("EXIT", t_man + 2.5))
t_carve = t_start("Carve in", 0, t_sd + 0.5)
t_pipe = t_start("Half-pipe", 0, S.get("RAMP", T_END - 6))
fp = RUNS.get("Foot push") or []
board = BB[:, 0]
inpipe = KIN["stage"] == 4
yR = float(np.median(board[inpipe, 1])) if inpipe.any() else PIPE["y0"] + 0.7
wall_h = float(np.max(PZ[np.abs(PX) > PIPE["x_hi"] - 0.1]))
x_lip = float(min(abs(x) for x, zz in zip(PX, PZ) if zz >= wall_h - 1e-4 and abs(x) > 0.3))
print(f"cuts: step {t_step:.2f} ride {t_ride:.2f} manual {t_man:.2f} set-down {t_sd:.2f} carve {t_carve:.2f} "
      f"pipe {t_pipe:.2f} foot-push {[(round(x, 2), round(y, 2)) for x, y in fp]} end {T_END:.2f}; yR {yR:.2f}, "
      f"lip |x| {x_lip:.3f}, wall {wall_h * 100:.1f} cm")


def plant_side(pl):
    k = int(round(0.5 * (pl["t0"] + pl["t1"]) / dt))
    d = RB[k, pl["body"]] - BB[k, 0]
    return 1.0 if (-math.sin(BYAW[k]) * d[0] + math.cos(BYAW[k]) * d[1]) >= 0 else -1.0


def plant_for(g0, g1):
    return next((p for p in PLANTS if g0 - 1e-6 <= p["t0"] <= g1 + 1e-6), None)


def F(d9, d45):
    """per-format value"""
    return d9 if V else d45


# camera kits (board frame offsets: along the board heading, to its left, up; metres)
REAR = dict(kind="follow", subject="trunk", off=(-0.80, 0.42, 0.15), fill=F(0.40, 0.36), fstop=4.0, smooth=0.5,
            look_z=0.03, lead=0.3)
SIDE = dict(kind="follow", subject="mix", off=(0.04, 0.80, 0.07), fill=F(0.46, 0.42), fstop=4.0, smooth=0.6,
            look_z=0.05)
LOWSIDE = dict(kind="follow", subject="mix", off=(0.30, 0.66, 0.035), fill=F(0.50, 0.45), fstop=4.0, smooth=0.6,
               look_z=0.06)
HERO = dict(kind="follow", subject="trunk", off=(0.95, -0.55, 0.10), off1=(0.80, -0.62, 0.10), fill=F(0.36, 0.32),
            fstop=4.0, smooth=0.7, look_z=0.03)
# the pipe: on the axis outside the open end, panning with the rider across the pipe (x tracked, smoothed)
y_open = PIPE["y0"] if PIPE.get("trim", {}).get("entry_end", "y0").startswith("y0") else PIPE["y1"]
sgn = 1.0 if y_open == PIPE["y0"] else -1.0
PIPE_AXIS = dict(kind="fixed", subject="board", eye=(0.0, yR - sgn * F(1.7, 2.0), F(0.17, 0.22)),
                 target=(0.0, yR + sgn * 0.05, F(0.15, 0.09)), track=(0.96, 0.6, 0.0), smooth=0.3,
                 focal=F(74.0, 70.0), fstop=8.0)
# elevated three-quarter of the whole short pipe (the outside of the turn), rider tracked loosely
PIPE_34 = dict(kind="fixed", subject="board", eye=(-1.9, yR - sgn * 1.5, 0.60),
               target=(0.2, yR + sgn * 0.1, 0.07), track=(0.95, 0.85, 0.0), smooth=0.4,
               focal=F(82.0, 82.0), fstop=8.0)


# 9x16 standard: a closer, higher three-quarter over the short pipe so the transitions read as a mini-ramp
PIPE_HI = dict(kind="fixed", subject="board", eye=(-1.05, yR - sgn * 0.95, 0.55),
               target=(0.1, yR + sgn * 0.05, 0.07), track=(0.92, 0.8, 0.0), smooth=0.4, focal=46.0, fstop=8.0)


def carve_wide(t0):
    """behind and above the approach, panning with the rider toward the pipe (the opening ahead in frame)."""
    return dict(kind="anchor", subject="trunk", at=t0, off=(-1.0, 0.32, 0.45), fill=F(0.36, 0.34), zoom=1.6,
                fstop=8.0, smooth=0.6, look_z=0.0, lead=0.5)


def pipe_section(tp, events, fill_cams, min_s=0.5):
    """The half-pipe from tp to the end: the events (dict name / t / cam [/ slow]) in time order, clipped so they never
    overlap or leave the clip, with filler shots between them cycling through fill_cams (a filler > 3 s is split in
    two). A gap shorter than min_s is absorbed by the next event (or the previous one at the end)."""
    out, k = [], 0
    for e in sorted(events, key=lambda e: e["t"][0]):
        e0, e1 = max(e["t"][0], tp), min(e["t"][1], T_END)
        if e1 - e0 < 0.4:
            continue
        if e0 - tp >= min_s:
            g = e0 - tp
            parts = [(tp, tp + g / 2), (tp + g / 2, e0)] if g > 3.0 else [(tp, e0)]
            for p0, p1 in parts:
                out.append(dict(name=f"pipe_{k}", t=(p0, p1), cam=fill_cams[k % len(fill_cams)]))
                k += 1
        else:
            e0 = tp
        if e.get("slow"):
            w0, w1 = e["slow"]["t"]
            e = dict(e, slow=dict(e["slow"], t=(max(w0, e0), min(w1, e1))))
        out.append(dict(e, t=(e0, e1)))
        tp = e1
    if T_END - tp >= min_s or not out:
        out.append(dict(name=f"pipe_{k}", t=(tp, T_END), cam=fill_cams[k % len(fill_cams)]))
    elif T_END - tp > 1e-6:
        out[-1] = dict(out[-1], t=(out[-1]["t"][0], T_END))
    return out


def shift_aim(eye, aim, cam, n_out):
    """move the aim point so the subject sits (dx, dy) NDC right / up of the centre (overlay-safe framing); uses the
    shot's focal (fill shots: the focal is computed from the distance, which re-aiming barely changes)"""
    dx, dy = cam["shift"]
    fwd = aim - eye
    dist = np.linalg.norm(fwd, axis=1, keepdims=True)
    fwd = fwd / dist
    right = np.cross(fwd, np.array([0, 0, 1.0]))
    right /= np.linalg.norm(right, axis=1, keepdims=True)
    up = np.cross(right, fwd)
    if "fill" in cam:
        foc = cam["fill"] * 36.0 * float(dist[0, 0]) / 0.27
    else:
        foc = float(cam.get("focal", 35.0))
    th, tv = APERTURE[0] / 2 / foc, APERTURE[1] / 2 / foc
    return aim - right * (dx * th * dist) - up * (dy * tv * dist)


def fp_standard_cam(pl, sd):
    """standard-cut foot push: a camera planted on the push side at the contact (higher, wider than the rider's
    eye line), panning loosely so the board visibly rolls off after the foot touches down; the whole robot in frame"""
    if pl is None:
        return dict(kind="follow", subject="board", off=(0.06, sd * 0.90, 0.22), fill=F(0.34, 0.34), fstop=5.6,
                    smooth=0.4, look_z=0.08, aim_subject="mix", yaw_smooth=1.0)
    tc = 0.5 * (pl["t0"] + pl["t1"])
    return dict(kind="anchor", subject="board", at=tc, off=(0.10, sd * 0.95, 0.30), fill=F(0.33, 0.36), fstop=5.6,
                smooth=0.5, look_z=F(0.12, 0.10), aim_subject="mix", lead=0.0)


def shots_standard():
    sh = [dict(name="intro", t=(0.0, 0.0), hold=True, dur=a.intro_s, card="title",
               cam=dict(REAR, off=(-0.95, 0.62, 0.24), off1=(-0.74, 0.50, 0.18), look_z=F(0.16, 0.12),
                        fill=F(0.30, 0.28), smooth=0.5)),
          dict(name="push", t=(0.0, t_step - 0.25), cam=REAR),
          dict(name="step_on", t=(t_step - 0.25, t_man - 0.1), cam=SIDE),
          dict(name="manual", t=(t_man - 0.1, t_sd + 0.7), cam=LOWSIDE)]
    tc = t_sd + 0.7
    t_mid = max(tc + 1.5, t_pipe - 3.6)
    # the long flat roll to the pipe, real time but moving: a camera swinging round the rider from ahead-right to its
    # side, a board-level tracking shot, then the wide behind-and-above shot with the pipe opening ahead
    t_w = max(tc + 3.0, t_pipe - 3.0)
    t_s = tc + 0.5 * (t_w - tc)
    sh.append(dict(name="carve_swing", t=(tc, t_s), cam=dict(HERO, off=(0.95, -0.55, 0.10), off1=(0.80, -0.50, 0.14),
                                                             az=0.0, az1=-1.35, fill=F(0.38, 0.34))))
    sh.append(dict(name="carve_track", t=(t_s, t_w), cam=dict(kind="follow", subject="mix", off=(0.20, 0.62, 0.02),
                                                              off1=(-0.05, 0.60, 0.03), fill=F(0.46, 0.42), fstop=4.0,
                                                              smooth=0.6, look_z=0.06)))
    sh.append(dict(name="carve_wide", t=(t_w, t_pipe + 0.3), cam=carve_wide(t_w)))
    pc = PIPE_HI if V else PIPE_34
    # foot-push: a tighter tracking camera on the side of the planting foot, low, the whole robot and the ground beside
    # the board in frame (the plant is ~80 ms: it must not happen behind the board or at the frame edge)
    ev = []
    for i, (g0, g1) in enumerate(fp):
        pl = plant_for(g0, g1)
        sd = plant_side(pl) if pl else -1.0
        ev.append(dict(name=f"footpush_{i}", t=(g0 - 0.3, g1 + 0.3),
                       cam=fp_standard_cam(pl, sd)))
    sh += pipe_section(t_pipe + 0.3, ev, [pc])
    sh.append(dict(name="end", t=(T_END, T_END), hold=True, dur=a.hold_s, card="end", cam=pc))
    return sh


def shots_cinematic():
    sh = []
    # cold, in order: the start pose held (time frozen) under the title while the camera drifts round at deck height
    sh.append(dict(name="open", t=(0.0, 0.0), hold=True, dur=a.intro_s + 0.4, card="title",
                   cam=dict(kind="orbit", subject="trunk", r=0.62, r1=0.50, h=0.10, h1=0.07, az0=math.radians(-150),
                            rate=math.radians(16), fill=F(0.34, 0.30), fstop=2.8, look_z=F(0.13, 0.12), smooth=0.3)))
    t1 = min(1.4, t_step - 1.2)
    sh.append(dict(name="push_low", t=(0.0, t1), cam=dict(REAR, off=(-0.66, 0.32, 0.05), fill=F(0.42, 0.42),
                                                          fstop=2.8, look_z=0.06)))
    # wheels: the rear truck and wheels at 6 cm, tracking beside the board while it pushes
    sh.append(dict(name="push_wheels", t=(t1, t_step - 0.3),
                   cam=dict(kind="follow", subject="rear_truck", off=(-0.13, 0.17, 0.016), off1=(0.02, 0.19, 0.02),
                            focal=F(36.0, 31.0), fstop=5.6, smooth=0.25, look_z=0.012, aim_off=(0.05, 0.0, 0.0))))
    # step on: deck + feet in profile, ramped to 0.5x around the push foot landing on the deck
    t_so = min(t_ride + 0.25, t_man - 0.9)
    sh.append(dict(name="step_on", t=(t_step - 0.3, t_so),
                   slow=dict(t=(t_step - 0.05, t_step + 0.5), k=0.5, ease=0.25),
                   cam=dict(kind="follow", subject="feet", off=(0.06, 0.42, 0.06), fill=F(0.46, 0.55), fstop=4.0,
                            smooth=0.4, look_z=0.03, aim_subject="mix")))
    # head: the face, from ahead and a little below, rolling toward camera (the ride before the manual)
    sh.append(dict(name="head", t=(t_so, t_man - 0.3),
                   cam=dict(kind="follow", subject="head", off=(0.50, 0.17, -0.03), focal=F(30.0, 36.0), fstop=2.8,
                            smooth=0.35, look_z=-0.035)))
    tm0 = sh[-1]["t"][1]
    # manual: deck level, rear wheels and the tail; slow ramp (0.3x) through the nose lift
    tm1 = min(t_sd - 0.4, tm0 + 2.2)
    if t_sd - 0.4 - tm1 < 0.9:          # no sliver of a wide shot: the low shot runs to the set-down
        tm1 = t_sd - 0.4
    sh.append(dict(name="manual_low", t=(tm0, tm1), slow=dict(t=(t_man - 0.05, t_man + 0.5), k=0.3, ease=0.3),
                   cam=dict(kind="follow", subject="board", off=(0.0, 0.40, 0.012), off1=(0.10, 0.43, 0.02),
                            fill=F(0.56, 0.50), fstop=4.0, smooth=0.4, look_z=0.05, aim_subject="mix")))
    if t_sd - 0.4 > tm1 + 0.05:
        sh.append(dict(name="manual_wide", t=(tm1, t_sd - 0.4), cam=dict(LOWSIDE, off=(0.55, 0.85, 0.06))))
    # set-down: the front truck touching back down, 0.4x
    sh.append(dict(name="set_down", t=(t_sd - 0.4, t_sd + 0.9), slow=dict(t=(t_sd - 0.15, t_sd + 0.25), k=0.4,
                                                                            ease=0.2),
                   cam=dict(kind="follow", subject="front_truck", off=(0.16, 0.16, 0.02), focal=F(32.0, 28.0),
                            fstop=5.6, smooth=0.3, look_z=0.01, aim_off=(-0.03, 0.0, 0.0))))
    tc = t_sd + 0.9
    t_mid = max(tc + 2.5, t_pipe - 2.0)
    t_q = min(tc + 1.2, t_mid - 1.0)
    sh.append(dict(name="carve_hero", t=(tc, t_q), cam=dict(HERO, off=(1.1, -0.45, 0.06), off1=(0.85, -0.6, 0.12),
                                                           fstop=2.8)))
    # worm's-eye: the lens rolling just ahead of the nose at deck height, the rider towering over the board; the long
    # flat roll toward the pipe is played up to 3x (badged "SPEED-UP 3×"), ramping back to real time before the cut
    fw0, fw1 = t_q + 0.4, t_mid - 0.4
    sh.append(dict(name="carve_low", t=(t_q, t_mid),
                   slow=dict(t=(fw0, fw1), k=3.0, ease=0.4) if fw1 - fw0 > 0.6 else None,
                   cam=dict(kind="follow", subject="board", off=(0.16, 0.40, 0.035), off1=(-0.06, 0.40, 0.04),
                            fill=F(0.34, 0.46), fstop=2.8, smooth=0.25, eye_smooth=0.15, look_z=0.07,
                            aim_subject="mix")))
    sh.append(dict(name="carve_wide", t=(t_mid, t_pipe + 0.3), cam=dict(carve_wide(t_mid), off=(-1.4, 0.45, 0.75),
                                                                       off1=(-1.1, 0.2, 0.42))))
    ev = []
    # the best wall ride: in 0.3x from the platform beside the coping, the rider climbing toward the lens
    if best is not None:
        ta = apex_t[best]
        kx = int(round(ta / dt))
        side = 1.0 if board[kx, 0] > 0 else -1.0
        ya = float(board[kx, 1])
        ev.append(dict(name="wall_ride", t=(ta - 1.0, ta + 0.9), slow=dict(t=(ta - 0.35, ta + 0.25), k=0.3, ease=0.3),
                       cam=dict(kind="fixed", subject="trunk", eye=(side * (x_lip + 0.07), ya + sgn * 0.75,
                                                                    wall_h + 0.07),
                                track=(1.0, 1.0, 1.0), target=(0, 0, 0), fill=F(0.40, 0.36), fstop=4.0, smooth=0.25,
                                look_z=-0.02)))
    # foot-push: a low close-up on the side of the planting foot, the floor beside the board in frame; slow motion
    # (0.25x) timed on the actual floor contact (export_chain robot_floor), real time either side
    for i, (g0, g1) in enumerate(fp):
        pl = plant_for(g0, g1)
        sd = plant_side(pl) if pl else -1.0
        print(f"foot-push camera on the board's {'left' if sd > 0 else 'right'} (plant body {RB_NAMES[pl['body']] if pl else '-'})")
        sl = dict(t=(pl["t0"] - 0.2, pl["t1"] + 0.2), k=0.25, ease=0.25) if pl else None
        e = dict(name=f"footpush_{i}", t=(g0 - 0.3, g1 + 0.3),
                 cam=dict(kind="follow", subject="board", off=(-0.06, sd * 0.50, 0.03), off1=(0.0, sd * 0.52, 0.035),
                          focal=F(30.0, 32.0), fstop=4.0, smooth=0.4, aim_smooth=0.25, look_z=F(0.07, 0.05),
                          aim_subject="plant_mix", yaw_smooth=1.0))
        if sl:
            e["slow"] = sl
        ev.append(e)
    head_pipe = dict(kind="follow", subject="head", off=(0.44, -0.18, -0.01), focal=F(30.0, 36.0), fstop=2.8,
                     smooth=0.35, look_z=-0.035)
    sh += pipe_section(t_pipe + 0.3, ev, [head_pipe, PIPE_AXIS if V else PIPE_34])
    sh.append(dict(name="end", t=(T_END, T_END), hold=True, dur=a.hold_s, card="end",
                   cam=PIPE_AXIS if V else PIPE_34))
    return sh


SHOTS = shots_standard() if a.cut == "standard" else shots_cinematic()
# order + coverage: source time never goes back and nothing is skipped (held frames repeat their own time only)
_live = [s for s in SHOTS if not s.get("hold")]
for s0, s1 in zip(_live[:-1], _live[1:]):
    assert abs(s0["t"][1] - s1["t"][0]) < 1e-6, f"gap / overlap between {s0['name']} and {s1['name']}"
assert abs(_live[0]["t"][0]) < 1e-9 and abs(_live[-1]["t"][1] - T_END) < 1e-6
if a.cut == "standard":
    assert all(not s.get("slow") and s.get("speed", 1.0) == 1.0 for s in SHOTS), "standard cut must be 1x"
assert all(s.get("slow") is None or "t" in s["slow"] for s in SHOTS)


# ── bloopers (--bloopers) ────────────────────────────────────────────────────
def _take(kin, lab):
    rb, bb = kin["robot_body_pos"], kin["board_body_pos"]
    rn_, bn_ = [str(n) for n in kin["robot_body_names"]], [str(n) for n in kin["board_body_names"]]
    subj = {"trunk": kin["root_pos"], "head": rb[:, rn_.index("jaw_soft")], "board": bb[:, 0],
            "front_truck": bb[:, bn_.index("front_truck")], "rear_truck": bb[:, bn_.index("rear_truck")],
            "feet": 0.5 * (rb[:, rn_.index("ankle_left")] + rb[:, rn_.index("ankle_right")]),
            "mix": 0.55 * kin["root_pos"] + 0.45 * bb[:, 0]}
    subj["plant"], subj["plant_mix"] = subj["feet"], 0.5 * subj["feet"] + 0.5 * bb[:, 0]
    q = kin["board_body_quat_wxyz"][:, 0]
    byaw = np.unwrap(np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2)))
    return dict(KIN=kin, RB=rb, BB=bb, SUBJ=subj, BYAW=byaw, NSTEP=len(kin["t"]), LAB=lab)


TAKES = {"chain": dict(KIN=KIN, RB=RB, BB=BB, SUBJ=SUBJ, BYAW=BYAW, NSTEP=NSTEP, LAB=LAB)}


def use_take(name):
    """point the module globals the camera / sampling code reads at one take (the chain or a blooper)"""
    global KIN, RB, BB, SUBJ, BYAW, NSTEP, LAB
    t = TAKES[name]
    KIN, RB, BB, SUBJ, BYAW, NSTEP, LAB = t["KIN"], t["RB"], t["BB"], t["SUBJ"], t["BYAW"], t["NSTEP"], t["LAB"]


BLOOPERS = []
if a.bloopers:
    for i, it in enumerate(json.load(open(a.bloopers))["items"]):
        kin = {k: v for k, v in np.load(it["kin"]).items()}
        assert abs(float(kin["t"][1] - kin["t"][0]) - dt) < 1e-9
        cap = it.get("caption") or f"Blooper {i + 1}"
        if cap not in NAMES:
            NAMES.append(cap)
        TAKES[f"b{it['ep']}"] = _take(kin, np.full(len(kin["t"]), NAMES.index(cap)))
        BLOOPERS.append((f"b{it['ep']}", it, cap))


def blooper_shots():
    """each failed run: its last `blooper_s` s (cinematic: the last 0.55 s ramped to 0.35x), then frozen on the
    step the eval ended it. On the pipe: along the pipe axis from the open end (no wall between lens and rider);
    on the flat: a fixed camera beside the board, panning with the fall."""
    out = []
    for i, (name, it, cap) in enumerate(BLOOPERS):
        use_take(name)
        te = float(KIN["t"][-1])
        t0 = max(0.0, te - a.blooper_s)
        k0 = int(t0 / dt)
        bx, by = float(BB[-1, 0, 0]), float(BB[-1, 0, 1])
        if PIPE["y0"] - 0.2 <= by <= PIPE["y1"] + 0.2 and int(KIN["stage"][-1]) >= 4:
            ye = by - sgn * F(0.95, 1.05)            # ~1 m before where the run ends, on the pipe axis
            ys = RB[k0:, :, 1]
            ye = min(ye, ys.min() - 0.45) if sgn > 0 else max(ye, ys.max() + 0.45)   # never on top of the rider
            cam = dict(kind="fixed", subject="trunk", eye=(0.3 * bx, ye, F(0.22, 0.26)),
                       target=(bx, by, 0.08), track=(0.85, 0.7, 0.6), smooth=0.25, focal=F(52.0, 46.0), fstop=4.0,
                       look_z=0.0)
        else:
            cam = dict(kind="anchor", subject="trunk", at=0.5 * (t0 + te), off=(0.30, -0.85, 0.15),
                       fill=F(0.33, 0.30), fstop=4.0, smooth=0.25, look_z=0.0)
        sh = dict(name=f"blooper_{i}", take=name, blooper=i, caption=cap, t=(t0, te), freeze_s=a.freeze_s, cam=cam)
        if a.cut == "cinematic":
            sh["slow"] = dict(t=(te - 0.55, te), k=0.35, ease=0.25)
        out.append(sh)
    use_take("chain")
    return out


if BLOOPERS:
    SHOTS = SHOTS[:-1] + blooper_shots() + SHOTS[-1:]
    print("bloopers:", [(b[0], b[2], round(float(TAKES[b[0]]["KIN"]["t"][-1]), 2)) for b in BLOOPERS])


# ── build ────────────────────────────────────────────────────────────────────
REFRAMED = {}
# the bottom-left phase label in NDC (compose_cine geometry: baseline H-440 / H-120, box up to ~446 px wide, 80 px tall)
_H = 1920 if V else 1350
LABEL_TOP_NDC = 1 - 2 * ((_H - (440 if V else 120)) - 80) / _H
LABEL_RIGHT_NDC = 2 * 446 / 1080 - 1


def bbox_ndc(eye, aim, focal, rb, bp, r_robot=0.07, r_board=0.04):
    """(n, 4) NDC bbox (x0, y0, x1, y1; y up) of robot links + board bodies inflated by r (head shell / deck)"""
    out = np.zeros((len(eye), 4))
    for k in range(len(eye)):
        fwd = (aim[k] - eye[k]) / np.linalg.norm(aim[k] - eye[k])
        right = np.cross(fwd, [0, 0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        th, tv = APERTURE[0] / 2 / focal[k], APERTURE[1] / 2 / focal[k]
        pts = np.concatenate([rb[k], bp[k]])
        rad = np.r_[np.full(len(rb[k]), r_robot), np.full(len(bp[k]), r_board)]
        d = pts - eye[k]
        z = np.maximum(d @ fwd, 0.02)
        sx, sy = (d @ right) / z / th, (d @ up) / z / tv
        ex, ey = rad / z / th, rad / z / tv
        out[k] = ((sx - ex).min(), (sy - ey).min(), (sx + ex).max(), (sy + ey).max())
    return out


def label_clear_shift(eye, aim, focal, rb, bp, margin=0.03, head=0.96):
    """the (dx, dy) NDC move that clears the robot + board off the bottom-left label (90% of the shot's frames)
    without cutting the head or the far side: up first (headroom permitting), then to the right; None if clear"""
    bb = bbox_ndc(eye, aim, focal, rb, bp)
    hit = bb[:, 0] < LABEL_RIGHT_NDC
    need_up = np.where(hit, LABEL_TOP_NDC + margin - bb[:, 1], 0.0)
    dy_need = max(0.0, float(np.percentile(need_up, 90)))
    if dy_need < 0.01:
        return None
    dy = min(dy_need, max(0.0, float(np.percentile(head - bb[:, 3], 5))))
    dx = 0.0
    if dy < dy_need - 0.01:
        still = (bb[:, 1] + dy) < LABEL_TOP_NDC + margin
        need_r = np.where(hit & still, LABEL_RIGHT_NDC + margin - bb[:, 0], 0.0)
        dx_need = max(0.0, float(np.percentile(need_r, 90)))
        dx = min(dx_need, max(0.0, float(np.percentile(head - bb[:, 2], 5))))
    if dx < 0.01 and dy < 0.01:
        return None
    return dx, dy


def label_overlap(eye, aim, focal, rb, bp):
    """mean fraction of the bottom-left label box covered by the robot + board bbox"""
    bb = bbox_ndc(eye, aim, focal, rb, bp)
    w = np.clip(np.minimum(bb[:, 2], LABEL_RIGHT_NDC) - np.maximum(bb[:, 0], -1 + 2 * 40 / 1080), 0, None)
    h = np.clip(np.minimum(bb[:, 3], LABEL_TOP_NDC) - np.maximum(bb[:, 1], LABEL_TOP_NDC - 2 * 102 / _H), 0, None)
    return float(np.mean(w * h) / ((LABEL_RIGHT_NDC + 1 - 2 * 40 / 1080) * 2 * 102 / _H))


def build():
    fr = {k: [] for k in ("root_pos", "root_quat", "joint_pos", "board_pos", "board_quat", "robot_body_pos", "eye",
                          "target", "focal", "fstop", "shot", "src_t", "speed", "phase")}
    meta = []
    for si, sh in enumerate(SHOTS):
        use_take(sh.get("take", "chain"))
        rq = edit.fix_quat_signs(KIN["root_quat_wxyz"])
        bq = edit.fix_quat_signs(KIN["board_body_quat_wxyz"])
        src_t, spd = time_map(sh)
        n = len(src_t)
        f = np.clip(src_t / dt, 0, NSTEP - 1)
        fr["root_pos"].append(sample(KIN["root_pos"], f))
        fr["root_quat"].append(edit.nquat(sample(rq, f)))
        fr["joint_pos"].append(sample(KIN["joint_pos"], f))
        fr["robot_body_pos"].append(sample(RB, f))
        fr["board_pos"].append(sample(BB, f))
        fr["board_quat"].append(edit.nquat(sample(bq, f)))
        eye, aim, focal = camera(sh["cam"], f, n, sh["t"][0])
        if not sh.get("hold") and not sh.get("no_reframe"):
            cam = sh["cam"]
            for _ in range(2):     # overlay-safe framing: lift / slide the subject clear of the bottom-left label
                dxy = label_clear_shift(eye, aim, focal, sample(RB, f), sample(BB, f))
                if dxy is None:
                    break
                sx0, sy0 = cam.get("shift", (0.0, 0.0))
                cam = dict(cam, shift=(sx0 + dxy[0], sy0 + dxy[1]))
                eye, aim, focal = camera(cam, f, n, sh["t"][0])
            sub = SUBJ[cam.get("subject", "trunk")]
            if label_overlap(eye, aim, focal, sample(RB, f), sample(BB, f)) > 0.15 and sub.ndim == 2:
                for _ in range(2):
                    dxy = label_clear_shift(eye, aim, focal, sample(sub, f)[:, None], np.zeros((n, 0, 3)))
                    if dxy is None:
                        break
                    sx0, sy0 = cam.get("shift", (0.0, 0.0))
                    cam = dict(cam, shift=(sx0 + dxy[0], sy0 + dxy[1]))
                    eye, aim, focal = camera(cam, f, n, sh["t"][0])
            if cam is not sh["cam"]:
                REFRAMED[sh["name"]] = tuple(round(v, 3) for v in cam["shift"])
        fr["eye"].append(eye)
        fr["target"].append(aim)
        fr["focal"].append(focal)
        fr["fstop"].append(np.full(n, float(sh["cam"].get("fstop", 4.0))))
        fr["shot"].append(np.full(n, si))
        fr["src_t"].append(src_t)
        fr["speed"].append(spd)
        ph = LAB[np.clip(np.round(f).astype(int), 0, NSTEP - 1)]
        fr["phase"].append(np.full(n, -1) if sh.get("hold") else ph)
        meta.append(dict(name=sh["name"], take=sh.get("take", "chain"), blooper=sh.get("blooper"),
                         caption_bl=sh.get("caption"), frames=n, caption=None, card=sh.get("card"),
                         speed=float(spd.min()) if n and not sh.get("hold") else 0.0, t=list(sh["t"]),
                         hold=bool(sh.get("hold")), trail=False, cap_right=False,
                         subject=sh["cam"].get("subject", "trunk")))
    use_take("chain")
    out = {k: np.concatenate(v) for k, v in fr.items()}
    out.update(joint_names=KIN["joint_names"], board_body_names=KIN["board_body_names"], fps=a.fps,
               shots=json.dumps(meta), robot_body_names=KIN["robot_body_names"], aperture=np.array(APERTURE),
               format=a.format, phase_names=json.dumps(NAMES), cut=a.cut)
    out.update({k: v for k, v in KIN.items() if k.startswith("board_geom_")})
    return out, meta


tl, shots_out = build()
inf, vis, scr = edit.visibility(tl)
tl["robot_in_frame"], tl["robot_visible"], tl["robot_screen"] = inf, vis, scr


def subject_bbox(r_robot=0.07, r_board=0.04):
    """per frame: NDC bbox (x0, y0, x1, y1; y up, -1..1) of every robot link and board body origin, each inflated
    by a sphere of r (the head shell / deck extent around the origins)"""
    out = np.zeros((len(tl["shot"]), 4))
    for k in range(len(out)):
        eye, tgt = tl["eye"][k], tl["target"][k]
        fwd = (tgt - eye) / np.linalg.norm(tgt - eye)
        right = np.cross(fwd, [0, 0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        th, tv = APERTURE[0] / 2 / tl["focal"][k], APERTURE[1] / 2 / tl["focal"][k]
        pts = np.concatenate([tl["robot_body_pos"][k], tl["board_pos"][k]])
        rad = np.r_[np.full(len(tl["robot_body_pos"][k]), r_robot), np.full(len(tl["board_pos"][k]), r_board)]
        d = pts - eye
        z = np.maximum(d @ fwd, 0.02)
        sx, sy = (d @ right) / z / th, (d @ up) / z / tv
        ex, ey = rad / z / th, rad / z / tv
        out[k] = ((sx - ex).min(), (sy - ey).min(), (sx + ex).max(), (sy + ey).max())
    return out


tl["subj_bbox"] = subject_bbox()
print("reframed for the label (shift NDC dx, dy):", REFRAMED)


# per-shot subject check: the subject point in frame, line of sight, and the lens never inside the robot / board
def subject_check():
    res = []
    acc = 0
    for si, sh in enumerate(SHOTS):
        n = shots_out[si]["frames"]
        sl = slice(acc, acc + n)
        use_take(sh.get("take", "chain"))
        subj = sample(SUBJ[sh["cam"].get("subject", "trunk")], np.clip(tl["src_t"][sl] / dt, 0, NSTEP - 1))
        eye, tgt, foc = tl["eye"][sl], tl["target"][sl], tl["focal"][sl]
        ok, dmin = 0, 9.0
        for k in range(n):
            fwd = (tgt[k] - eye[k]) / np.linalg.norm(tgt[k] - eye[k])
            right = np.cross(fwd, [0, 0, 1.0])
            right /= np.linalg.norm(right)
            up = np.cross(right, fwd)
            d = subj[k] - eye[k]
            zz = d @ fwd
            sx, sy = (d @ right) / zz / (APERTURE[0] / 2 / foc[k]), (d @ up) / zz / (APERTURE[1] / 2 / foc[k])
            ok += (zz > 0) and abs(sx) < 0.92 and abs(sy) < 0.92
            pts = np.concatenate([tl["robot_body_pos"][acc + k], tl["board_pos"][acc + k]])
            dmin = min(dmin, float(np.min(np.linalg.norm(pts - eye[k], axis=1))))
        res.append((ok / max(n, 1), dmin))
        acc += n
    return res


SC = subject_check()
use_take("chain")
POLICIES = ("push", "manual", "half-pipe", "foot-push")
# chain extras for compose_cine.py: the end card (the run-selection and chain-complete disclosures from meta.json)
m = META
n_, cc = m.get("n"), m.get("chain_complete")
n_ok = int(round(cc * n_)) if (cc is not None and n_) else None
# r3 (2026-11-22, author: fewer words): 3 short lines + one footnote; same disclosures (pick, rate, sim-only, short pipe)
lines = [(f"{len(POLICIES)} RL policies, one run", "l1"),
         (f"{n_ok} / {n_} sim runs finish" if n_ok is not None else "Simulated runs", "l2"),
         ("Not on hardware yet", "l3"),
         ("Top-scoring run shown" + (" · pipe shortened for render" if PIPE.get("trim") else ""), "foot")]
tl["card_lines"] = json.dumps(lines)
tl["title_sub"] = "Simulated skate run"
tl["pipe"] = json.dumps(PIPE)
np.savez_compressed(a.out, **tl)
json.dump(shots_out, open(os.path.splitext(a.out)[0] + "_shots.json", "w", encoding="utf-8"), indent=1,
          ensure_ascii=False)

N = len(tl["shot"])
print(f"{a.out}: {a.cut} {a.format}, {N} frames, {N / a.fps:.1f} s at {a.fps} fps (source {T_END:.2f} s)")
bb_ = tl["subj_bbox"]
WPX, HPX = (1080, 1920) if V else (1080, 1350)
LY = HPX - (440 if V else 120)
LBL = (40, LY - 80, 40 + 26 + 380, LY + 22)         # the bottom-left phase label box ("Half-pipe" / "Foot push", px)
TAGR = (30, (172 if V else 36) - 12, 640, (172 if V else 36) + 44)
pxb = np.stack([(bb_[:, 0] + 1) / 2 * WPX, (1 - bb_[:, 3]) / 2 * HPX, (bb_[:, 2] + 1) / 2 * WPX,
                (1 - bb_[:, 1]) / 2 * HPX], 1)


def _ov(r, q):
    return max(0.0, min(r[2], q[2]) - max(r[0], q[0])) * max(0.0, min(r[3], q[3]) - max(r[1], q[1]))


lab_ov = np.array([_ov(LBL, q) for q in pxb]) / ((LBL[2] - LBL[0]) * (LBL[3] - LBL[1]))
tag_ov = np.array([_ov(TAGR, q) for q in pxb]) / ((TAGR[2] - TAGR[0]) * (TAGR[3] - TAGR[1]))
whole = (bb_[:, 0] > -1) & (bb_[:, 1] > -1) & (bb_[:, 2] < 1) & (bb_[:, 3] < 1)
hgt = (np.minimum(bb_[:, 3], 1) - np.maximum(bb_[:, 1], -1)) / 2
acc = 0
for si, s in enumerate(shots_out):
    sl = slice(acc, acc + s["frames"])
    d = np.linalg.norm(tl["eye"][sl] - tl["target"][sl], axis=1)
    sp = tl["speed"][sl]
    print(f"  {acc / a.fps:6.2f}s  {s['name']:12s} {[round(v, 2) for v in s['t']]} speed {sp.min():.2f}-{sp.max():.2f} "
          f"{s['frames']:4d} fr  rider in-frame {inf[sl].mean():4.0%} visible {vis[sl].mean():4.0%}  subject in-frame "
          f"{SC[si][0]:4.0%}  lens>=robot {SC[si][1] * 100:.0f} cm  dist {d.min():.2f}-{d.max():.2f} m  focal "
          f"{tl['focal'][sl].min():.0f}-{tl['focal'][sl].max():.0f}  whole robot+board in frame {whole[sl].mean():4.0%} "
          f"height {hgt[sl].mean():.0%}  label-box on robot {lab_ov[sl].mean():.0%} tag {tag_ov[sl].mean():.0%}")
    acc += s["frames"]
