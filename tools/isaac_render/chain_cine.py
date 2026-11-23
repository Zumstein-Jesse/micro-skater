# micro-skater tools/isaac_render: the author's Isaac Sim replay pipeline. Apache-2.0 (see LICENSE).
"""One command: chain demo_qpos.npz -> the STANDARD (1x) and CINEMATIC cuts in native 9x16 and 4x5 (no 16x9).

    python chain_cine.py QPOS --label NAME [--stills | --test] [--from STEP] [--only STEP] [--cuts standard,cinematic]
                         [--formats 9x16,4x5] [--prefix microduck_foot_push] [--replace] [--no-wait]

Steps (each prints its report; any failure stops the run). Paths are relative to the repo root; work files go to tools\\isaac_render\\chain_cine\\NAME. Windows only (the
render step runs the Isaac launcher through cmd):
    export  train venv   export_chain.py QPOS -> chain_cine\\NAME\\chain.kin.npz, pipe.json, meta.json (+contacts)
    trim    .usdvenv     pipe_trim.py -> pipe_render.json (the rendered pipe cut to the part the run uses)
    park    .usdvenv     build_park.py --layout pipe --look cine --pipe pipe_render.json -> scenes\\skatepark_cine\\NAME
    edit    .usdvenv     edit_cine.py x cuts x formats -> chain_cine\\NAME\\timeline_<cut>_<fmt>.npz
    bake    .usdvenv     bake_replay.py -> scenes\\skatepark_cine\\NAME\\replay_<cut>_<fmt>.usda
    check   .usdvenv     check_chain.py: links / board vs MuJoCo, wheel gap to the rendered pipe, trimmed-pipe coverage
    render  Isaac Sim    render_replay.py rt 32 subframes, native 1080x1920 / 1080x1350 -> output\\chain\\NAME\\
                         frames_<cut>_<fmt> (held frames rendered once and copied; resumes; waits while a training run uses the GPU)
    compose .usdvenv     compose_cine.py -> output\\videos\\<prefix>_<cut>_<fmt>.mp4 (never overwrites an existing file
                         unless --replace; the PROTECTED file names never)
--stills: one frame per shot (+ each slow-motion peak) for every cut / format, composited -> output\\chain\\NAME\\stills_comp
--test:   a short section of each cut (TEST_RANGES) rendered and encoded -> output\\chain\\NAME\\test_<cut>_<fmt>.mp4
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(HERE))  # the repo root: scenes/, output/ and assets/ live there
# the python of the training venv (micro_skater + microduck_rl installed); default: this python
TRAIN_PY = os.environ.get("MICRO_SKATER_TRAIN_PY", sys.executable)
# a python with usd-core (pip install usd-core numpy pillow); default: .usdvenv next to this file
USD_PY = os.environ.get("MICRO_SKATER_USD_PY", os.path.join(HERE, ".usdvenv", "Scripts", "python.exe"))
# Isaac Sim's python launcher (for example <isaac-sim>/): --isaac or ISAAC_SIM_PYTHON; only the render
# step needs it
ISAAC = os.environ.get("ISAAC_SIM_PYTHON", "")
PROTECTED: set[str] = set()   # output file names this pipeline must never overwrite
STEPS = ["export", "trim", "park", "edit", "bake", "check", "render", "compose"]
RES = {"9x16": (1080, 1920), "4x5": (1080, 1350)}

ap = argparse.ArgumentParser()
ap.add_argument("qpos")
ap.add_argument("--label", required=True)
ap.add_argument("--stills", action="store_true")
ap.add_argument("--test", action="store_true")
ap.add_argument("--from", dest="start", default="export", choices=STEPS)
ap.add_argument("--only", default=None, choices=STEPS)
ap.add_argument("--cuts", default="standard,cinematic")
ap.add_argument("--formats", default="9x16,4x5")
ap.add_argument("--prefix", default="microduck_foot_push")
ap.add_argument("--replace", action="store_true", help="allow overwriting this pipeline's own earlier output mp4")
ap.add_argument("--no-wait", action="store_true")
ap.add_argument("--subframes", type=int, default=32)
ap.add_argument("--margin", type=float, default=0.7, help="pipe_trim margin (m)")
ap.add_argument("--test-s", type=float, default=4.0)
ap.add_argument("--isaac", default=None, help="Isaac Sim python launcher (default: env ISAAC_SIM_PYTHON)")
# r3 (2026-11-22, additive / default off): bloopers + multi-pass renders
ap.add_argument("--bloopers", action="store_true", help="edit with chain_cine/NAME/bloopers/bloopers.json "
                "(make_bloopers.py): failed runs of the same eval before the end card")
ap.add_argument("--passes", type=int, default=1, help="independent render passes per frame, averaged in compose "
                "(pass 0 -> frames_<cut>_<fmt><tag>, pass j -> ..._p<j>)")
ap.add_argument("--frames-tag", default="", help="suffix of the frame dirs (a new timeline needs new dirs)")
a = ap.parse_args()
ISAAC = a.isaac or ISAAC

CUTS = a.cuts.split(",")
FMTS = a.formats.split(",")
D = os.path.join(HERE, "chain_cine", a.label)
SC = os.path.join(PROJECT, "scenes", "skatepark_cine", a.label)
OUT = os.path.join(PROJECT, "output", "chain", a.label)
for p in (D, SC, OUT):
    os.makedirs(p, exist_ok=True)
env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
TL = lambda c, f: os.path.join(D, f"timeline_{c}_{f}.npz")
RP = lambda c, f: os.path.join(SC, f"replay_{c}_{f}.usda")
FR = lambda c, f, j=0: os.path.join(OUT, f"frames_{c}_{f}{a.frames_tag}" + (f"_p{j}" if j else ""))
FRS = lambda c, f: ",".join(FR(c, f, j) for j in range(a.passes))


def run(cmd, log=None):
    print(">>", " ".join(cmd), flush=True)
    if log:
        with open(log, "w", encoding="utf-8") as fh:
            r = subprocess.run(cmd, cwd=HERE, env=env, stdout=fh, stderr=subprocess.STDOUT)
        tail = open(log, encoding="utf-8", errors="replace").read().splitlines()
        for ln in [ln for ln in tail if ln.startswith(("RENDER", "BAKE", "PARK"))][-6:]:
            print("   ", ln)
    else:
        r = subprocess.run(cmd, cwd=HERE, env=env)
    if r.returncode != 0:
        raise SystemExit(f"step failed ({r.returncode}): {' '.join(cmd)}" + (f"  (log {log})" if log else ""))


def wanted(step):
    return step == a.only if a.only else STEPS.index(step) >= STEPS.index(a.start)


def gpu_wait():
    while not a.no_wait:
        busy = "train.exe" in subprocess.run(["tasklist", "/FI", "IMAGENAME eq train.exe"], capture_output=True,
                                             text=True).stdout
        try:
            used = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                      capture_output=True, text=True).stdout.split()[0])
        except Exception:  # noqa: BLE001
            used = 0
        if not busy and used < 9000:
            return
        print(time.strftime("%H:%M:%S"), f"GPU busy (train.exe {busy}, {used} MiB used): waiting (poll 120 s)",
              flush=True)
        time.sleep(120)


def shot_picks(c, f):
    shots = json.load(open(os.path.splitext(TL(c, f))[0] + "_shots.json", encoding="utf-8"))
    z = np.load(TL(c, f))
    sp = z["speed"]
    picks, acc = [], 0
    for s in shots:
        n = s["frames"]
        picks.append(acc + (n - 30 if s.get("hold") else n // 2))
        seg = sp[acc:acc + n]
        if not s.get("hold") and seg.min() < 0.99:
            picks.append(acc + int(np.argmin(seg)))
        acc += n
    return sorted(set(picks))


def unique_frames(c, f, ks):
    """frames whose pose / camera differ from the previous frame (held frames render once and are copied)"""
    z = np.load(TL(c, f))
    keys = ("root_pos", "root_quat", "joint_pos", "board_pos", "board_quat", "eye", "target", "focal", "fstop")
    src = {}
    for k in ks:
        if k > 0 and all(np.array_equal(z[n][k], z[n][k - 1]) for n in keys):
            src[k] = src.get(k - 1, k - 1)
        else:
            src[k] = k
    return src


def test_range(c, f):
    """a few seconds that show the cut's character: the standard cut across the carve-in -> half-pipe cut (a camera cut,
    a label change, the short pipe), the cinematic cut through the manual slow-motion ramp"""
    z = np.load(TL(c, f))
    shots = json.load(open(os.path.splitext(TL(c, f))[0] + "_shots.json", encoding="utf-8"))
    acc, st = 0, {}
    for s in shots:
        st[s["name"]] = (acc, acc + s["frames"])
        acc += s["frames"]
    fps = int(z["fps"])
    n = int(a.test_s * fps)
    pipe0 = min(v[0] for k_, v in st.items() if k_.startswith("pipe_"))
    k0 = pipe0 - int(1.5 * fps) if c == "standard" else st["manual_low"][0]
    return max(0, k0), min(len(z["shot"]), max(0, k0) + n)


if wanted("export"):
    run([TRAIN_PY, "export_chain.py", a.qpos, "--out", os.path.join(D, "chain")])
if wanted("trim"):
    run([USD_PY, "pipe_trim.py", "--dir", D, "--margin", str(a.margin)])
if wanted("park"):
    run([USD_PY, "build_park.py", "--layout", "pipe", "--look", "cine", "--pipe", os.path.join(D, "pipe_render.json"),
         "--paths", os.path.join(D, "*.kin.npz"), "--out", os.path.join(SC, "park.usda")],
        log=os.path.join(OUT, "park.log"))
if wanted("edit"):
    for c in CUTS:
        for f in FMTS:
            run([USD_PY, "edit_cine.py", "--cut", c, "--format", f, "--dir", D, "--out", TL(c, f)]
                + (["--bloopers", os.path.join(D, "bloopers", "bloopers.json")] if a.bloopers else []))
if wanted("bake"):
    for c in CUTS:
        for f in FMTS:
            run([USD_PY, "bake_replay.py", "--timeline", TL(c, f), "--park", os.path.join(SC, "park.usda"),
                 "--out", RP(c, f)], log=os.path.join(OUT, f"bake_{c}_{f}.log"))
if wanted("check"):
    for c in CUTS:
        for f in FMTS:
            run([USD_PY, "check_chain.py", "--replay", RP(c, f), "--timeline", TL(c, f), "--every", "1"])
if wanted("render"):
    for c in CUTS:
        for f in FMTS:
            z = np.load(TL(c, f))
            N = len(z["shot"])
            if a.stills:
                ks = shot_picks(c, f)
                dst = FR(c, f) + "_stills"
            elif a.test:
                r0, r1 = test_range(c, f)
                ks = list(range(r0, r1))
                dst = FR(c, f)
            else:
                ks = list(range(N))
                dst = FR(c, f)
            src = unique_frames(c, f, ks)
            for j in range(1 if (a.stills or a.test) else a.passes):
                dj = dst if j == 0 else FR(c, f, j)
                todo = sorted({src[k] for k in ks if not os.path.exists(os.path.join(dj, f"frame_{src[k]:05d}.png"))})
                if todo:
                    if not ISAAC:
                        raise SystemExit("render step: set ISAAC_SIM_PYTHON or pass --isaac (Isaac Sim's python launcher)")
                    gpu_wait()
                    w, h = RES[f]
                    run(["cmd", "/c", ISAAC, os.path.join(HERE, "render_replay.py"), "--stage", RP(c, f), "--out", dj,
                         "--mode", "rt", "--subframes", str(a.subframes), "--width", str(w), "--height", str(h),
                         "--pick", ",".join(map(str, todo))], log=os.path.join(OUT, f"render_{c}_{f}{a.frames_tag}_p{j}.log"))
                for k in ks:
                    if src[k] != k and not os.path.exists(os.path.join(dj, f"frame_{k:05d}.png")):
                        shutil.copyfile(os.path.join(dj, f"frame_{src[k]:05d}.png"), os.path.join(dj, f"frame_{k:05d}.png"))
            print(f"render {c} {f}: {len(ks)} frames ({len(set(src.values()))} unique) in {dst}", flush=True)
if wanted("compose"):
    for c in CUTS:
        for f in FMTS:
            if a.stills:
                run([USD_PY, "compose_cine.py", "--frames", FR(c, f) + "_stills", "--timeline", TL(c, f),
                     "--still", ",".join(map(str, shot_picks(c, f))), "--still-dir", os.path.join(OUT, "stills_comp")])
            elif a.test:
                r0, r1 = test_range(c, f)
                dst = os.path.join(OUT, f"test_{c}_{f}.mp4")
                run([USD_PY, "compose_cine.py", "--frames", FR(c, f), "--timeline", TL(c, f), "--range", f"{r0}:{r1}",
                     "--out", dst])
            else:
                dst = os.path.join(PROJECT, "output", "videos", f"{a.prefix}_{c}_{f}.mp4")
                assert os.path.basename(dst) not in PROTECTED
                if os.path.exists(dst) and not a.replace:
                    raise SystemExit(f"{dst} exists: refusing to overwrite (pass --replace for this pipeline's own file)")
                run([USD_PY, "compose_cine.py", "--frames", FRS(c, f), "--timeline", TL(c, f), "--out", dst])
print("chain_cine done:", OUT, flush=True)
