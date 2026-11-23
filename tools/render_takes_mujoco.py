# micro-skater tools: MuJoCo-only renderer for recorded play-sim takes. Apache-2.0 (see LICENSE).
"""Render a recorded play-sim take (.npz from skate_play.Recorder) with MuJoCo's offscreen renderer.

The take stores qpos after every 50 Hz control step. This script sets qpos for each frame, calls mj_forward (no
physics is stepped) and renders with mujoco.Renderer. A free camera follows the robot trunk (smoothed). Frames are
piped to ffmpeg as an H.264 mp4 in real time (no speed-up).

    python render_takes_mujoco.py --npz takes/hero.npz --model takes/model.mjb --out walk.mp4 \
        --start 2.0 --end 5.0 [--width 640 --height 360 --fps 25 --distance 1.2 --elevation -15 --azimuth 90]

Times are seconds from the start of the take (step index / 50). --list prints the take's events and mode changes.
--azimuth is the camera azimuth in degrees; with --follow-heading it is added to the smoothed heading of the trunk
(so 90 is a side view, 180 a front view, 0 a rear chase view).

The model has no skybox, so far-away pixels (found from a depth render) are painted with a blue gradient and distance
fog blends the far floor into the horizon colour (--no-sky turns both off). The model must have the bodies
"robot/trunk_base" and "board/skateboard" (the play-sim bundle's model). Needs: mujoco, numpy, ffmpeg on PATH.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess

import mujoco
import numpy as np


def list_take(z):
    meta = json.loads(str(z["meta"]))
    print("events (label, step, s):", [(e[0], e[1], round(e[1] / 50, 2)) for e in meta.get("events", [])])
    prev = None
    for i, (a, b) in enumerate(zip(z["mode"], z["skate_mode"])):
        if (a, b) != prev:
            print(f"  step {i:5d}  t={i / 50:6.2f}s  mode={a} skate_mode={b}")
            prev = (a, b)


def yaw_of(q):
    w, x, y, zq = q
    return math.atan2(2 * (w * zq + x * y), 1 - 2 * (y * y + zq * zq))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--fps", type=int, default=25, choices=[10, 25, 50])
    ap.add_argument("--distance", type=float, default=1.2)
    ap.add_argument("--elevation", type=float, default=-15.0)
    ap.add_argument("--azimuth", type=float, default=90.0)
    ap.add_argument("--follow-heading", action="store_true", help="azimuth relative to the trunk heading")
    ap.add_argument("--smooth", type=float, default=0.15, help="look-at smoothing time constant (s)")
    ap.add_argument("--smooth-heading", type=float, default=0.8, help="heading smoothing time constant (s)")
    ap.add_argument("--freeze-heading-after", type=float, default=None,
                    help="stop following the heading after this time (s), e.g. at a fall")
    ap.add_argument("--no-sky", action="store_true", help="keep MuJoCo's black background")
    ap.add_argument("--target", choices=["robot", "board", "mid"], default="robot")
    ap.add_argument("--lookat-z", type=float, default=0.08, help="height offset of the look-at point (m)")
    ap.add_argument("--crf", type=int, default=18)
    a = ap.parse_args()

    z = np.load(a.npz)
    if a.list:
        list_take(z)
        return
    qpos = z["qpos"].astype(np.float64)
    dt = 0.02
    i0 = int(round(a.start / dt))
    i1 = len(qpos) if a.end is None else min(len(qpos), int(round(a.end / dt)))
    stride = 50 // a.fps

    m = mujoco.MjModel.from_binary_path(a.model)
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, a.width)
    m.vis.global_.offheight = max(m.vis.global_.offheight, a.height)
    d = mujoco.MjData(m)
    sky_top, sky_hor = np.array([118, 160, 210], np.float32), np.array([214, 226, 238], np.float32)
    if not a.no_sky:   # fog the far terrain into the horizon colour; the background is painted from the depth buffer
        m.vis.rgba.fog[:3] = sky_hor / 255.0
        m.vis.map.fogstart, m.vis.map.fogend = 2.5, 8.0   # x stat.extent (2 m)
    r = mujoco.Renderer(m, a.height, a.width)
    rd = None if a.no_sky else mujoco.Renderer(m, a.height, a.width)
    if rd is not None:
        rd.enable_depth_rendering()
    zfar = m.vis.map.zfar * m.stat.extent
    ramp = np.linspace(0.0, 1.0, a.height, dtype=np.float32)[:, None, None] ** 0.7
    sky = (sky_top * (1 - ramp) + sky_hor * ramp) * np.ones((1, a.width, 1), np.float32)
    opt = mujoco.MjvOption()
    if not a.no_sky:
        opt.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = 1
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.elevation = a.distance, a.elevation
    trunk = m.body("robot/trunk_base").id
    board = m.body("board/skateboard").id

    def target(i):
        d.qpos[:] = qpos[i]
        mujoco.mj_forward(m, d)
        p = d.xpos[trunk].copy()
        if a.target == "board":
            p = d.xpos[board].copy()
        elif a.target == "mid":
            p = 0.5 * (p + d.xpos[board])
        return p, yaw_of(qpos[i, 3:7])

    # Pre-warm the smoothed camera on the pre-roll before i0 so the first frame is already settled.
    alpha = 1.0 - math.exp(-dt / max(a.smooth, 1e-3))
    j0 = max(0, i0 - 100)
    look, head = target(j0)
    head_c, head_s = math.cos(head), math.sin(head)
    beta = 1.0 - math.exp(-dt / max(a.smooth_heading, 1e-3))
    proc = None
    if a.out:
        proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{a.width}x{a.height}",
             "-r", str(a.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(a.crf),
             "-preset", "medium", "-movflags", "+faststart", a.out], stdin=subprocess.PIPE)
    n = 0
    for i in range(j0, i1):
        p, h = target(i)
        look += alpha * (p - look)
        if a.freeze_heading_after is None or i * dt < a.freeze_heading_after:
            head_c += beta * (math.cos(h) - head_c)
            head_s += beta * (math.sin(h) - head_s)
        if i < i0 or (i - i0) % stride:
            continue
        cam.lookat[:] = look + np.array([0.0, 0.0, a.lookat_z])
        az = a.azimuth + (math.degrees(math.atan2(head_s, head_c)) if a.follow_heading else 0.0)
        cam.azimuth = az
        r.update_scene(d, camera=cam, scene_option=opt)
        if rd is None:
            img = r.render()
        else:
            r.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1
            img = r.render().astype(np.float32)
            rd.update_scene(d, camera=cam, scene_option=opt)
            mask = rd.render() >= 0.99 * zfar
            img[mask] = sky[mask]
            img = img.astype(np.uint8)
        if proc:
            proc.stdin.write(img.tobytes())
        n += 1
    if proc:
        proc.stdin.close()
        proc.wait()
    print(f"rendered {n} frames ({i0 * dt:.2f}-{i1 * dt:.2f} s, {a.fps} fps) -> {a.out}")


if __name__ == "__main__":
    main()
