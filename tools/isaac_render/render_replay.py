# micro-skater tools/isaac_render: the author's Isaac Sim replay pipeline. Apache-2.0 (see LICENSE).
"""Render the baked skatepark replay through /World/ShotCamera to a PNG sequence (Isaac Sim python).

    <isaac-sim python> tools\\isaac_render\\render_replay.py --out DIR [--frames 0:2859 | --pick 10,200,...]

Renderer: RTX path tracing (--mode pt, default: denoised, --spp samples per frame, fixed exposure) or the real-time path
tracer (--mode rt). Depth of field comes from the camera's f-stop / focus distance (baked per frame). Frames already in
DIR are skipped, so a stopped render resumes.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent  # the repo root: scenes/, output/ and assets/ live there
parser = argparse.ArgumentParser()
parser.add_argument("--stage", default=str(PROJECT / "scenes" / "skatepark" / "replay.usda"))
parser.add_argument("--out", required=True)
parser.add_argument("--width", type=int, default=1920)
parser.add_argument("--height", type=int, default=1080)
parser.add_argument("--frames", default=None, help="a:b range of timeline frames")
parser.add_argument("--pick", default=None, help="comma separated frames")
parser.add_argument("--every", type=int, default=1)
parser.add_argument("--mode", choices=["pt", "rt"], default="pt")
parser.add_argument("--spp", type=int, default=64, help="pt: samples per pixel per frame")
parser.add_argument("--subframes", type=int, default=1, help="rt: accumulation passes per frame")
parser.add_argument("--exposure", type=float, default=None, help="override tonemapper exposure (EV)")
parser.add_argument("--dof", type=int, default=1)
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "renderer": "PathTracing" if args.mode == "pt" else "RealTimePathTracing",
                     "width": args.width, "height": args.height})

import carb  # noqa: E402
import isaacsim.core.experimental.utils.app as app_utils  # noqa: E402

s = carb.settings.get_settings()
if args.mode == "pt":
    s.set("/rtx/rendermode", "PathTracing")
    s.set("/rtx/pathtracing/spp", min(args.spp, 32))
    s.set("/rtx/pathtracing/totalSpp", args.spp)
    s.set("/rtx/pathtracing/adaptiveSampling/enabled", False)
    s.set("/rtx/pathtracing/optixDenoiser/enabled", True)
    s.set("/rtx/pathtracing/maxBounces", 6)
    s.set("/rtx/pathtracing/maxSpecularAndTransmissionBounces", 6)
    s.set("/rtx/pathtracing/fireflyFilter/enabled", True)
    s.set("/rtx/pathtracing/fireflyFilter/maxIntensityPerSample", 40.0)
    s.set("/rtx/pathtracing/enableDof", bool(args.dof))
else:
    s.set("/rtx/post/dof/enabled", bool(args.dof))
s.set("/rtx/post/histogram/enabled", False)
s.set("/rtx/post/motionblur/enabled", False)
s.set("/rtx/post/aa/op", 0 if args.mode == "pt" else 3)
s.set("/rtx/post/tonemap/op", 6)                # ACES
if args.exposure is not None:
    s.set("/rtx/post/tonemap/exposure", args.exposure)
s.set("/rtx/raytracing/fractionalCutoutOpacity", False)
app_utils.enable_extension("omni.replicator.core")
app.update()

import omni.replicator.core as rep  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image  # noqa: E402

ctx = omni.usd.get_context()
ctx.open_stage(str(Path(args.stage).resolve()))
for _ in range(30):
    app.update()
while ctx.get_stage_loading_status()[2] > 0:
    app.update()
stage = ctx.get_stage()
fps = stage.GetTimeCodesPerSecond()
n = int(stage.GetEndTimeCode()) + 1
if args.pick:
    frames = [int(v) for v in args.pick.split(",")]
elif args.frames:
    a, b = map(int, args.frames.split(":"))
    frames = list(range(a, min(b, n), args.every))
else:
    frames = list(range(0, n, args.every))

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
product = rep.create.render_product("/World/ShotCamera", (args.width, args.height))
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach(product)
timeline = omni.timeline.get_timeline_interface()
timeline.set_current_time(frames[0] / fps)
for _ in range(20):     # textures, MDL compile, BVH
    rep.orchestrator.step(rt_subframes=1, delta_time=0.0, pause_timeline=True)
t_start = time.time()
done = 0
for k in frames:
    dst = out / f"frame_{k:05d}.png"
    if dst.exists():
        continue
    timeline.set_current_time(k / fps)
    rep.orchestrator.step(rt_subframes=(args.subframes if args.mode == "rt" else max(1, args.spp // min(args.spp, 32))),
                          delta_time=0.0, pause_timeline=True)
    img = np.asarray(rgb.get_data())[:, :, :3].astype(np.uint8)
    if img.shape[:2] != (args.height, args.width):
        raise RuntimeError(f"frame {k}: {img.shape}")
    Image.fromarray(img).save(dst)
    done += 1
    if done % 20 == 1 or args.pick:
        el = time.time() - t_start
        print(f"RENDER frame {k} ({done}/{len(frames)}) mean {img.mean():.1f}  {el / done:.2f} s/frame", flush=True)
rgb.detach()
rep.orchestrator.wait_until_complete()
print(f"RENDER done: {done} frames in {time.time() - t_start:.0f} s -> {out}", flush=True)
app.close()
