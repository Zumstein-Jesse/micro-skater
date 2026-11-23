# micro-skater: part of the author's skateboard add-on for microduck_rl. Apache-2.0 (see LICENSE).
"""Run the final four-policy chain (push p16 -> manual m6 -> half-pipe r10 <-> foot push p17e) like the final eval.

    python -m micro_skater.eval.run_chain [--n 32] [--seed 3] [--out out/chain_s3] [--no-render] [--logs] [--cpu]

Sets the env vars and arguments of the final evaluation and calls micro_skater.eval.chain_play (with --logs:
micro_skater.eval.chain_logged, which also saves per-step logs) in a child process; the task configs read the env
vars when they are imported. The final evaluation was 6 batches of 32 runs with seeds 3, 4 and 5, plus three more
batches with seed 5 (GPU physics is not bit-deterministic, so reruns of a seed differ). Seeds 0-2 were used to record
the hand-over bank of the p17d stage, so they are not held out.

Outputs in --out: funnel.json (per-stage funnel and the "complete" count), picks.json, demo_qpos.npz (the demo run's
joint states, the input of tools/isaac_render) and the compiled model.mjb, also with --no-render; without --no-render
also the MuJoCo videos chain_typical.mp4 / chain_best.mp4 / chain_demo.mp4.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

from micro_skater.launch import MANUAL_STAGES, PUSH_STAGES, RAMP_STAGES, REALISM
from micro_skater.paths import checkpoint_path, data_path

# The env of the final eval batches, as recorded in their env files.
CHAIN_ENV = {
    **PUSH_STAGES, **MANUAL_STAGES, **RAMP_STAGES, **REALISM,
    "MICRODUCK_PIPE_H": "0.08",                                       # 8 cm half-pipe
    "MICRODUCK_RAMP_HANDOFF": str(data_path("ramp_bank_p13.npz")),
    "MICRODUCK_CHAIN_START_GAP": "9",
    "MICRODUCK_CHAIN_SQUAT": "1",            # "complete" requires a clean end (no squat on the board)
    "MICRODUCK_CHAIN_RAMP_FOLLOW": "0.75",   # video camera only
    "MICRODUCK_CHAIN_RAMP_DIST": "0.95",     # video camera only
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=32, help="parallel runs (the final eval used 32 per batch)")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--logs", action="store_true", help="also save per-step logs (chain_logs.npz; bloopers input)")
    ap.add_argument("--cpu", action="store_true", help="run on the CPU (slow; for a quick check use --n 2)")
    ap.add_argument("--push", default=str(checkpoint_path("push_p16/model.pt")))
    ap.add_argument("--manual", default=str(checkpoint_path("manual_m6/model.pt")))
    ap.add_argument("--ramp", default=str(checkpoint_path("halfpipe_r10/model.pt")))
    ap.add_argument("--stroke", default=str(checkpoint_path("footpush_p17e/model.pt")))
    a, extra = ap.parse_known_args()
    label = a.label or f"final_s{a.seed}"
    out = a.out or os.path.join("out", f"chain_{label}")
    os.makedirs(out, exist_ok=True)
    module = "micro_skater.eval.chain_logged" if a.logs else "micro_skater.eval.chain_play"
    cmd = [sys.executable, "-m", module,
           "--push", a.push, "--manual", a.manual, "--ramp", a.ramp, "--n", str(a.n), "--seconds", str(a.seconds),
           "--stance-gate", "--manual-clock", "4", "--seq-override", "settle_s=(1.2, 1.5)",
           "--seq-override", "ramp_gate_x=0.8", "--rolling", "0.3", "--out", out, "--label", label,
           "--seed", str(a.seed),
           "--stroke", a.stroke, "--stroke-deadline", "3.0", "--stroke-vmin", "0.20", "--stroke-vmax", "0.75",
           "--stroke-abort-s", "0.8", "--stroke-alpha", "1.0", *extra]
    if a.no_render:
        cmd.append("--no-render")
    env = {k: v for k, v in os.environ.items() if not k.startswith("MICRODUCK_")}
    env.update(CHAIN_ENV)
    env.setdefault("WANDB_MODE", "disabled")
    if a.cpu:
        env["MICRO_SKATER_DEVICE"] = "cpu"
    print("[run_chain]", " ".join(cmd))
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    sys.exit(main())
