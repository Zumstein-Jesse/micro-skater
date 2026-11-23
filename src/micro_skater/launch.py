# micro-skater: part of the author's skateboard add-on for microduck_rl. Apache-2.0 (see LICENSE).
"""Training launch recipes for the released policies (used by scripts/train_*.py).

Each released policy was the LAST stage of a long chain of resumed runs (p1 -> ... -> p16, r1 -> ... -> r10, ...).
Every stage kept the earlier stages' switches on, so a stage is "the task + this set of env vars + resume from the
previous stage's checkpoint". The env vars below are exactly the ones the final runs used (copied from their run
logs); docs/training_history.md tells the story of the stages.

What you can run from this repo:
  * by default the recipe resumes from the RELEASED checkpoint of that stage (checkpoints/<name>/model.pt), i.e. it
    continues training the final policy under its final curriculum (``--iterations`` more iterations);
  * ``--init PATH`` resumes from any other checkpoint of the same task (for example your own previous stage);
  * the earlier stages' checkpoints are not released, so re-running the whole chain from scratch means re-creating
    every stage (see docs/training_history.md). That is a multi-day GPU job and has not been re-tested from this repo.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys

from micro_skater.paths import checkpoint_path, data_path

# Realism switches shared by every phase policy (docs/realism.md).
REALISM = {
    "MICRODUCK_BAM_MAX_CURRENT": "1.0",          # firmware current limit 1000 mA (BAM default 1.75 A)
    "MICRODUCK_BAM_DATASHEET_SPEED": "1",        # datasheet no-load speed (BAM was ~25 % fast)
    "MICRODUCK_SKATE_HEAT": "1",                 # servo heat cost on
    "MICRODUCK_SKATE_HEAT_W": "-0.3",
    "MICRODUCK_SKATE_MASS_RANGE": "0.98,1.10",   # whole-robot mass randomisation
}
PUSH_STAGES = {
    "MICRODUCK_SEQPUSH_P10_IT0": "2100", "MICRODUCK_SEQPUSH_HEADING": "1", "MICRODUCK_SEQPUSH_HEAD_IT0": "2200",
    "MICRODUCK_SEQPUSH_P12": "1", "MICRODUCK_SEQPUSH_P12_IT0": "2800",
    "MICRODUCK_SEQPUSH_P13": "1", "MICRODUCK_SEQPUSH_P13_IT0": "3406",
    "MICRODUCK_SEQPUSH_P14": "1", "MICRODUCK_SEQPUSH_P14_IT0": "3806",
    "MICRODUCK_SEQPUSH_P15": "1", "MICRODUCK_SEQPUSH_P15_IT0": "4256",
    "MICRODUCK_SEQPUSH_P16": "1", "MICRODUCK_SEQPUSH_P16_IT0": "4656",
}
MANUAL_STAGES = {
    "MICRODUCK_SEQMANUAL_HEADING": "1", "MICRODUCK_SEQMANUAL_HEAD_IT0": "1000",
    "MICRODUCK_SEQMANUAL_M5": "1", "MICRODUCK_SEQMANUAL_M5_IT0": "1500",
    "MICRODUCK_SEQMANUAL_M6": "1", "MICRODUCK_SEQMANUAL_M6_IT0": "2002",
}
RAMP_STAGES = {
    "MICRODUCK_RAMP_R2_IT0": "600", "MICRODUCK_RAMP_R3_IT0": "900",
    "MICRODUCK_RAMP_PUMP": "1", "MICRODUCK_RAMP_PUMP_IT0": "1500",
    "MICRODUCK_RAMP_R5": "1", "MICRODUCK_RAMP_R5_IT0": "2099", "MICRODUCK_RAMP_R5PUMP": "1",
    "MICRODUCK_RAMP_R6": "1", "MICRODUCK_RAMP_R6_IT0": "2798",
    "MICRODUCK_RAMP_R7": "1", "MICRODUCK_RAMP_R7_IT0": "3597",
    "MICRODUCK_RAMP_R9": "1", "MICRODUCK_RAMP_R9_IT0": "4303",
}

STAGES = {
    "p16": dict(
        task="Mjlab-Skate-SeqPush-MicroDuck", ckpt="push_p16", envs=2048, iterations=1000, save=100,
        resumed_from="p15 iteration 4645 (not released)",
        env={**PUSH_STAGES, "MICRODUCK_SEQPUSH_P16_AMPS": "1", **REALISM}),
    "m6": dict(
        task="Mjlab-Skate-SeqManual-MicroDuck", ckpt="manual_m6", envs=2048, iterations=600, save=100,
        resumed_from="m5 iteration 1998 (not released)",
        env={**MANUAL_STAGES, "MICRODUCK_SEQMANUAL_HANDOFF": str(data_path("bank_p13.npz")), **REALISM}),
    "r10": dict(
        task="Mjlab-Skate-SeqRamp-MicroDuck", ckpt="halfpipe_r10", envs=2048, iterations=500, save=50,
        resumed_from="r9 iteration 4500 (not released)",
        env={**RAMP_STAGES, "MICRODUCK_RAMP_HANDOFF": str(data_path("ramp_bank_p13.npz")),
             "MICRODUCK_RAMP_R10": "1", "MICRODUCK_RAMP_R10_IT0": "4509",
             "MICRODUCK_RAMP_R10_BANK": str(data_path("bank_hb.npz")), "MICRODUCK_RAMP_R10_SHARE": "0.35",
             "MICRODUCK_RAMP_R10_HYERR_W": "-0.1", "MICRODUCK_PIPE_H": "0.08", **REALISM}),
    "p17e": dict(
        task="Mjlab-Skate-SeqPush-MicroDuck", ckpt="footpush_p17e", envs=2048, iterations=600, save=100,
        resumed_from="p17d iteration 7840 (not released)",
        env={**PUSH_STAGES, "MICRODUCK_SEQPUSH_P16_AMPS": "1", "MICRODUCK_SEQPUSH_P16_AMPS_K0": "3",
             "MICRODUCK_SEQPUSH_P16_HEAT_ON": "1", "MICRODUCK_SEQPUSH_P17": "1", "MICRODUCK_SEQPUSH_P17_IT0": "5656",
             "MICRODUCK_SEQPUSH_P17B": "1", "MICRODUCK_SEQPUSH_P17B_GATE": "floor", "MICRODUCK_SEQPUSH_P17B_K0": "10",
             "MICRODUCK_SEQPUSH_P17C": "1", "MICRODUCK_SEQPUSH_P17D": "1", "MICRODUCK_SEQPUSH_P17E": "1",
             "MICRODUCK_SEQPUSH_P17E_LIMITS_W": "-0.75", "MICRODUCK_SEQPUSH_P17E_KNEE_W": "-0.45",
             "MICRODUCK_PIPE_H": "0.08", **REALISM}),
    "runner": dict(
        task="Mjlab-Velocity-Run-Flat-MicroDuck", ckpt="runner_run_v2", envs=4096, iterations=750, save=250,
        resumed_from="run_v1 final, iteration 6498 (not released)", env={}),
    "v7f": dict(
        task="Mjlab-Skate-Flat-MicroDuck", ckpt="skater_v7f", envs=4096, iterations=500, save=50,
        resumed_from="skate_v7e iteration 4200 (not released)", env={}),
}
# Switches of earlier experiments that must be OFF for these recipes (they default off; cleared in case your shell
# still has them from another run).
CLEAR = ("MICRODUCK_WARM_START",)


def main(policy: str | None = None, argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    if policy is None:
        ap.add_argument("policy", choices=sorted(STAGES))
    ap.add_argument("--init", default=None, help="checkpoint to resume from (default: the released one)")
    ap.add_argument("--iterations", type=int, default=None, help="iterations to train (default: the original run's)")
    ap.add_argument("--num-envs", type=int, default=None, help="parallel envs (default: the original run's)")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--smoke", action="store_true", help="64 envs, 5 iterations: check that everything builds")
    ap.add_argument("--dry-run", action="store_true", help="print the env vars and the command, do not run")
    a, extra = ap.parse_known_args(argv)
    name = policy or a.policy
    st = STAGES[name]
    init = a.init or str(checkpoint_path(f"{st['ckpt']}/model.pt"))
    envs = 64 if a.smoke else (a.num_envs or st["envs"])
    its = 5 if a.smoke else (a.iterations or st["iterations"])
    run_name = a.run_name or (f"{name}_smoke" if a.smoke else f"{name}_continued")
    cmd = [sys.executable, "-m", "micro_skater.train", st["task"], "--env.scene.num-envs", str(envs),
           "--agent.logger", "tensorboard", "--agent.run-name", run_name, "--agent.max-iterations", str(its),
           "--agent.save-interval", str(st["save"]), "--init-checkpoint", init, *extra]
    env = {k: v for k, v in os.environ.items() if not (k.startswith("MICRODUCK_") or k in CLEAR)}
    env.update(st["env"])
    env.setdefault("WANDB_MODE", "disabled")
    print(f"[launch] {name}: {st['task']}; the released run resumed from {st['resumed_from']}")
    for k in sorted(st["env"]):
        print(f"  {k}={st['env'][k]}")
    print("  " + " ".join(shlex.quote(c) for c in cmd))
    if a.dry_run:
        return 0
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    sys.exit(main())
