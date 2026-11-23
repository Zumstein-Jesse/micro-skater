# How to run, in detail

← [back to README](../README.md)

## What was tested, and how

Everything below was checked on Windows 11 with Python 3.12 and microduck_rl's own venv (mjlab 1.3.0, rsl_rl 5.0.1,
torch 2.9.1, MuJoCo Warp). microduck_rl was a clean copy of commit `bae39f5`, and this repo was on `PYTHONPATH`.

| Check | Result |
|---|---|
| `import micro_skater.tasks` registers all 8 task ids | yes |
| Each of the 7 checkpoints builds its task env **on the CPU** (1 env), loads through the task's runner as `play` does, and takes 5 steps | yes: obs sizes 114 / 79 / 61, 14 finite actions |
| The same checkpoints load the way a resume does (`runner.load(path)`, no optimizer state in the file) | yes: actor identical, iteration restored |
| `scripts/eval_chain.py --cpu --n 4 --seconds 30 --no-render` (seed 3) | 4/4 reached the half-pipe, 3 foot-push strokes, 2/4 complete; 3.5 min |
| `python -m micro_skater.train ... --init-checkpoint X` consumes the flag | yes |
| `scripts/train_*.py --dry-run` prints the original runs' env vars | yes |

**Not tested yet (TODO):**

- `uv pip install -e micro-skater` into a fresh microduck_rl checkout. The package metadata is there, but only the
  `PYTHONPATH` route was tested.
- An actual training run from this repo.
- `micro_skater.play` with a viewer or `--video`.
- A rendered (GPU) chain eval from this repo.
- Linux.

## Commands

All commands run from the repo root with microduck_rl's venv Python. Set `WANDB_MODE=disabled` unless you use W&B.

### Chain evaluation

```bash
python scripts/eval_chain.py [--n 32] [--seed 3] [--seconds 30] [--out out/chain_s3] [--no-render] [--logs] [--cpu]
                             [--push ...] [--manual ...] [--ramp ...] [--stroke ...]   # other checkpoints
```

The script sets the final eval's env vars (`micro_skater/eval/run_chain.py`, `CHAIN_ENV`) and the scheduler arguments:

- manual clock 4 s;
- the stance gate;
- foot-push window 0.20–0.75 m/s;
- 3 s foot-push deadline;
- board rolling resistance 0.3.

It then runs `python -m micro_skater.eval.chain_play`.

Read `funnel.json` → `agg.chain_complete` (the share of runs that finish) and `agg.stroke` (the foot-push
statistics). The final results in the README come from 6 such batches of 32 runs, with seeds 3, 4, 5, 5, 5 and 5: seed
5 was rerun three times, and GPU physics is not bit-deterministic, so each rerun is a different sample. Seeds 0–2 were
used to record the chain hand-over bank that the foot-push policy (p17d, then p17e) trained on, so they are not
held-out seeds.

`--no-render` only skips the MuJoCo videos: `demo_qpos.npz` (the demo run's joint states) and `model.mjb` (the compiled
model) are still written; they are the input of `tools/isaac_render/`. `--logs` also writes `chain_logs.npz` (per-step
logs of every run, for `tools/isaac_render/make_bloopers.py`).

### Single-policy evaluation

`python -m micro_skater.eval.sequence_eval CKPT --task <task id> --n 64` runs per-spawn test batteries for one
specialist, for example push only or manual only. Set the policy's env vars first; `scripts/train_<x>.py --dry-run`
prints them.

### Training

`src/micro_skater/launch.py` holds one recipe per released policy:

- the task id;
- the env vars of the final run;
- the parallel env count, iterations and save interval;
- the stage it was resumed from.

| Script | Task | Envs × iterations | Original run resumed from |
|---|---|---|---|
| `train_p16.py` | `Mjlab-Skate-SeqPush-MicroDuck` | 2048 × 1000 | p15 iteration 4645 |
| `train_m6.py` | `Mjlab-Skate-SeqManual-MicroDuck` | 2048 × 600 | m5 iteration 1998 |
| `train_r10.py` | `Mjlab-Skate-SeqRamp-MicroDuck` | 2048 × 500 | r9 iteration 4500 |
| `train_p17e.py` | `Mjlab-Skate-SeqPush-MicroDuck` | 2048 × 600 | p17d iteration 7840 |
| `train_runner.py` | `Mjlab-Velocity-Run-Flat-MicroDuck` | 4096 × 750 | run_v1, iteration 6498 |
| `train_v7f.py` | `Mjlab-Skate-Flat-MicroDuck` | 4096 × 500 | skate v7e iteration 4200 |

The predecessor checkpoints are not released. By default each script resumes from the **released** checkpoint, which
continues training the final policy. `--init PATH` resumes from a different checkpoint of the same task.

Curricula are keyed to absolute env steps (`iteration × 24`), and a resume restores the step counter. Starting from a
fresh network therefore needs the whole stage chain; [training_history.md](training_history.md) describes it.

Flags: `--smoke` (64 envs, 5 iterations), `--dry-run`, `--iterations N`, `--num-envs N`, `--run-name NAME`. Any
other argument is passed on to mjlab's `train`.

### Environment variables

| Variable | Meaning |
|---|---|
| `MICRODUCK_*` | Task switches, read when the task configs are imported. The launch scripts set them; leftover values in your shell are cleared |
| `MICRO_SKATER_CHECKPOINTS` | Folder of the released checkpoints (default `checkpoints/`) |
| `MICRO_SKATER_WALKER` | The walking policy the skate task's walk-up uses (default `checkpoints/walker/model.pt`) |
| `MICRO_SKATER_DEVICE` | Device of the chain / sequence eval (default `cuda:0`; `eval_chain.py --cpu` sets `cpu`) |

## Notes on the code

- Comments in the task files name earlier training stages (p9b, m4, r7, ...) and the checkpoints that were measured
  (`model_1499` and so on). Those runs are not released; the version story is in
  [training_history.md](training_history.md).
- `micro_skater.tasks.mdp` and `micro_skater.robot.microduck_constants` re-export microduck_rl's modules of the same
  name and add the skate terms. Nothing of microduck_rl is copied.
- `micro_skater.actuator.FrictionDRBamActuator` subclasses microduck_rl's BAM actuator. It adds the current limit
  override, the datasheet back-EMF and the current log used by the heat cost.
- Some code paths are for experiments that are off by default (the sequence task's older stages, the r9c bank, the
  student and pump hooks in `mdp.py`). They are kept so the released configs resolve exactly as trained. Data files
  that only the off-by-default r9c experiment uses (`bank_strokes.npz`) are not shipped.
