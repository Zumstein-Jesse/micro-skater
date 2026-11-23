# Released checkpoints

Seven policies, one folder each. Every `model.pt` is a **stripped copy** of the training checkpoint:

- **Kept:** actor and critic weights, including their observation normalizers, plus the iteration number and the env
  step counter. The step counter tells the chain eval which curriculum stage the policy was trained at.
- **Removed:** optimizer state, pickled configs, file paths and machine names.
- The files load with `torch.load(path, weights_only=True)`.

Next to each `model.pt`:

- `meta.json`: task id, iteration, observation and action sizes, network shape.
- `params/env.yaml` and `params/agent.yaml`: the resolved training configs of the final run. Data file paths are
  rewritten to `micro_skater/data/...`. `micro_skater.eval.chain_play` reads `env.yaml` to replay the settings each
  policy was trained with, so keep `params/` next to the model.
  - They are the configs as resolved at **iteration 0** of that run; curricula change many weights later (see
    [docs/rewards.md](../docs/rewards.md)).
  - Python tags (`!!python/name:...`) point to `micro_skater.*` for functions this repo adds, and to
    `mjlab_microduck.*` / `mjlab.*` for upstream ones, so they resolve with both packages importable.
  - Plain `yaml.unsafe_load` of an `env.yaml` still fails on two entries that mjlab 1.3.0 itself writes (the nested
    enum `ViewerConfig.OriginType` and a default lambda), as for any mjlab run folder. `chain_play` uses a safe loader
    that ignores python tags.

| Folder | Policy | Task id | Iteration | Actor obs → actions | Critic obs |
|---|---|---|---|---|---|
| `push_p16/` | Phase 1: push from rest, step on, ride (p16) | `Mjlab-Skate-SeqPush-MicroDuck` | 5644 | 114 → 14 | 127 |
| `manual_m6/` | Phase 2: manual on the back wheels (m6) | `Mjlab-Skate-SeqManual-MicroDuck` | 2597 | 114 → 14 | 127 |
| `halfpipe_r10/` | Phase 3: exit, carve in, ride the half-pipe walls (r10) | `Mjlab-Skate-SeqRamp-MicroDuck` | 4999 | 114 → 14 | 127 |
| `footpush_p17e/` | Phase 4: one foot-push stroke on the pipe floor (p17e) | `Mjlab-Skate-SeqPush-MicroDuck` (p17 flags on) | 8439 | 114 → 14 | 127 |
| `runner_run_v2/` | Running (play-sim Shift-run), run_v2 | `Mjlab-Velocity-Run-Flat-MicroDuck` | 6750 | 61 → 14 | 76 |
| `skater_v7f/` | Keyboard play-sim skater, v7f | `Mjlab-Skate-Flat-MicroDuck` | 4350 | 79 → 14 | 114 |
| `walker/` | Walking: trained with microduck_rl's stock velocity task, unchanged | `Mjlab-Velocity-Flat-MicroDuck` (from microduck_rl) | 4999 | 61 → 14 | 76 |

All actors are MLPs with layers 512-256-128, ELU activations and an empirical observation normalizer. They run at
50 Hz and output 14 joint-position targets.

- **The phase policies** (p16, m6, r10, p17e) expect the realism env vars of their training run. Those are listed in
  `src/micro_skater/launch.py`. `scripts/eval_chain.py` sets them for you.
- **The skate task (v7f)** uses `walker/` to drive the walk-up to the board. `MICRO_SKATER_WALKER` points it at another
  file.

## Using them

```bash
# the four-policy chain, as in the final eval
python scripts/eval_chain.py --n 32 --seed 3
# look at a single policy in mjlab's viewer
python -m micro_skater.play Mjlab-Velocity-Run-Flat-MicroDuck --checkpoint-file checkpoints/runner_run_v2/model.pt
# keep training a released policy (resumes iteration + curriculum, fresh optimizer)
python scripts/train_r10.py --iterations 200
```

If you pass these files to `--init-checkpoint` or to `runner.load()`, the missing optimizer state is handled by
`micro_skater.checkpoint_compat`: actor, critic and iteration load, and the optimizer starts fresh. This patch is
installed when `micro_skater.tasks` is imported.

## Not released

The intermediate stages (p1–p15, p17–p17d, m1–m5, r1–r9, run_v1, skate v1–v7e) and the optimizer states are not
released. To retrain a stage from scratch you would rebuild the chain of stages; see `docs/training_history.md`.

## License

The weights are released under Apache-2.0, like the code (see `../LICENSE`). They were trained in simulation only and
have never run on the real robot.
