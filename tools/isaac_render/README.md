# Isaac Sim replay pipeline (optional, Windows only)

These scripts made the skatepark videos in `media/` (`hero.gif`, `phase*.gif`, `blooper_*.gif`, `full_run_*.mp4`).

**This is not physics.** Isaac Sim only replays motion that MuJoCo already computed:

1. A chain eval records the joint and board states of its demo run. `scripts/eval_chain.py` writes `demo_qpos.npz`
   and `model.mjb`, with or without `--no-render`.
2. The pipeline bakes those frames into a USD scene and renders them.
3. A checker compares the baked replay with MuJoCo.

**Windows only.** The driver runs the Isaac launcher through `cmd /c`, waits for the GPU with `tasklist` /
`nvidia-smi`, and expects the default usd-core venv at `.usdvenv\Scripts\python.exe`. Porting it to Linux means
changing those three places. It has not been tried.

> **Not included:** NVIDIA's scene assets are not included. That covers the vegetation USDs, the MDL materials and
> textures under `assets/skatepark/`, and the sky HDRIs (`assets/skatepark/hdri/`). `build_park.py` references the
> vegetation from NVIDIA's public asset server at render time (`--no-remote` skips it). The materials, textures and
> HDRIs have to be in `assets/skatepark/` locally. You need your own Isaac Sim install (tested with 6.x), and you must
> follow the license terms of the assets you use. The robot meshes come from the installed microduck_rl package
> (CC BY-NC-SA 4.0), so renders of the robot are CC BY-NC-SA 4.0 too.

## One-time setup

All paths are relative to the **repo root**: `assets/`, `scenes/` and `output/` are created there (they are in
`.gitignore`). Work files of a run go to `tools/isaac_render/chain_cine/<label>/`.

1. **A usd-core python** for every step except render: `python -m venv tools\isaac_render\.usdvenv` and then
   `tools\isaac_render\.usdvenv\Scripts\pip install usd-core numpy pillow`. Or point `MICRO_SKATER_USD_PY` at any
   python that has those packages.
2. **The robot as USD.** `bake_replay.py` reads
   `assets/microduck_usd/robot_allcollisions/robot_allcollisions.usda`. It is not shipped. Generate it once with Isaac
   Sim's MJCF importer from the microduck_rl robot model:

   ```bat
   <isaac-sim>\python.bat <isaac-sim>\standalone_examples\api\isaacsim.asset.importer.mjcf\mjcf_import.py ^
       --mjcf <microduck_rl>\src\mjlab_microduck\robot\microduck\robot_allcollisions.xml ^
       --usd-path assets\microduck_usd --no-fix-base
   ```

   Run it from the repo root. `MICRODUCK_ROBOT_USD` overrides the path. The importer's script location can change
   between Isaac Sim versions.
3. **Isaac Sim's python launcher** for the render step: set `ISAAC_SIM_PYTHON=<isaac-sim>\python.bat` or pass
   `--isaac <launcher>` to `chain_cine.py`. There is no default; the render step stops with an error without it.
4. The NVIDIA scene assets above, in `assets/skatepark/`.

## Pieces

| Step | Script | Runs with | Output |
|---|---|---|---|
| export | `export_chain.py` (+ `chain_model.py` for a fresh `model.mjb`) | the training venv | per-frame link poses (`chain.kin.npz`), `pipe.json` |
| trim | `pipe_trim.py` | usd-core python | `pipe_render.json` (the pipe cut to the part the run uses) |
| park | `build_park.py`, `make_textures.py` | usd-core python | the skatepark USD |
| edit | `edit_cine.py` (helpers in `cine_geom.py`), `make_bloopers.py` | usd-core python | timelines (cuts, cameras, slow motion) |
| bake | `bake_replay.py` | usd-core python | `replay_<cut>_<fmt>.usda` |
| check | `check_chain.py` | usd-core python | replay vs MuJoCo error (the released videos: max 0.34 mm) |
| render | `render_replay.py` | **Isaac Sim python** | frames |
| compose | `compose_cine.py` | usd-core python + Pillow + ffmpeg | the mp4s |

`make_bloopers.py` needs the per-step logs of every run in the eval: run the eval with
`scripts/eval_chain.py --logs`, which also writes `chain_logs.npz`.

`chain_cine.py` runs all of them in order:

```bat
set MICRO_SKATER_TRAIN_PY=<python of the training venv>        & rem default: the python running chain_cine.py
set MICRO_SKATER_USD_PY=<python with usd-core, numpy, pillow>  & rem default: tools\isaac_render\.usdvenv
set ISAAC_SIM_PYTHON=<isaac-sim>\python.bat                    & rem or pass --isaac
python tools\isaac_render\chain_cine.py out\chain_s3\demo_qpos.npz --label demo --cuts standard --formats 4x5
```

The scripts were moved here from the author's working folder and their machine paths removed. The full pipeline has
**not** been re-run from this location (TODO). The MuJoCo-only renders in `media/mj_*.gif` come from
`tools/render_takes_mujoco.py` instead and need no Isaac Sim.
