# micro-skater: part of the author's skateboard add-on for microduck_rl. Apache-2.0 (see LICENSE).
"""Microduck run: the velocity (walking) recipe with a fast forward-command range, for the skate play sim's
Shift-to-run.

Resumed from the play walker (velocity 2026-11-01_19-01-57 model_4999) with --init-checkpoint, so every velocity
curriculum continues at its final stage. Changes vs the velocity recipe:

  - forward command range up to RUN_VX_MAX, with a run bucket (RUN_ENV_SHARE of non-standing resamples draw vx in
    RUN_VX_RANGE). NOTE (review): the rest draws vx from (-0.4, RUN_VX_MAX), not the walker's (-0.4, 0.4), and the
    bucket can also hit turn-in-place envs, so walker-range vx is ~37 % of resamples and backward ~19 % (walker ~50 %)
  - run_speed_tracking: a narrow forward-speed Gaussian on run commands only (the walker reached ~0.39 m/s at a
    1.0 m/s command: the shared std^2 0.1 tracking term is too flat to pay for the extra effort)

Obs (61D), actions, DR, noise, pushes and the robot model are the velocity recipe's: the policy is a drop-in walker.
The pose reward is unchanged: its speed regimes key on |v_xy| + |w_z|, which can't tell a run from a turning walk.
"""

from copy import deepcopy

from mjlab.managers import RewardTermCfg

from micro_skater.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    MicroduckRlCfg,
    make_microduck_velocity_env_cfg,
)

RUN_VX_MAX = 1.0
RUN_VX_RANGE = (0.5, RUN_VX_MAX)
RUN_ENV_SHARE = 0.35
RUN_VX_GATE = 0.45                # run_speed_tracking pays on commands above this
RUN_TRACK_STD = 0.2
RUN_TRACK_WEIGHT = 3.0
# v2 (resumed from run_v1 model_6498): yaw tracking while moving forward fast. run_v1 turned 0.16-0.25 rad/s on a
# 0.8 rad/s command at 0.75 m/s and walked turns at ~0.5 (the walker did 0.7-1.0). MICRODUCK_RUN_V2=0 turns it off.
import os as _os

RUN_V2 = _os.environ.get("MICRODUCK_RUN_V2", "1") != "0"
RUN_YAW_GATE = 0.3                # forward command above this (covers the play W walk at 0.4 and the run)
RUN_YAW_STD = 0.3
RUN_YAW_WEIGHT = 2.0


def make_microduck_run_env_cfg(play: bool = False):
    cfg = make_microduck_velocity_env_cfg(play=play)

    old = deepcopy(cfg.commands["twist"])
    cmd = microduck_mdp.RunVelocityCommandCfg(**{k: getattr(old, k) for k in vars(old)})
    cmd.ranges.lin_vel_x = (-0.4, RUN_VX_MAX)
    cmd.rel_run_envs = RUN_ENV_SHARE
    cmd.run_vx_range = RUN_VX_RANGE
    cfg.commands["twist"] = cmd

    cfg.rewards["run_speed_tracking"] = RewardTermCfg(
        func=microduck_mdp.run_speed_tracking,
        weight=RUN_TRACK_WEIGHT,
        params={"std": RUN_TRACK_STD, "command_name": "twist", "vx_gate": RUN_VX_GATE},
    )
    if RUN_V2:
        cfg.rewards["run_yaw_tracking"] = RewardTermCfg(
            func=microduck_mdp.run_yaw_tracking,
            weight=RUN_YAW_WEIGHT,
            params={"std": RUN_YAW_STD, "command_name": "twist", "vx_gate": RUN_YAW_GATE},
        )
    return cfg


MicroduckRunRlCfg = deepcopy(MicroduckRlCfg)
MicroduckRunRlCfg.experiment_name = "velocity_run"
MicroduckRunRlCfg.run_name = "velocity_run"
