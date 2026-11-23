# micro-skater tools/isaac_render: the author's Isaac Sim replay pipeline. Apache-2.0 (see LICENSE).
"""Save the chain env's compiled MuJoCo model (robot + sequence skateboard + half-pipe terrain) next to a
demo_qpos.npz, for export_chain.py (train venv; builds the SeqRamp chain env cfg with 1 env, no policies).

    python chain_model.py out\\chain_XXX\\demo_qpos.npz [--device cpu]

Only needed when the board XML / pipe changed since the last model.mjb (export_chain.py checks the board geoms against
skateboard_sequence.xml and fails loudly when stale). Uses the env cfg chain_play.build_chain uses
(make_microduck_skate_seq_ramp_env_cfg(play=True, chain=True)); the terrain does not depend on the checkpoints.
--device cpu keeps it off the GPU (MuJoCo Warp on CPU; slower, but one env and no stepping).
"""
from __future__ import annotations

import argparse
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("qpos")
ap.add_argument("--device", default="cpu")
a = ap.parse_args()

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import micro_skater.tasks  # noqa: E402,F401
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from micro_skater.tasks.microduck_skate_seq_ramp_env_cfg import make_microduck_skate_seq_ramp_env_cfg  # noqa: E402

cfg = make_microduck_skate_seq_ramp_env_cfg(play=True, chain=True)
cfg.scene.num_envs = 1
env = ManagerBasedRlEnv(cfg=cfg, device=a.device)
m = env.sim.mj_model
z = np.load(a.qpos)
jn = [m.joint(j).name for j in range(m.njnt)]
assert list(map(str, z["jnt_names"])) == jn and np.array_equal(z["jnt_qposadr"], m.jnt_qposadr), \
    "the env's joints differ from demo_qpos.npz"
dst = os.path.join(os.path.dirname(os.path.abspath(a.qpos)), "model.mjb")
mujoco.mj_saveModel(m, dst)
print(f"saved {dst}: nq {m.nq}, ngeom {m.ngeom}")
