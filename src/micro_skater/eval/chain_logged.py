# micro-skater: part of the author's skateboard add-on for microduck_rl. Apache-2.0 (see LICENSE).
"""chain_play.py with the per-step logs saved: same arguments as chain_play.py.

Additionally writes <out>/chain_logs.npz (qpos, stage, rise and board xyz per step for every env, episode end and
reason, dt), which tools/isaac_render/make_bloopers.py reads. With MICRODUCK_CHAIN_TAU_LOG=1 it also records, at
physics-substep resolution, the robot's motor torque (N*m, output side = kt*I), the joint velocity and the supply
voltage after sag -> <out>/chain_tau.npz (tau, vel: (S, N, 14) float16; vin: (S, N); step_sub: the substep index at
each policy call; joint names; max_current). The final evaluation ran through this script.

    python -m micro_skater.eval.chain_logged <chain_play arguments>     (or: scripts/eval_chain.py --logs)
"""
from __future__ import annotations

import os
import sys

import numpy as np

from micro_skater.eval import chain_play as cp
from micro_skater.eval import sequence_eval as se

_orig = se.run_spawn
TAU_LOG = os.environ.get("MICRODUCK_CHAIN_TAU_LOG", "") not in ("", "0")
_TAU = dict(tau=[], vel=[], vin=[], marks=[], names=None, imax=None)

if TAU_LOG:
    from micro_skater.actuator.friction_dr_bam import FrictionDRBamActuator

    _orig_compute = FrictionDRBamActuator.compute

    def _compute_logged(self, cmd):
        tau = _orig_compute(self, cmd)
        if tau.shape[-1] == 14:                          # the robot's 14 servos (not the board / other entities)
            if _TAU["names"] is None:
                _TAU["names"] = list(self._target_names)
                _TAU["imax"] = self._bam_model.actuator.max_current
            vin = self._bam_model.actuator.vin           # (N, 1) per-env effective vin set in this compute (with sag)
            _TAU["tau"].append(tau.detach().half())
            _TAU["vel"].append(cmd.vel.detach().half())
            _TAU["vin"].append(vin.detach().reshape(-1).half())
        return tau

    FrictionDRBamActuator.compute = _compute_logged


def _out_dir():
    a = sys.argv
    if "--out" in a:
        return a[a.index("--out") + 1]
    lab = a[a.index("--label") + 1] if "--label" in a else "chain"
    return os.path.join(cp.OUT_ROOT, f"chain_{lab}")


def run_spawn(vec, policy, spawn, seconds):
    pol = policy
    if TAU_LOG:
        def pol(obs):                                    # mark the substep index at every policy call
            _TAU["marks"].append(len(_TAU["tau"]))
            return policy(obs)
    L, end, reason, apex_n, apex_rise = _orig(vec, pol, spawn, seconds)
    out = _out_dir()
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, "chain_logs.npz"), qpos=L["qpos"].astype(np.float32), stage=L["stage"],
                        rise=L["rise"], xyz=L["x"], end=end, reason=np.array(reason), dt=vec.unwrapped.step_dt)
    print(f"[logged] wrote {out}/chain_logs.npz", flush=True)
    if TAU_LOG and _TAU["tau"]:
        import torch
        marks = _TAU["marks"] + [len(_TAU["tau"])]
        np.savez_compressed(os.path.join(out, "chain_tau.npz"),
                            tau=torch.stack(_TAU["tau"]).cpu().numpy(), vel=torch.stack(_TAU["vel"]).cpu().numpy(),
                            vin=torch.stack(_TAU["vin"]).cpu().numpy(), step_sub=np.array(marks),
                            names=np.array(_TAU["names"]), max_current=np.float32(_TAU["imax"] or np.nan),
                            physics_dt=vec.unwrapped.physics_dt if hasattr(vec.unwrapped, "physics_dt") else np.nan)
        print(f"[logged] wrote {out}/chain_tau.npz ({len(_TAU['tau'])} substeps, max_current {_TAU['imax']})",
              flush=True)
    return L, end, reason, apex_n, apex_rise


se.run_spawn = run_spawn
if __name__ == "__main__":
    cp.main()
