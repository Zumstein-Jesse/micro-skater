# micro-skater: part of the author's skateboard add-on for microduck_rl. Apache-2.0 (see LICENSE).
"""Let rsl_rl load the released (stripped) checkpoints.

The files in checkpoints/ hold the actor and critic weights (with their observation normalizers), the iteration and
the env step counter, but no optimizer state. ``PPO.load`` with no ``load_cfg`` (what a resume / ``--init-checkpoint``
does) would look for ``optimizer_state_dict`` and fail; this patch then loads actor + critic + iteration and keeps the
fresh optimizer. Full checkpoints (with an optimizer) load exactly as before.
"""
from __future__ import annotations

_PATCHED = False


def install() -> None:
    global _PATCHED
    if _PATCHED:
        return
    from rsl_rl.algorithms import PPO

    orig = PPO.load

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        if load_cfg is None and "optimizer_state_dict" not in loaded_dict:
            print("[micro_skater] checkpoint has no optimizer state (released file): loading actor + critic, "
                  "fresh optimizer")
            load_cfg = {"actor": True, "critic": True, "optimizer": False, "iteration": True, "rnd": False}
        return orig(self, loaded_dict, load_cfg, strict)

    PPO.load = load
    _PATCHED = True
