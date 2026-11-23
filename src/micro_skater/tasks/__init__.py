# micro-skater: part of the author's skateboard add-on for microduck_rl. Apache-2.0 (see LICENSE).
"""Register the micro-skater tasks with mjlab, on top of microduck_rl.

mjlab imports this module through the ``mjlab.tasks`` entry point (pyproject.toml), so ``train <TASK_ID>`` and
``play <TASK_ID>`` see these ids once the package is installed (or on PYTHONPATH and imported).

| task id                                    | what it trains                                              |
|--------------------------------------------|-------------------------------------------------------------|
| Mjlab-Velocity-Run-Flat-MicroDuck          | running: the stock velocity recipe with fast forward commands |
| Mjlab-Skate-Flat-MicroDuck                 | the keyboard "play sim" skater (v7f): walk up, step on, push, ride, carve, step off |
| Mjlab-Skate-Trick-DoubleKick-MicroDuck     | tricks on a double-kick board, sideways stance (exploration) |
| Mjlab-Skate-Trick-DoubleKickFwd-MicroDuck  | tricks, forward stance (manual, kick-turn)                  |
| Mjlab-Skate-Sequence-MicroDuck             | the one-run sequence task (the chain eval builds this one)  |
| Mjlab-Skate-SeqPush-MicroDuck              | phase 1 push specialist (p16) and the foot-push stroke (p17e) |
| Mjlab-Skate-SeqManual-MicroDuck            | phase 2 manual specialist (m6)                              |
| Mjlab-Skate-SeqRamp-MicroDuck              | phase 3 half-pipe specialist (r10)                          |
"""
from __future__ import annotations

from mjlab.tasks.registry import list_tasks, register_mjlab_task

from micro_skater import checkpoint_compat

checkpoint_compat.install()  # the released checkpoints have no optimizer state

# microduck_rl's runner (importing its tasks package also registers its own tasks and its --init-checkpoint hook).
from mjlab_microduck.tasks import MicroduckOnPolicyRunner


def _register(task_id: str, make_env, rl_cfg) -> None:
    if task_id in list_tasks():  # e.g. an older local install that already defines the same id
        print(f"[micro_skater] task {task_id} is already registered: kept the existing one")
        return
    register_mjlab_task(
        task_id=task_id,
        env_cfg=make_env(),
        play_env_cfg=make_env(play=True),
        rl_cfg=rl_cfg,
        runner_cls=MicroduckOnPolicyRunner,
    )


from .microduck_run_env_cfg import MicroduckRunRlCfg, make_microduck_run_env_cfg  # noqa: E402
from .microduck_skate_env_cfg import MicroduckSkateRlCfg, make_microduck_skate_env_cfg  # noqa: E402
from .microduck_skate_trick_env_cfg import (  # noqa: E402
    make_microduck_skate_trick_env_cfg,
    make_microduck_skate_trick_fwd_env_cfg,
    make_microduck_skate_trick_ppo_runner_cfg,
)
from .microduck_skate_sequence_env_cfg import (  # noqa: E402
    make_microduck_skate_sequence_env_cfg,
    make_microduck_skate_sequence_ppo_runner_cfg,
)
from .microduck_skate_seq_push_env_cfg import (  # noqa: E402
    make_microduck_skate_seq_push_env_cfg,
    make_microduck_skate_seq_push_ppo_runner_cfg,
)
from .microduck_skate_seq_manual_env_cfg import (  # noqa: E402
    make_microduck_skate_seq_manual_env_cfg,
    make_microduck_skate_seq_manual_ppo_runner_cfg,
)
from .microduck_skate_seq_ramp_env_cfg import (  # noqa: E402
    make_microduck_skate_seq_ramp_env_cfg,
    make_microduck_skate_seq_ramp_ppo_runner_cfg,
)

_register("Mjlab-Velocity-Run-Flat-MicroDuck", make_microduck_run_env_cfg, MicroduckRunRlCfg)
_register("Mjlab-Skate-Flat-MicroDuck", make_microduck_skate_env_cfg, MicroduckSkateRlCfg)
_register("Mjlab-Skate-Trick-DoubleKick-MicroDuck", make_microduck_skate_trick_env_cfg,
          make_microduck_skate_trick_ppo_runner_cfg())
_register("Mjlab-Skate-Trick-DoubleKickFwd-MicroDuck", make_microduck_skate_trick_fwd_env_cfg,
          make_microduck_skate_trick_ppo_runner_cfg("skate_trick_fwd"))
_register("Mjlab-Skate-Sequence-MicroDuck", make_microduck_skate_sequence_env_cfg,
          make_microduck_skate_sequence_ppo_runner_cfg())
_register("Mjlab-Skate-SeqPush-MicroDuck", make_microduck_skate_seq_push_env_cfg,
          make_microduck_skate_seq_push_ppo_runner_cfg())
_register("Mjlab-Skate-SeqManual-MicroDuck", make_microduck_skate_seq_manual_env_cfg,
          make_microduck_skate_seq_manual_ppo_runner_cfg())
_register("Mjlab-Skate-SeqRamp-MicroDuck", make_microduck_skate_seq_ramp_env_cfg,
          make_microduck_skate_seq_ramp_ppo_runner_cfg())

TASK_IDS = (
    "Mjlab-Velocity-Run-Flat-MicroDuck",
    "Mjlab-Skate-Flat-MicroDuck",
    "Mjlab-Skate-Trick-DoubleKick-MicroDuck",
    "Mjlab-Skate-Trick-DoubleKickFwd-MicroDuck",
    "Mjlab-Skate-Sequence-MicroDuck",
    "Mjlab-Skate-SeqPush-MicroDuck",
    "Mjlab-Skate-SeqManual-MicroDuck",
    "Mjlab-Skate-SeqRamp-MicroDuck",
)
