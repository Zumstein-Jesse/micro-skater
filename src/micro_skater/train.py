# micro-skater: part of the author's skateboard add-on for microduck_rl. Apache-2.0 (see LICENSE).
"""``python -m micro_skater.train <TASK_ID> [mjlab train args] [--init-checkpoint PATH]``

Same as microduck_rl's ``train`` command. It exists because microduck_rl only handles ``--init-checkpoint`` when the
recognise), because it registers this package's tasks even without an installed entry point (PYTHONPATH checkout), and because
importing them installs the loader patch for the released checkpoints (no optimizer state).
"""
from __future__ import annotations

import sys


def main() -> int | None:
    sys.argv[0] = "train"                       # what microduck_rl's train hook expects (before mjlab is imported)
    import micro_skater.tasks  # noqa: F401  (registers the tasks even when the package is only on PYTHONPATH)
    from mjlab_microduck.train_cli import main as microduck_train_main

    return microduck_train_main()


if __name__ == "__main__":
    sys.exit(main())
