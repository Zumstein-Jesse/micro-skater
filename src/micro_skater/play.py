# micro-skater: part of the author's skateboard add-on for microduck_rl. Apache-2.0 (see LICENSE).
"""``python -m micro_skater.play <TASK_ID> --checkpoint-file checkpoints/<name>/model.pt [mjlab play args]``

mjlab's ``play`` (interactive viewer, or ``--video`` to record) with this package's tasks registered, also when the
package is only on PYTHONPATH.
"""
from __future__ import annotations

import sys


def main() -> int | None:
    import micro_skater.tasks  # noqa: F401
    from mjlab.scripts.play import main as mjlab_play_main

    return mjlab_play_main()


if __name__ == "__main__":
    sys.exit(main())
