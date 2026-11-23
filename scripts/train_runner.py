# micro-skater. Apache-2.0 (see LICENSE).
"""Train / continue the running policy (play sim Shift-run) (runner).

    python scripts/train_runner.py [--dry-run] [--smoke] [--init CKPT] [--iterations N] [--num-envs N]

Recipe (task, env vars, resume point): micro_skater/launch.py, STAGES["runner"].
"""
import sys

from micro_skater.launch import main

if __name__ == "__main__":
    sys.exit(main("runner"))
