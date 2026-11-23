# micro-skater. Apache-2.0 (see LICENSE).
"""Train / continue the keyboard play-sim skater (v7f).

    python scripts/train_v7f.py [--dry-run] [--smoke] [--init CKPT] [--iterations N] [--num-envs N]

Recipe (task, env vars, resume point): micro_skater/launch.py, STAGES["v7f"].
"""
import sys

from micro_skater.launch import main

if __name__ == "__main__":
    sys.exit(main("v7f"))
