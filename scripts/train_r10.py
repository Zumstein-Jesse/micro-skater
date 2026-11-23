# micro-skater. Apache-2.0 (see LICENSE).
"""Train / continue the phase 3 half-pipe specialist (r10).

    python scripts/train_r10.py [--dry-run] [--smoke] [--init CKPT] [--iterations N] [--num-envs N]

Recipe (task, env vars, resume point): micro_skater/launch.py, STAGES["r10"].
"""
import sys

from micro_skater.launch import main

if __name__ == "__main__":
    sys.exit(main("r10"))
