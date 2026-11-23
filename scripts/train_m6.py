# micro-skater. Apache-2.0 (see LICENSE).
"""Train / continue the phase 2 manual specialist (m6).

    python scripts/train_m6.py [--dry-run] [--smoke] [--init CKPT] [--iterations N] [--num-envs N]

Recipe (task, env vars, resume point): micro_skater/launch.py, STAGES["m6"].
"""
import sys

from micro_skater.launch import main

if __name__ == "__main__":
    sys.exit(main("m6"))
