# micro-skater. Apache-2.0 (see LICENSE).
"""Train / continue the phase 1 push specialist (push from rest, step on, ride) (p16).

    python scripts/train_p16.py [--dry-run] [--smoke] [--init CKPT] [--iterations N] [--num-envs N]

Recipe (task, env vars, resume point): micro_skater/launch.py, STAGES["p16"].
"""
import sys

from micro_skater.launch import main

if __name__ == "__main__":
    sys.exit(main("p16"))
