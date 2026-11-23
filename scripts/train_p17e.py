# micro-skater. Apache-2.0 (see LICENSE).
"""Train / continue the phase 4 foot-push stroke specialist (p17e).

    python scripts/train_p17e.py [--dry-run] [--smoke] [--init CKPT] [--iterations N] [--num-envs N]

Recipe (task, env vars, resume point): micro_skater/launch.py, STAGES["p17e"].
"""
import sys

from micro_skater.launch import main

if __name__ == "__main__":
    sys.exit(main("p17e"))
