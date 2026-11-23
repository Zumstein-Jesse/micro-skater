# micro-skater tools/isaac_render: the author's Isaac Sim replay pipeline. Apache-2.0 (see LICENSE).
r"""Bloopers for the chain cine cut: real failed runs of the SAME chain eval (chain_logs.npz from scripts/eval_chain.py --logs, all N envs),
exported like the demo (export_chain.py) so edit_cine.py --bloopers can append them before the end card.

    train venv:  python make_bloopers.py RUN_DIR --label NAME --eps 13,3,31 [--captions "First push,Foot push,Wall ride"]

Each episode is written as <video>\chain_cine\NAME\bloopers\ep<i>\demo_qpos.npz (qpos up to the step the chain
eval ended it: the run's own fall / tilt termination), then export_chain.py -> blooper.kin.npz there.
bloopers.json lists them (episode, reason from funnel.json, end time, caption). Nothing is simulated beyond what the
eval recorded: each clip ends where the eval ended that run (the cut freezes on that frame).
"""
import argparse, json, os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--label", required=True)
ap.add_argument("--eps", required=True)
ap.add_argument("--captions", default=None)
a = ap.parse_args()
L = np.load(os.path.join(a.run, "chain_logs.npz"))
demo = np.load(os.path.join(a.run, "demo_qpos.npz"))
funnel = json.load(open(os.path.join(a.run, "funnel.json")))["episodes"]
eps = [int(e) for e in a.eps.split(",")]
caps = a.captions.split(",") if a.captions else [""] * len(eps)
out_dir = os.path.join(HERE, "chain_cine", a.label, "bloopers")
items = []
for ep, cap in zip(eps, caps):
    n = int(L["end"][ep])
    assert not funnel[ep]["chain_complete"], f"ep {ep} completed the chain: not a blooper"
    d = os.path.join(out_dir, f"ep{ep}")
    os.makedirs(d, exist_ok=True)
    np.savez(os.path.join(d, "demo_qpos.npz"), qpos=L["qpos"][:n, ep], dt=L["dt"], stage=L["stage"][:n, ep],
             owner=np.zeros(n, np.int64), apex=np.zeros(n, bool), origin=demo["origin"],
             jnt_names=demo["jnt_names"], jnt_qposadr=demo["jnt_qposadr"], jnt_type=demo["jnt_type"])
    subprocess.run([sys.executable, os.path.join(HERE, "export_chain.py"), os.path.join(d, "demo_qpos.npz"),
                    "--model", os.path.join(a.run, "model.mjb"), "--out", os.path.join(d, "blooper")], check=True)
    items.append(dict(ep=ep, reason=L["reason"][ep].item(), end_s=(n - 1) * float(L["dt"]), steps=n, caption=cap,
                      squat=bool(funnel[ep].get("squat")), kin=os.path.join(d, "blooper.kin.npz")))
    print("blooper", items[-1])
json.dump(dict(run=a.run, n_eps=int(L["qpos"].shape[1]), items=items), open(os.path.join(out_dir, "bloopers.json"), "w"),
          indent=1)
