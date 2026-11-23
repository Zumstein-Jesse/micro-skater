# micro-skater: part of the author's skateboard add-on for microduck_rl. Apache-2.0 (see LICENSE).
"""Play the three-specialist skate chain: push + step-on -> manual -> approach + half-pipe, one policy per scheduler phase.

    python chain_play.py --push CKPT --manual CKPT --ramp CKPT [--n 256] [--seconds 30] [--out DIR] [--label L]
                         [--blend 4] [--exit-gate wheels|stage] [--curriculum-step S] [--seed 0] [--no-render]
                         [--seq-override KEY=VALUE ...] [--spawn start] [--stroke CKPT]

The env is Mjlab-Skate-SeqRamp-MicroDuck's chain=True cfg (32 s episodes from
START, the push / manual specialists' scheduler settings, hold-ended MANUAL, the ramp specialist's entry line). One
runner loads the three checkpoints in turn; each loaded policy is DEEP-COPIED (ppo.get_policy() returns the actor
module itself, so three loads would otherwise leave three references to the last one).

Owner (per env, latched with max, only goes up; cleared where episode_length_buf == 0):
  0 push   -> 1 manual  when stage >= SETTLE
  1 manual -> 2 ramp    when stage == EXIT and (wheels_on >= 0.15 s or 0.8 s in EXIT) (--exit-gate wheels: the manual
                        policy owns the set-down), or at EXIT entry (--exit-gate stage); always once in RAMP
All three policies run every step; for --blend K steps after a switch a = (1 - k/K) a_old + k/K a_new.
chain p17 (2026-11-21, DEFAULT OFF): --stroke CKPT (or env MICRODUCK_CHAIN_STROKE) adds owner 3 = STROKE, a p17 push
checkpoint that takes ONE foot-push stroke on forward half-pipe flat crossings (2 -> 3 -> 2; see Stroke below).
Without it the chain is unchanged.

Outputs in --out (default out/chain_<label>/): chain_typical.mp4 (median eval score), chain_best.mp4 (top score),
chain_demo.mp4 (demo score, below), funnel.json (sequence_eval.funnel + hand-off survival / y at hand-off /
chain_complete), picks.json (checkpoints, indices, seed), demo_qpos.npz (the demo env's qpos for the Isaac Sim replay).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from dataclasses import asdict

os.environ.setdefault("WANDB_MODE", "disabled")
import numpy as np  # noqa: E402
import torch  # noqa: E402

from micro_skater.eval import sequence_eval as se  # noqa: E402

DEVICE = se.DEVICE
TASK = "Mjlab-Skate-SeqRamp-MicroDuck"
OWNER_NAMES = ("push", "manual", "ramp", "stroke")    # chain p17: "stroke" (owner 3) only with --stroke
SWITCHES = ("push->manual", "manual->ramp")
OUT_ROOT = os.path.join(os.getcwd(), "out")
# chainfix (2026-11-20): p9b model_2100 turns ~+65 deg left during PUSH from the start line (manual m2 then
# ~-35 deg), so the plain start reached EXIT 5-8 m beside the pipe (x -4.6 .. -8.4) heading ~+25..+65 deg; in_pipe gated
# on y only, so RAMP began on the floor beside the pipe (rise 0, past_pipe_end at the far end). Calibrated aimed start
# (dx m, dy m, dyaw deg) and the EXIT -> RAMP opening gate |x| < CHAINFIX_GATE_X (m; flat bottom half-width 0.5).
CHAINFIX_START_POSE = (0.6, -6.0, -60.0)
CHAINFIX_GATE_X = 0.8
# render6: a half-pipe pass counts on screen / in chain_complete / in the demo score only at >= PASS_MIN_CM of apex rise
# (the env's detector, st["apex_evt"], fires from apex_min_z ~1 cm; its raw count stays in funnel.json "passes").
# chain_complete = reached RAMP, >= COMPLETE_PASSES such passes, clean end (chain_complete_old: >= 3 raw passes).
PASS_MIN_CM = 2.0
COMPLETE_PASSES = 2
# render9 (chain_final_h8 review): the demo pick prefers a clean half-pipe. Over the RAMP frames of the clip (to
# clip_end), every deck contact with the terrain (the env's tail_floor sensor or a deck tail / core / nose geom touching it
# in the render copy = exactly what raises the TAIL DOWN / DECK SCRAPE chips) costs DEMO_W_EVENT per contact (onset) +
# DEMO_W_STEP per step, and the board's max roll relative to the surface above DEMO_ROLL_FREE_DEG costs DEMO_W_ROLL per
# deg (h8 n128 calibration: completed runs 0-3 contact events, max relative roll 3-11 deg; one extra wall ride = +1).
DEMO_W_EVENT, DEMO_W_STEP, DEMO_W_ROLL, DEMO_ROLL_FREE_DEG = 1.0, 0.1, 0.15, 8.0


# ── build ─────────────────────────────────────────────────────────────────────────────────────────────────────────────


def spawn_clock_offset(spawn: str):
    """Trick-obs clock offset for a chain that starts at `spawn`: None from the chain's own start (push / step_on),
    the manual specialist's for glide / manual spawns, the ramp specialist's for exit / ramp spawns."""
    from micro_skater.tasks import microduck_skate_seq_manual_env_cfg as man
    from micro_skater.tasks import microduck_skate_seq_ramp_env_cfg as rcfg
    if spawn in ("glide", "manual"):
        return man.CLOCK_OFFSET
    if spawn in ("exit", "ramp_bottom", "ramp_wall"):
        return rcfg.CLOCK_OFFSET
    return None


def push_ramp_of(ckpt: str):
    """p9rev (2026-11-20): the push_ramp (slewed push target, m/s per s) the push checkpoint was trained at, or None
    (a policy trained without it). Read from its run's params/env.yaml (reset_sequence seq_overrides + the push_ramp
    seq_cfg_curriculum stages) at the checkpoint's common_step_counter. The step command (phase obs -> RIDE) and
    push_done are gated by the ramp, so the chain must replay the same gate or the observed phase flips at another time."""
    import yaml
    path = os.path.join(os.path.dirname(os.path.abspath(ckpt)), "params", "env.yaml")
    if not os.path.isfile(path):
        return None

    class _L(yaml.SafeLoader):
        pass
    _L.add_multi_constructor("", lambda loader, suffix, node: None)    # python tags (funcs, tuples) -> None
    with open(path, encoding="utf-8") as f:
        y = yaml.load(f, Loader=_L) or {}
    ov = (((y.get("events") or {}).get("reset_sequence") or {}).get("params") or {}).get("seq_overrides") or {}
    rate = ov.get("push_ramp")
    if not rate:
        return None
    stages = (((y.get("curriculum") or {}).get("push_ramp") or {}).get("params") or {}).get("stages") or []
    if stages:
        step = int(torch.load(ckpt, map_location="cpu", weights_only=False).get("infos", {}).get("env_state", {})
                   .get("common_step_counter", 0))
        for s in stages:                                # = skate_sequence.seq_cfg_curriculum (latest stage reached)
            if step >= int(s["step"]):
                rate = s["cfg"]["push_ramp"]
    return float(rate)


def ramp_carve_of(ckpt: str):
    """r2rev (2026-11-20): the RAMP carve command cfg (skate_seq_ramp.CARVE_FLAT / CARVE_WALL dict, or None = kappa 0 in
    RAMP, the r1 contract) the ramp checkpoint was trained at: its run's params/env.yaml curriculum.ramp_carve stages at
    the checkpoint's common_step_counter. A run without the ramp_carve curriculum (r1 and older) -> None. Read from the
    run, not from this cfg's CARVE_STAGES (those depend on MICRODUCK_RAMP_R2_IT0 and on --curriculum-step)."""
    import yaml
    path = os.path.join(os.path.dirname(os.path.abspath(ckpt)), "params", "env.yaml")
    if not os.path.isfile(path):
        return None

    class _L(yaml.SafeLoader):
        pass
    _L.add_multi_constructor("", lambda loader, suffix, node: None)
    with open(path, encoding="utf-8") as f:
        y = yaml.load(f, Loader=_L) or {}
    stages = (((y.get("curriculum") or {}).get("ramp_carve") or {}).get("params") or {}).get("stages") or []
    step = int(torch.load(ckpt, map_location="cpu", weights_only=False).get("infos", {}).get("env_state", {})
               .get("common_step_counter", 0))
    cur = None
    for s in stages:                                    # = skate_seq_ramp.carve_curriculum (latest stage reached)
        if step >= int(s["step"]):
            cur = s.get("cfg")
    return dict(cur) if cur else None


def build_chain(ckpts: list[str], n: int, seed: int | None = None, seq_override: dict | None = None,
                spawn: str = "start", start_pose: tuple | None = None):
    """start_pose (chainfix): (dx, dy, dyaw_deg) for the start spawns (skate_sequence.reset_sequence_state start_pose).
    sequence_eval.build with the SeqRamp chain cfg and one deep-copied policy per checkpoint. seq_override: extra
    seq_cfg values on top of the chain cfg's (e.g. step_cmd=False to play a policy trained without the step command)."""
    import micro_skater.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
    from micro_skater.tasks.microduck_skate_seq_ramp_env_cfg import make_microduck_skate_seq_ramp_env_cfg

    cfg = make_microduck_skate_seq_ramp_env_cfg(play=True, chain=True)
    cfg.scene.num_envs = n
    if seed is not None:
        cfg.seed = seed
    # p9rev: the push policy's trained step-command gate (None = trained without the ramp: the chain cfg's p8 gate)
    pr = push_ramp_of(ckpts[0])
    cfg.events["reset_sequence"].params["seq_overrides"]["push_ramp"] = pr
    print(f"[chain] push_ramp {pr} (from {ckpts[0]})", flush=True)
    if seq_override:
        cfg.events["reset_sequence"].params["seq_overrides"].update(seq_override)
    # review (rrev): a chain started mid-way (--spawn) gets the clock offset its owner specialist trained with (the
    # trick-obs clock t/10; the chain cfg has none, so an exit spawn read t ~ 0 where the ramp policy saw 0.5-0.8)
    co = spawn_clock_offset(spawn)
    cfg.events["reset_sequence"].params["clock_offset"] = co
    print(f"[chain] spawn {spawn}  clock_offset {co}", flush=True)
    if start_pose is not None:                          # chainfix: aimed start (None: the cfg's start, unchanged)
        cfg.events["reset_sequence"].params["start_pose"] = tuple(start_pose)
        print(f"[chain] start_pose {tuple(start_pose)}", flush=True)
    print("[chain] seq_overrides " + str(cfg.events["reset_sequence"].params["seq_overrides"]), flush=True)
    for k in [k for k, t in cfg.curriculum.items() if (t.params or {}).get("event_name") == "reset_sequence"]:
        cfg.curriculum.pop(k)
    # r2rev: the ramp policy's trained RAMP carve command (obs kappa slot), pinned whatever --curriculum-step says
    rck = ckpts[2] if len(ckpts) > 2 else None             # passes the push policy only
    rc = ramp_carve_of(rck) if rck else None
    if "ramp_carve" in cfg.curriculum:
        cfg.curriculum["ramp_carve"].params["stages"] = [{"step": 0, "cfg": rc}]
    print(f"[chain] ramp carve {rc} (from {rck})", flush=True)
    env = ManagerBasedRlEnv(cfg=cfg, device=DEVICE)
    agent_cfg = load_rl_cfg(TASK)
    vec = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = load_runner_cls(TASK)(vec, asdict(agent_cfg), device=DEVICE)
    pols = []
    for ck in ckpts:
        runner.load(ck, load_cfg={"actor": True}, strict=True, map_location=DEVICE)
        pols.append(copy.deepcopy(runner.get_inference_policy(device=DEVICE)).eval())
    # distinct modules with distinct storage (the deep copies), and distinct normalizers when the checkpoints differ
    assert len({id(p) for p in pols}) == len(pols), "chain policies are not distinct objects"
    ptrs = [next(p.parameters()).data_ptr() for p in pols]
    assert len(set(ptrs)) == len(pols), "chain policies share parameter storage"
    means = [p.obs_normalizer._mean.detach().clone() for p in pols]
    for i in range(len(pols)):
        for j in range(i + 1, len(pols)):
            same = torch.equal(means[i], means[j])
            if os.path.abspath(ckpts[i]) != os.path.abspath(ckpts[j]):
                assert not same, f"obs_normalizer means of {ckpts[i]} and {ckpts[j]} are identical"
            elif same:
                print(f"[chain] note: policies {i} and {j} come from the same checkpoint (smoke test)", flush=True)
    return vec, pols, cfg.seed, agent_cfg


# ── chain p17: STROKE owner (2026-11-21, DEFAULT OFF: --stroke CKPT / env MICRODUCK_CHAIN_STROKE) ─────────────────────────
# On a forward half-pipe flat
# crossing the ramp policy (owner 2) hands the board to a p17 push checkpoint (owner 3 = STROKE) for ONE foot-push stroke,
# then takes it back. The chain env stays in stage RAMP; the STROKE policy's input is rewritten to what its p17 training
# showed: the push task's PUSH / SETTLE scheduler + p17 tracker are MIRRORED here per env (Stroke.update = seq_update's
# PUSH branch with MICRODUCK_SEQPUSH_P17 + _p17_update, in the push env's order: scheduler first, then the tracker),
# the trick / seq block slots follow from that state, and the placement slots -9..-3 come from the SAME functions the
# p17c training obs used (sq.p17_floor_ref at the hand-over, sq.p17_floor_obs every step). Success = training's
# p17_success with the stance features recorded AT THE HAND-OVER (sq.p17_stance_feats / sq.p17_stance_err_ref).
# Bridge semantics: p17b trained the stance bridge alpha 0 (p16's own pre-stroke stance) -> 1 (the ramp policy's riding
# stance); the chain always hands over from the real riding stance = alpha 1.0, and the one-stroke target is set from
# min(board, trunk) speed (the bridge spawns' single baseline).
STROKE_REC_VMIN = 0.2       # m/s: the entry frame's nose-first speed floor (p17d: = the bank_chain
                            # recording's window floor)
STROKE_ALPHA = 1.0          # the chain hands over from the ramp policy's real riding stance (p17b bridge alpha 1.0)


def stroke_cfg_of(ckpt: str) -> dict:
    """The push-task scheduler / p17 values the STROKE checkpoint trained with, from its run's params/env.yaml:
    sq.SEQ_DEFAULTS + reset_sequence seq_overrides (scalars) + push_ramp at the checkpoint's step (push_ramp_of), the
    success term's on_s / wheels_s, p17_fail's wall_rise, reset_p17's floor_obs (p17c: placement slots rewritten)."""
    import yaml
    from micro_skater.tasks import skate_sequence as sq
    path = os.path.join(os.path.dirname(os.path.abspath(ckpt)), "params", "env.yaml")
    assert os.path.isfile(path), f"--stroke: no params/env.yaml next to {ckpt}"

    class _L(yaml.SafeLoader):
        pass
    _L.add_multi_constructor("", lambda loader, suffix, node: None)
    with open(path, encoding="utf-8") as f:
        y = yaml.load(f, Loader=_L) or {}
    ev = y.get("events") or {}
    ov = ((ev.get("reset_sequence") or {}).get("params") or {}).get("seq_overrides") or {}
    cfg = dict(sq.SEQ_DEFAULTS)
    cfg.update({k: v for k, v in ov.items() if v is not None and not isinstance(v, (list, dict))})
    tm = y.get("terminations") or {}
    assert "p17_fail" in tm and "reset_p17" in ev, f"--stroke {ckpt}: not a p17 push checkpoint (no p17_fail / reset_p17)"
    sp = (tm.get("success") or {}).get("params") or {}
    fp = (tm.get("p17_fail") or {}).get("params") or {}
    rp = (ev.get("reset_p17") or {}).get("params") or {}
    return dict(v_gap=float(cfg["v_gap"]), both_on_s=float(cfg["both_on_s"]), push_max_s=float(cfg["push_max_s"]),
                last_stage=int(cfg.get("last_stage", sq.ST_RAMP)), stance_hold_s=float(cfg.get("stance_hold_s", 0.2)),
                start_gap=float(cfg["start_gap"]), step_cmd=bool(cfg.get("step_cmd", False)),
                push_ramp=push_ramp_of(ckpt), on_s=float(sp.get("on_s", 0.3)), wheels_s=float(sp.get("wheels_s", 0.2)),
                wall_rise=float(fp.get("wall_rise", 0.02)), floor_obs=bool(rp.get("floor_obs", False)),
                # p17d: the low-speed wall rule (sq.p17_wall_turn) and the pipe deadline the checkpoint trained with
                wall_turn=bool(fp.get("wall_turn", False)),
                deadline_pipe=(float(fp["deadline_pipe"]) if fp.get("deadline_pipe") is not None else None),
                bridge=rp.get("bridge") is not None or "p17b_alpha" in (y.get("curriculum") or {}))


class StrokeGate:
    """The STROKE hand-over gate's crossing entries (shared by Stroke, so the recorded bank_chain
    states are exactly the chain's hand-over frames). An env is ARMED by entering RAMP (p17d: the first flat entry from
    the approach counts as a crossing) and by every wall visit (board centre |x| > L/2, |y| < W/2 - 0.4, in RAMP); the
    ENTRY is the first later frame that passes geometry (RAMP, |y| < W/2 - 0.4, nose-first board speed
    >= rec_vmin, |cos yaw| >= 0.9, both axles on the flat, four wheels down, both soles on the deck, no foot on the floor)
    and disarms it: one entry per crossing. kind = "approach" (no wall visit yet this episode) or "crossing"."""

    def __init__(self, env, rec_vmin: float = STROKE_REC_VMIN):
        n, dev = env.num_envs, env.device
        self.env, self.rec_vmin = env, rec_vmin
        self.armed = torch.zeros(n, dtype=torch.bool, device=dev)
        self.seen_ramp = torch.zeros(n, dtype=torch.bool, device=dev)
        self.walled = torch.zeros(n, dtype=torch.bool, device=dev)   # a wall visit this episode
        self.n_armed = 0                        # arm events (RAMP entries + wall visits arming an unarmed env)

    def reset(self, mask) -> None:
        self.armed &= ~mask
        self.seen_ramp &= ~mask
        self.walled &= ~mask

    def step(self, st, block=None) -> dict:
        """Once per env step (before the action): returns dict(entry (B,) bool, vb, first (B,) bool = an approach
        entry, xf, d_rem). block (B,) bool: envs that may neither arm nor enter (in STROKE)."""
        from mjlab.utils.lab_api.math import quat_apply
        from micro_skater.tasks import skate_sequence as sq
        from micro_skater.tasks import skate_trick as tk
        env = self.env
        n, dev = env.num_envs, env.device
        blk = torch.zeros(n, dtype=torch.bool, device=dev) if block is None else block
        pipe = st["pipe"]
        L2, W2 = pipe["L"] / 2, pipe["W"] / 2
        p = sq.pipe_xyz(env)
        ex = quat_apply(tk._board(env).data.root_link_quat_w, torch.tensor([1.0, 0.0, 0.0], device=dev).expand(n, 3))
        xf, xr = p[:, 0] + ex[:, 0] * sq.BOARD["axle_x"], p[:, 0] - ex[:, 0] * sq.BOARD["axle_x"]
        vb = sq.board_v_fwd(env)
        lf, rf = sq._feet_floor(env)
        ramp = st["stage"] == sq.ST_RAMP
        inp = p[:, 1].abs() < W2 - 0.4
        wall = ramp & inp & (p[:, 0].abs() > L2) & ~blk
        first = ramp & ~self.seen_ramp & ~blk
        arm = wall | first
        self.n_armed += int((arm & ~self.armed).sum())
        self.armed |= arm
        self.seen_ramp |= ramp
        self.walled |= wall
        geo = (ramp & inp & (vb >= self.rec_vmin) & (ex[:, 0].abs() >= 0.9) & (xf.abs() < L2) & (xr.abs() < L2)
               & sq._wheels_all_down(env) & tk.soles_on_deck(env) & ~lf & ~rf)
        entry = self.armed & geo & ~blk
        self.armed &= ~entry
        return dict(entry=entry, vb=vb, first=entry & ~self.walled, xf=xf,
                    d_rem=L2 - xf * torch.sign(ex[:, 0]))


class Stroke:
    """Per-env STROKE state for the chain (see the block comment). begin() at a hand-over, update() once per later step,
    obs() rewrites the last 25 actor slots (trick block -25..-14, seq block -13..-1) of the envs in STROKE."""

    def __init__(self, env, cfg: dict, vmin: float = 0.4, vmax: float = 0.75, deadline: float = 3.0,
                 abort_s: float = 0.8, dv: float = 0.07, t_settle: float = 0.25, train_alpha=None):
        from micro_skater.tasks import skate_sequence as sq
        from micro_skater.tasks import skate_trick as tk
        self.env, self.cfg = env, dict(cfg)
        self.vmin, self.vmax, self.deadline, self.abort_s, self.dv, self.t_settle = vmin, vmax, deadline, abort_s, dv, t_settle
        self.train_alpha = train_alpha
        n, dev = env.num_envs, env.device
        z = lambda: torch.zeros(n, device=dev)
        b = lambda: torch.zeros(n, dtype=torch.bool, device=dev)
        assert int(cfg["last_stage"]) == sq.ST_SETTLE, "Stroke mirrors only the push task's PUSH -> SETTLE advance"
        self.active, self.cmd, self.p_rf, self.ok, self.succ, self.fail = b(), b(), b(), b(), b(), b()
        self.g = StrokeGate(env)                # the crossing entries (shared)
        self.fail_wall = b()
        self.t0, self.t_stage, self.stage_t, self.v_tgt, self.v0, self.vs = z(), z(), z(), z(), z(), z()
        self.both_on, self.wheels_on, self.tdv, self.gain, self.ntd, self.p_t, self.st_t = z(), z(), z(), z(), z(), z(), z()
        self.T_settle = z() + float(t_settle)
        self.stage = torch.full((n,), sq.ST_PUSH, dtype=torch.long, device=dev)
        self.obs_ph = torch.full((n,), sq.PUSH, dtype=torch.long, device=dev)
        self.ref = torch.zeros(n, len(sq.P17_REF_FIELDS), device=dev)
        self.fref = torch.zeros(n, 6, device=dev)
        self.last_rise = torch.full((n,), float("nan"), device=dev)
        self._step = 0
        self.open: list = [None] * n            # the open record per env
        self.pending: list = [None] * n         # a closed record waiting for its next wall apex (rise after)
        self.records: list[dict] = []
        self.entries: list[dict] = []           # every gated crossing entry (started or not): env, k, v, started, first
        self._phase = torch.tensor(sq.STAGE_PHASE, device=dev)
        # update() runs INSIDE env.step, right after the termination compute - where the push task runs seq_update /
        # _p17_update (one physics substep stale kinematics, contact data of the previous sense()); chain_play's own
        # loop only sees the post-step state, so the mirror would otherwise latch / count a step apart
        tm = env.termination_manager
        self._tm_compute = tm.compute

        def _compute(*args, **kw):
            out = self._tm_compute(*args, **kw)
            self.update(self.active.clone(), tk._t(env))
            return out
        tm.compute = _compute

    # -- gate ---------------------------------------------------------------------------------------------------------
    def gate(self, st, base_owner) -> tuple[torch.Tensor, torch.Tensor]:
        """(start, entry): entry = a StrokeGate crossing entry (the first gated frame of a nose-first flat crossing:
        the approach's first one or one after a wall visit); start = entry with the ramp policy in charge and the board
        speed inside [vmin, vmax] (the trained pipe hand-over window)."""
        g = self.g.step(st, block=self.active)
        entry, vb = g["entry"], g["vb"]
        start = entry & (base_owner == 2) & (vb >= self.vmin) & (vb <= self.vmax)
        if bool(entry.any()):
            for i in entry.nonzero().squeeze(1).tolist():
                self.entries.append(dict(env=i, k=self._step, v=round(float(vb[i]), 4), started=bool(start[i]),
                                         first=bool(g["first"][i])))
        return start, entry

    # -- state ----------------------------------------------------------------------------------------------------------
    def begin(self, mask, t, step: int, init: dict | None = None) -> None:
        """Hand-over: the push task's p17 spawn state (reset_p17 + the spawn step's obs). init (tests only) overrides
        v_tgt / v0 / vs / T_settle / ref / fref with a training env's own values."""
        from micro_skater.tasks import skate_sequence as sq
        from micro_skater.tasks import skate_trick as tk
        if not bool(mask.any()):
            return
        env, c = self.env, self.cfg
        st = sq._st(env)
        # the one-stroke baseline as training set it: min(board, trunk) for p17b bridge spawns, the board speed for p17
        u = torch.nan_to_num(sq._push_v(env, st)) if c["bridge"] else sq.board_v_fwd(env)
        W = st["pipe"]["W"]
        p, yaw = sq.pipe_xyz(env), tk._board_yaw(env)
        v_tgt = u + c["v_gap"] + self.dv
        v0, vs, ts = u, u, torch.full_like(u, float(self.t_settle))
        ref = sq.p17_stance_feats(env)
        fref = sq.p17_floor_ref(p, yaw, W, c["start_gap"])
        if init:
            v_tgt, v0, vs = init.get("v_tgt", v_tgt), init.get("v0", v0), init.get("vs", vs)
            ts, ref, fref = init.get("T_settle", ts), init.get("ref", ref), init.get("fref", fref)
        m, m1 = mask, mask.unsqueeze(1)
        w = lambda new, old: torch.where(m, new, old)
        self.v_tgt, self.v0, self.vs, self.T_settle = w(v_tgt, self.v_tgt), w(v0, self.v0), w(vs, self.vs), w(ts, self.T_settle)
        self.ref, self.fref = torch.where(m1, ref, self.ref), torch.where(m1, fref, self.fref)
        self.t0, self.t_stage = w(t, self.t0), w(t, self.t_stage)
        zero = torch.zeros_like(u)
        for k in ("stage_t", "both_on", "wheels_on", "tdv", "gain", "ntd", "p_t", "st_t"):
            setattr(self, k, w(zero, getattr(self, k)))
        for k in ("cmd", "p_rf", "ok", "succ", "fail"):
            setattr(self, k, getattr(self, k) & ~m)
        self.stage = torch.where(m, torch.full_like(self.stage, sq.ST_PUSH), self.stage)
        self.obs_ph = torch.where(m, torch.full_like(self.obs_ph, sq.PUSH), self.obs_ph)
        self.active = self.active | m
        vb = sq.board_v_fwd(env)
        for i in m.nonzero().squeeze(1).tolist():
            self.pending[i] = None
            self.open[i] = dict(env=i, k0=step, t0=float(t[i]), x0=float(p[i, 0]), y0=float(p[i, 1]),
                                yaw0_deg=round(math.degrees(float(yaw[i])), 2), v_before=float(vb[i]), pv_before=float(u[i]),
                                v_tgt=float(self.v_tgt[i]),
                                rise_before_cm=(None if math.isnan(float(self.last_rise[i]))
                                                else round(float(self.last_rise[i]) * 100, 2)))

    def update(self, m, t) -> None:
        """One env step for the envs in m (in STROKE; called from the termination-compute hook, see __init__): the push
        env's per-step order - seq_update's PUSH branch (with the p17 latch / push_done gate, reading the tracker's
        previous values), then _p17_update, then p17_success / p17_fail. Sets self.succ / self.fail."""
        from micro_skater.tasks import skate_sequence as sq
        from micro_skater.tasks import skate_trick as tk
        if not bool(m.any()):
            self.succ, self.fail, self.fail_wall = self.succ & m, self.fail & m, self.fail_wall & m
            return
        env, c, dt = self.env, self.cfg, self.env.step_dt
        st = sq._st(env)
        w = lambda new, old: torch.where(m, new, old)
        v = sq.board_v_fwd(env)
        pv = torch.nan_to_num(sq._push_v(env, st))
        lf, rf = sq._feet_floor(env)
        on = tk.soles_on_deck(env)
        # seq_update (PUSH / SETTLE part; last_stage caps SETTLE)
        both_on = torch.where(on & ~rf & ~lf, self.both_on + dt, torch.zeros_like(self.both_on))
        wheels_on = torch.where(sq._wheels_all_down(env), self.wheels_on + dt, torch.zeros_like(self.wheels_on))
        ts = t - self.t_stage
        thr = self.v_tgt - c["v_gap"]
        fast = v >= thr
        fast_cmd = fast
        if c["push_ramp"]:
            v_ramp = self.v0.clamp(min=0.0) + float(c["push_ramp"]) * self.stage_t
            fast = fast & (v_ramp >= thr)
            fast_cmd = fast & (rf | (self.v0 >= thr))
        fast_cmd = fast_cmd & (pv >= thr) & ((pv - self.tdv) >= sq.P17_GAIN_MIN) & self.p_rf
        stage = self.stage
        push_done = (stage == sq.ST_PUSH) & ((fast & (both_on >= c["both_on_s"]))
                                             | (ts > c["push_max_s"]) & (both_on >= c["both_on_s"])) & self.ok
        adv = push_done & (stage < c["last_stage"])
        stage_t = torch.where(adv, torch.zeros_like(self.stage_t), self.stage_t + dt)
        stage = torch.where(adv, stage + 1, stage)
        t_stage = torch.where(adv, t, self.t_stage)
        trick = self._phase[stage]
        if c["step_cmd"]:
            go = (stage == sq.ST_PUSH) & (fast_cmd | (stage_t > c["push_max_s"]))
            cmd = (self.cmd | go) & (stage == sq.ST_PUSH)
            obs_ph = torch.where(cmd, torch.full_like(trick, sq.RIDE), trick)
        else:
            cmd, obs_ph = self.cmd, trick
        # _p17_update
        td = rf & ~self.p_rf
        lo = ~rf & self.p_rf
        tdv = torch.where(td, pv, self.tdv)
        gain = pv - tdv
        left = tk._contact_any(env, "left_deck")
        ntd = self.ntd + td.float()
        ok = self.ok | (lo & left & (gain >= sq.P17_GAIN_MIN) & (pv >= self.vs + sq.P17_GAIN_MIN))
        gmax = torch.where(lo, torch.maximum(self.gain, gain), self.gain)
        p_t = self.p_t + dt
        e = sq.p17_stance_err_ref(env, self.ref)
        tol = sq.P17_STANCE_TOL
        sok = ((stage == sq.ST_SETTLE) & on & (e[:, 0] < tol[0]) & (e[:, 1] < tol[1]) & (e[:, 2] < tol[2])
               & (e[:, 3] < tol[3]) & (e[:, 4] < tol[4]))
        st_t = torch.where(sok, self.st_t + dt, torch.zeros_like(self.st_t))
        # p17_success / p17_fail
        succ = ((stage == sq.ST_SETTLE) & (stage_t >= self.T_settle) & (both_on >= c["on_s"])
                & (wheels_on >= c["wheels_s"]) & ok & (st_t >= c["stance_hold_s"] - 1e-6))
        pp = sq.pipe_xyz(env)
        rise = pp[:, 2] - sq.DECK_Z
        wall = rise > c["wall_rise"]
        if c.get("wall_turn"):   # p17d: reached the far wall and turned back (travel sign = the hand-over's fref)
            wall = wall | sq.p17_wall_turn(pp, tk._board(env).data.root_link_lin_vel_w[:, 0], self.fref[:, 2],
                                           float(st["pipe"]["L"]))
        fail = ((p_t > self.deadline) | wall) & ~succ
        self.fail_wall = fail & wall & m     # the label, decided on the same (in-step) state
        for k, val in (("both_on", both_on), ("wheels_on", wheels_on), ("stage_t", stage_t), ("stage", stage),
                       ("t_stage", t_stage), ("cmd", cmd), ("obs_ph", obs_ph), ("tdv", tdv), ("ntd", ntd), ("ok", ok),
                       ("gain", gmax), ("p_t", p_t), ("st_t", st_t)):
            setattr(self, k, w(val, getattr(self, k)))
        self.p_rf = w(rf, self.p_rf)
        self.succ, self.fail = succ & m, fail & m

    def obs(self, actor: torch.Tensor, t) -> torch.Tensor:
        """The STROKE policy's actor obs: a copy with the last 25 slots of the envs in STROKE set from the mirrored
        state (seq_trick_obs / seq_obs layout; placement slots via sq.p17_floor_obs when the checkpoint trained with
        floor_obs, else the real pipe placement as in p17 / p17b training)."""
        from micro_skater.tasks import skate_sequence as sq
        from micro_skater.tasks import skate_trick as tk
        env = self.env
        a = actor.clone()
        ph = self.obs_ph
        oh = (torch.nn.functional.one_hot(ph.clamp(max=tk.N_TRICK - 1), tk.N_TRICK).float()
              * (ph < tk.N_TRICK).float().unsqueeze(1))
        z = torch.zeros(env.num_envs, 1, device=env.device)
        clock = ((t - self.t0) / 10.0).clamp(max=0.8).unsqueeze(1)
        trick = torch.cat([oh, z + 1.0, z, z, z, z + 1.0, z, clock], dim=1)       # dir 1, kappa 0
        seq = a[:, -13:].clone()
        push = (ph == sq.PUSH).float()
        seq[:, 0] = push
        seq[:, 1] = (ph == sq.RAMP).float()
        seq[:, 2] = self.v_tgt * push
        seq[:, 3] = ((t - self.t_stage) / 5.0).clamp(max=2.0)
        if self.cfg["floor_obs"]:
            seq = sq.p17_floor_obs(seq, sq.pipe_xyz(env), tk._board_yaw(env), self.fref, sq._st(env)["pipe"]["W"])
        new = torch.cat([trick, seq], dim=1)
        a[:, -25:] = torch.where(self.active.unsqueeze(1), new, a[:, -25:])
        return a

    def finish(self, m, step: int, outcome: str) -> None:
        """Close the open records of the envs in m (STROKE -> RAMP)."""
        from micro_skater.tasks import skate_sequence as sq
        from mjlab.utils.lab_api.math import quat_apply
        from micro_skater.tasks import skate_trick as tk
        if not bool(m.any()):
            return
        env = self.env
        st = sq._st(env)
        p = sq.pipe_xyz(env)
        n = env.num_envs
        ex = quat_apply(tk._board(env).data.root_link_quat_w, torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(n, 3))
        xf = p[:, 0] + ex[:, 0] * sq.BOARD["axle_x"]
        d_rem = st["pipe"]["L"] / 2 - xf * torch.sign(ex[:, 0])
        vb = sq.board_v_fwd(env)
        pv = torch.nan_to_num(sq._push_v(env, st))
        for i in m.nonzero().squeeze(1).tolist():
            r = self.open[i]
            if r is None:
                continue
            if outcome == "episode_end":                  # the env has already reset: no physical after-values
                r.update(k1=step, outcome=outcome, dur_s=round((step - r["k0"]) * env.step_dt, 3),
                         touchdowns=int(self.ntd[i]), legal_stroke=bool(self.ok[i]),
                         reached_settle=bool(self.stage[i] == sq.ST_SETTLE), rise_after_cm=None)
                self.records.append(r)
                self.open[i] = None
                self.pending[i] = None
                continue
            r.update(k1=step, outcome=outcome, dur_s=round((step - r["k0"]) * env.step_dt, 3), v_after=float(vb[i]),
                     pv_after=float(pv[i]), gain=round(float(vb[i]) - r["v_before"], 4),
                     pv_gain=round(float(pv[i]) - r["pv_before"], 4), touchdowns=int(self.ntd[i]),
                     legal_stroke=bool(self.ok[i]), best_lift_gain=round(float(self.gain[i]), 4),
                     reached_settle=bool(self.stage[i] == sq.ST_SETTLE), d_rem=round(float(d_rem[i]), 3),
                     x1=float(p[i, 0]), rise_after_cm=None)
            self.records.append(r)
            self.open[i] = None
            self.pending[i] = r if outcome != "episode_end" else None
        self.active = self.active & ~m

    def apex(self, evt, rise) -> None:
        """A rewarded wall pass (st["apex_evt"], st["ap_rise"] m): the next pass after a hand-back is its rise_after."""
        if rise is None or not bool(evt.any()):
            return
        for i in evt.nonzero().squeeze(1).tolist():
            if self.pending[i] is not None:
                self.pending[i]["rise_after_cm"] = round(float(rise[i]) * 100, 2)
                self.pending[i] = None
        self.last_rise = torch.where(evt, rise, self.last_rise)

    def clear(self, fresh, step: int) -> None:
        """New episodes: an open STROKE ends with the episode ("episode_end"); everything is cleared."""
        if not bool(fresh.any()):
            return
        self.finish(fresh & self.active, step, "episode_end")
        self.g.reset(fresh)
        self.last_rise = torch.where(fresh, torch.full_like(self.last_rise, float("nan")), self.last_rise)
        for i in fresh.nonzero().squeeze(1).tolist():
            self.pending[i] = None

    def close_all(self, step: int) -> None:
        self.finish(self.active.clone(), step, "run_end")

    def note(self) -> str:
        ta = "unknown" if self.train_alpha is None else f"{self.train_alpha:g}"
        return (f"STROKE hands over from the ramp policy's real riding stance = p17b bridge alpha {STROKE_ALPHA:g} "
                f"(checkpoint trained to alpha {ta}; floor_obs {self.cfg['floor_obs']}, bridge {self.cfg['bridge']})")


def stroke_summary(recs: list[dict], eps: list[dict], apex_log, rise_log, end, dt) -> dict:
    """funnel agg["stroke"]: per-STROKE outcomes, speed / apex rise before and after, ends during STROKE, survival
    1 s after the hand-back; the wall passes right after a hand-back vs every other counted pass."""
    one_s = int(round(1.0 / dt))
    out = dict(n=len(recs), per_episode_mean=float(np.mean([len(r.get("p17_strokes", [])) for r in eps])) if eps else 0.0)
    if not recs:
        return out
    oc = {}
    for r in recs:
        oc[r["outcome"]] = oc.get(r["outcome"], 0) + 1
    med = lambda xs: (float(np.median(xs)) if len(xs) else None)
    q = lambda xs: ([round(float(v), 3) for v in np.quantile(xs, [0.1, 0.5, 0.9])] if len(xs) else None)
    succ = [r for r in recs if r["outcome"] == "success"]
    back = [r for r in recs if r["outcome"] not in ("episode_end", "run_end")]
    out.update(outcomes=oc, success_rate=len(succ) / len(recs),
               v_before_q=q([r["v_before"] for r in recs]), v_after_q=q([r["v_after"] for r in recs if "v_after" in r]),
               gain_q=q([r["gain"] for r in recs if "gain" in r]), gain_success_q=q([r["gain"] for r in succ]),
               pv_gain_q=q([r["pv_gain"] for r in recs if "pv_gain" in r]),
               touchdowns_mean=float(np.mean([r.get("touchdowns", 0) for r in recs])),
               legal_stroke=float(np.mean([r.get("legal_stroke", False) for r in recs])),
               dur_q=q([r["dur_s"] for r in recs if "dur_s" in r]),
               d_rem_q=q([r["d_rem"] for r in recs if "d_rem" in r]),
               ended_during_stroke=sum(r["outcome"] == "episode_end" for r in recs),
               ended_reasons=sorted({r.get("end_reason", "") for r in recs if r["outcome"] == "episode_end"}),
               alive_1s_after=(float(np.mean([r["alive_1s_after"] for r in back])) if back else None),
               rise_before_cm_median=med([r["rise_before_cm"] for r in recs if r.get("rise_before_cm") is not None]),
               rise_after_cm_median=med([r["rise_after_cm"] for r in recs if r.get("rise_after_cm") is not None]),
               rise_after_success_cm_median=med([r["rise_after_cm"] for r in succ if r.get("rise_after_cm") is not None]))
    # the wall passes right after a hand-back vs the other passes (first episode, every raw rewarded pass)
    after, other = [], []
    if apex_log is not None and rise_log is not None:
        for i, ep in enumerate(eps):
            e1 = int(end[i]) + 1
            ks = np.nonzero(apex_log[:e1, i])[0]
            firsts = set()
            for r in ep.get("p17_strokes", []):
                if r.get("k1") is None or r["outcome"] in ("episode_end", "run_end"):
                    continue
                nxt = ks[ks >= r["k1"]]
                if len(nxt):
                    firsts.add(int(nxt[0]))
            for k in ks:
                z = float(rise_log[k, i]) * 100
                if not math.isnan(z):
                    (after if int(k) in firsts else other).append(z)
    out.update(apex_after_stroke_cm_median=med(after), apex_after_stroke_n=len(after),
               apex_other_cm_median=med(other), apex_other_n=len(other))
    return out


# ── policy selection ──────────────────────────────────────────────────────────────────────────────────────────────────


class Chain:
    """Callable policy(obs) over the three specialists (owner latch + action blend), with per-step logs."""

    def __init__(self, env, pols, blend: int = 4, exit_gate: str = "wheels", wheels_s: float = 0.15,
                 exit_max_s: float = 0.8, settle_gate: float | None = None, settle_on_s: float = 0.3,
                 settle_wheels_s: float = 0.2, manual_clock: float | None = None, stance_gate: bool = False,
                 stance_max_s: float = 1.5, stroke: Stroke | None = None):
        assert exit_gate in ("wheels", "stage")
        # chain p17: stroke = a Stroke (owner 3; the 4th policy = the p17 checkpoint). None = the chain as before.
        self.stroke = stroke
        self._stroke_checked = False
        # p9brev: stance_gate hands over once the push policy has held the p9b hand-off stance (seq_cfg stance_tol,
        # st["stance_t"] >= stance_hold_s; needs stance_tol in the chain's seq_overrides), or after stance_max_s in
        # SETTLE (its trained SETTLE overtime), or at MANUAL entry at the latest. Overrides settle_gate.
        self.stance_gate, self.stance_max_s = stance_gate, stance_max_s
        # handoff: manual_clock C (s) adds C to the trick-obs clock (t/10, clipped 0.8; actor obs slot -14: the trick
        # block's last value, the 13D seq block follows) in the MANUAL policy's input only. The manual specialist
        # trained with clock_offset U(3, 7) s (its SETTLE started at a clock of 3-7 s); in the chain it takes over at
        # t ~ 1.3-2 s. None = unchanged.
        self.manual_clock = manual_clock
        self._clock_checked = False
        self.env, self.pols, self.K, self.exit_gate = env, pols, max(1, int(blend)), exit_gate
        # handoff: settle_gate S (s) keeps the push policy in SETTLE until stage_t >= S, both soles on the deck >=
        # settle_on_s and four wheels down >= settle_wheels_s (the push specialist's own success state); MANUAL entry
        # hands over at the latest (the push policy never trained in MANUAL). None = hand over at SETTLE entry.
        self.settle_gate, self.settle_on_s, self.settle_wheels_s = settle_gate, settle_on_s, settle_wheels_s
        self.wheels_s, self.exit_max_s = wheels_s, exit_max_s
        n, dev = env.num_envs, env.device
        self.owner = torch.zeros(n, dtype=torch.long, device=dev)
        self.prev = torch.zeros(n, dtype=torch.long, device=dev)
        self.k = torch.full((n,), self.K, dtype=torch.long, device=dev)
        self.eff = torch.zeros(n, dtype=torch.long, device=dev)      # chain p17: owner incl. STROKE (= owner without)
        self.reset_log()

    def reset_log(self):
        self.step = 0
        self.log_owner, self.log_apex, self.log_alpha, self.log_rise = [], [], [], []
        self.handoffs: list[dict] = []

    def _want(self, st) -> torch.Tensor:
        from micro_skater.tasks import skate_sequence as sq
        stage = st["stage"]
        if self.stance_gate:
            hold = st["seq_cfg"]["stance_hold_s"] - 1e-6
            g1 = (((stage == sq.ST_SETTLE) & ((st["stance_t"] >= hold) | (st["stage_t"] >= self.stance_max_s)))
                  | (stage >= sq.ST_MANUAL))
        elif self.settle_gate is None:
            g1 = stage >= sq.ST_SETTLE
        else:
            g1 = ((stage == sq.ST_SETTLE) & (st["stage_t"] >= self.settle_gate) & (st["both_on"] >= self.settle_on_s)
                  & (st["wheels_on"] >= self.settle_wheels_s)) | (stage >= sq.ST_MANUAL)
        want = torch.where(g1, torch.ones_like(stage), torch.zeros_like(stage))
        if self.exit_gate == "wheels":
            g2 = (stage == sq.ST_EXIT) & ((st["wheels_on"] >= self.wheels_s) | (st["stage_t"] >= self.exit_max_s))
            g2 = g2 | (stage > sq.ST_EXIT)
        else:
            g2 = stage >= sq.ST_EXIT
        return torch.where(g2, torch.full_like(stage, 2), want)

    @torch.no_grad()
    def __call__(self, obs):
        if self.stroke is not None:                                       # chain p17 (default off)
            return self._call_stroke(obs)
        from micro_skater.tasks import skate_sequence as sq
        from micro_skater.tasks import skate_trick as tk
        env = self.env
        st = sq.seq_update(env)
        if self.manual_clock is None:
            acts = torch.stack([p(obs) for p in self.pols])              # (3, B, A): every policy, every step
        else:
            ob_m = obs.clone()
            a_ = ob_m["actor"]
            if not self._clock_checked:                                   # the slot really is the clock
                ref = ((tk._t(env) + st["t_off"]) / 10.0).clamp(max=0.8)
                assert torch.allclose(a_[:, -14], ref, atol=1e-4), "actor obs slot -14 is not the trick clock"
                self._clock_checked = True
            a_[:, -14] = (a_[:, -14] + self.manual_clock / 10.0).clamp(max=0.8)
            acts = torch.stack([p(ob_m if i == 1 else obs) for i, p in enumerate(self.pols)])
        fresh = env.episode_length_buf == 0                               # new episode: clear the latch
        want = self._want(st)
        owner0 = torch.where(fresh, want, self.owner)                     # a fresh env starts at its stage's owner
        new = torch.maximum(owner0, want)
        sw = (new != owner0) & ~fresh
        self.prev = torch.where(fresh, new, torch.where(sw, owner0, self.prev))
        self.k = torch.where(fresh, torch.full_like(self.k, self.K), torch.where(sw, torch.ones_like(self.k), self.k))
        self.owner = new
        ar = torch.arange(env.num_envs, device=env.device)
        alpha = (self.k.float() / self.K).clamp(max=1.0).unsqueeze(1)
        a = (1.0 - alpha) * acts[self.prev, ar] + alpha * acts[new, ar]
        if bool(sw.any()):
            p = sq.pipe_xyz(env)
            v = sq.board_v_fwd(env)
            pitch = torch.rad2deg(tk.front_up_pitch(env))
            W = st["pipe"]["W"]
            t = tk._t(env)
            for i in sw.nonzero().squeeze(1).tolist():
                self.handoffs.append(dict(env=i, step=self.step, frm=int(owner0[i]), to=int(new[i]),
                                          stage=int(st["stage"][i]), t=float(t[i]), v=float(v[i]),
                                          y_open=float(p[i, 1] + W / 2), x=float(p[i, 0]), pitch=float(pitch[i]),
                                          wheels_on=float(st["wheels_on"][i])))
        self.log_owner.append(new.clone())
        self.log_apex.append(st["apex_evt"].clone())
        rz = st.get("ap_rise")                                            # render6: the pass's apex rise (m)
        self.log_rise.append(torch.where(st["apex_evt"], rz, torch.zeros_like(rz)) if rz is not None
                             else torch.full(st["apex_evt"].shape, float("nan"), device=env.device))
        self.log_alpha.append(alpha.squeeze(1).clone())
        self.k = (self.k + 1).clamp(max=self.K)
        self.step += 1
        return a

    @torch.no_grad()
    def _call_stroke(self, obs):
        """chain p17: __call__ with the STROKE owner. The base owner (0..2) latches as before; owner 3 overrides it
        while Stroke.active (RAMP -> STROKE at a gated flat-crossing entry, STROKE -> RAMP on p17 success, the 2 cm far
        wall rise, the deadline or no floor touchdown by abort_s). The action blend runs on the effective owner."""
        from micro_skater.tasks import skate_sequence as sq
        from micro_skater.tasks import skate_trick as tk
        env, S = self.env, self.stroke
        st = sq.seq_update(env)
        a0 = obs["actor"]
        if not self._stroke_checked:                                      # the last 25 slots really are the blocks
            assert torch.allclose(a0[:, -13:], sq.seq_obs(env), atol=1e-4), "actor obs [-13:] is not seq_obs"
            assert torch.allclose(a0[:, -25:-13], sq.seq_trick_obs(env), atol=1e-4), \
                "actor obs [-25:-13] is not seq_trick_obs"
            self._stroke_checked = True
        t = tk._t(env)
        fresh = env.episode_length_buf == 0
        S.clear(fresh, self.step)
        want = self._want(st)
        owner0 = torch.where(fresh, want, self.owner)
        base = torch.maximum(owner0, want)
        self.owner = base
        m = S.active & ~fresh                  # (S.update already ran inside the env step: the termination hook)
        S.apex(st["apex_evt"], st.get("ap_rise"))
        wall = S.fail & S.fail_wall
        abort = m & ~S.succ & ~S.fail & (S.ntd == 0) & (S.p_t >= S.abort_s)
        S.finish(S.succ, self.step, "success")
        S.finish(wall, self.step, "wall")
        S.finish(S.fail & ~wall, self.step, "deadline")
        S.finish(abort, self.step, "abort")
        S.finish(S.active & ~fresh & ((base != 2) | (st["stage"] != sq.ST_RAMP)), self.step, "owner")
        S._step = self.step
        start, _ = S.gate(st, base)
        S.begin(start, t, self.step)
        ob_s = obs.clone()
        ob_s["actor"][:] = S.obs(a0, t)
        if self.manual_clock is None:
            ob_m = obs
        else:
            ob_m = obs.clone()
            ob_m["actor"][:, -14] = (ob_m["actor"][:, -14] + self.manual_clock / 10.0).clamp(max=0.8)
        acts = torch.stack([self.pols[0](obs), self.pols[1](ob_m), self.pols[2](obs), self.pols[3](ob_s)])
        eff = torch.where(S.active, torch.full_like(base, 3), base)
        eff0 = torch.where(fresh, eff, self.eff)
        sw = (eff != eff0) & ~fresh
        self.prev = torch.where(fresh, eff, torch.where(sw, eff0, self.prev))
        self.k = torch.where(fresh, torch.full_like(self.k, self.K), torch.where(sw, torch.ones_like(self.k), self.k))
        self.eff = eff
        ar = torch.arange(env.num_envs, device=env.device)
        alpha = (self.k.float() / self.K).clamp(max=1.0).unsqueeze(1)
        a = (1.0 - alpha) * acts[self.prev, ar] + alpha * acts[eff, ar]
        if bool(sw.any()):
            p = sq.pipe_xyz(env)
            v = sq.board_v_fwd(env)
            pitch = torch.rad2deg(tk.front_up_pitch(env))
            W = st["pipe"]["W"]
            for i in sw.nonzero().squeeze(1).tolist():
                self.handoffs.append(dict(env=i, step=self.step, frm=int(eff0[i]), to=int(eff[i]),
                                          stage=int(st["stage"][i]), t=float(t[i]), v=float(v[i]),
                                          y_open=float(p[i, 1] + W / 2), x=float(p[i, 0]), pitch=float(pitch[i]),
                                          wheels_on=float(st["wheels_on"][i])))
        self.log_owner.append(eff.clone())
        self.log_apex.append(st["apex_evt"].clone())
        rz = st.get("ap_rise")
        self.log_rise.append(torch.where(st["apex_evt"], rz, torch.zeros_like(rz)) if rz is not None
                             else torch.full(st["apex_evt"].shape, float("nan"), device=env.device))
        self.log_alpha.append(alpha.squeeze(1).clone())
        self.k = (self.k + 1).clamp(max=self.K)
        self.step += 1
        return a

    def logs(self) -> dict:
        return dict(owner=torch.stack(self.log_owner).cpu().numpy(), apex=torch.stack(self.log_apex).cpu().numpy(),
                    alpha=torch.stack(self.log_alpha).cpu().numpy(),
                    apex_rise=torch.stack(self.log_rise).cpu().numpy())


# ── funnel / picks ────────────────────────────────────────────────────────────────────────────────────────────────────


def counted_passes(apex_col, rise_col, rise_any=None):
    """render6: (T,) bool: the rewarded passes (apex_col) whose apex rise (rise_col, m) is >= PASS_MIN_CM. rise_col
    nan (an env without st["ap_rise"]): the wall-visit max of rise_any (L["rise"]) since the previous pass."""
    apex_col = np.asarray(apex_col, bool)
    rz = np.asarray(rise_col, float).copy()
    if rise_any is not None and np.isnan(rz[apex_col]).any():
        k_prev = 0
        for k in np.nonzero(apex_col)[0]:
            rz[k] = float(np.max(rise_any[k_prev:k + 1]))
            k_prev = k + 1
    return apex_col & (np.nan_to_num(rz) * 100.0 >= PASS_MIN_CM - 1e-6), rz


SQUAT_HEAD_M = 0.10      # render10: head_camera site this close to the deck top (along the deck normal) = a squat


def squat_check(m, qpos, end, eps, agg, thr: float = SQUAT_HEAD_M) -> None:
    """render10 (2026-11-22, --squat-check / MICRODUCK_CHAIN_SQUAT=1, default off; step-5 viewer review: a squat with
    the head on the deck got the green tick). CPU forward kinematics of every logged frame up to each env's end: the
    head_camera site's height above the board's deck_top site along the deck normal (healthy riding ~0.21 m; r9's
    post-STROKE joint-limit squat 0.035-0.08 m). An env below `thr` anywhere is not a clean end
    and not chain_complete (rec["squat"], rec["head_min_cm"]); agg gains squat_envs and the recomputed rates."""
    import mujoco
    d = mujoco.MjData(m)
    hc, dk = m.site("robot/head_camera").id, m.site("board/deck_top").id
    for i, rec in enumerate(eps):
        hmin = np.inf
        for k in range(min(int(end[i]) + 1, qpos.shape[0])):
            d.qpos[:] = qpos[k, i]
            mujoco.mj_kinematics(m, d)
            n = d.site_xmat[dk].reshape(3, 3)[:, 2]
            hmin = min(hmin, float(np.dot(d.site_xpos[hc] - d.site_xpos[dk], n)))
        rec["head_min_cm"] = round(hmin * 100.0, 2)
        rec["squat"] = bool(hmin < thr)
        if rec["squat"]:
            rec["clean_end"] = False
            rec["chain_complete"] = rec["chain_complete_old"] = False
    agg["squat_envs"] = [i for i, r in enumerate(eps) if r["squat"]]
    agg["squat_check"] = f"head_camera < {thr * 100:.0f} cm above deck_top = not clean, not complete"
    agg["chain_complete"] = float(np.mean([r["chain_complete"] for r in eps]))
    agg["chain_complete_old"] = float(np.mean([r["chain_complete_old"] for r in eps]))
    print(f"[squat] {len(agg['squat_envs'])} / {len(eps)} envs squat (head < {thr * 100:.0f} cm): "
          f"{agg['squat_envs']}; chain_complete -> {agg['chain_complete']:.3f}", flush=True)


def chain_funnel(L, end, reason, apex_n, apex_rise, dt, handoffs, apex_log=None, rise_log=None, strokes=None,
                 entries=None, entry_win=(0.4, 0.75), armed_n=None):
    """apex_log: (T, B) the rewarded passes (st["apex_evt"], Chain.logs()["apex"]); rise_log: their apex rise (m,
    Chain.logs()["apex_rise"]). render6: "passes" = the detector's raw count, "passes_2cm" / "pass_cm" = the passes
    >= PASS_MIN_CM (and every pass height); chain_complete uses passes_2cm (chain_complete_old: >= 3 raw passes)."""
    agg, eps = se.funnel(L, end, reason, apex_n, apex_rise, dt)
    tos = L["timeout_terms"]
    one_s = int(round(1.0 / dt))
    for rec, i in zip(eps, range(len(eps))):
        rec["clean_end"] = se._ended_ok(reason[i], tos)
        e1 = int(end[i]) + 1
        if apex_log is not None:
            rec["passes"] = int(apex_log[:e1, i].sum())
            rl = rise_log[:e1, i] if rise_log is not None else np.full(min(e1, len(apex_log)), np.nan)
            ok_, rz = counted_passes(apex_log[:e1, i], rl, L["rise"][:e1, i])
            rec["pass_cm"] = [round(float(z) * 100, 2) for z in rz[apex_log[:e1, i].astype(bool)]]
            rec["passes_2cm"] = int(ok_.sum())
        passes = rec.get("passes", rec["apexes"])
        p2 = rec.get("passes_2cm", passes)
        rec["chain_complete_old"] = bool(rec["max_stage"] >= 4 and passes >= 3 and rec["clean_end"])
        rec["chain_complete"] = bool(rec["max_stage"] >= 4 and p2 >= COMPLETE_PASSES and rec["clean_end"])
        rec["handoffs"] = []
    ho = {}
    for h in handoffs:
        i = h["env"]
        if h["step"] >= int(end[i]):                    # after the env's first episode ended (auto-reset): ignore
            continue
        h = dict(h, alive_1s=bool(int(end[i]) > h["step"] + one_s or eps[i]["clean_end"]))
        eps[i]["handoffs"].append(h)
        ho.setdefault(f"{OWNER_NAMES[h['frm']]}->{OWNER_NAMES[h['to']]}", []).append(h)
    hs = {}
    for name, lst in ho.items():
        ys = np.array([h["y_open"] for h in lst])
        hs[name] = dict(n=len(lst), survival_1s=float(np.mean([h["alive_1s"] for h in lst])),
                        y_open_q=[float(q) for q in np.quantile(ys, [0.0, 0.1, 0.5, 0.9, 1.0])],
                        t_median=float(np.median([h["t"] for h in lst])),
                        v_median=float(np.median([h["v"] for h in lst])),
                        pitch_median=float(np.median([h["pitch"] for h in lst])),
                        wheels_on_median=float(np.median([h["wheels_on"] for h in lst])))
    agg["handoff"] = hs
    if strokes is not None:                             # chain p17 (default None: no "stroke" keys)
        for r in strokes:
            i = r["env"]
            if r["k0"] >= int(end[i]):                  # after the env's first episode ended: ignore
                continue
            r = dict(r)
            if r["outcome"] == "episode_end":
                r["end_reason"] = reason[i]
            else:
                r["alive_1s_after"] = bool(int(end[i]) > r["k1"] + one_s or eps[i]["clean_end"])
            eps[i].setdefault("p17_strokes", []).append(r)
        for rec in eps:
            rec.setdefault("p17_strokes", [])
        agg["stroke"] = stroke_summary([r for rec in eps for r in rec["p17_strokes"]], eps, apex_log, rise_log, end, dt)
        if entries is not None:                         # every gated crossing entry, first episode
            en = [e_ for e_ in entries if e_["k"] < int(end[e_["env"]])]
            vs_ = np.array([e_["v"] for e_ in en]) if en else np.zeros(0)
            agg["stroke"]["entries"] = dict(
                n=len(en), started=int(sum(e_["started"] for e_ in en)),
                v_q=[round(float(x), 3) for x in np.quantile(vs_, [0.0, 0.1, 0.5, 0.9, 1.0])] if len(vs_) else None,
                below_window=int(sum(not e_["started"] and e_["v"] < entry_win[0] for e_ in en)),
                above_window=int(sum(not e_["started"] and e_["v"] > entry_win[1] for e_ in en)),
                arm_events=armed_n,
                approach=int(sum(bool(e_.get("first")) for e_ in en)),
                approach_v_q=([round(float(x), 3) for x in np.quantile([e_["v"] for e_ in en if e_.get("first")],
                                                                        [0.0, 0.5, 1.0])]
                              if any(e_.get("first") for e_ in en) else None),
                approach_started=int(sum(bool(e_.get("first")) and e_["started"] for e_ in en)))
    agg["chain_complete"] = float(np.mean([r["chain_complete"] for r in eps]))
    agg["chain_complete_old"] = float(np.mean([r["chain_complete_old"] for r in eps]))
    agg["chain_complete_def"] = (f"reached RAMP, >= {COMPLETE_PASSES} passes >= {PASS_MIN_CM} cm apex rise, clean end "
                                 "(old: >= 3 raw passes)")
    p2 = [r["passes_2cm"] for r in eps if "passes_2cm" in r]
    if p2:
        agg["passes_2cm_median"], agg["passes_2cm_mean"] = float(np.median(p2)), float(np.mean(p2))
        agg["passes_raw_mean"] = float(np.mean([r["passes"] for r in eps]))
    return agg, eps


def demo_clean_penalty(r: dict) -> float:
    """render9: the RAMP cleanliness terms of demo_score (ramp_clean_terms, 0 without them)."""
    return (DEMO_W_EVENT * (r.get("ramp_deck_events") or 0) + DEMO_W_STEP * (r.get("ramp_deck_steps") or 0)
            + DEMO_W_ROLL * max(0.0, (r.get("ramp_roll_max_deg") or 0.0) - DEMO_ROLL_FREE_DEG))


def demo_score(r: dict, old: bool = False, clean: bool = True) -> float:
    """render6: counts the passes >= PASS_MIN_CM (old=True: the detector's raw passes, the chain v1 pick, no clean
    terms). render9: clean=True subtracts demo_clean_penalty (deck contacts / board roll in RAMP)."""
    p = r.get("passes", r["apexes"]) if old else r.get("passes_2cm", r.get("passes", r["apexes"]))
    sc = (10.0 * r["clean_end"] + 4.0 * r["max_stage"] + min(p, 4) + 0.2 * r["apex_rise_cm"]
          + 2.0 * (r.get("manual_frac") or 0.0) + float((r.get("manual_longest_s") or 0.0) >= 1.0)
          - 0.5 * (r.get("exit_tail_strike_s") or 0.0))
    return sc - (demo_clean_penalty(r) if clean and not old else 0.0)


def ramp_clean_terms(env, L, CL, end, eps, dt):
    """render9: per episode, over its RAMP frames up to clip_end (what the clip shows): deck contact steps / events
    (tail_floor sensor or deck geoms touching the terrain in the render copy, = the HUD chips), of which tail / core+nose,
    and the max board roll relative to the surface (deg). Written into eps[i]."""
    rs = se.RenderScene.get(env)
    for i, rec in enumerate(eps):
        e_i = int(end[i])
        e = clip_end(e_i, env_passes(L, CL, i, e_i)[0], dt)
        ks = [k for k in range(e) if int(L["stage"][k, i]) == 4]
        con, tail, scr, roll = [], 0, 0, 0.0
        for k in ks:
            rs.set_state(L["qpos"][k, i])
            tch = rs.deck_touch()
            t_ = bool(L["tail"][k, i]) or "tail" in tch
            c_ = bool(tch & {"core", "nose"})
            tail += t_
            scr += c_
            con.append(t_ or c_)
            roll = max(roll, rs.board_roll_rel())
        c = np.array(con, bool)
        rec["ramp_deck_steps"] = int(c.sum())
        rec["ramp_deck_events"] = int(c[0] + (c[1:] & ~c[:-1]).sum()) if len(c) else 0
        rec["ramp_tail_steps"], rec["ramp_scrape_steps"] = int(tail), int(scr)
        rec["ramp_roll_max_deg"] = round(roll, 2)
        rec["demo_clean_penalty"] = round(demo_clean_penalty(rec), 3)


def make_picks(eps, old: bool = False, clean: bool = True) -> dict:
    sc = np.array([r["score"] for r in eps])
    order = np.argsort(sc, kind="stable")
    # render10 (tech review 2026-11-22): with --squat-check, "best" skips squat runs (score ignores the squat)
    ok_ = [int(j) for j in order if not eps[int(j)].get("squat", False)]
    return dict(typical=int(order[len(order) // 2]), best=ok_[-1] if ok_ else int(order[-1]),
                demo=int(np.argmax([demo_score(r, old=old, clean=clean) for r in eps])))


# ── render ────────────────────────────────────────────────────────────────────────────────────────────────────────────

# stage cameras, the render-time scene copy, the camera rig and the HUD live in sequence_eval (shared with its videos)
CAMS = se.CAMS
STAGE_CAM = se.STAGE_CAM


CLIP_AFTER_APEX_S = 0.6    # s: render8: the clip ends this long after the last counted wall apex


def clip_end(end_i: int, apex_col: np.ndarray, dt: float, after_s: float = CLIP_AFTER_APEX_S) -> int:
    """Frames to show: cut after_s after the LAST counted pass of apex_col (render6: the counted passes >= PASS_MIN_CM,
    counted_passes), up to the episode end; the whole episode without a counted pass. render8 (chain_final review: the
    clip ran 1.2 s past the 3rd pass and, with fewer passes, to the episode end, rolling on the flat): the policy run
    is unchanged, only the clip is trimmed."""
    k = np.nonzero(apex_col[:end_i])[0]
    if len(k):
        return min(end_i, int(k[-1]) + int(round(after_s / dt)) + 1)
    return end_i


def _runs_of(mask):
    """[(start, end)) index runs of a bool array (None -> [])."""
    if mask is None:
        return []
    m = np.asarray(mask, bool)
    d = np.diff(np.r_[0, m.astype(int), 0])
    return list(zip(np.nonzero(d == 1)[0].tolist(), np.nonzero(d == -1)[0].tolist()))


def env_passes(L, CL, idx, end_i: int):
    """render6: (counted (T,) bool, apex rise (T,) m) of env idx's rewarded passes up to its episode end; counted =
    apex rise >= PASS_MIN_CM."""
    e1 = min(int(end_i) + 1, CL["apex"].shape[0])
    rl = CL["apex_rise"][:e1, idx] if "apex_rise" in CL else np.full(e1, np.nan)
    return counted_passes(CL["apex"][:e1, idx], rl, L["rise"][:e1, idx])


def render_chain(env, L, CL, end, idx, out, dt, footer, label="", end_msg="", title_s=None, hold_s=None,
                 seconds_cap=None, title="Microduck · push → manual → half-pipe (RL)", end_ok="auto",
                 end_sub="", credits=None, fast2x: bool = False, n_pol: int = 3):
    """Render one env of the chain on sequence_eval's render-time scene copy (textured floor, shadows, half-pipe
    copings) with sequence_eval's offline camera plan (no lag; rear three-quarter through PUSH / STEP ON, a slow orbit
    to the side profile for MANUAL and the set-down, then a slow blend to the ramp camera once the rider carves in) and
    its shared frame pipeline (se.clip_frames, render5): the clip starts se.PUSH_LEAD_S before the first push
    (se.clip_start) on an INTRO_STILL_S still with the title card, which hands over to a persistent caption at the same
    top-left anchor; stage panel bottom-left (per-stage telemetry), honesty note bottom-right, active-policy tag
    top-right; the clip ends 1.2 s after the 3rd pass, then the se.END_CARD_S end card (freeze, slow fade).
    title_s / label: unused (kept for callers; the output name carries the label).
    render6: footer "" = no running honesty note (the disclosure moves to the closing card: credits, a list of
    (text, style) for se.credits_frames, se.CREDITS_S after the end card); passes on screen = the passes >= PASS_MIN_CM
    (env_passes); fast2x: flat coasting stretches > se.FAST_MIN_S play at 2x with a "2x speed" chip (se.fast_mask).
    render10 (2026-11-21, author: real time only): fast2x defaults to False = every frame at 1x, no 2x badge; opt back in
    with --speedup / MICRODUCK_CHAIN_SPEEDUP=1."""
    import imageio.v2 as imageio

    rs = se.RenderScene.get(env)
    F = se.hud_fonts()
    origin = env.scene.env_origins[idx].detach().cpu().numpy()
    cp, _ = env_passes(L, CL, idx, int(end[idx]))
    e = clip_end(int(end[idx]), cp, dt)
    if seconds_cap is not None:
        e = min(e, int(round(seconds_cap / dt)))
    stage = L["stage"][:e, idx].astype(int)
    # render8: "wall ride k" counts from the START of the counted ride's wall visit (the apex event fires after the
    # apex, so the old count still said 2 through the climb of ride 3)
    rl_ = L["rise"][:e, idx]
    mark = np.zeros(e, int)
    for kp in np.nonzero(cp[:e])[0]:
        k_ = int(kp)
        while k_ > 0 and float(rl_[k_ - 1]) > 0.002:
            k_ -= 1
        mark[k_] += 1
    passes = np.cumsum(mark)
    setdown, t_wd = se.setdown_track(stage, L["wheels"][:e, idx], dt)
    # render7: CARVE IN on the high carve camera (se.CAMS["carve"]), the ramp camera from RAMP entry
    keys = se.cam_keys(stage, setdown, t_wd, dt, ramp_ok=rs.pipe is not None, carve_cam=True)
    ff = (L["lf"][:e, idx].astype(bool) | L["rf"][:e, idx].astype(bool)) if "lf" in L and "rf" in L else None
    labels = se.stage_labels(stage, L["cmd"][:e, idx] if "cmd" in L else None,
                             L["both_on"][:e, idx] if "both_on" in L else None, setdown, dt, passes=passes, carve=True,
                             foot_floor=ff,                # render7: STEP ON from the last floor contact (the feet)
                             front_down=L["fwd"][:e, idx] if "fwd" in L else None,      # render8: MANUAL / SET-DOWN
                             wheels=L["wheels"][:e, idx])                              # on the wheel contacts
    poses = [rs.set_state(L["qpos"][k, idx]) for k in range(e)]
    kc = next((k for k in range(e) if keys[k] == "carve"), None)
    carve_dir = 90.0
    if kc is not None:                                     # render7: heading board -> pipe centre at the carve start
        bxy = poses[kc][0][:2] - origin[:2]
        carve_dir = math.degrees(math.atan2(-bxy[1], -bxy[0])) if float(np.hypot(*bxy)) > 0.3 else poses[kc][1]
    cams = se.CamRig(dt, origin, carve_dir=carve_dir).plan(keys, [p[0] for p in poses], [p[1] for p in poses])
    # render7: real frames from se.START_STILL_S before the first push (or frame 0), topped up with a held first frame
    k0 = se.clip_start(stage, L["v"][:e, idx], dt, lead_s=se.START_STILL_S)
    n_still = se.intro_hold_frames(stage, L["v"][:e, idx], dt, k0)
    # render9 (review: the half-pipe was off-screen until ~f440): an establishing wide shot under the title card,
    # se.EST_S of held frame-k0 state from high behind the start line looking down the run to the pipe (far fog, a slow
    # push-in, a "half-pipe ahead" pointer), then a cut to the push camera at k0 (0.2 s before the first push). It
    # replaces the held start frames (the real frames k0 .. first push stay).
    est_cams = est_mark = None
    if rs.pipe is not None and len(stage) and int(stage[0]) == 0:         # a start-line clip
        k0 = se.clip_start(stage, L["v"][:e, idx], dt, lead_s=se.EST_LEAD_S)   # the cut: EST_LEAD_S before the push
        n_real = int(np.argmax(L["v"][:e, idx] > se.PUSH_V_START)) - k0                  # real frames k0 .. first push
        n_still = int(round(se.EST_S / dt))
        b0 = poses[k0][0]
        P = rs.pipe
        mouth = np.array([P["cx"], P["cy"] - P["W"] / 2 + 0.9, float(origin[2]) + 0.04])
        u = mouth[:2] - b0[:2]
        D = float(np.hypot(*u))
        hd = math.degrees(math.atan2(u[1], u[0]))
        # long lens (se.EST_FOVY) from ~5.5 m behind the start, looking down the run: rider lower-middle, the pipe's
        # two walls upper-middle (framing solved with se.cam_project: rider ~460-560 px, pipe ~300 px at D ~ 10 m)
        look = np.array([b0[0] + 0.5 * u[0], b0[1] + 0.5 * u[1], float(origin[2]) + 0.10])
        d0 = 0.5 * D + 6.0
        est_cams = [(hd + 6.0, -7.0, d0 * (1.0 - 0.06 * se._smoothstep(j / max(1, n_still - 1))), look)
                    for j in range(n_still)]
        est_mark = mouth
        print(f"[render] establishing shot {n_still * dt:.2f} s (+ {n_real * dt:.2f} s real before the push): start "
              f"-> pipe {D:.1f} m, cam az {hd + 6:.0f} dist {d0:.1f}", flush=True)
    fast = (se.fast_mask(stage, labels, L["rise"][:e, idx], t_wd, dt, keys=keys, min_s=se.FAST_MIN_S_CHAIN,
                         pipe_flat=True)                   # render8: + flat stretches between the walls, labelled
            if fast2x else None)

    if bool((CL["owner"][:e, idx] == 3).any()):          # chain p17: the STROKE owner's frames read FOOT PUSH
        labels = list(labels)
        for k_ in range(e):
            if int(CL["owner"][k_, idx]) == 3:
                labels[k_] = "FOOT PUSH"

    def tag_fn(k):                                       # render9: small, in the stage panel, no "(blending)"
        return "policy: " + ("push", "manual", "half-pipe", "foot-push")[int(CL["owner"][k, idx])]
    nf2 = int(fast.sum()) if fast is not None else 0
    def runs(lab_):
        r_, k_ = [], 0
        while k_ < e:
            if labels[k_] == lab_:
                j_ = k_
                while j_ < e and labels[j_] == lab_:
                    j_ += 1
                r_.append(f"{k_ * dt:.2f}-{j_ * dt:.2f}")
                k_ = j_
            else:
                k_ += 1
        return ",".join(r_)
    print(f"[render] {os.path.basename(out)}: clip from {k0 * dt:.2f} s (+ {n_still * dt:.2f} s held first frame) to "
          f"{e * dt:.2f} s; 2x speed over {nf2 * dt:.2f} s of it; carve cam from "
          f"{'-' if kc is None else f'{kc * dt:.2f}'} s (dir {carve_dir:.0f} deg); STEP ON {runs('STEP ON')} s; "
          f"CARVE IN {runs('CARVE IN')} s; MANUAL {runs('MANUAL')} s; SET-DOWN {runs('SET-DOWN')} s; 2x frames "
          f"{','.join(f'{a_ * dt:.2f}-{b_ * dt:.2f}' for a_, b_ in _runs_of(fast))} s", flush=True)
    last = None
    with imageio.get_writer(out, fps=int(round(1 / dt)), codec="libx264", quality=8, macro_block_size=8) as w:
        # render9: no "phase scheduler" sub line, no top-right chip (the title card ran into it at f20), no persistent
        # caption after the card: at most the stage panel + the 2x badge on screen
        for fr, raw in se.clip_frames(rs, F, L, idx, origin, stage, labels, cams, setdown, dt, k0=k0, title=title,
                                      sub=f"one robot · {n_pol} learned policies, one after another", note=footer,
                                      tag_fn=None, ptag_fn=tag_fn, fast=fast, n_still=n_still, est_cams=est_cams,
                                      est_mark=est_mark, caption=False, fast_big=True):
            w.append_data(fr)
            last = raw                                   # the end card starts from the frame without the HUD
        if last is not None:
            # render6: end_ok True / False / None (neutral icon) is used as given; "auto" = the old guess from end_msg
            ok = end_ok if end_ok != "auto" else (True if end_msg.endswith("passes") else
                                                  (None if "time-out" in end_msg else False))
            for fr in se.end_card_frames(last, end_msg, end_sub, dt, hold_s, F, ok=ok, foot_l=title, foot_r=footer,
                                         tag=None):
                w.append_data(fr)
            if credits:
                for fr in se.credits_frames(credits, dt, F):
                    w.append_data(fr)
    return e


# ── main ──────────────────────────────────────────────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", required=True)
    ap.add_argument("--manual", required=True)
    ap.add_argument("--ramp", required=True)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--out", default="")
    ap.add_argument("--label", default="chain")
    ap.add_argument("--blend", type=int, default=4)
    ap.add_argument("--exit-gate", default="wheels", choices=("wheels", "stage"))
    ap.add_argument("--curriculum-step", type=int, default=None,
                    help="env.common_step_counter after the loads (default: the SeqRamp run's last step, "
                         "max_iterations x 24 = every curriculum at its final stage; the RAMP carve command is "
                         "pinned to the --ramp checkpoint's own stage, ramp_carve_of)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--manual-clock", type=float, default=None, metavar="C",
                    help="add C s to the trick-obs clock in the manual policy's input only (it trained with a clock "
                         "offset of U(3, 7) s; in the chain it takes over at t ~ 1.3-2 s)")
    ap.add_argument("--settle-gate", type=float, default=None, metavar="S",
                    help="push->manual hand-off: the push policy keeps SETTLE until stage_t >= S and both soles on the "
                         "deck >= 0.3 s and four wheels down >= 0.2 s; MANUAL entry hands over at the latest "
                         "(default: hand over at SETTLE entry)")
    ap.add_argument("--stance-gate", action="store_true",
                    help="p9b push policies: hand over once the hand-off stance (stance_tol, default the push cfg's "
                         "final P9B stage) is held stance_hold_s, or after --stance-max-s in SETTLE, or at MANUAL "
                         "entry; pair with --seq-override 'settle_s=(1.2, 1.5)' so the scheduler does not start "
                         "MANUAL before the stance can be reached")
    ap.add_argument("--stance-max-s", type=float, default=1.5)
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--render", default="typical,best,demo", help="clips to render (comma list; render5)")
    ap.add_argument("--start-pose", default=None, metavar="DX,DY,DYAW",
                    help="chainfix: shift / turn the start spawn (m, m, deg): aim the start so the push / manual "
                         "policies' own drift ends on the approach line (see --chainfix)")
    ap.add_argument("--chainfix", action="store_true",
                    help="chainfix preset: --start-pose CHAINFIX_START_POSE (unless given) and seq_cfg ramp_gate_x "
                         "CHAINFIX_GATE_X (EXIT -> RAMP only in front of the pipe opening)")
    ap.add_argument("--spawn", default="start",
                    help="spawn type the chain starts from (default start = the whole chain; e.g. step_on / exit to "
                         "exercise a hand-off with policies that do not complete the earlier phases)")
    ap.add_argument("--title", default="Microduck · push → manual → half-pipe (RL)",
                    help="clean video title (footer + opening title); render only. render10: with --stroke and the "
                         "default title, the title names the foot-push")
    ap.add_argument("--squat-check", action="store_true",
                    default=os.environ.get("MICRODUCK_CHAIN_SQUAT", "0") == "1",
                    help="render10: an env whose head drops below SQUAT_HEAD_M above the deck is not a clean end / "
                         "not chain_complete (no green tick, not picked as a clean run); default off")
    ap.add_argument("--rolling", type=float, default=None, metavar="X",
                    help="pin the board wheel rolling-friction scale (event board_rolling, trained range 0.3-1.0) to X "
                         "for playback, e.g. 0.3 = smooth concrete / good bearings (low end of every policy's range)")
    ap.add_argument("--speedup", action="store_true",
                    default=os.environ.get("MICRODUCK_CHAIN_SPEEDUP", "0") == "1",
                    help="render10: play flat coasting / carve-in stretches at a labelled 2x (render6-9 behaviour). "
                         "Default off = real time only (author request 2026-11-21); env MICRODUCK_CHAIN_SPEEDUP=1")
    ap.add_argument("--no-2x", action="store_true",
                    help="render6: force 1x (now the default; overrides --speedup)")
    ap.add_argument("--stroke", default=os.environ.get("MICRODUCK_CHAIN_STROKE") or None, metavar="CKPT",
                    help="chain p17 (default off): a p17 push checkpoint as owner 3 = STROKE, one foot-push stroke per "
                         "forward half-pipe flat crossing (env MICRODUCK_CHAIN_STROKE)")
    ap.add_argument("--stroke-vmin", type=float, default=0.4, help="STROKE hand-over window: min board speed (m/s)")
    ap.add_argument("--stroke-vmax", type=float, default=0.75, help="... max board speed (the trained pipe window)")
    ap.add_argument("--stroke-deadline", type=float, default=None,
                    help="STROKE hand-back deadline (s): the p17 deadline the checkpoint reached (Metrics/seq_p17_deadline)")
    ap.add_argument("--stroke-abort-s", type=float, default=0.8, help="hand back if no push-foot touchdown by then (s)")
    ap.add_argument("--stroke-dv", type=float, default=0.07, help="one-stroke target above the speed (training U(0.04, 0.10))")
    ap.add_argument("--stroke-alpha", type=float, default=None,
                    help="the p17b bridge alpha the STROKE checkpoint trained to (log / funnel note only)")
    ap.add_argument("--seq-override", action="append", default=[], metavar="KEY=VALUE",
                    help="extra seq_cfg value (Python literal), repeatable; e.g. step_cmd=False for a push policy "
                         "trained without the step command (s2c)")
    a = ap.parse_args()
    _rf = os.environ.get("MICRODUCK_CHAIN_RAMP_FOLLOW")  # render10 (viewer review: the rider sat cut off at the frame
    if _rf:                                              # edge during the foot push); default unset = se.RAMP_FOLLOW 0.3
        se.RAMP_FOLLOW = float(_rf)
    _rd = os.environ.get("MICRODUCK_CHAIN_RAMP_DIST")    # render10: closer half-pipe camera (viewer review: rider ~170 px,
    if _rd:                                              # the 80 ms foot plant unreadable); default unset = 1.55 m
        se.CAMS["ramp"] = se.CAMS["ramp"][:2] + (float(_rd),) + se.CAMS["ramp"][3:]
    if a.stroke and a.title == ap.get_default("title"):   # render10 (viewer review: the title omitted the foot-push)
        a.title = "Microduck · push → manual → half-pipe + foot-push (RL)"
    import ast
    seq_ov = {kv.split("=", 1)[0]: ast.literal_eval(kv.split("=", 1)[1]) for kv in a.seq_override}
    start_pose = tuple(float(v) for v in a.start_pose.split(",")) if a.start_pose else None
    if a.chainfix:
        start_pose = start_pose or CHAINFIX_START_POSE
        seq_ov.setdefault("ramp_gate_x", CHAINFIX_GATE_X)
    if a.stance_gate:                                   # p9brev: track the hand-off stance in the chain env
        from micro_skater.tasks import microduck_skate_seq_push_env_cfg as pcfg
        seq_ov.setdefault("stance_tol", tuple(pcfg.P9B_TOL_STAGES[-1][1]))
        seq_ov.setdefault("stance_hold_s", pcfg.P9B_HOLD_S)
    out = a.out or os.path.join(OUT_ROOT, f"chain_{a.label}")
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    ckpts = [a.push, a.manual, a.ramp] + ([a.stroke] if a.stroke else [])
    vec, pols, seed, agent_cfg = build_chain(ckpts, a.n, a.seed, seq_ov, spawn=a.spawn, start_pose=start_pose)
    env = vec.unwrapped
    roll_trained = None
    if a.rolling is not None:                           # rolling: pin wheel rolling friction (in-distribution value)
        roll_trained = tuple(float(v) for v in env.event_manager.get_term_cfg("board_rolling").params["ranges"])
        env.event_manager.get_term_cfg("board_rolling").params["ranges"] = (a.rolling, a.rolling)
        print(f"[chain] board_rolling pinned to {a.rolling}", flush=True)
    dt = env.step_dt
    env.common_step_counter = a.curriculum_step if a.curriculum_step is not None else agent_cfg.max_iterations * 24
    print(f"[chain] n {a.n}  seconds {a.seconds}  common_step_counter {env.common_step_counter}  "
          f"exit_gate {a.exit_gate}  blend {a.blend}", flush=True)
    stroke = None
    if a.stroke:                                        # chain p17 (default off)
        scfg = stroke_cfg_of(a.stroke)
        # p17d review: default = the pipe deadline the checkpoint trained with (p17d deadline_pipe), else 3.0 s
        if a.stroke_deadline is None:
            a.stroke_deadline = scfg["deadline_pipe"] if scfg.get("deadline_pipe") is not None else 3.0
        elif scfg.get("deadline_pipe") is not None and abs(a.stroke_deadline - scfg["deadline_pipe"]) > 1e-6:
            print(f"[stroke] WARNING --stroke-deadline {a.stroke_deadline} != the trained pipe deadline "
                  f"{scfg['deadline_pipe']}", flush=True)
        stroke = Stroke(env, scfg, vmin=a.stroke_vmin, vmax=a.stroke_vmax, deadline=a.stroke_deadline,
                        abort_s=a.stroke_abort_s, dv=a.stroke_dv, train_alpha=a.stroke_alpha)
        print(f"[stroke] {a.stroke}\n[stroke] cfg {json.dumps(scfg)}\n[stroke] window {a.stroke_vmin}-{a.stroke_vmax} "
              f"m/s, deadline {a.stroke_deadline} s, abort {a.stroke_abort_s} s, dv {a.stroke_dv}\n[stroke] "
              + stroke.note(), flush=True)
    chain = Chain(env, pols, blend=a.blend, exit_gate=a.exit_gate, settle_gate=a.settle_gate,
                  manual_clock=a.manual_clock, stance_gate=a.stance_gate, stance_max_s=a.stance_max_s, stroke=stroke)
    chain.reset_log()
    L, end, reason, apex_n, apex_rise = se.run_spawn(vec, chain, a.spawn, a.seconds)
    CL = chain.logs()
    assert CL["owner"].shape[0] == L["stage"].shape[0], "chain log out of step with run_spawn"
    if stroke is not None:
        stroke.close_all(chain.step)
    agg, eps = chain_funnel(L, end, reason, apex_n, apex_rise, dt, chain.handoffs, apex_log=CL["apex"],
                            rise_log=CL["apex_rise"], strokes=stroke.records if stroke is not None else None,
                            entries=stroke.entries if stroke is not None else None,
                            entry_win=(a.stroke_vmin, a.stroke_vmax), armed_n=stroke.g.n_armed if stroke is not None else None)
    if stroke is not None:
        agg["stroke"].update(note=stroke.note(), cfg=stroke.cfg, window=[a.stroke_vmin, a.stroke_vmax],
                             deadline=a.stroke_deadline, abort_s=a.stroke_abort_s, dv=a.stroke_dv,
                             hand_over_alpha=STROKE_ALPHA, train_alpha=a.stroke_alpha)
    if a.squat_check:                                   # render10 (default off): a squat is not a clean ride
        squat_check(env.sim.mj_model, L["qpos"], end, eps, agg)
        print("STROKE " + json.dumps(agg["stroke"]), flush=True)
    print("RESULT chain: " + json.dumps(agg), flush=True)
    # handoff sweep summary (reporting only, outside the funnel): hand-off stage / timing, set-down = four wheels down
    # >= se.SETDOWN_WHEELS_S in EXIT or later (sequence_eval.setdown_track, the chain's wheels gate)
    sd = []
    for i in range(len(eps)):
        e_ = int(end[i])
        done, _ = se.setdown_track(L["stage"][:e_, i].astype(int), L["wheels"][:e_, i], dt)
        sd.append(bool(done.any()))
    pm = [h for r in eps for h in r["handoffs"] if h["frm"] == 0]
    summ = dict(settle_gate=a.settle_gate, manual_clock=a.manual_clock, n=len(eps),
                handoff_survival_1s=float(np.mean([h["alive_1s"] for h in pm])) if pm else None,
                handoff_in_manual=float(np.mean([h["stage"] >= 2 for h in pm])) if pm else None,
                reached_MANUAL=agg["reached_MANUAL"], reached_EXIT=agg["reached_EXIT/approach"],
                setdown_ok=float(np.mean(sd)), end_reasons=agg["end_reasons"])
    print("HANDOFF " + json.dumps(summ), flush=True)
    from micro_skater.tasks import skate_sequence as _sq
    ramp_clean_terms(env, L, CL, end, eps, dt)          # render9: deck contacts / roll in RAMP (demo score terms)
    picks = make_picks(eps)
    picks_v8 = make_picks(eps, clean=False)             # render9: the render8 demo rule (no clean terms), reported only
    rank = sorted(range(len(eps)), key=lambda j: -demo_score(eps[j]))[:8]
    print("DEMO_RANK " + json.dumps([dict(env=j, score=round(demo_score(eps[j]), 3),
                                          score_v8=round(demo_score(eps[j], clean=False), 3),
                                          passes_2cm=eps[j].get("passes_2cm"), complete=eps[j]["chain_complete"],
                                          **{kk: eps[j].get(kk) for kk in ("ramp_deck_steps", "ramp_deck_events",
                                                                            "ramp_tail_steps", "ramp_scrape_steps",
                                                                            "ramp_roll_max_deg", "demo_clean_penalty")})
                                     for j in rank]), flush=True)
    print(f"[demo] pick env {picks['demo']} (render8 rule: env {picks_v8['demo']})", flush=True)
    picks_old = make_picks(eps, old=True)               # render6: the chain v1 rule (raw passes), reported only
    for k_, i in picks.items():
        print(f"pick {k_}: env {i} " + json.dumps({kk: v for kk, v in eps[i].items() if kk not in ("handoffs", "p17_strokes")}), flush=True)
    json.dump(dict(agg=agg, episodes=eps, handoff_summary=summ), open(os.path.join(out, "funnel.json"), "w"), indent=1)
    pk = dict(checkpoints=dict(zip(OWNER_NAMES, ckpts)), picks=picks, seed=seed, n=a.n, seconds=a.seconds,
                   exit_gate=a.exit_gate, blend=a.blend, seq_override=seq_ov, spawn=a.spawn, settle_gate=a.settle_gate,
                   manual_clock=a.manual_clock, start_pose=start_pose,
                   common_step_counter=int(env.common_step_counter),
                   demo_score={k_: demo_score(eps[i]) for k_, i in picks.items()},
                   picks_old=picks_old, demo_score_old={k_: demo_score(eps[i], old=True) for k_, i in picks_old.items()},
                   # render9: the demo score's clean terms (weights, the picks' values) and the render8 pick without them
                   demo_clean_weights=dict(event=DEMO_W_EVENT, step=DEMO_W_STEP, roll_per_deg=DEMO_W_ROLL,
                                           roll_free_deg=DEMO_ROLL_FREE_DEG),
                   demo_clean_terms={k_: {kk: eps[i].get(kk) for kk in ("ramp_deck_steps", "ramp_deck_events",
                                                                        "ramp_tail_steps", "ramp_scrape_steps",
                                                                        "ramp_roll_max_deg", "demo_clean_penalty")}
                                     for k_, i in picks.items()},
                   picks_v8=picks_v8, demo_score_v8={k_: demo_score(eps[i], clean=False) for k_, i in picks_v8.items()},
                   pass_min_cm=PASS_MIN_CM, chain_complete=agg["chain_complete"],
                   chain_complete_old=agg["chain_complete_old"], rolling=a.rolling, rolling_trained=roll_trained,
                   # pipe_h (2026-11-21): the TRUE pipe the chain ran on (MICRODUCK_PIPE_H) and the observed one
                   pipe=dict(_sq.PIPE), obs_pipe=dict(_sq.OBS_PIPE), pipe_h_env=os.environ.get("MICRODUCK_PIPE_H"))
    if stroke is not None:                              # chain p17: only with --stroke (picks.json unchanged without)
        pk.update(stroke=dict(ckpt=a.stroke, cfg=stroke.cfg, window=[a.stroke_vmin, a.stroke_vmax],
                              deadline=a.stroke_deadline, abort_s=a.stroke_abort_s, dv=a.stroke_dv,
                              hand_over_alpha=STROKE_ALPHA, train_alpha=a.stroke_alpha))
    json.dump(pk, open(os.path.join(out, "picks.json"), "w"), indent=1)
    # demo qpos for the Isaac Sim replay (cut like the clip)
    m = env.sim.mj_model
    # pipe_h: the compiled model (this run's terrain) next to demo_qpos.npz, so export_chain.py exports the pipe the run
    # used (its fallback model.mjb is chain_v1's 12 cm pipe)
    import mujoco
    mujoco.mj_saveModel(m, os.path.join(out, "model.mjb"))
    di = picks["demo"]
    e = clip_end(int(end[di]), env_passes(L, CL, di, int(end[di]))[0], dt)
    np.savez_compressed(os.path.join(out, "demo_qpos.npz"), qpos=L["qpos"][:e, di], dt=dt, stage=L["stage"][:e, di],
                        owner=CL["owner"][:e, di], apex=CL["apex"][:e, di],
                        origin=env.scene.env_origins[di].detach().cpu().numpy(),
                        jnt_names=np.array([m.joint(j).name for j in range(m.njnt)]), jnt_qposadr=m.jnt_qposadr,
                        jnt_type=m.jnt_type)
    if a.no_render:
        return
    cc = agg["chain_complete"] * 100
    for which in a.render.split(","):
        i = picks[which]
        # render6: no running footer; the disclosure is a closing card after the end card
        sp_note = "" if a.spawn == "start" else f" · from the {a.spawn} spawn"
        # render7: one accurate wording per pick (the demo and best clips are different runs: the demo pick maximises
        # demo_score, "best" the eval score) and a short closing card: 2 lines + 1 small footnote, larger type
        # render8 (chain_final review): a 2-line closing card + 1 footnote, plain words; the rate was measured on the
        # same runs (same pinned friction), and each pick's wording says how it was chosen
        # render9 (review): plain English. "most complete" = the demo pick (top demo score: phases reached, wall rides,
        # manual, a clean half-pipe); the rate is chain_complete over the same runs (same settings)
        # review B: the eval score (stage + 0.25 x raw apexes + 0.01 x episode length) effectively picks the longest run;
        # the demo pick is a showcase pick (few runs reach 3 counted rides), said so plainly
        best_txt = (f"Shown: the longest complete run of {a.n}" if eps[picks["best"]]["chain_complete"]
                    else f"Shown: the longest run of {a.n}")
        if a.squat_check:                    # render10 (viewer review): the eval score is not episode length; say what it is
            best_txt = f"Shown: the top eval-score run of {a.n}" + ("" if eps[picks["best"]]["chain_complete"]
                                                                    else " (did not finish)")
        shown = {"typical": f"Shown: a typical run (median of {a.n})", "best": best_txt,
                 "demo": f"Shown: a showcase run, picked from {a.n}"}[which]
        credits = [(f"Learned in simulation · {len(pols)} AI policies chained", "head"),
                   (f"{shown}{sp_note}", "body"),
                   (f"{cc:.0f}% of the {a.n} runs finish with {COMPLETE_PASSES}+ wall rides", "body")]
        if a.rolling is not None:
            lo, hi = roll_trained if roll_trained else (a.rolling, a.rolling)
            where = ("low end of" if abs(a.rolling - lo) < 1e-6 else "high end of" if abs(a.rolling - hi) < 1e-6
                     else "within" if lo - 1e-6 <= a.rolling <= hi + 1e-6 else "outside")
            credits.append((f"wheel rolling resistance {a.rolling:g}× ({where} trained range)", "fine"))
        r_ = eps[i]
        e_i = int(end[i])
        cp, rz = env_passes(L, CL, i, e_i)
        e_clip = clip_end(e_i, cp, dt)
        # render7: pass heights from the same source as the HUD (the logged wall height L["rise"], the HUD value and its
        # peak-hold line): the peak of each counted pass's wall visit, same .1f rounding (chain_r6: HUD 5.8, card 5.7)
        rl_ = L["rise"][:e_clip, i]
        hts = []
        for kp in np.nonzero(cp[:e_clip])[0]:
            k_ = int(kp)
            while k_ > 0 and float(rl_[k_ - 1]) > 0.002:
                k_ -= 1
            hts.append(float(rl_[k_:int(kp) + 1].max()) * 100)
        reached = r_["max_stage"] >= 4
        # render8 (plain language): "3 wall rides · highest 6.3 cm" instead of "half-pipe · 6.3 / 2.3 / 4.1 cm"
        rides = (f"{len(hts)} wall ride{'' if len(hts) == 1 else 's'} · highest {max(hts):.1f} cm" if hts else "")
        if reached and r_["clean_end"]:
            msg = rides or "half-pipe · no wall ride ≥ 2 cm"
            sub = f"a wall ride counts from {PASS_MIN_CM:g} cm up the wall" if hts else f"highest {r_['apex_rise_cm']:.1f} cm"
        elif r_["clean_end"]:
            msg, sub = "time-out", ""
        else:
            msg = "ended early"
            sub = r_["reason"].replace("_", " ").replace(",", ", ")
            if hts:
                sub = rides + " · " + sub
        path = os.path.join(out, f"chain_{which}.mp4")
        # render6: green check only for chain_complete (RAMP, >= 2 passes >= 2 cm, clean end); otherwise neutral
        nf = render_chain(env, L, CL, end, i, path, dt, "", label=a.label, end_msg=msg, title=a.title,
                          end_ok=True if r_["chain_complete"] else None, end_sub=sub, credits=credits,
                          fast2x=a.speedup and not a.no_2x, n_pol=len(pols))
        print(f"wrote {path} ({nf} frames + end card + closing card) end card: {msg} | {sub} | "
              f"complete {r_['chain_complete']}", flush=True)


if __name__ == "__main__":
    main()
