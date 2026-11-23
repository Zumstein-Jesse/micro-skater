# micro-skater: part of the author's skateboard add-on for microduck_rl. Apache-2.0 (see LICENSE).
"""Skate SEQUENCE approach + half-pipe SPECIALIST (2026-11-20). Task Mjlab-Skate-SeqRamp-MicroDuck.

The third of three specialists (push +
step-on / manual / approach + ramp) with the SAME 114D obs and actions as Mjlab-Skate-Sequence-MicroDuck, chained by
scheduler stage at playback (micro_skater.eval.chain_play, which builds this cfg with
chain=True). MDP terms new to this task: skate_seq_ramp.py (balance reference up_eff in RAMP, apex counter, park cost,
energy potential, ramp_done success).

  - spawns: EXIT (the manual hand-off, 85 % on four wheels), ramp_bottom (70 % arriving along the pipe), ramp_wall;
    clock offset U(5, 9) s on the trick-obs clock (the chain reaches EXIT at t ~ 7-9 s); 12 s episodes.
  - entry: pursuit target x 0.35 -> 0.45, 1.0 -> 0.6 m inside the open end, kappa cap 0.4 -> 0.3 (enter at ~37 deg).
  - success (time_out=True): ramp_done = 8 s in RAMP or 5 alternating apexes. past_pipe_end stays a time-out.
  - upright / posture / riding_tilt / fell_over use up_eff in RAMP.
  - pumping is paid through ramp_energy (signed potential on v^2/2 + g * rise, from 600 it), not through a CoM pattern.
  - review (rrev): exit_time -2 /s in the EXIT stage (EXIT paid ~4.2 /s vs ~2.3-2.6 /s in RAMP on the warm start: the
    ride terms pay only in RIDE, so delaying the RAMP entry paid ~2 /s).

Warm start (MICRODUCK_WARM_START=1, iteration counter restarts at 0) from s2c model_2500 with the action std clamped to
<= 0.6:

Curriculum stages are in iterations of THIS run (x 24 env steps) and replace the parent's REV_STEP0 stages wholesale.

r2 (2026-11-20, rampdiag diagnosis; resume of r1, the step counter continues; R2_IT0 = the resume iteration):
  - passes: latched detector (rp.apex_update: max rise of the wall visit, fires 3 mm below it; re-armed on the inner
    flat), alternating or same-wall returns, apex_min_z 0.01 -> 0.0075, z_cap 0.07 -> 0.04 (the target first apex
    is 2.5-4 cm; a frozen rider at 0.85 m/s gets ~3.2 / 1.1 / 0.5 cm, so the 3rd pass needs pumping).
  - past_pipe_end at the ENTRY end = a failure (fall price); the far end stays a time-out (past_pipe_end).
  - faster entry: exit spawns 0.6-1.0 -> 0.85-1.2 m/s; ramp_energy 2.0 from R2_IT0 (was 600).
  - RAMP carve command (rp.ramp_carve, observed kappa) + ramp_track 1.5 from R2_IT0: stage 1 CARVE_FLAT (steer the
    crossing line on the flat toward the wall it heads to), stage 2 CARVE_WALL from R2_IT0 + 500 (+ the turn back on
    the wall). Chain playback (chain_play.py: common_step_counter = max_iterations x 24) sees the final stage.

r3 (2026-11-20; resume of r2 model_900, R3_IT0 = the resume iteration):
  - measured: pumping cannot sustain height at this scale (stiff-rider proxy: a timed 2-3 cm crouch/extend adds
    +0.01-0.03 per-pass retention against a 0.33-0.38 loss; scripted pumps on the policy only add falls). A stiff rider
    keeps 0.62-0.67 of the pass height per wall visit, the r2 policy 0.29: the rider is the trainable gap.
  - ramp_keep 4.0 from R3_IT0 (rp.ramp_keep: per-pass height retention, gated to >= 2 cm-visible passes); ramp_energy
    2.0 -> 0 (its sum telescopes to e(end) - e(entry) < 0 for any unpumped ride: a small, noisy, unavoidable cost).
  - CARVE_WALL dropped (needs 2.6-9 1/m on this wall vs the 0.5 cap): CARVE_FLAT stays; MICRODUCK_RAMP_R2_IT0 default
    400 -> 600 (the actual r2 value; any 400-600 gives the same stages from 900 on now that CARVE_WALL is gone).
  - ramp_bottom spawn speed 0.5-1.0 -> 0.6-1.2 m/s.
  - r3rev: ramp_keep ratio denominator min(previous pass, APEX_Z_CAP): 1st-pass height above the apex cap no longer
    lowers the pass-2 keep pay (r2 model_900: pass 2 ~1.2-2 cm whatever pass 1, so the raw ratio favoured a lower pass 1).

rampbank (2026-11-20, default off): a chain dry run
reached the HALF-PIPE in 34 % (r3b alone from its exit spawns: 94 %). MICRODUCK_RAMP_HANDOFF=<bank.npz> (kind "ramp")
turns a share MICRODUCK_RAMP_HANDOFF_PROB (default 0.45) of the EXIT spawns into recorded chain hand-off states
(skate_sequence._ramp_handoff_reset: pipe-frame pose, twist, joints, scheduler stage / timers / pursuit side / clock),
raises the EXIT spawn share to MICRODUCK_RAMP_HANDOFF_EXIT_P (default 0.5; ramp_bottom / ramp_wall keep their ratio)
and adds the chain's EXIT -> RAMP gate ramp_gate_x 0.8 (chain_play --seq-override). Hand-off spawns also get the
chain's tilt guard from their first step (riding_tilt after_s applies to the other spawns only: in the chain t ~ 7 s at
the switch, so the guard is already live; rp.riding_tilt_guard handoff_after_s).
  Diagnosis (p11 / m2 / r3b, n 32): 15 of 22 failures end in EXIT 1.7-7.5 s after the switch,
  stalled (v ~0) against the wall's end face at y_open ~0, |x| 0.57-1.06, then riding_tilt / foot_on_floor; hand-off
  |x| 0.17-2.2 m (p10-p90; 53 % outside the 0.8 gate), heading -33..+21 deg, v 0.73-1.0, on either side of the axis
  (training exit spawns: |x| 0.45-0.75 on the far side, heading 0 or turned <= 30 deg in, v 0.85-1.2, stage_t 0).

  r4 (resume of r3b model_1499; the bank recorded with the final m4 manual):

  r5 (resume of r4 model_2098; default off, see _add_r5): head hold,
  trunk band, dense pump direction. Same env vars as r4 plus:
"""

from __future__ import annotations
from micro_skater.paths import data_path as _data_path

import math
import os

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import CurriculumTermCfg, EventTermCfg, MetricsTermCfg, RewardTermCfg, TerminationTermCfg

from micro_skater.tasks import mdp as microduck_mdp
from micro_skater.tasks import skate_seq_ramp as rp
from micro_skater.tasks import skate_sequence as sq
from micro_skater.tasks.microduck_skate_sequence_env_cfg import (make_microduck_skate_sequence_env_cfg,
                                                                    make_microduck_skate_sequence_ppo_runner_cfg)

EPISODE_LENGTH_S = 12.0
CHAIN_EPISODE_S = 32.0
# headrev (2026-11-20, chain playback only; ramp TRAINING keeps start_gap 5): the chain starts CHAIN_START_GAP m before
# the pipe's open end (was 5). A straight chain (chaindiag base: PUSH 2.6 m + SETTLE 1.7 m + MANUAL 3.0 m = 7.2 m
# median arc, p9b) then enters EXIT at y_open ~ -2 (-3 .. -1 for 6-8 m): inside this task's exit spawns (y_open -3.5 ..
# -1, |x| 0.45-0.75 on the far side of the target, heading +y or turned in <= 30 deg). With 5 m it reached EXIT ~2 m
# INSIDE the pipe at |x| ~0.6 (the wall's end face). y obs clips at y_open < -3, so the longer floor is unobserved.
# The push / manual specialists train the same geometry with their heading flags (start_gap 9 / 12, the target below).
# MICRODUCK_CHAIN_START_GAP=5 restores the old start.
CHAIN_START_GAP = float(os.environ.get("MICRODUCK_CHAIN_START_GAP", "9.0"))
IT = 24                                   # env steps per iteration (NUM_STEPS_PER_ENV)

SEQ_OVERRIDES = dict(target_x=0.45, target_in=0.6, kappa_max=0.3, apex_min_z=0.0075)   # r2: apex_min_z 0.01 -> 0.0075
# r4rev (2026-11-20): EXIT pursuit gate point (sq.approach_kappa gate_pt = (tx, y_in, switch, kappa_max)): until 0.3 m
# before the open end steer to (entry_s * 0.2, the open end) with cap 0.45 1/m (the RAMP carve command already goes to
# 0.5), then the old target / cap 0.3. Kinematic pursuit from the m4 hand-off bank: 52 -> 75 % cross the open end at
# |x| < 0.5 (|x| > 0.8: 24 -> 44 %); a higher cap alone gives 54 %. In SEQ_OVERRIDES, so chain_play (_chain_overrides)
# sees the same command. MICRODUCK_RAMP_GATE=0 restores the r3b command.
EXIT_GATE_PT = (0.2, 0.0, 0.3, 0.45)
if os.environ.get("MICRODUCK_RAMP_GATE", "1") == "1":
    SEQ_OVERRIDES["gate_pt"] = EXIT_GATE_PT
SPAWN_PROBS = dict(exit=0.40, ramp_bottom=0.35, ramp_wall=0.25)
SPAWN_STANCES = dict(exit=["tail", "tail_back", "glide_step", "glide"], ramp_bottom=["glide", "glide_fwd"],
                     ramp_wall=["glide", "glide_fwd", "tail"])
# r2: exit 0.6-1.0 -> 0.85-1.2 m/s (r1 entered RAMP at 0.63-0.69 m/s: a 1.5-1.9 cm first rise)
FLAT_SPEEDS = dict(push=(0.1, 0.8), glide=(0.5, 1.1), manual=(0.5, 1.0), exit=(0.85, 1.2))
EXIT_TILT_PROB = 0.15
RAMP_SPEED = (0.6, 1.2)                   # r3 0.5-1.0 -> 0.6-1.2: 3 passes >= 2 cm need ~1 m/s across
RAMP_CHAIN_PROB = 0.7
WALL_FRAC = (0.05, 0.35)
CLOCK_OFFSET = (5.0, 9.0)
# rampbank (default off): recorded chain manual -> ramp hand-off states at a share of the EXIT spawns
RAMP_HANDOFF = os.environ.get("MICRODUCK_RAMP_HANDOFF") or None
RAMP_HANDOFF_PROB = float(os.environ.get("MICRODUCK_RAMP_HANDOFF_PROB", "0.45"))
RAMP_HANDOFF_EXIT_P = float(os.environ.get("MICRODUCK_RAMP_HANDOFF_EXIT_P", "0.5"))
RAMP_HANDOFF_GATE_X = 0.8                 # chain_play --seq-override ramp_gate_x=0.8 (CHAINFIX_GATE_X)

RAMP_DONE = dict(ramp_s=8.0, max_apex=5)
EXIT_TIME_W = -2.0                        # review (rrev): EXIT ~4.2 /s vs RAMP ~2.3 /s measured (warm start, final weights)
APEX_Z_CAP = 0.04                          # r2 0.07 -> 0.04: the target first apex is 2.5-4 cm
RAMP_TRACK_W = 1.5
R2_IT0 = int(os.environ.get("MICRODUCK_RAMP_R2_IT0", "600"))   # r2 resume iteration (r3: default = the actual r2 value)
CARVE_WALL_IT = R2_IT0 + 500                                       # r3: stage no longer scheduled (CARVE_STAGES)
# r3 (2026-11-20): resume of r2; R3_IT0 = the resume iteration
R3_IT0 = int(os.environ.get("MICRODUCK_RAMP_R3_IT0", "900"))
RAMP_KEEP_W = 4.0                          # r3: ramp_keep (pass-height retention) weight from R3_IT0

PUSH_MANUAL_TERMS = ("push_progress", "push_time", "push_speed", "push_stroke", "push_step_on", "push_step_reach",
                     "manual", "manual_shape")
PUSH_MANUAL_METRICS = ("seq_push", "seq_liftoff", "seq_stroke_q", "seq_stepon", "seq_manual", "seq_manual_band")


def _stages(*pairs) -> list[dict]:
    """(iteration, weight) pairs -> reward_weight stages in env steps."""
    return [{"step": it * IT, "weight": w} for it, w in pairs]


RAMP_APEX_STAGES = _stages((0, 3.0), (300, 6.0))
RAMP_LATERAL_STAGES = _stages((0, 3.0), (600, 2.0))
RAMP_ENERGY_STAGES = _stages((0, 0.0), (min(600, R2_IT0), 2.0), (R3_IT0, 0.0))   # r2: from the resume; r3: off
RAMP_KEEP_STAGES = _stages((0, 0.0), (R3_IT0, RAMP_KEEP_W))
RAMP_TRACK_STAGES = _stages((0, 0.0), (R2_IT0, RAMP_TRACK_W))
# r3: CARVE_WALL dropped (a turn back on this wall needs ~2.6-9 1/m vs the 0.5 1/m command cap and the ~0.3-0.5 1/m
# the board reaches by lean; passes are fakie shuttles): CARVE_FLAT stays the final stage
CARVE_STAGES = [{"step": 0, "cfg": None}, {"step": R2_IT0 * IT, "cfg": dict(rp.CARVE_FLAT)}]
ACTION_RATE_STAGES = _stages((0, -0.08), (800, -0.12))
BOARD_WOBBLE_STAGES = _stages((0, -0.1), (800, -0.2))

# pump (2026-11-20, default off): MICRODUCK_RAMP_PUMP=1 adds rp.ramp_pump
# (pumping work sum (N_eff - g) dh of the rider's CoM along the deck normal: extend in the loaded transition, crouch at
# the unloaded apex; 0 on the flat, telescoping when held) from MICRODUCK_RAMP_PUMP_IT0 (= the resume iteration), and
# relaxes ramp_keep (PUMP_KEEP: retention above the 0.8 stiff-rider cap pays, the visibility gate 2 -> 1.5 cm).
RAMP_PUMP = os.environ.get("MICRODUCK_RAMP_PUMP", "0") == "1"
PUMP_IT0 = int(os.environ.get("MICRODUCK_RAMP_PUMP_IT0", "1500"))
RAMP_PUMP_W = float(os.environ.get("MICRODUCK_RAMP_PUMP_W", "15.0"))
RAMP_PUMP_STAGES = _stages((0, 0.0), (PUMP_IT0, RAMP_PUMP_W))
PUMP_KEEP = dict(cap=1.0, z_ref=0.015)


def _add_pump(cfg, rw) -> None:
    """pump (MICRODUCK_RAMP_PUMP=1): reward + its weight stage, reset event (after reset_ramp), ramp_keep relaxation,
    metric seq_pump (J/kg per episode)."""
    cfg.events["reset_pump"] = EventTermCfg(func=rp.reset_pump_state, mode="reset")
    cfg.rewards["ramp_pump"] = RewardTermCfg(func=rp.ramp_pump, weight=RAMP_PUMP_STAGES[0]["weight"])
    cfg.curriculum["ramp_pump"] = rw("ramp_pump", RAMP_PUMP_STAGES)
    cfg.rewards["ramp_keep"].params.update(PUMP_KEEP)
    cfg.metrics["seq_pump"] = MetricsTermCfg(func=rp.m_pump)


# r5 (2026-11-20, default off): MICRODUCK_RAMP_R5=1 adds, from
# MICRODUCK_RAMP_R5_IT0 (= the resume iteration; r4 model_2098 stores common_step_counter 50472 = it 2103, so any
# R5_IT0 <= 2103 is live from the first iteration of the resume):
#   r5_head_hold  (EXIT / RAMP) head joints vs the hand-off head pose + head angular speed vs the board (apex whip)
#   r5_trunk_band (RAMP) trunk deck-frame height outside 7.5 (8.0 from +200)-10.5 cm (r4: 7.1 cm frozen squat)
#   ramp_pump_dir (RAMP) dense pump direction: CoM travel along the deck normal x sign-squashed (N_eff - g)
# head / band weights are staged in over 200 iterations (the r4 policy falls 81 % when scripted 2.3 cm taller).
RAMP_R5 = os.environ.get("MICRODUCK_RAMP_R5", "0") == "1"
R5_IT0 = int(os.environ.get("MICRODUCK_RAMP_R5_IT0", "2099"))
R5_HEAD_STAGES = _stages((0, 0.0), (R5_IT0, -0.25), (R5_IT0 + 200, -0.5))
R5_BAND_STAGES = _stages((0, 0.0), (R5_IT0, -0.25), (R5_IT0 + 200, -0.5))
R5_PUMP_DIR_STAGES = _stages((0, 0.0), (R5_IT0, 0.5))
# r5rev (pre-launch review): the band's lower edge 7.5 cm with the -0.25 stage, 8.0 cm with -0.5 (r4 rides at 7.1 cm and
# its squat is part of its balance: first ask for +0.4 cm, then +0.9)
R5_BAND_ZLO = [(0, 0.075), (R5_IT0 * IT + 200 * IT, 0.080)]


def _add_r5(cfg, rw) -> None:
    """r5 (MICRODUCK_RAMP_R5=1): head hold, trunk band, dense pump direction + their stages, reset event, metrics."""
    R = RewardTermCfg
    cfg.events["reset_r5"] = EventTermCfg(func=rp.reset_r5_state, mode="reset")
    cfg.rewards["r5_head_hold"] = R(func=rp.r5_head_hold, weight=R5_HEAD_STAGES[0]["weight"])
    cfg.rewards["r5_trunk_band"] = R(func=rp.r5_trunk_band, weight=R5_BAND_STAGES[0]["weight"],
                                     params={"z_lo_stages": list(R5_BAND_ZLO)})
    cfg.rewards["ramp_pump_dir"] = R(func=rp.ramp_pump_dir, weight=R5_PUMP_DIR_STAGES[0]["weight"])
    cfg.curriculum["r5_head_hold"] = rw("r5_head_hold", R5_HEAD_STAGES)
    cfg.curriculum["r5_trunk_band"] = rw("r5_trunk_band", R5_BAND_STAGES)
    cfg.curriculum["ramp_pump_dir"] = rw("ramp_pump_dir", R5_PUMP_DIR_STAGES)
    cfg.metrics["seq_pump_dir"] = MetricsTermCfg(func=rp.m_pump_dir)
    cfg.metrics["seq_head_dev"] = MetricsTermCfg(func=rp.m_head_dev)
    cfg.metrics["seq_trunk_z"] = MetricsTermCfg(func=rp.m_trunk_z)


# r5pump (2026-11-21, default off): MICRODUCK_RAMP_R5PUMP=1 (with
# MICRODUCK_RAMP_R5=1) makes pumping pay through its OUTCOME and helps exploration:
#   ramp_retain   (RAMP) energy kept per wall visit: q = 1 + d / (g pass rise), d = change of the system-CoM v^2/2 between successive flat-edge
#                 crossings (J/kg), q paid [0.5, 1.0] -> [0, 1] x min(pass rise / 2.5 cm, 1) when the visit had a pass
#                 >= 1.5 cm (review: a plain v^2 ratio was inflated by along-pipe speed, a raw d paid low passes); weight
#                 R5PUMP_RETAIN_W from R5PUMP_IT0
#   ramp_pump_dir 0.5 -> 2.0 at R5PUMP_IT0, annealed 1.5 / 1.0 / 0.75 / 0.5 at +100 / +200 / +300 / +400 (review: 2.5
#                 thinned the fall margin)
#   spawns (training only): EXIT share 0.5 -> 0.4 with the hand-off bank (0.4 -> 0.3 without), ramp_bottom speed
#                 0.6-1.2 -> 0.45-1.2 m/s (pumping needed for height); ramp_done 8 s / 5 passes -> 10 s / 8 passes
RAMP_R5PUMP = os.environ.get("MICRODUCK_RAMP_R5PUMP", "0") == "1"
R5PUMP_IT0 = int(os.environ.get("MICRODUCK_RAMP_R5PUMP_IT0", str(R5_IT0)))
R5PUMP_RETAIN_W = float(os.environ.get("MICRODUCK_RAMP_R5PUMP_RETAIN_W", "8.0"))
R5PUMP_RETAIN_STAGES = _stages((0, 0.0), (R5PUMP_IT0, R5PUMP_RETAIN_W))
R5PUMP_DIR_STAGES = _stages((0, 0.0), (R5PUMP_IT0, 2.0), (R5PUMP_IT0 + 100, 1.5), (R5PUMP_IT0 + 200, 1.0),
                            (R5PUMP_IT0 + 300, 0.75), (R5PUMP_IT0 + 400, 0.5))
R5PUMP_RAMP_SPEED = (0.45, 1.2)
R5PUMP_EXIT_P = 0.4                       # with the hand-off bank (was RAMP_HANDOFF_EXIT_P 0.5); without: 0.3 (was 0.4)
R5PUMP_RAMP_DONE = dict(ramp_s=10.0, max_apex=8)


def _add_r5pump(cfg, rw) -> None:
    """r5pump (MICRODUCK_RAMP_R5PUMP=1): ramp_retain + its stage, reset event (after reset_ramp), stronger annealed
    ramp_pump_dir (needs _add_r5 first), metrics seq_retain / seq_retain_n."""
    cfg.events["reset_retain"] = EventTermCfg(func=rp.reset_retain_state, mode="reset")
    cfg.rewards["ramp_retain"] = RewardTermCfg(func=rp.ramp_retain, weight=R5PUMP_RETAIN_STAGES[0]["weight"])
    cfg.curriculum["ramp_retain"] = rw("ramp_retain", R5PUMP_RETAIN_STAGES)
    if "ramp_pump_dir" in cfg.rewards:
        cfg.curriculum["ramp_pump_dir"] = rw("ramp_pump_dir", R5PUMP_DIR_STAGES)
    cfg.metrics["seq_retain"] = MetricsTermCfg(func=rp.m_retain)
    cfg.metrics["seq_retain_n"] = MetricsTermCfg(func=rp.m_retain_n)


def _r5pump_spawns(cfg) -> None:
    """r5pump, training only: more / slower in-pipe spawns, a longer RAMP window."""
    p = cfg.events["reset_sequence"].params
    p["ramp_speed"] = R5PUMP_RAMP_SPEED
    ex = R5PUMP_EXIT_P if RAMP_HANDOFF else 0.3
    rest = 1.0 - SPAWN_PROBS["exit"]
    p["spawn_probs"] = {k: (ex if k == "exit" else v * (1.0 - ex) / rest) for k, v in SPAWN_PROBS.items()}
    cfg.terminations["ramp_done"].params.update(R5PUMP_RAMP_DONE)


# r6 (2026-11-21, default off): MICRODUCK_RAMP_R6=1 (with R5 + R5PUMP),
# from MICRODUCK_RAMP_R6_IT0 (= the resume iteration; r5 model_2797 stores common_step_counter 67272 = it 2803, so any
# R6_IT0 <= 2803 is live from the first iteration of the resume):
#   r6_head        (EXIT / RAMP) SQUARED head-joint deviation from the hand-off head pose (yaw x2) + squared head rate
#                  vs the board; on top of r5_head_hold. Staged in (r5 with a frozen head falls 47 % from ramp_bottom
#                  spawns vs 6 %: its nod is part of its balance, the legs must take it over)
#   r6_trunk_band  (RAMP) replaces r5_trunk_band: lower edge 8.0 cm where loaded, 6.5 cm where unloaded (N_eff - g <=
#                  -0.75 m/s^2 = a 3 cm apex at rest: the pump's crouch), upper 11 cm (r5: 8.0-10.5 everywhere)
#   ramp_retain    8 -> R6_RETAIN_W (14)
#   r6_keep        (RAMP) pass k >= 2 height vs an entry-energy anchor (0.8 e_1 / g), R6_KEEP_W per full pass
#   ramp_pump_dir  0.5 -> R6_PUMP_DIR_W (0.75)
RAMP_R6 = os.environ.get("MICRODUCK_RAMP_R6", "0") == "1"
R6_IT0 = int(os.environ.get("MICRODUCK_RAMP_R6_IT0", "2798"))
R6_HEAD_STAGES = _stages((0, 0.0), (R6_IT0, -0.2), (R6_IT0 + 100, -0.4), (R6_IT0 + 200, -0.6))
R6_BAND_W = -0.5
R6_RETAIN_W = float(os.environ.get("MICRODUCK_RAMP_R6_RETAIN_W", "14.0"))
R6_KEEP_W = float(os.environ.get("MICRODUCK_RAMP_R6_KEEP_W", "3.0"))
R6_PUMP_DIR_W = 0.75


def _add_r6(cfg, rw) -> None:
    """r6 (MICRODUCK_RAMP_R6=1, needs _add_r5 + _add_r5pump first): head cost, load-dependent trunk band (r5 band off),
    stronger outcome pay, pass-height maintenance + their stages, reset event (after reset_ramp), metrics."""
    assert "r5_trunk_band" in cfg.rewards and "ramp_retain" in cfg.rewards, "r6 needs MICRODUCK_RAMP_R5=1 + R5PUMP=1"
    R = RewardTermCfg
    cfg.events["reset_r6"] = EventTermCfg(func=rp.reset_r6_state, mode="reset")
    cfg.rewards["r6_head"] = R(func=rp.r6_head, weight=R6_HEAD_STAGES[0]["weight"])
    cfg.curriculum["r6_head"] = rw("r6_head", R6_HEAD_STAGES)
    cfg.rewards["r6_trunk_band"] = R(func=rp.r6_trunk_band, weight=0.0)
    cfg.curriculum["r6_trunk_band"] = rw("r6_trunk_band", _stages((0, 0.0), (R6_IT0, R6_BAND_W)))
    # the r5 band's own stages with an off stage at R6_IT0 (its last weight before: -0.5)
    cfg.curriculum["r5_trunk_band"] = rw("r5_trunk_band", R5_BAND_STAGES + _stages((R6_IT0, 0.0)))
    cfg.curriculum["ramp_retain"] = rw("ramp_retain", R5PUMP_RETAIN_STAGES + _stages((R6_IT0, R6_RETAIN_W)))
    cfg.rewards["r6_keep"] = R(func=rp.r6_keep, weight=0.0)
    cfg.curriculum["r6_keep"] = rw("r6_keep", _stages((0, 0.0), (R6_IT0, R6_KEEP_W)))
    cfg.curriculum["ramp_pump_dir"] = rw("ramp_pump_dir", R5PUMP_DIR_STAGES + _stages((R6_IT0, R6_PUMP_DIR_W)))
    cfg.metrics["seq_r6_ratio"] = MetricsTermCfg(func=rp.m_r6_keep)
    cfg.metrics["seq_r6_n"] = MetricsTermCfg(func=rp.m_r6_n)
    cfg.metrics["seq_head_rate"] = MetricsTermCfg(func=rp.m_head_rate)


# r7 (2026-11-21, default off): MICRODUCK_RAMP_R7=1 (with R5 + R5PUMP + R6),
# from MICRODUCK_RAMP_R7_IT0 (= the resume iteration; r6 model_3596 stores common_step_counter 86472 = it 3603, so any
# R7_IT0 <= 3603 is live from the first iteration of the resume). r6's pump was a neck / head fold at the apex that the
# whole-body-CoM pump terms paid:
#   r7_pump_dir   replaces ramp_pump_dir (same weight 0.75): CoM EXCLUDING the neck + head subtree (legs / hips only)
#   r7_pump       replaces ramp_pump (same weight RAMP_PUMP_W): idem for the pumping-work term
#   r7_head       (EXIT / RAMP) head off level vs world up beyond 20 deg + neck / head pitch beyond 0.3 rad of the riding
#                 pose, squared, cap 4 per step; R7_HEAD_W staged in over 150 iterations (r5: a frozen head falls 47 %)
RAMP_R7 = os.environ.get("MICRODUCK_RAMP_R7", "0") == "1"
R7_IT0 = int(os.environ.get("MICRODUCK_RAMP_R7_IT0", "3597"))
#   r7_retain / r7_keep replace ramp_retain / r6_keep (same weights): x the visit head factor (1 - max r7_head / 4 over
#                 the wall visit; review r7: the fold's real work still bought ~7-10 outcome pay per visit)
# r7 audit (r6 model_3596): the neck + head subtree is 38 % of the mass and makes 76-80 % of the
# pump-measured CoM motion and of the pumping work; r7_head raw 4.2-4.4 per episode for r6's fold -> x R7_HEAD_W -3.0
# = 13 per episode (on top of ~5-6 of r5 / r6 head costs). r7_pump_dir 0.75 -> 0.9, r7_pump x 1.25 (review: a knee
# crouch moves the below-neck CoM ~0.6 d vs ~0.75 d for the whole body: same pay per cm of leg pump).
R7_HEAD_W = float(os.environ.get("MICRODUCK_RAMP_R7_HEAD_W", "-3.0"))
R7_HEAD_STAGES = _stages((0, 0.0), (R7_IT0, R7_HEAD_W / 3.0), (R7_IT0 + 75, 2.0 * R7_HEAD_W / 3.0),
                         (R7_IT0 + 150, R7_HEAD_W))
R7_PUMP_DIR_W = 0.9
R7_PUMP_W = 1.25 * RAMP_PUMP_W


def _add_r7(cfg, rw) -> None:
    """r7 (MICRODUCK_RAMP_R7=1, needs _add_r6 first): below-neck pump terms replace the whole-body ones, head-steadiness
    cost + their stages, reset event (after reset_ramp), metrics."""
    assert "r6_head" in cfg.rewards and "ramp_pump_dir" in cfg.rewards, "r7 needs MICRODUCK_RAMP_R5/R5PUMP/R6=1"
    R = RewardTermCfg
    cfg.events["reset_r7"] = EventTermCfg(func=rp.reset_r7_state, mode="reset")
    cfg.rewards["r7_head"] = R(func=rp.r7_head, weight=0.0)
    cfg.curriculum["r7_head"] = rw("r7_head", R7_HEAD_STAGES)
    cfg.rewards["r7_pump_dir"] = R(func=rp.r7_pump_dir, weight=0.0)
    cfg.curriculum["r7_pump_dir"] = rw("r7_pump_dir", _stages((0, 0.0), (R7_IT0, R7_PUMP_DIR_W)))
    cfg.curriculum["ramp_pump_dir"] = rw("ramp_pump_dir", R5PUMP_DIR_STAGES + _stages((R6_IT0, R6_PUMP_DIR_W),
                                                                                       (R7_IT0, 0.0)))
    if "ramp_pump" in cfg.rewards:
        cfg.rewards["r7_pump"] = R(func=rp.r7_pump, weight=0.0)
        cfg.curriculum["r7_pump"] = rw("r7_pump", _stages((0, 0.0), (R7_IT0, R7_PUMP_W)))
        cfg.curriculum["ramp_pump"] = rw("ramp_pump", RAMP_PUMP_STAGES + _stages((R7_IT0, 0.0)))
    # r7rev: outcome pay x the visit head factor (ramp_retain / r6_keep off at R7_IT0, r7_retain / r7_keep on)
    cfg.events["reset_r7_visit"] = EventTermCfg(func=rp.reset_r7_visit, mode="reset")
    cfg.rewards["r7_retain"] = R(func=rp.r7_retain, weight=0.0)
    cfg.curriculum["r7_retain"] = rw("r7_retain", _stages((0, 0.0), (R7_IT0, R6_RETAIN_W)))
    cfg.curriculum["ramp_retain"] = rw("ramp_retain", R5PUMP_RETAIN_STAGES + _stages((R6_IT0, R6_RETAIN_W),
                                                                                   (R7_IT0, 0.0)))
    cfg.rewards["r7_keep"] = R(func=rp.r7_keep, weight=0.0)
    cfg.curriculum["r7_keep"] = rw("r7_keep", _stages((0, 0.0), (R7_IT0, R6_KEEP_W)))
    cfg.curriculum["r6_keep"] = rw("r6_keep", _stages((0, 0.0), (R6_IT0, R6_KEEP_W), (R7_IT0, 0.0)))
    cfg.metrics["seq_r7_hf"] = MetricsTermCfg(func=rp.m_r7_hf)
    cfg.metrics["seq_r7_head"] = MetricsTermCfg(func=rp.m_r7_head)
    cfg.metrics["seq_r7_lev"] = MetricsTermCfg(func=rp.m_r7_lev)
    cfg.metrics["seq_r7_pump_dir"] = MetricsTermCfg(func=rp.m_r7_pump_dir)
    cfg.metrics["seq_r7_pump"] = MetricsTermCfg(func=rp.m_r7_pump)


# r8b (2026-11-21, default off): MICRODUCK_RAMP_R8B=1 (with R5 + R5PUMP + R6
# + R7; trained with MICRODUCK_PIPE_H=0.08), from MICRODUCK_RAMP_R8B_IT0 (= the resume iteration; r7 model_4295 stores
# common_step_counter 103272 = it 4303, so any R8B_IT0 <= 4303 is live from the first iteration of the resume). A CLEAN,
# composed wall apex (chain_final_h8 review: deck scraping, board slew / tail down, rear sole unweighting, head fold at
# the apexes). One RAMP cost r8b_clean = min(sum k_i part_i, 8 /s) (rp.R8B_K), parts measured against the pipe SURFACE
# under the board (rp.r8b_update):
#   deck  8  deck geom on the terrain (new contact sensor "deck_terrain", deck geoms only, not the wheels)
#   slew  3  board yaw rate about the deck normal beyond 1 rad/s (per rad/s, cap 2)
#   tilt  2  deck roll vs the surface beyond 15 deg + pitch vs the surface beyond 12 deg (per 10 deg, cap 2)
#   soles 3  a sole off the deck / the lighter sole below 20 % of the soles' load (0..1)
#   head  1  head over the ankles below 15.5 cm (per cm) + head pitch / yaw rate vs the board beyond 3 rad/s (per 3), cap 3
#   lip   2  the leading wheel within 1 cm of the coping (per cm, cap 2): a soft brake before over_coping's fall
# (r7 4295, h 8, ramp_bottom n 32): slew > 1.5 rad/s at 25 / 29 first passes (with or without the lip:
# the steep top of the arc, not the coping), the lighter sole < 20 % at 23 / 29, lead wheel within 5 mm of the coping
# 17 / 29, deck / tail on the terrain 4 / 29; flat-bottom values ~0. Pumping is infeasible:
# r7_pump / r7_pump_dir (which paid the apex crouch r8b_head charges; review r8b) are off from R8B_IT0.
# Half weight for >= 75 iterations (the counter is already at 4303), then full. Audit:
RAMP_R8B = os.environ.get("MICRODUCK_RAMP_R8B", "0") == "1"
R8B_IT0 = int(os.environ.get("MICRODUCK_RAMP_R8B_IT0", "4296"))
R8B_W = float(os.environ.get("MICRODUCK_RAMP_R8B_W", "-1.0"))
R8B_STAGES = _stages((0, 0.0), (R8B_IT0, 0.5 * R8B_W), (R8B_IT0 + 82, R8B_W))


def _r8b_sensor():
    from mjlab.sensor import ContactMatch, ContactSensorCfg
    return ContactSensorCfg(name="deck_terrain",
                            primary=ContactMatch(mode="geom", pattern=r"^deck.*_collision$", entity="board"),
                            secondary=ContactMatch(mode="body", pattern="terrain"), fields=("found",),
                            reduce="netforce", num_slots=1)


def _add_r8b(cfg, rw) -> None:
    """r8b (MICRODUCK_RAMP_R8B=1, needs _add_r7 first): the clean-apex cost + its stages, the deck_terrain sensor, pump
    terms off, metrics (raw part values per step)."""
    assert "r7_head" in cfg.rewards and "r7_pump_dir" in cfg.rewards, "r8b needs MICRODUCK_RAMP_R5/R5PUMP/R6/R7=1"
    cfg.scene.sensors = tuple(cfg.scene.sensors) + (_r8b_sensor(),)
    cfg.rewards["r8b_clean"] = RewardTermCfg(func=rp.r8b_clean, weight=0.0)
    cfg.curriculum["r8b_clean"] = rw("r8b_clean", R8B_STAGES)
    cfg.curriculum["r7_pump_dir"] = rw("r7_pump_dir", _stages((0, 0.0), (R7_IT0, R7_PUMP_DIR_W), (R8B_IT0, 0.0)))
    if "r7_pump" in cfg.rewards:
        cfg.curriculum["r7_pump"] = rw("r7_pump", _stages((0, 0.0), (R7_IT0, R7_PUMP_W), (R8B_IT0, 0.0)))
    for k, f in rp.R8B_PARTS.items():
        cfg.metrics["seq_r8b_" + k] = MetricsTermCfg(func=f)


# r9 (2026-11-21, default off): MICRODUCK_RAMP_R9=1 (with R5 + R5PUMP + R6
# + R7; R8B optional - r8b was rejected, the r9 base is r7 model_4295, R9_IT0 4303; trained with MICRODUCK_PIPE_H=0.08
# and MICRODUCK_BAM_MAX_CURRENT=1.0), from MICRODUCK_RAMP_R9_IT0 (= the resume iteration: the checkpoint's
# common_step_counter / 24, so the stages are live from the first iteration). FOOT-PUSHING on the flat bottom on the
# FORWARD crossings (the passes are fakie shuttles:), RL only (p15's stroke bends the push knee
# backwards, so it is not a reference; p15 zero-shot from a riding stance falls 56 %:). Terms (rp r9
# block): r9_gain (cm of cross-pipe height gained on a pushed forward crossing, capped 3 cm and at a 4.5 cm target),
# r9_reach (potential), r9_stroke (annealed exploration bonus), r9_drag, r9_late, r9_knee (knee on its normal side, every stage; the author flagged the
# reversed knee as unnatural, it is mechanically possible). No joint-speed cost (XL330 no-load 14-22 rad/s; rp.r9_jvel
# is kept unused). Terminations: foot_on_floor / board_lost relaxed for the push foot on the flat (r9_foot_on_floor /
# r9_board_lost); with R8B on, r8b_clean's sole-off part is exempt in the push zone. Training spawns: ramp_bottom 0.3-1.2 m/s, ramp_done 10 s / 12 passes.
RAMP_R9 = os.environ.get("MICRODUCK_RAMP_R9", "0") == "1"
R9_IT0 = int(os.environ.get("MICRODUCK_RAMP_R9_IT0", "4303"))
R9_W = dict(gain=float(os.environ.get("MICRODUCK_RAMP_R9_GAIN_W", "3.0")), reach=1.0, drag=-1.0, late=-3.0,
            knee=-1.0, stroke=2.0)
R9_TERMS = dict(gain=rp.r9_gain, reach=rp.r9_reach, drag=rp.r9_drag, late=rp.r9_late, knee=rp.r9_knee,
                stroke=rp.r9_stroke)
# review r9: a clean completed stroke (touch + both soles back on at the wall) pays 2 for 100 iterations, 1 to +150, then
# 0 (exploration past the coasting optimum: a push that falls costs ~-30..-36 mid-episode; gate the run on seq_r9_n)
R9_STROKE_STAGES = ((0, 0.0), (R9_IT0, 2.0), (R9_IT0 + 100, 1.0), (R9_IT0 + 150, 0.0))
# the knee cost comes in at half weight for 50 iterations
R9_HALF_FIRST = ("knee",)
R9_RAMP_SPEED = (0.3, 1.2)
R9_RAMP_DONE = dict(ramp_s=10.0, max_apex=12)
R9_RAMP_OFF_S = 1.0


def _add_r9(cfg, rw) -> None:
    """r9 (MICRODUCK_RAMP_R9=1, needs _add_r7 first; after _add_r8b if R8B is on): the push terms + their stages, reset
    event (after reset_ramp), the r8b sole exemption (R8B on), metrics. Terminations / spawns: _r9_terminations."""
    assert "r7_head" in cfg.rewards, "r9 needs MICRODUCK_RAMP_R5/R5PUMP/R6/R7=1"
    R = RewardTermCfg
    cfg.events["reset_r9"] = EventTermCfg(func=rp.reset_r9_state, mode="reset")
    for k, f in R9_TERMS.items():
        name = "r9_" + k
        cfg.rewards[name] = R(func=f, weight=0.0)
        st = [(0, 0.0), (R9_IT0, 0.5 * R9_W[k]), (R9_IT0 + 50, R9_W[k])] if k in R9_HALF_FIRST else [(0, 0.0), (R9_IT0, R9_W[k])]
        if k == "stroke":
            st = list(R9_STROKE_STAGES)
        cfg.curriculum[name] = rw(name, _stages(*st))
    if "r8b_clean" in cfg.rewards:
        cfg.rewards["r8b_clean"].params = dict(cfg.rewards["r8b_clean"].params or {}, r9_exempt=True)
    for k, f in (("gain", rp.m_r9_gain), ("n", rp.m_r9_n), ("td", rp.m_r9_td), ("knee", rp.m_r9_knee),
                 ("stroke", rp.m_r9_stroke)):
        cfg.metrics["seq_r9_" + k] = MetricsTermCfg(func=f)


def _r9_terminations(cfg, chain: bool) -> None:
    """r9: the push foot may touch the flat bottom / leave the deck longer in RAMP (chain playback too); training only:
    slower ramp_bottom spawns (more forward crossings that need a push), a longer RAMP window."""
    T = TerminationTermCfg
    cfg.terminations["foot_on_floor"] = T(func=rp.r9_foot_on_floor)
    cfg.terminations["board_lost"] = T(func=rp.r9_board_lost, params={"ramp_off_s": R9_RAMP_OFF_S})
    if not chain:
        cfg.events["reset_sequence"].params["ramp_speed"] = R9_RAMP_SPEED
        cfg.terminations["ramp_done"].params.update(R9_RAMP_DONE)


# r9b (2026-11-21, default off): MICRODUCK_RAMP_R9B=1 (with R9=1 and its
# realism vars), TRAINING only (the chain playback is unchanged). r9 (r7 4295 -> r9 4500 at 1.0 A) never made one
# push-foot touchdown (seq_r9_td = 0 for 200 iterations; foot_on_floor fired at falls - left foot, tilt > 40 deg on the
# wall / at the turn-around: - not at a push in the zone). REVERSE CURRICULUM (the push specialist's
# step_mid lesson): rp.reset_r9b_push re-places a share of the ramp_bottom spawns MID-PUSH on the flat (rolling 0.3-0.7
# m/s nose-first toward a wall, the push foot on / just over the floor in normal-knee IK stances);
# the spawn crossing is measured by r9_gain from the spawn energy; r9_stroke's bonus is gated on OWN strokes (riding
# crossings, not the spawn's first visit) instead of iterations; the push sole's off-deck allowance 1.0 -> 1.5 s in the
# zone (a second stroke + the step-on, p15: contact 0.08 + air 0.6 + step-on 0.3 s). Base: r9 model_4500 (stores
# common_step_counter 108216 = it 4509 -> R9B_IT0 4509: live from the first iteration).
RAMP_R9B = os.environ.get("MICRODUCK_RAMP_R9B", "0") == "1"
R9B_IT0 = int(os.environ.get("MICRODUCK_RAMP_R9B_IT0", "4509"))
# mid-push share of the ramp_bottom spawns (x the ramp_bottom spawn prob 0.45: 34 % -> 13.5 % of all spawns)
R9B_SHARE = [(R9B_IT0 * IT, 0.75), ((R9B_IT0 + 150) * IT, 0.6), ((R9B_IT0 + 300) * IT, 0.45), ((R9B_IT0 + 500) * IT, 0.3)]
R9B_STANCES = dict(push=0.3, mount=0.15, push_sweep=0.15, push_low=0.25, push_lift=0.15)
R9B_SPAWN_PROBS = dict(exit=0.35, ramp_bottom=0.45, ramp_wall=0.20)     # r5pump / r9: exit 0.4 / 0.35 / 0.25
R9B_RAMP_OFF_S = 1.5
# board centre along the travel from the flat's centre (flat |x| < 0.5): 0.2-0.9 m of flat ahead; the near end (0.3) is
# the "nearly done" state (~0.3-0.6 s to the wall: lift the foot and step back on)
R9B_X = (-0.40, 0.30)
R9B_STROKE = dict(w_hi=2.0, rate_on=1.0, prog_its=100, it0=R9B_IT0, it_end=R9B_IT0 + 600)


def _add_r9b(cfg) -> None:
    """r9b (MICRODUCK_RAMP_R9B=1, training only, after _add_r9 / _r9_terminations): the mid-push spawn event right
    after reset_sequence, the spawn mix, the gated stroke bonus, the longer zone off-deck allowance, metrics."""
    assert RAMP_R9 and "r9_stroke" in cfg.rewards, "r9b needs MICRODUCK_RAMP_R9=1"
    ev = EventTermCfg(func=rp.reset_r9b_push, mode="reset", params=dict(share_stages=list(R9B_SHARE),
                                                                         stances=dict(R9B_STANCES), x_along=R9B_X))
    events = {}
    for k, v in cfg.events.items():               # dict order = event order: right after reset_sequence
        events[k] = v
        if k == "reset_sequence":
            events["reset_r9b"] = ev
    assert "reset_r9b" in events
    cfg.events = events
    cfg.events["reset_sequence"].params["spawn_probs"] = dict(R9B_SPAWN_PROBS)
    cfg.curriculum["r9_stroke"] = CurriculumTermCfg(func=rp.r9b_stroke_gate,
                                                    params=dict(reward_name="r9_stroke", **R9B_STROKE))
    cfg.curriculum["r9b_rate"] = CurriculumTermCfg(func=rp.r9b_rate_log, params={})
    cfg.curriculum["r9b_share"] = CurriculumTermCfg(func=rp.r9b_share_log, params={"share_stages": list(R9B_SHARE)})
    cfg.terminations["board_lost"].params["ramp_off_s"] = R9B_RAMP_OFF_S
    for k, f in (("own", rp.m_r9b_own), ("spawn_stroke", rp.m_r9b_spawn_stroke), ("spawn", rp.m_r9b_spawn)):
        cfg.metrics["seq_r9b_" + k] = MetricsTermCfg(func=f)


# r9c (2026-11-21, default off): MICRODUCK_RAMP_R9C=1 (with R9 + R9B and
# their realism vars), TRAINING only. r9b's IK mid-push stances are not viable states (0 / 32 alive at
# 0.5 s for every key) and its mini-runs made 0 strokes. r9c keeps r9b's schedule (share stages, gated stroke bonus, stall
# margin, 1.5 s zone off-deck allowance) and replaces the spawn by REAL p16 (1.0 A) mid-stroke states from a
# bank (rp.reset_r9c_bank, adaptive per-phase weights), and makes a failed attempt cheap: r9c_fall (a fall
# in the attempt window costs R9C_ATT_LO of a fall until own strokes exist, -> full price with r9b_stroke_gate's
# progress, ramped to full over the 100 iterations before the stroke bonus' hard end) and the HEAD style costs off inside
# the attempt window (rp.r9c_unless_attempt; r6_trunk_band stays on), plus rp.r9c_push (the stroke impulse: braking
# charged at once, gains paid when the push sole is back on the deck). Review fixes:
RAMP_R9C = os.environ.get("MICRODUCK_RAMP_R9C", "0") == "1"
R9C_BANK = os.environ.get("MICRODUCK_RAMP_R9C_BANK", str(_data_path("bank_strokes.npz")))
R9C_ATT_LO = float(os.environ.get("MICRODUCK_RAMP_R9C_ATT_LO", "0.3"))
# rung base weights (rp.R9C_PHASES: tau_on bins, s to the step-on; adaptive frontier weighting on top)
R9C_PHASE_W = dict(t20=1.0, t35=1.0, t50=1.0, t70=1.0, t100=1.0, t300=0.5)
if os.environ.get("MICRODUCK_RAMP_R9C_PHASE_W"):       # e.g. "t20:0.2,t35:1,t50:1" (smoke tests)
    R9C_PHASE_W = {k: float(v) for k, v in (kv.split(":") for kv in os.environ["MICRODUCK_RAMP_R9C_PHASE_W"].split(","))}
R9C_ADAPTIVE = None if os.environ.get("MICRODUCK_RAMP_R9C_ADAPTIVE", "1") == "0" else dict(floor=0.15, decay=0.995)
# right knee <= 0.5 rad in the spawn states (0.3 left no last-stroke contacts: bank tau_on 0.2-0.35 s 611 vs 2785 states;
# r9_knee still charges the reversed knee from the first step)
R9C_KNEE_MAX = float(os.environ.get("MICRODUCK_RAMP_R9C_KNEE_MAX", "0.5"))
R9C_X = (-0.45, 0.0)
# spawn velocity scale (rp.reset_r9c_bank v_scale): MICRODUCK_RAMP_R9C_VSCALE="lo,hi" (default "1,1" = as recorded)
R9C_VSCALE = tuple(float(x) for x in os.environ.get("MICRODUCK_RAMP_R9C_VSCALE", "1,1").split(","))
# trunk_low limit inside the attempt window (rp.r9c_trunk_low; 0.083 elsewhere)
R9C_TRUNK_ATT = float(os.environ.get("MICRODUCK_RAMP_R9C_TRUNK_ATT", "0.072"))
# head style costs only: r6_trunk_band stays on (with it off in the window the policy crouched into
# trunk_low, 0.3-1.1 ends per iteration)
R9C_STYLE = ("r7_head", "r6_head", "r5_head_hold")
# dense stroke impulse (rp.r9c_push, per cm of wall height gained over a push-foot floor contact in the zone; a brake
# costs the same): immediate credit for the push vs the braking plant the zero-shot r9 policy makes (
# 19-21 of 21 foot_on_floor ends = the right foot planted, board braked to v_fwd -0.1)
R9C_PUSH_W = float(os.environ.get("MICRODUCK_RAMP_R9C_PUSH_W", "2.0"))
# smoke-test only (default 0 = off): ramp_bottom spawns only, this share of them from the bank
R9C_FOCUS = float(os.environ.get("MICRODUCK_RAMP_R9C_FOCUS", "0"))


def _add_r9c(cfg) -> None:
    """r9c (MICRODUCK_RAMP_R9C=1, training only, after _add_r9b): bank spawns, the attempt-priced fall, the style costs
    off in the attempt window, loggers / metrics."""
    assert RAMP_R9B and "reset_r9b" in cfg.events, "r9c needs MICRODUCK_RAMP_R9B=1"
    ev = cfg.events["reset_r9b"]
    ev.func = rp.reset_r9c_bank
    ev.params = dict(share_stages=list(R9B_SHARE), phase_w=dict(R9C_PHASE_W), bank_path=R9C_BANK, knee_max=R9C_KNEE_MAX,
                     x_along=R9C_X, adaptive=dict(R9C_ADAPTIVE) if R9C_ADAPTIVE else None,
                     v_scale=R9C_VSCALE)
    fall = cfg.rewards["fall"]
    assert fall.func is sq.seq_fall, f"r9c: unexpected fall term {fall.func}"
    fall.func = rp.r9c_fall
    fall.params = dict(fall.params or {}, att_lo=R9C_ATT_LO, it_end=R9B_STROKE["it_end"])
    for k in R9C_STYLE:
        if k in cfg.rewards:
            t = cfg.rewards[k]
            assert getattr(rp, t.func.__name__, None) is t.func, f"r9c: {k} is not a skate_seq_ramp function"
            t.params = dict(func=t.func.__name__, params=dict(t.params or {}))
            t.func = rp.r9c_unless_attempt
    tl = cfg.terminations["trunk_low"]
    tl.params = dict(minimum_height=float((tl.params or {}).get("minimum_height", 0.083)), att_height=R9C_TRUNK_ATT)
    tl.func = rp.r9c_trunk_low
    for ph in rp.R9C_PHASES:
        cfg.curriculum["r9c_p_" + ph] = CurriculumTermCfg(func=rp.r9c_phase_log, params={"phase": ph})
    cfg.curriculum["r9c_prog"] = CurriculumTermCfg(func=rp.r9c_prog_log, params={})
    cfg.rewards["r9c_push"] = RewardTermCfg(func=rp.r9c_push, weight=R9C_PUSH_W)
    cfg.metrics["seq_r9c_att"] = MetricsTermCfg(func=rp.m_r9c_att)
    cfg.metrics["seq_r9c_push"] = MetricsTermCfg(func=rp.m_r9c_push)
    cfg.metrics["seq_r9c_gain"] = MetricsTermCfg(func=rp.m_r9c_gain_cm)
    cfg.metrics["seq_r9c_own"] = MetricsTermCfg(func=rp.m_r9c_own)
    cfg.metrics["seq_r9c_own_ride"] = MetricsTermCfg(func=rp.m_r9c_own_ride)
    if R9C_FOCUS:                                      # smoke-test only: every spawn a ramp_bottom, mostly bank states
        cfg.events["reset_sequence"].params["spawn_probs"] = dict(ramp_bottom=1.0)
        ev.params["share_stages"] = [(0, R9C_FOCUS)]


# r10 (2026-11-21, default off): MICRODUCK_RAMP_R10=1 (with R9 and
# its realism vars; R9B / R9C off), TRAINING only (the chain playback cfg is unchanged). Resume of r9 model_4500 (stores
# common_step_counter 108216 = it 4509 -> R10_IT0 4509: live from the first iteration). The chain's STROKE (p17d foot-push)
# hands the board back at 0.4-0.6 m/s on a nose-first crossing; r9 then folds into a joint-limit squat on the next wall
# (5 / 40 nose-first visits >= 0.4 m/s vs 1 / 163 below; never tail-first). r10:
#   reset_r10_bank  a share R10_SHARE of all spawns restore a recorded chain hand-back state (bank,
#                   MICRODUCK_RAMP_R10_BANK; the rest = the r9 spawn mix, incl. the p13 EXIT hand-off bank)
#   r10_head_low    head camera < 15 cm over the deck (EXIT / RAMP), squared per 2 cm, cap 9
#   r10_jlim        qpos joint-limit proximity / violation (legs + neck / head pitch; hip yaw / roll margin 4 deg, the rest
#                   10 deg), squared per 5 deg, cap 10 (hardware audit: hip roll / yaw pressed past the MJCF stop)
#   r10_hy_heat     hip-yaw copper loss above 0.65 W (2 s low-pass), -0.3 per W on top of skate_heat (audit: both hip yaws stall at +-1.0 A ~99 %)
# every r10 cost at half weight for R10_HALF iterations, then full.
RAMP_R10 = os.environ.get("MICRODUCK_RAMP_R10", "0") == "1"
R10_IT0 = int(os.environ.get("MICRODUCK_RAMP_R10_IT0", "4509"))
R10_BANK = os.environ.get("MICRODUCK_RAMP_R10_BANK") or None
R10_SHARE = float(os.environ.get("MICRODUCK_RAMP_R10_SHARE", "0.35"))
R10_W = dict(head_low=float(os.environ.get("MICRODUCK_RAMP_R10_HEAD_W", "-1.0")),
             jlim=float(os.environ.get("MICRODUCK_RAMP_R10_JLIM_W", "-0.5")),
             # -0.3 per W = skate_heat's own price again on the two hip yaws (r9 reward mass: positives ~3.5, costs ~-2,
             # skate_heat -0.83 per normalised s; -0.7 would push the net toward 0 = an early-termination incentive)
             hy_heat=float(os.environ.get("MICRODUCK_RAMP_R10_HY_W", "-0.3")))
R10_HALF = int(os.environ.get("MICRODUCK_RAMP_R10_HALF", "50"))
R10_HYERR_W = float(os.environ.get("MICRODUCK_RAMP_R10_HYERR_W", "0"))   # 0 = off (rp.r10_hy_err)
R10_NOISE = dict(xy=0.04, yaw_deg=3.0, joint=0.02, v_scale=0.0, dv=0.12)


def _add_r10(cfg, rw) -> None:
    """r10 (MICRODUCK_RAMP_R10=1, training only, after _r9_terminations): the hand-back bank spawn event right after
    reset_sequence (when MICRODUCK_RAMP_R10_BANK is set), the posture / limit / hip-yaw heat costs + stages, metrics."""
    assert RAMP_R9 and "r9_gain" in cfg.rewards, "r10 needs MICRODUCK_RAMP_R9=1"
    assert not RAMP_R9B and not RAMP_R9C, "r10 resumes r9 (R9B / R9C off)"
    for w in R10_W.values():
        assert w <= 0.0, "r10 weights are costs (<= 0)"
    if R10_BANK:
        assert os.path.isfile(R10_BANK), f"r10 bank missing: {R10_BANK}"
        assert RAMP_HANDOFF, "r10 bank spawns rely on the rampbank tilt guard (MICRODUCK_RAMP_HANDOFF)"
        ev = EventTermCfg(func=rp.reset_r10_bank, mode="reset", params=dict(path=R10_BANK, share=R10_SHARE,
                                                                             noise=dict(R10_NOISE)))
        events = {}
        for k, v in cfg.events.items():               # dict order = event order: right after reset_sequence
            events[k] = v
            if k == "reset_sequence":
                events["reset_r10"] = ev
        assert "reset_r10" in events
        cfg.events = events
        cfg.metrics["seq_r10_bank"] = MetricsTermCfg(func=rp.m_r10_bank)
    for k, f in (("head_low", rp.r10_head_low), ("jlim", rp.r10_jlim), ("hy_heat", rp.r10_hy_heat)):
        name = "r10_" + k
        cfg.rewards[name] = RewardTermCfg(func=f, weight=0.0)
        cfg.curriculum[name] = rw(name, _stages((0, 0.0), (R10_IT0, 0.5 * R10_W[k]), (R10_IT0 + R10_HALF, R10_W[k])))
    for k, f in (("head_h", rp.m_r10_head_h), ("squat", rp.m_r10_squat), ("jlim", rp.m_r10_jlim),
                 ("over", rp.m_r10_over), ("hy_amp", rp.m_r10_hy_amp), ("hy_near", rp.m_r10_hy_near)):
        cfg.metrics["seq_r10_" + k] = MetricsTermCfg(func=f)
    # r10 hip-yaw target-error cost (rp.r10_hy_err; the heat cost has no gradient at the 1.0 A clamp). Registered only
    # when MICRODUCK_RAMP_R10_HYERR_W != 0 (default 0: the r10 cfg is unchanged), same half -> full stages.
    if R10_HYERR_W != 0.0:
        assert R10_HYERR_W < 0.0, "r10_hy_err is a cost (weight < 0)"
        cfg.rewards["r10_hy_err"] = RewardTermCfg(func=rp.r10_hy_err, weight=0.0)
        cfg.curriculum["r10_hy_err"] = rw("r10_hy_err", _stages((0, 0.0), (R10_IT0, 0.5 * R10_HYERR_W),
                                                                (R10_IT0 + R10_HALF, R10_HYERR_W)))
        cfg.metrics["seq_r10_hy_err"] = MetricsTermCfg(func=rp.m_r10_hy_err)


def _chain_overrides() -> dict:
    """Scheduler settings of the full chain at playback: the push and manual specialists' (hold-ended MANUAL at its
    hardened stage) without their stage caps / floor length, then this task's entry line. Includes the push
    specialist's step_cmd (phase obs RIDE in PUSH once at speed): the push policy's obs contract."""
    from micro_skater.tasks import microduck_skate_seq_manual_env_cfg as man
    from micro_skater.tasks import microduck_skate_seq_push_env_cfg as push

    ov = {**push.SEQ_OVERRIDES, **man.SEQ_OVERRIDES, **man.HOLD_STAGES[-1][1], **SEQ_OVERRIDES}
    for k in ("last_stage", "start_gap"):
        ov.pop(k, None)
    ov["start_gap"] = CHAIN_START_GAP                  # headrev: the chain's own floor length (playback only)
    return ov


def make_microduck_skate_seq_ramp_env_cfg(play: bool = False, chain: bool = False) -> ManagerBasedRlEnvCfg:
    """chain=True: the playback cfg of the three-specialist chain (32 s episodes from START, no clock offset, the
    push / manual specialists' scheduler settings)."""
    cfg = make_microduck_skate_sequence_env_cfg(play=play)
    cfg.episode_length_s = CHAIN_EPISODE_S if chain else EPISODE_LENGTH_S

    # ── spawns + scheduler overrides ──
    p = cfg.events["reset_sequence"].params
    base_stances = p["spawn_stances"]
    p["spawn_stances"] = {k: list(v) for k, v in SPAWN_STANCES.items()}
    p["flat_speeds"] = dict(FLAT_SPEEDS)
    p["step_blend"] = None
    p["exit_tilt_prob"] = EXIT_TILT_PROB
    p["ramp_speed"] = RAMP_SPEED
    p["ramp_chain_prob"] = RAMP_CHAIN_PROB
    p["wall_frac"] = WALL_FRAC
    if chain:
        p["spawn_probs"] = dict(start=1.0)
        # every spawn type keeps a stance list (chain_play --spawn can start the chain mid-way)
        p["spawn_stances"] = {**{k: list(v) for k, v in base_stances.items()}, **p["spawn_stances"]}
        p["seq_overrides"] = _chain_overrides()
        p["clock_offset"] = None
    else:
        p["spawn_probs"] = dict(SPAWN_PROBS)
        p["seq_overrides"] = dict(SEQ_OVERRIDES)
        p["clock_offset"] = CLOCK_OFFSET
        if RAMP_HANDOFF:                               # rampbank (default off)
            p["handoff_path"] = RAMP_HANDOFF
            p["handoff_prob"] = RAMP_HANDOFF_PROB
            rest = 1.0 - SPAWN_PROBS["exit"]
            p["spawn_probs"] = {k: (RAMP_HANDOFF_EXIT_P if k == "exit" else v * (1.0 - RAMP_HANDOFF_EXIT_P) / rest)
                                for k, v in SPAWN_PROBS.items()}
            p["seq_overrides"]["ramp_gate_x"] = RAMP_HANDOFF_GATE_X
    # apex counter / energy potential reset (after reset_sequence: dict order = event order)
    cfg.events["reset_ramp"] = EventTermCfg(func=rp.reset_ramp_state, mode="reset")

    # r2: the trick block's kappa slot carries the RAMP carve command (rp.ramp_carve; EXIT / other stages unchanged)
    for grp in ("actor", "critic"):
        cfg.observations[grp].terms["trick"].func = rp.seq_trick_obs

    # ── rewards (per second). Kept from the chain task: on_deck 0.5, fall -10, body_on_board -1, head_pose -0.5,
    # ride 3, ride_wheels 1, tail_strike -1, head_steady, dof_pos_limits. Every PUSH / MANUAL term dropped ──
    R = RewardTermCfg
    for k in PUSH_MANUAL_TERMS:
        cfg.rewards.pop(k)
    cfg.rewards["upright"] = R(func=rp.seq_upright, weight=1.0)
    cfg.rewards["posture"] = R(func=rp.posture, weight=0.75)
    cfg.rewards["head_steady"].weight = -0.01
    cfg.rewards["ramp_lateral"].weight = RAMP_LATERAL_STAGES[0]["weight"]
    cfg.rewards["ramp_apex"] = R(func=rp.ramp_apex, weight=RAMP_APEX_STAGES[0]["weight"], params={"z_cap": APEX_Z_CAP})
    cfg.rewards["ramp_park"] = R(func=rp.ramp_park, weight=-0.5)
    cfg.rewards["ramp_energy"] = R(func=rp.ramp_energy, weight=RAMP_ENERGY_STAGES[0]["weight"])
    cfg.rewards["ramp_track"] = R(func=rp.ramp_track, weight=RAMP_TRACK_STAGES[0]["weight"])   # r2 carve tracking
    cfg.rewards["ramp_keep"] = R(func=rp.ramp_keep, weight=RAMP_KEEP_STAGES[0]["weight"],      # r3 pass retention
                                params={"den_cap": APEX_Z_CAP})   # r3rev: ratio vs min(previous pass, apex cap)
    # review (rrev): EXIT out-paid RAMP by ~2 /s on the warm start (ride + ride_wheels pay only in RIDE): price it
    cfg.rewards["exit_time"] = R(func=rp.exit_time, weight=EXIT_TIME_W)
    cfg.rewards["action_rate_l2"].weight = ACTION_RATE_STAGES[0]["weight"]
    cfg.rewards["board_wobble"].weight = BOARD_WOBBLE_STAGES[0]["weight"]

    # ── curriculum: wholesale replacement, stages in iterations of this run ──
    def rw(name, stages):
        return CurriculumTermCfg(func=microduck_mdp.reward_weight, params={"reward_name": name, "weight_stages": stages})

    cfg.curriculum = {
        "ramp_apex": rw("ramp_apex", RAMP_APEX_STAGES),
        "ramp_lateral": rw("ramp_lateral", RAMP_LATERAL_STAGES),
        "ramp_energy": rw("ramp_energy", RAMP_ENERGY_STAGES),
        "ramp_track": rw("ramp_track", RAMP_TRACK_STAGES),
        "ramp_keep": rw("ramp_keep", RAMP_KEEP_STAGES),
        "action_rate": rw("action_rate_l2", ACTION_RATE_STAGES),
        "board_wobble": rw("board_wobble", BOARD_WOBBLE_STAGES),
        # r2: RAMP carve command stages (rp.carve_curriculum -> st["rk_cfg"])
        "ramp_carve": CurriculumTermCfg(func=rp.carve_curriculum, params={"stages": CARVE_STAGES}),
    }

    if RAMP_PUMP:                                      # pump (default off)
        _add_pump(cfg, rw)
    if RAMP_R5:                                        # r5 (default off)
        _add_r5(cfg, rw)
    if RAMP_R5PUMP:                                    # r5pump (default off; after _add_r5: replaces its pump_dir stages)
        _add_r5pump(cfg, rw)
    if RAMP_R6:                                        # r6 (default off; after _add_r5 / _add_r5pump)
        _add_r6(cfg, rw)
    if RAMP_R7:                                        # r7 (default off; after _add_r6)
        _add_r7(cfg, rw)
    if RAMP_R8B:                                       # r8b (default off; after _add_r7)
        _add_r8b(cfg, rw)
    if RAMP_R9:                                        # r9 (default off; after _add_r7 / _add_r8b)
        _add_r9(cfg, rw)

    # ── metrics ──
    if not chain:
        for k in PUSH_MANUAL_METRICS:
            cfg.metrics.pop(k, None)
    cfg.metrics["seq_rise"] = MetricsTermCfg(func=rp.m_rise)
    cfg.metrics["seq_pass_rise"] = MetricsTermCfg(func=rp.m_pass_rise)
    cfg.metrics["seq_carve_on"] = MetricsTermCfg(func=rp.m_carve_on)
    cfg.metrics["seq_keep"] = MetricsTermCfg(func=rp.m_keep)          # r3: seq_keep / seq_pass2 = mean retention
    cfg.metrics["seq_pass2"] = MetricsTermCfg(func=rp.m_pass2)

    # ── terminations: tilt guards against up_eff in RAMP, ramp_done success; the rest as the chain task ──
    T = TerminationTermCfg
    cfg.terminations["fell_over"] = T(func=rp.seq_fell_over, params={"limit_angle": math.radians(60.0)})
    cfg.terminations["riding_tilt"] = T(func=rp.riding_tilt_guard, params={"limit_deg": 50.0})
    if RAMP_HANDOFF and not chain:                     # rampbank: hand-off spawns are guarded from their first step
        cfg.terminations["riding_tilt"].params["handoff_after_s"] = 0.0
    cfg.terminations["ramp_done"] = T(func=rp.ramp_done, time_out=True, params=dict(RAMP_DONE))
    if RAMP_R5PUMP and not chain:                      # r5pump: training spawns / RAMP window (never the chain playback)
        _r5pump_spawns(cfg)
    # r2: rolling back out of the entry end is a failure (seq_fall: -10); the far end keeps the name and the time-out
    cfg.terminations["past_pipe_end"] = T(func=rp.past_pipe_end_far, time_out=True)
    cfg.terminations["past_pipe_end_entry"] = T(func=rp.past_pipe_end_entry)
    if RAMP_R9:                                        # r9 (default off): after r5pump's spawns / ramp_done
        _r9_terminations(cfg, chain)
    if RAMP_R9B and not chain:                         # r9b (default off, training only): after _r9_terminations
        _add_r9b(cfg)
    if RAMP_R9C and not chain:                         # r9c (default off, training only): after _add_r9b
        _add_r9c(cfg)
    if RAMP_R10 and not chain:                         # r10 (default off, training only): after _r9_terminations
        _add_r10(cfg, rw)
    return cfg


def make_microduck_skate_seq_ramp_ppo_runner_cfg():
    rl = make_microduck_skate_sequence_ppo_runner_cfg()
    rl.experiment_name = "skate_seq_ramp"
    rl.max_iterations = 1500
    rl.save_interval = 100
    rl.algorithm.entropy_coef = 0.002
    return rl
