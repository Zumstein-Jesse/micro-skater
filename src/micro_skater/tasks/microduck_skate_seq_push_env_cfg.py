# micro-skater: part of the author's skateboard add-on for microduck_rl. Apache-2.0 (see LICENSE).
"""Skate SEQUENCE push + step-on SPECIALIST (2026-11-20). Task Mjlab-Skate-SeqPush-MicroDuck.

One of three specialists (push + step-on / manual / approach + ramp) with the SAME 114D obs and actions as
Mjlab-Skate-Sequence-MicroDuck, chained by scheduler stage at playback (
chain_play.py). Built on make_microduck_skate_sequence_env_cfg (obs / actions / sensors / DR untouched); the scheduler is
capped at SETTLE (seq_overrides last_stage), so MANUAL is never observed.

Two flaws of the chain stack fixed here:
  - push_step_on paid 6/s on every PUSH step with both soles on the deck: brief taps onto the deck earned ~+2/s without
    finishing the step-on. Here it is an event (stepon_evt, the PUSH -> SETTLE transition, once per episode).
  - no terminal price for refusing to step on: seq_overtime (PUSH > 8 s, SETTLE > 2.5 s) is a failure (seq_fall -10).
Success = a time_out=True termination (not a bonus): SETTLE, ts >= T_settle, both soles on >= 0.3 s, wheels >= 0.2 s.

stepfix (2026-11-20, MICRODUCK_SEQPUSH_STEPFIX=0 restores p5): p5 never
stepped on because (measured) the step-on is a whole-body move the push stance cannot start (CoM 5.7 cm outside the
board sole, the trunk 2 cm too low for the push sole to clear the 24 mm deck), under the command the policy kept its
stroke cycle beside the board (sole 80 mm up at y -0.10, CoM 22 mm outside the board sole), and its reverse curriculum
(step_blend, a joint-space blend toward glide_step) spawned impossible states (board sole 6-10 mm inside the deck, push
foot jammed into the deck edge; 17/32 ended within 0.1 s even at zero action). Fix: step_mid spawns from VALID
whole-body-IK mid-step stances (step_over / step_edge / step_lift, zero-shot step-ons s2c 98 / 71 / 46 %, p5
79 / 44 / 0 %, vs 2 % for
the blend), annealed toward the real starts; push_step_path (via-point + weight-shift potential under the command)
replaces push_step_reach; step_blend off.

Warm start from s2c model_2500 with MICRODUCK_WARM_START=1 (iteration counter restarts at 0); the curriculum below is in
iterations of THIS run (x 24 env steps) and replaces the parent's REV_STEP0-offset stages wholesale.
"""

from __future__ import annotations

import os

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import CurriculumTermCfg, MetricsTermCfg, RewardTermCfg, TerminationTermCfg

from micro_skater.tasks import mdp as microduck_mdp
from micro_skater.tasks import skate_sequence as sq
from micro_skater.tasks.microduck_skate_sequence_env_cfg import (make_microduck_skate_sequence_env_cfg,
                                                                    make_microduck_skate_sequence_ppo_runner_cfg)

EPISODE_LENGTH_S = 11.0
IT = 24                                   # env steps per iteration (NUM_STEPS_PER_ENV)

CMD_MAX_S = 2.5
SEQ_OVERRIDES = dict(step_cmd=True, stroke_air=(0.25, 0.65), stroke_sweep=(0.02, 0.06), settle_s=(0.5, 1.0), push_max_s=5.0,
                     push_late_s=4.5, last_stage=sq.ST_SETTLE)
SPAWN_PROBS = dict(start=0.45, push=0.20, step_on=0.30, glide=0.05)
SPAWN_STANCES = dict(start=["mount"], push=["push", "push_back", "mount"], step_on=["push_back", "push"],
                     glide=["glide_step"])
FLAT_SPEEDS = dict(push=(0.1, 0.8), glide=(0.6, 1.0), manual=(0.5, 1.0), exit=(0.5, 1.0))
STEP_BLEND = ("glide_step", (0.2, 1.0), 0.7)

SUCCESS = dict(stage=sq.ST_SETTLE, on_s=0.3, wheels_s=0.2)
OVERTIME_LIMITS = ((sq.ST_PUSH, 8.0), (sq.ST_SETTLE, 2.5))   # p5: push_max_s 5 + CMD_MAX_S 2.5 < 8 (cmd timeout first)
RIDE_OFF_S = 0.25                         # push foot off the deck outside PUSH -> board_lost

# ── stepfix (see the docstring); MICRODUCK_SEQPUSH_STEPFIX=0 = the p5 task ──
ENABLE_STEPFIX = os.environ.get("MICRODUCK_SEQPUSH_STEPFIX", "1") == "1"
# step_mid stance weights: step_over is already ~80 % zero-shot, so the harder edge / lift slices get 2x
STEPFIX_STANCES = ["step_over", "step_edge", "step_edge", "step_lift", "step_lift"]
# reverse curriculum (iterations of this run): step_mid share annealed toward the real starts (start / step_on at speed)
STEPFIX_SPAWN_STAGES = (
    (0, dict(start=0.35, push=0.10, step_on=0.15, step_mid=0.35, glide=0.05)),
    (300, dict(start=0.40, push=0.15, step_on=0.20, step_mid=0.20, glide=0.05)),
    (700, dict(start=0.45, push=0.20, step_on=0.20, step_mid=0.10, glide=0.05)),
)
# ── bridge (2026-11-20; MICRODUCK_SEQPUSH_BRIDGE=0 = stepfix only): p6 it100
# was 100 % from step_mid but 0 % from start - its own stroke-air states under the command are the missing rung. The
# stroke_air spawn draws whole recorded p6 states (deck-relative), blended toward step_lift by rung (BRIDGE_RUNGS).
ENABLE_BRIDGE = ENABLE_STEPFIX and os.environ.get("MICRODUCK_SEQPUSH_BRIDGE", "1") == "1"
BRIDGE_RUNGS = dict(r0=1.0, r50=1.0, r67=2.0, r75=3.0, r83=3.0, r92=1.0)  # p6 it100: 0 / 0 / 0 / 2 / 40 / 98 % (r83 = frontier)
# bridgerev (2026-11-20, MICRODUCK_SEQPUSH_BRIDGE_ADAPT=0 = the fixed weights above): frontier tracking. Per-rung success
# estimate p (decayed counts of the `success` termination, ~8-iteration window); rung weight = base * (0.1 + 4 p (1 - p)),
# so the rungs near 50 % get most of the share and the frontier walks down the ladder by itself. All rungs base 1
# (r33 added: the gap between the real start r0 and r50). Logged as Metrics/seq_bridge_frontier (lowest rung >= 50 %).
ENABLE_BRIDGE_ADAPT = ENABLE_BRIDGE and os.environ.get("MICRODUCK_SEQPUSH_BRIDGE_ADAPT", "1") == "1"
# bridge2: finer rungs in the p7 r33-r50 gap (p7 it400: r33 0 %, r50 46 %); bank re-recorded from p7 model_500
BRIDGE_RUNGS_ADAPT = dict(r0=1.0, r10=1.0, r20=1.0, r30=1.0, r35=1.0, r40=1.0, r45=1.0, r50=1.0, r58=1.0, r67=1.0, r75=1.0,
                          r83=1.0, r92=1.0)
BRIDGE_ADAPT = dict(floor=0.1, decay=0.995)
BRIDGE_SPAWN_STAGES = (
    (0, dict(start=0.30, push=0.10, step_on=0.15, step_mid=0.25, stroke_air=0.15, glide=0.05)),
    (300, dict(start=0.35, push=0.15, step_on=0.15, step_mid=0.15, stroke_air=0.15, glide=0.05)),
    (700, dict(start=0.40, push=0.15, step_on=0.15, step_mid=0.10, stroke_air=0.15, glide=0.05)),
)
STEP_PATH_W = 20.0                        # = push_step_reach's weight; the banked sum is <= ~20 x 0.13 m = 2.6
# ── landrule (2026-11-20, MICRODUCK_SEQPUSH_LANDRULE=0 = off): under the step command the push foot's next landing must
# be on the deck (seq_cmd_landing, charged at the full fall price = the overtime-equivalent: an attempt that falls costs
# the attempt-fall ramp 0.5-1.0, so attempting is never dearer than another stroke); push_progress pays no speed gain
# under the command and charges no speed loss there (it paid a stroke's gain, and its 10x loss charge on the trunk
# slowing made a failed attempt (-10.3..-11.7 discounted, p7 m700) dearer than the overtime refusal (-4.6..-8.2)).
ENABLE_LANDRULE = ENABLE_BRIDGE and os.environ.get("MICRODUCK_SEQPUSH_LANDRULE", "1") == "1"
LAND_SCALE = 1.0
# landrev review: the push sole must touch the deck within LAND_REACH_S of the command (successes: 0.10-0.24 s, p7
# m700) - closes the planted-foot / foot-held-up refusals that otherwise ran to the 2.5 s overtime (~0.28 discounted)
LAND_REACH_S = 1.0

# ── p9 polish (2026-11-20; MICRODUCK_SEQPUSH_P9=0 = the p8 task). Two
# video reviews failed p8 model_1200 for demo use. Measured from start (48 envs, det / stoch): PUSH lasts 0.88 s
# to the command + 0.40 s step-on; strokes are an initial shove, a double tap (touchdowns 0.06 s apart), one big kick
# (contact 0.08 s, +0.26 m/s per contact, board accel p99 15 m/s^2); recovery foot peaks 61-64 mm (main strokes);
# trunk 86 mm above the deck, dipping to 66-70 at touchdowns; head rate median 3.1 rad/s in PUSH, the head swings 70 deg
# through the step-on; edge crossed 7.5 mm up; trunk-vs-board speed lurch 0.4-0.5 m/s at the step. Why it rushes: with
# gamma 0.99 every second spent pushing discounts the ~13 of step-on value by ~40 %, so nothing in the stack could
# buy a deliberate push. Fixes:
#   - push_ramp (SEQ_DEFAULTS, a slewed target): the command needs the ramp spawn speed +
#     rate * t to reach v_tgt - v_gap too, and push_progress / push_speed pay min(speed, ramp). Rushing no longer
#     brings the step-on sooner and ahead-of-ramp speed pays nothing; following the ramp with a stroke inside the 0.6 s
#     window pays it all. Curriculum 0.8 -> 0.35 m/s^2 (command from rest at ~1.0 s -> 2.0-2.6 s = 3-4 strokes at ~1.3 Hz).
#   - stroke_sweep (0.02, 0.06) -> (0.03, 0.08): q = 1 needs an 80 mm sweep of the planted sole = leg extension.
#   - costs (skate_sequence p9 block, all >= 0, negative weights): push_foot_height (recovery above 35 mm),
#     push_trunk_floor (trunk < 80 mm above the deck in PUSH), head_level / head_rate gated to the PUSH stage, step_lurch
#     (|trunk - board| > 0.1 m/s under the command), step_clearance (sole < 15 mm over the edge band).
ENABLE_P9 = os.environ.get("MICRODUCK_SEQPUSH_P9", "1") == "1"
P9_IT0 = int(os.environ.get("MICRODUCK_SEQPUSH_P9_IT0", "1200"))   # iteration counter of the resume (p8 model_1200)
P9_SWEEP = (0.03, 0.08)
P9_RAMP_STAGES = ((0, 0.8), (200, 0.55), (400, 0.45), (600, 0.35))   # (iterations after P9_IT0, m/s per s); model_1200
# zero-shot at 0.8: 92 % success (= p8), at 0.35: 44 % (5 strokes, 1.08 m/s overshoot) - the ramp must be paced in
# p8 model_1200 per-episode raw (det / stoch, start): foot_height 0.64 / 0.81, trunk_floor ~0.2, head_level (PUSH)
# 0.17 / 0.25, head_rate (PUSH, free 2) 1.6 / 2.7, lurch 0.23 / 0.27 (command), clearance ~0.1 -> ~-3.5..-4.5 per
# episode in total at these weights against ~+17 of task return; only step_clearance is attempt-only (<= ~0.5).
P9_W = dict(foot_height=-1.5, trunk_floor=-2.0, head_level=-2.0, head_rate=-0.5, lurch=-2.0, clearance=-4.0)
P9_HEAD_LEVEL_STAGES = ((0, -1.0), (200, -2.0))
P9_HEAD_RATE_STAGES = ((0, -0.25), (200, -0.5))
P9_HEAD = dict(free_deg=10.0, rate_free=2.0)
# ── p9b hand-off stance (2026-11-20; MICRODUCK_SEQPUSH_P9B=0 = off). In the chain the MANUAL specialist takes over at
# SETTLE and survives only 6-31 % of p8 hand-offs: p8 settles with the board (left) sole at the deck centre (+0.7 / +0.4
# cm), soles 4.8 cm apart, trunk rolled 16 deg, CoM 10.4 cm; the manual rides from left sole (-4.3, +4.2) cm, 9.2 cm
# apart, roll ~2 deg, tilt +9 deg, CoM 12.5 cm (sq.HANDOFF_STANCE; out/handoff_sweep/). Success now also
# needs that stance held >= 0.2 s (seq_success stance=True), settle_stance_path (a bounded potential, charged back on
# a SETTLE failure) leads into it, and SETTLE overtime 2.5 -> 1.5 s, priced 1.5 x the fall (see below).
# Pricing: a step-on that misses the stance ends at the SETTLE overtime (-15, 1.5 s after the banked step-on, i.e.
# ~-7 discounted at the step-on): worse than reaching the stance, still better than refusing the step-on.
# Tolerances (left sole m, right sole m, CoM height m, tilt x deg, tilt y deg) start where p8's hand-off passes (p90:
# 0.075, 0.016, 0.022, 10.4, 16.4) and close to ~2x the manual's own spread (p90 0.013, 0.011, 0.004, 2.6, 3.2);
# stages sit 100 iterations off the ramp stages so the two curricula do not step together.
ENABLE_P9B = ENABLE_P9 and os.environ.get("MICRODUCK_SEQPUSH_P9B", "1") == "1"
# model_1200 zero-shot (best 0.2 s window in SETTLE): the binding components are the trunk roll
# across the deck (|err| median 16-17 deg from start / r50) and the forward tilt (12-13 deg from step_mid / r50), then
# the board sole (51-67 mm). (0.08, 0.03, 0.025, 12, 18) passed start 71 % / step_mid 65 % / r50 33 %, (0.06, .., 10,
# 13) only 2-17 %: stage 0 widened on tilt, finer steps after.
# p9brev: (a) the stage steps are now MINIMUM iterations - stance_tol_curriculum also needs >= P9B_TOL_GATE success
# among the episodes ending in SETTLE before it tightens (a time-only stage that outran the policy left success at 0 %
# with no signal: the potential is charged back on the overtime); (b) the final stage stops at 8 deg / 8 deg and
# 3 / 2 cm / 1.3 cm (~2.5-3x the manual's own p90 spread) instead of 5 / 5 deg: the manual specialist is also retrained
# on recorded hand-off states, so the push needs the basin, not the manual's median (zero-shot model_1200 is 0 %
# already at the old 250-iteration stage: ty p50 16.7 deg from start).
P9B_TOL_STAGES = ((0, (0.08, 0.03, 0.025, 16.0, 22.0)), (100, (0.07, 0.03, 0.022, 13.0, 17.0)),
                  (250, (0.06, 0.03, 0.019, 11.0, 13.0)), (400, (0.05, 0.027, 0.017, 10.0, 11.0)),
                  (550, (0.04, 0.023, 0.015, 9.0, 9.5)), (700, (0.03, 0.02, 0.013, 8.0, 8.0)))
P9B_TOL_GATE = 0.6
P9B_HOLD_S = 0.2
# review of the first p9b audit (finaltol.json): with a 2.0 s SETTLE limit at the plain fall
# price the dense SETTLE pay (~3 /s) collected until the overtime made a missed stance ~ a success (step_mid +10.8 vs
# +10.9); 1.5 s (the request) at 1.5 x the fall price restores a clear gap and still leaves a missed stance far above
# refusing the step-on (~-7 at the step-on against -6..-10 for the refusals).
P9B_SETTLE_OVERTIME_S = 1.5
P9B_SETTLE_OVERTIME_SCALE = 1.5
P9B_PATH_W = 0.25                          # x stance_distance (tolerance units): p8 hand-off ~15.7 -> <= ~3.9 banked
# ── p10 "deliberate push" (2026-11-20; MICRODUCK_SEQPUSH_P10=0 = p9b).
# A video review failed p9b model_2000: 0.38 s touchdown interval, 60 ms contacts (a tap), head bob 21 mm / roll ~5 deg
# per stroke, swing sole pitched ~31 deg and twisted, head pitched back (head_pitch p90 1.2 rad) and turned (+0.30 rad
# head yaw) in SETTLE. Scale: leg ~0.10 m (hip-pitch axis to sole) vs ~0.9 m for a human, Froude time factor
# sqrt(0.10 / 0.9) = 0.33, so a human 0.8-1.2 s stroke is 0.27-0.40 s here: p9b's cadence is already dynamically
# human-like; a push that LOOKS deliberate in real time needs ~1.7x that (target touchdown interval >= 0.65 s, i.e. a
# human-equivalent lazy 2 s push with a glide). Contact is reach-limited (a planted sole slides back at the board speed:
# 12 cm of leg sweep = 0.12 s at 1 m/s, 0.24 s at 0.5 m/s), so the duty target is ~30-35 % from rest, ~20 % at speed.
# Why p9's stroke costs did not move the cadence: push_stroke's q = q_air * q_sweep sat
# at 0.11-0.15 the whole run (air 0.30 s vs its 0.25-0.65 ramp -> q_air 0.12, sweep 47 mm vs 30-80 -> 0.34): a flat
# ~1.06 tax per stroke with almost no gradient in either factor; strokes 4-7 (+0.07-0.13 m/s each) ride ABOVE the ramp
# (pre-command top speed 1.46 m/s vs a 0.7-0.9 command) where nothing pays or charges them; the command latches only
# during a floor contact, so tapping minimises the latch wait. Fixes (skate_sequence p10 block, costs, weights ramped):
# push_cadence (minimum touchdown interval), push_stroke_shape (additive contact + sweep quality, reach-limited contact
# target), push_overspeed (v > v_tgt + 0.15 in PUSH: the free strokes past the ramp), push_swing_foot (sole pitch /
# twist / clearance in the swing), head_bob (PUSH: head vz + roll rate), settle_head_pose (SETTLE: the 4 head joints to
# HANDOFF_STANCE). The SETTLE feet stay as they are: HANDOFF_STANCE (the manual's riding state) itself has both soles at
# deck x -0.035..-0.043 = over the rear truck, side by side; changing it would break the hand-off.
ENABLE_P10 = ENABLE_P9B and os.environ.get("MICRODUCK_SEQPUSH_P10", "1") == "1"
P10_IT0 = int(os.environ.get("MICRODUCK_SEQPUSH_P10_IT0", "2100"))   # iteration counter of the resume (p9b model_2100)
P10_W = dict(cadence=-1.0, shape=-0.5, overspeed=-0.5, swing=-0.3, head_bob=-0.3, settle_head=-2.0)
P10_FRAC_STAGES = ((0, 0.25), (100, 0.5), (250, 0.75), (400, 1.0))    # (iterations after P10_IT0, share of P10_W)
P10_TERMS = dict(cadence="push_cadence", shape="push_stroke_shape", overspeed="push_overspeed",
                 swing="push_swing_foot", head_bob="push_head_bob", settle_head="settle_head_pose")
# ── heading (2026-11-20; DEFAULT OFF: MICRODUCK_SEQPUSH_HEADING=1).
# p9b model_2100 from start (32 det): heading +61 deg LEFT in PUSH before the command (+5 under it, ~0 in SETTLE),
# yaw rate +32 deg/s while the push foot is up vs +5 with it down, deck rolled ~1.5 deg left between strokes (the board
# foot's lean): x -1.6 m, heading +68 deg at SETTLE, so the chain never reached the pipe. heading_line_cost prices the
# bearing error to the scheduler's approach target (the point EXIT's pursuit steers to) in PUSH + SETTLE; line_noise
# spawns mis-aimed / off-line starts to correct. Stages count from HEAD_IT0 (= the iteration of the resume checkpoint;
# resume keeps the step counter), stage 0 at step 0.
ENABLE_HEADING = ENABLE_P10 and os.environ.get("MICRODUCK_SEQPUSH_HEADING", "0") == "1"
HEAD_IT0 = int(os.environ.get("MICRODUCK_SEQPUSH_HEAD_IT0", "2450"))
HEAD_W = -0.6                              # raw episode cost: drifting p9b ~7.5, straight ~0.4 -> -4.5 vs -0.2
HEAD_FRAC_STAGES = ((0, 0.33), (100, 0.67), (200, 1.0))   # (iterations after HEAD_IT0, share of HEAD_W)
HEAD_PARAMS = dict(free_deg=8.0, unit_deg=10.0, cap=3.0, v_full=0.3, stages=(sq.ST_PUSH, sq.ST_SETTLE))
HEAD_NOISE = {"start": (0.5, 20.0, 0.5), "push": (0.5, 20.0, 0.5), "step_on": (0.3, 15.0, 0.4),
              "step_mid": (0.3, 15.0, 0.4), "stroke_air": (0.3, 15.0, 0.4)}   # spawn -> (prob, +-yaw deg, +-x m)
HEAD_NOISE_STAGES = ((0, 0.5), (100, 1.0))  # share of each prob (a mis-aimed start is new to the policy)
# headrev (2026-11-20, with the heading flag only): the chain's geometry. start_gap 5 -> 9 (= the chain's start,
# microduck_skate_seq_ramp_env_cfg.CHAIN_START_GAP) and the bearing reference = the chain's EXIT pursuit target (the
# ramp specialist's target_x 0.45, target_in 0.6). From the 5 m start a STRAIGHT rider's bearing error to (0.35, -2) was
# 9 deg at the start and 13 deg by SETTLE (over the 8 deg dead band): the term trained a diagonal turn-in during PUSH
# that would arrive at EXIT on/over the axis (outside the ramp's exit spawns, far side |x| 0.45-0.75). From 9 m it is
# 6 -> 8 deg (free): the term charges the drift, not the straight line. PUSH / SETTLE y < -6 m, so the y obs (clipped
# at y_open < -3) reads -2 throughout, as in the chain.
HEAD_GEOMETRY = dict(start_gap=9.0, target_x=0.45, target_in=0.6)
# ── p12 posture polish (2026-11-20; DEFAULT OFF: MICRODUCK_SEQPUSH_P12=1).
# The p11 model_2799 review failed on head bob / tilt, cadence, swing sole and the step-on stand-up. Measured (32 envs
# from start, det / stoch): the head is flung as a reaction mass (per stroke head yaw joint 44 deg p2p,
# head pitch 20, neck 16, roll 15; head-joint speed sum p50 4.6 rad/s) while the head's own z moves only 8 mm and its
# vz rms is 0.04 m/s, under head_bob's 0.05 m/s dead band: head_bob priced the wrong signal. The trunk rocks 10.6 deg
# and bobs 10 mm per stroke. push_swing_foot is saturated (swing sole 50 deg toe-down vs its cap at 45 deg). Under the
# step command the trunk rises 26 mm over the push crouch and the head pitches to 0.97 rad / yaws 0.71 rad at 9.7 rad/s,
# outside every head cost (settle_head_pose is SETTLE-only). Pre-command touchdown gap p50 0.58 s (det) with only 3
# touchdowns: the robot is at 1.0 m/s by the 2nd touchdown (v at touchdown p50 1.01, command at 1.14, top 1.34 m/s), so
# contacts are reach-limited to 60 ms (sweep 45 mm). Feasibility: hip ~9 cm over the floor, leg ~10 cm -> planted-sole
# sweep <= ~0.10 m, contact <= sweep / v: >= 0.2 s only below ~0.5 m/s, ~0.10-0.12 s at 0.8-1.0 m/s. A 0.65-0.8 s gap is
# feasible (p90 board gain per contact 0.23 m/s >= ramp 0.35 m/s^2 x 0.7 s). So: lower the speed (v_t (0.8, 1.0) ->
# (0.75, 0.9), overspeed margin 0.15 -> 0.08 => hand-off <= ~1.0 m/s, inside the manual's 0.5-1.0 range), lengthen
# the touchdown interval (p12_cadence: t_ok 0.8), push_stroke_shape -0.5 -> -1.0 (sweep), and price the real signals.
ENABLE_P12 = ENABLE_P10 and os.environ.get("MICRODUCK_SEQPUSH_P12", "0") == "1"
P12_IT0 = int(os.environ.get("MICRODUCK_SEQPUSH_P12_IT0", "2800"))   # iteration counter of the resume (p11 model_2799)
# audit (full weights, start, p11 behaviour): swing_sole at -0.5 was only -0.10 discounted (the
# gated swing averages ~30 deg, 50 is the peak) -> -1.5; trunk_steady / head_steady priced the step-on's required rise /
# head move (-2.1 / -0.8, step_on spawns -3.4) -> both now PUSH-before-the-command only, the step-on is priced by
# p12_stepon_head and the rise overshoot (deck frame > 118 mm)
P12_W = dict(head_steady=-0.4, trunk_steady=-1.0, stepon_head=-1.5, swing_sole=-1.5, cadence=-1.5)
P12_TERMS = dict(head_steady="p12_head_steady", trunk_steady="p12_trunk_steady", stepon_head="p12_stepon_head",
                 swing_sole="p12_swing_sole", cadence="p12_cadence")
P12_FRAC_STAGES = ((0, 0.25), (100, 0.5), (200, 0.75), (300, 1.0))   # (iterations after P12_IT0, share of P12_W)
P12_CADENCE = dict(t_ok=0.80, t_bad=0.45)
P12_SHAPE_W = -1.0                                   # push_stroke_shape from P12_IT0 + 100 (p10 final -0.5)
P12_VT_STAGES = ((0, (0.8, 0.95)), (100, (0.75, 0.9)))   # SEQ_DEFAULTS v_t (p11: (0.8, 1.0))
P12_OVERSPEED_MARGIN = 0.08
P12_TRACK = dict(p12_track=True, p12_tau=0.5, p12_win_s=0.4)
P12_LAND_GRACE_S = 0.1     # p12rev: seq_cmd_landing spawn grace (0.1 s = _track_strokes' spawn-contact window)
# ── p13 posture polish 2 (2026-11-20; DEFAULT OFF: MICRODUCK_SEQPUSH_P13=1,
# needs P12=1). Two reviews failed p12 model_3398 on the step-on "jack-in-the-box" and the tilted swing foot, both
# unchanged through p11 -> p12. Why (see the skate_sequence p13 block): (1) the
# swing foot is ROLLED 47 deg (hip roll at its -22 deg limit + ~20 deg trunk roll), not pitched - every swing term priced
# pitch, which is already ~flat mid-swing; (2) push_step_path's trunk floor (0.112 m) asks for the 72 -> 111 mm stand-up
# that p12_trunk_steady's 118 mm cap never saw, and the trunk leans back 34 deg while tall; the IK step-on clears the
# deck edge (16 mm) down to an 81 mm trunk, so the floor drops to 0.090 (= the SETTLE height); (3) the trunk sinks
# 87 -> 77 mm over the strokes (72 at the latch); (4) the swing sole is parked 0.28 s / swing (p50). Fixes: p13_trunk_hold
# (push band 82-96 mm), p13_stepon_pose (window: trunk <= 98 mm, |pitch| <= 12 deg, neck / head pitch overshoot),
# p13_swing_tilt (sole roll + pitch tilt > 20 deg), p13_hover (sole < 0.15 m/s vs the deck past 0.20 s per swing).
ENABLE_P13 = ENABLE_P12 and os.environ.get("MICRODUCK_SEQPUSH_P13", "0") == "1"
P13_IT0 = int(os.environ.get("MICRODUCK_SEQPUSH_P13_IT0", "3406"))   # p13rev: = common_step_counter / 24 of p12 model_3398 (81744, not 3399 x 24)
P13_W = dict(trunk_hold=-1.0, stepon_pose=-1.5, swing_tilt=-0.5, hover=-2.0)
P13_TERMS = dict(trunk_hold="p13_trunk_hold", stepon_pose="p13_stepon_pose", swing_tilt="p13_swing_tilt",
                 hover="p13_hover")
P13_FRAC_STAGES = ((0, 0.25), (100, 0.5), (200, 0.75), (300, 1.0))   # (iterations after P13_IT0, share of P13_W)
P13_TRACK = dict(p13_track=True, p13_v_still=0.15, p13_hover_s=0.20)   # audit: 0.12 / 0.25 fired ~0 (start -0.04)
P13_PATH_TRUNK_Z_MIN = 0.090              # push_step_path trunk floor (bridge 0.112): the IK step-on passes at 81-91 mm
# ── p14 posture polish 3 (2026-11-21; DEFAULT OFF: MICRODUCK_SEQPUSH_P14=1,
# needs P13=1). p13 model_3797 fixed only the hover: step-on overshoot 19 mm / vz 0.27 m/s, swing sole ROLLED 47 deg (real,
# see the skate_sequence p14 block), head yaw 72 deg p2p. The p13 costs were linear and small (-1.8 / -1.8 per start
# episode against ~34 of positive return). p14: squared costs (gradient at the current behaviour), sized at full weight
# on p13 behaviour (start, undiscounted): stepon_pop ~-2.6, sole_roll ~-2.7, head_yaw ~-1.8 (~-7 of ~+16 total);
# the target behaviour (<= 8 mm, <= 0.15 m/s, roll <= 15 deg, head +-12 deg) costs < 0.3. Half weight, full at +150.
ENABLE_P14 = ENABLE_P13 and os.environ.get("MICRODUCK_SEQPUSH_P14", "0") == "1"
P14_IT0 = int(os.environ.get("MICRODUCK_SEQPUSH_P14_IT0", "3806"))
P14_W = dict(stepon_pop=-1.5, sole_roll=-0.35, head_yaw=-0.5)
P14_TERMS = dict(stepon_pop="p14_stepon_pop", sole_roll="p14_sole_roll", head_yaw="p14_head_yaw")
P14_FRAC_STAGES = ((0, 0.5), (150, 1.0))   # (iterations after P14_IT0, share of P14_W)
# ── p15 speed back (2026-11-21; DEFAULT OFF: MICRODUCK_SEQPUSH_P15=1, needs
# P14=1). p14 model_4246 calmed the step-on (overshoot 11.8 mm) and the head (25 deg p2p) but STALLS ~1.9 s with the
# push foot planted (v >= 0.2 m/s at 1.92 s vs 0.70, 2 strokes, chain complete 22 % vs 66 %): the step command is
# exogenous (v_ramp), push_progress is path-free, and every stroke pays ~10 stroke-shaped costs (p13 behaviour ~12 /
# episode pre-command vs ~3.8 for the planted stall). The airborne sole roll is not fixable (no ankle roll; hip roll at
# its -22 deg limit + trunk roll -20..-25 deg in the swing; IK: see the skate_sequence p15 block). p15, resumed from p14:
#   - p15_speed_deficit (new, full weight at once): (min(v_ramp, v_tgt) - v - 0.07) / 0.08, capped 2.5, PUSH pre-cmd
#     after 0.3 s (p15rev: free 0.10 -> 0.07 < v_gap prices a hover under the latch; the grace spares the push spawns'
#     drag): p14 stall ~-6.5 / episode, p13 ~-0.5. Max 3.75 / s (discounted ~-7.5 forever) < a fall (-10 + lost stream).
#   - p15_land_roll (new): per floor touchdown ((|roll| - 10) / 10)^2 capped 9 (p13 33 deg -> 5.3; flat -> 0).
#   - p14_sole_roll (whole-PUSH roll) and p13_swing_tilt (swing tilt) -> 0: they priced the unfixable swing roll, i.e.
#     taxed every stroke (the planted foot was the only escape).
#   - p14_head_yaw softened: free 0.10 -> 0.20 rad (+-11 deg; p2p <= 30 deg target), rate free 1.0 -> 1.5 rad/s, so a
#     stroking head is near free and standing still buys little.
ENABLE_P15 = ENABLE_P14 and os.environ.get("MICRODUCK_SEQPUSH_P15", "0") == "1"
P15_IT0 = int(os.environ.get("MICRODUCK_SEQPUSH_P15_IT0", "4256"))
P15_W = dict(speed=-1.5, land_roll=-0.15)
P15_TERMS = dict(speed="p15_speed_deficit", land_roll="p15_land_roll")
P15_FRAC_STAGES = dict(speed=((0, 1.0),), land_roll=((0, 0.5), (100, 1.0)))   # (iterations after P15_IT0, share)
P15_DROP = ("p14_sole_roll", "p13_swing_tilt")       # weight 0 from P15_IT0 (stage 0 at step 0 = already 0)
P15_HEAD_YAW = dict(free=0.20, rate_free=1.5)
# ── p16 real servo limits (2026-11-21; DEFAULT OFF: MICRODUCK_SEQPUSH_P16=1,
# needs P15=1). p15 model_4645 at the real 1.0 A firmware limit + datasheet speed (n 32): from start
# 72 % fall at 1.38 s median, all while stroking (pre-command), 3 % success; step_on 22 % success; at 1.4 A it still
# succeeds 84 % / 78 % (diag_1p4A.json) - the cliff is between 1.4 and 1.0 A. Saturated: the hip yaws (right 49 % of
# the strokes, 91 % under the command, 99.6 % in SETTLE = a static squeeze), right hip roll / knee / ankle 60-70 % in the
# last 0.5 s before a fall. The push leg also bends its knee backwards (right knee > +0.1 rad 41-50 % of the strokes),
# taps the floor (contact p50 0.08 s) and the neck cranes (neck pitch p10 -0.77 rad). p16, resumed from p15:
#   - current curriculum (MICRODUCK_SEQPUSH_P16_AMPS=1, separate so that evals with P16=1 keep the startup current):
#     sq.p16_current_curriculum sets bam max_current at runtime 1.4 -> 1.2 -> 1.1 -> 1.0 A, each step gated on the
#     training success share (>= P16_AMPS_GATE, p15 at 1.75 A trains at ~0.93) after a minimum, forced at a deadline;
#     the skate_heat weight (MICRODUCK_SKATE_HEAT=1 term) is switched on by the same curriculum once 1.0 A has run
#     P16_HEAT["delay"] iterations and the gate holds (forced at its deadline): half weight for 50 iterations, then full.
#   - costs (sq p16 block, NEGATIVE weights, half weight until +P16_FULL_AT): p16_knee_side (every stage after a 0.4 s
#     spawn grace; r9_knee's side test, linear to 1.2 rad), p16_stance_time (per pre-command lift-off, reach-limited 0.10-0.15 s target), p16_foot_lift (swing
#     sole > 2 cm above rest, squared), p16_neck (neck pitch < -0.35 rad while stroking). push_swing_foot's minimum
#     clearance 12 -> 6 mm (it would otherwise leave a 12-20 mm band under the 2 cm lift cap); p16_leg_limits (every
#     stage, coordinator /: leg joints within 3 deg of their MJCF stops, which the real runtime
#     does not enforce; hip yaws sit within 2 deg of a stop 9-15 % of the chain frames).
ENABLE_P16 = ENABLE_P15 and os.environ.get("MICRODUCK_SEQPUSH_P16", "0") == "1"
P16_IT0 = int(os.environ.get("MICRODUCK_SEQPUSH_P16_IT0", "4656"))
ENABLE_P16_AMPS = ENABLE_P16 and os.environ.get("MICRODUCK_SEQPUSH_P16_AMPS", "0") == "1"
# (amps, minimum iterations after P16_IT0, deadline iterations after P16_IT0)
P16_AMPS_STAGES = ((1.4, 0, 0), (1.2, 100, 250), (1.1, 200, 400), (1.0, 300, 550))
P16_AMPS_GATE = 0.70                                  # success share of START-spawn episodes (p15 @ 1.4 A: ~0.8)
# resume of p16 itself (default 0 / off): minimum current stage index (3 = 1.0 A) and heat at full weight from the start
P16_AMPS_K0 = int(os.environ.get("MICRODUCK_SEQPUSH_P16_AMPS_K0", "0"))
P16_HEAT_ON = os.environ.get("MICRODUCK_SEQPUSH_P16_HEAT_ON", "0") == "1"
P16_HEAT = dict(delay=100, deadline=750, half=50)    # iterations: after reaching 1.0 A / after P16_IT0 / at half weight
# review p16: knee cost linear per 0.1 rad up to 1.2 rad (cap 11) -> -0.27 keeps the worst rate at ~3 /s
P16_W = dict(knee=-0.27, stance=-0.4, lift=-0.75, neck=-0.25, limits=-0.5)
P16_TERMS = dict(knee="p16_knee_side", stance="p16_stance_time", lift="p16_foot_lift", neck="p16_neck",
                 limits="p16_leg_limits")
P16_FULL_AT = 150                                     # iterations after P16_IT0: half weight before, full after
P16_SWING_CLEAR = 0.006
# ── p17 one stroke from the RIDING stance (2026-11-21; DEFAULT
# OFF: MICRODUCK_SEQPUSH_P17=1, needs P16=1). Backup route for the half-pipe foot-push: on a forward flat crossing the
# chain would hand the push policy the ramp policy's riding state for ONE stroke + step back on (~1.5 s on the 1.0 m
# flat at 0.5-0.8 m/s), then hand back. p16 completes 32/32 from a rolling PUSH stance but only ~19 % from a both-feet
# glide (p15). p17 re-places P17_GLIDE_FRAC of the `glide` spawns (sq.reset_p17, a reset event after
# reset_sequence) into REAL forward-crossing riding states of the ramp specialist r9 model_4500 (
# bank P17_BANK; 0.4-0.9 m/s, both soles on the deck) on the pipe's flat bottom (the chain's geometry: launch with
# MICRODUCK_PIPE_H=0.08 = the bank's pipe, asserted) or, P17_RESET floor_share of them, transplanted onto the approach
# floor. One-stroke command: v_tgt - v_gap = spawn speed + U(dv_req), so the step command latches in the first push-foot
# floor contact that raised the speed by dv_req. Success = sq.p17_success (the usual SETTLE + both soles + wheels with
# T_settle U(0.2, 0.3) s for p17, AND a legal stroke: lift-off speed >= touchdown speed + 0.02 m/s with the board foot on
# the deck, AND the env's OWN recorded riding stance held 0.2 s - not HANDOFF_STANCE: the ramp policy rides ~10 deg more
# upright than the manual, sq.P17_STANCE_TOL; non-p17 envs keep the p9b rule); failure = sq.p17_fail (not successful
# 1.6 s after the spawn, or the board 2 cm up the far wall), priced by the existing seq_fall_push (-10, the attempt ramp
# under the command). On p17 envs heading_line and the HANDOFF settle_stance_path are masked and p17_stance_path (same
# potential, own stance) pays instead. Every p16 realism cost stays on. Spawn mix from P17_IT0 (P17_SPAWN_PROBS): start
# 0.40 -> 0.30, glide 0.05 -> 0.35 (x 0.87 = 0.30 p17 + 0.045 plain glide). Resume of p16 model_5644 (common_step_counter
# 135744 = it 5656 -> P17_IT0 5656; also P16_AMPS_K0=3 and P16_HEAT_ON=1 so the current stays at 1.0 A with heat).
ENABLE_P17 = ENABLE_P16 and os.environ.get("MICRODUCK_SEQPUSH_P17", "0") == "1"
P17_IT0 = int(os.environ.get("MICRODUCK_SEQPUSH_P17_IT0", "5656"))
P17_BANK = os.environ.get("MICRODUCK_SEQPUSH_P17_BANK", sq.P17_BANK)
P17_SPAWN_PROBS = dict(start=0.30, push=0.10, step_on=0.12, step_mid=0.05, stroke_air=0.08, glide=0.35)
P17_GLIDE_FRAC = 0.87
# floor placements 0.4-0.9 m/s; pipe placements 0.4-0.75 m/s (above that the board reaches the far wall in ~1.1 s; the
# r9 forward crossings themselves run 0.32 / 0.44 / 0.59 m/s p10 / p50 / p90)
P17_RESET = dict(floor_share=0.3, v_range=(0.4, 0.9), v_range_pipe=(0.4, 0.75), dv_req=(0.04, 0.10),
                 t_settle=(0.2, 0.3))
P17_FAIL = dict(deadline_s=1.6, wall_rise=0.02)
# p17rev (first 512-env mini-run: ~0 successes at 1.6 s from the start; p16 first touches the floor ~1.0-1.3 s after a
# riding spawn): the deadline starts long and steps down 3.0 -> 2.4 -> 2.0 -> 1.6 s, each step once >= 50 % of >= 200
# ended p17 episodes since the last step succeeded (sq.p17_deadline_curriculum; the wall gate still applies throughout)
P17_DEADLINE = dict(stages=(3.0, 2.4, 2.0, 1.6), gate=0.5, min_n=200.0)
# p17rev (review: refusing must not be cheaper than attempting): sq.p17_glide_tax (-2 /s while no push-foot touchdown
# yet, from 0.5 s) and the fall price through sq.p17_fall_push (a fall after a touchdown, outside the command, costs
# 0.5; the deadline failure keeps 1.0)
P17_GLIDE_TAX_W = -2.0
P17_ATTEMPT_SCALE = 0.5
# ── p17b stance bridge (2026-11-21, DEFAULT OFF: MICRODUCK_SEQPUSH_P17B=1, needs P17; sq.P17B_BANK block comment).
# The p17 spawns come from blend(alpha) states between p16's own pre-stroke glide states and the ramp riding states
# return-stance target blended with the same alpha; alpha climbs 0 -> 1 in rungs of 0.1 when
# >= P17B_ALPHA gate of >= min_n ended p17 episodes at the current rung succeeded; P17B_LOW_SHARE of the bridge spawns
# stay on lower rungs. Every p17 term (success rule, deadline curriculum, glide tax, fall discount) and every p16
# realism cost unchanged.
ENABLE_P17B = ENABLE_P17 and os.environ.get("MICRODUCK_SEQPUSH_P17B", "0") == "1"
P17B_BANK = os.environ.get("MICRODUCK_SEQPUSH_P17B_BANK", sq.P17B_BANK)
P17B_LOW_SHARE = 0.25
# gate_place: which riding spawns the alpha gate counts ("all" | "floor" | "pipe"; env MICRODUCK_SEQPUSH_P17B_GATE).
# p17b_mini (gate on all, 2026-11-21): rung 0 plateaued at ~0.28 = floor ~0.85 x 0.3 + pipe ~0.05 x 0.7 - the pipe-flat
# placement (seq obs x / y / heading along x, the wall ahead) is new to p16, not the stance; "floor" isolates the
# stance bridge.
# min_n 60 (was 150): a floor-only gate at 512 envs sees a decayed count of only ~95 (0.3 x ~310).
P17B_ALPHA = dict(gate=0.6, min_n=60.0, decay=0.995,
                  gate_place=os.environ.get("MICRODUCK_SEQPUSH_P17B_GATE", "all"),
                  start_k=int(os.environ.get("MICRODUCK_SEQPUSH_P17B_K0", "0")))   # resume: the rung reached
# ── p17c floor-equivalent placement obs (2026-11-21, DEFAULT OFF: MICRODUCK_SEQPUSH_P17C=1, needs P17B; sq block
# comment at sq.P17C_XF0). Pipe placements see seq slots -9..-3 rewritten to a matching floor placement (actor only;
# the critic keeps the true pipe obs), exactly what chain playback will feed (sq.p17_floor_obs). P17C_FLOOR_SHARE =
# share of riding spawns on the floor (p17 / p17b: 0.3 -> 0.5 pipe 50 %); P17C_V_PIPE = the pipe placements' speed
# range (env MICRODUCK_SEQPUSH_P17C_VPIPE "lo,hi"; p17 default 0.4-0.75).
ENABLE_P17C = ENABLE_P17B and os.environ.get("MICRODUCK_SEQPUSH_P17C", "0") == "1"
P17C_FLOOR_SHARE = float(os.environ.get("MICRODUCK_SEQPUSH_P17C_FLOOR_SHARE", "0.5"))
P17C_V_PIPE = tuple(float(x) for x in os.environ.get("MICRODUCK_SEQPUSH_P17C_VPIPE", "0.4,0.75").split(","))
if ENABLE_P17C:
    assert len(P17C_V_PIPE) == 2 and P17C_V_PIPE[0] <= P17C_V_PIPE[1], f"MICRODUCK_SEQPUSH_P17C_VPIPE lo,hi: {P17C_V_PIPE}"
    assert 0.0 <= P17C_FLOOR_SHARE <= 1.0, P17C_FLOOR_SHARE
if ENABLE_P17B:
    assert 0 <= P17B_ALPHA["start_k"] <= 10, f"MICRODUCK_SEQPUSH_P17B_K0 must be a bridge rung 0..10: {P17B_ALPHA['start_k']}"
# -- p17d chain hand-over states (2026-11-21, DEFAULT OFF: MICRODUCK_SEQPUSH_P17D=1, needs P17C; sq block comment at
# sq.P17D_BANK). The bridge bank becomes P17D_BANK (--ride bank_chain.npz bank_ride.npz --exact-top: alpha
# 1 = the chain's real STROKE hand-over states + the r9 ride rows, src 0 / 1); P17D_TRUE share of the pipe placements
# spawn a src-0 (chain) row at its recorded speed; pipe speed range P17D_V_PIPE (env MICRODUCK_SEQPUSH_P17D_VPIPE,
# default 0.25-0.75); pipe placements fail at P17D_DEADLINE_PIPE (the chain STROKE deadline; the floor keeps the
# curriculum) or once the board turned back on the far wall (sq.p17_wall_turn; < 0.63 m/s never rises 2 cm). The alpha
# curriculum continues from MICRODUCK_SEQPUSH_P17B_K0 on the new bank.
ENABLE_P17D = ENABLE_P17C and os.environ.get("MICRODUCK_SEQPUSH_P17D", "0") == "1"
P17D_BANK = os.environ.get("MICRODUCK_SEQPUSH_P17D_BANK", sq.P17D_BANK)
P17D_V_PIPE = tuple(float(x) for x in os.environ.get("MICRODUCK_SEQPUSH_P17D_VPIPE", "0.25,0.75").split(","))
P17D_TRUE = dict(share=float(os.environ.get("MICRODUCK_SEQPUSH_P17D_TRUE_SHARE", "0.7")), src=0)
P17D_DEADLINE_PIPE = 3.0
P17D_WALL_TURN = True
if ENABLE_P17D:
    assert len(P17D_V_PIPE) == 2 and P17D_V_PIPE[0] <= P17D_V_PIPE[1], f"MICRODUCK_SEQPUSH_P17D_VPIPE lo,hi: {P17D_V_PIPE}"
    assert 0.0 <= P17D_TRUE["share"] <= 1.0, P17D_TRUE
    assert os.path.isfile(P17D_BANK), f"p17d bank missing: {P17D_BANK}"
P17_BRIDGE_BANK = P17D_BANK if ENABLE_P17D else P17B_BANK    # the bridge bank reset_p17 + p17b_alpha read
# ── p17e (2026-11-21; DEFAULT OFF: MICRODUCK_SEQPUSH_P17E=1 on top of P17D). Hardware-realism
# audit of the chain STROKE (p17d model_7840, out/chain_stroke_p17d7840): the right hip roll / yaw are driven up to ~5 deg
# PAST their MJCF stops in 12.5 % of STROKE frames (the servo pushes into the soft stop at ~0.9 A; the real stops are
# unverified and the runtime enforces none) and the push knee reverses to +46 deg in 21 / 24 strokes. p17d's p16 costs
# are tiny (Episode_Reward p16_leg_limits -0.09, p16_knee_side -0.02) and p16_leg_limits' cap 4 is reached 3 deg past a stop
# = no gradient against going further. p17e (review p17e: the 3 deg band mostly taxes the floor push, which rests ON the
# stops ~50 % of frames): margin 3 -> 1 deg + cap 40, weight -0.5 = -0.5/s on a stop (as p17d), -2/s 1 deg past, -8/s
# 3 deg past, -18/s 5 deg past; knee -0.27 -> -0.45. Metric seq_p17e_past (any leg joint > 0.5 deg past its MJCF stop;
# no SETTLE exemption). Weights via MICRODUCK_SEQPUSH_P17E_LIMITS_W / _KNEE_W.
ENABLE_P17E = ENABLE_P17D and os.environ.get("MICRODUCK_SEQPUSH_P17E", "0") == "1"
P17E_W = dict(limits=float(os.environ.get("MICRODUCK_SEQPUSH_P17E_LIMITS_W", "-0.5")),
              knee=float(os.environ.get("MICRODUCK_SEQPUSH_P17E_KNEE_W", "-0.45")))
P17E_LIMITS_CAP = 40.0
P17E_LIMITS_MARGIN_DEG = 1.0
if ENABLE_P17E:
    assert ENABLE_P16, "p17e needs P16 (its cost terms)"
    P16_W = dict(P16_W, **P17E_W)


def m_p17e_past(env):
    """p17e metric: any of the 10 leg joints more than 0.5 deg past its MJCF range limit (0/1 per step)."""
    import math
    return (sq._p16_limit_excess(env, 0.0) > math.radians(0.5)).any(dim=1).float()


def _head_noise(frac: float) -> dict:
    return {s: (pr * frac, yd, xm) for s, (pr, yd, xm) in HEAD_NOISE.items()}


def _stages(*pairs) -> list[dict]:
    """(iteration, weight) pairs -> reward_weight stages in env steps."""
    return [{"step": it * IT, "weight": w} for it, w in pairs]


PUSH_STROKE_STAGES = _stages((0, -0.8), (400, -1.2))
POSTURE_STAGES = _stages((0, 0.75), (400, 1.0))
HEAD_STEADY_STAGES = _stages((0, -0.01), (400, -0.02), (800, -0.03))
ACTION_RATE_STAGES = _stages((0, -0.05), (400, -0.08), (800, -0.10))


def make_microduck_skate_seq_push_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_skate_sequence_env_cfg(play=play)
    cfg.episode_length_s = EPISODE_LENGTH_S

    # ── spawns + scheduler overrides (cap at SETTLE) ──
    p = cfg.events["reset_sequence"].params
    p["spawn_probs"] = dict(SPAWN_PROBS)
    p["spawn_stances"] = {k: list(v) for k, v in SPAWN_STANCES.items()}
    p["flat_speeds"] = dict(FLAT_SPEEDS)
    p["step_blend"] = STEP_BLEND
    p["seq_overrides"] = dict(SEQ_OVERRIDES)
    if ENABLE_STEPFIX:
        p["stance_path"] = sq.SEQ_STANCE_STEPFIX_JSON          # stance_poses_seq.json + the step_mid IK stances
        p["spawn_stances"]["step_mid"] = list(STEPFIX_STANCES)
        p["spawn_probs"] = dict(STEPFIX_SPAWN_STAGES[0][1])
        p["step_blend"] = None                                  # the joint-space blend spawned impossible states
    if ENABLE_BRIDGE:
        p["bridge_bank"] = sq.SEQ_BRIDGE_BANK
        p["bridge_rungs"] = dict(BRIDGE_RUNGS)
        if ENABLE_BRIDGE_ADAPT:
            p["bridge_rungs"] = dict(BRIDGE_RUNGS_ADAPT)
            p["bridge_adaptive"] = dict(BRIDGE_ADAPT)
        p["spawn_probs"] = dict(BRIDGE_SPAWN_STAGES[0][1])

    # ── rewards: kept as the chain task (upright 1, on_deck 0.5, head_pose -0.5, body_on_board -1, fall -10,
    # push_progress 10, push_time -1, push_speed 2, push_step_reach 20, ride 3, ride_wheels 1, tail_strike -1,
    # dof_pos_limits); changed / new / dropped below ──
    R = RewardTermCfg
    for k in ("manual", "manual_shape", "ramp_lateral", "ramp_apex"):
        cfg.rewards.pop(k)
    cfg.rewards["push_step_on"] = R(func=sq.push_step_on, weight=8.0, params={"event": True})   # p4: 4 -> 8
    # p4 (p3 it100: 0/48 step-ons from rest, every start ended at the 8 s overtime): the overtime was discounted to ~0,
    # so ignoring the step command was free while an attempt risked an immediate -10. Now: waiting costs now (cmd_wait),
    # the command times out after CMD_MAX_S, and a fall during the command costs half.
    cfg.rewards["cmd_wait"] = R(func=sq.cmd_wait, weight=-1.0)
    cfg.rewards["fall"] = R(func=sq.seq_fall_push, weight=-10.0, params={"cmd_scale": 0.5, "cmd_max_s": CMD_MAX_S})
    if ENABLE_LANDRULE:
        cfg.rewards["fall"].params["land_scale"] = LAND_SCALE
        cfg.rewards["push_progress"].params = dict(cfg.rewards["push_progress"].params or {}, cmd_gate=True)
    # review (push p0): late x3 made a late fall (-10) cheaper than continuing for a non-stepping policy; x2 here
    cfg.rewards["push_time"].params = {"phase": sq.PUSH, "late_mult": 2.0}
    cfg.rewards["push_stroke"].weight = PUSH_STROKE_STAGES[0]["weight"]
    cfg.rewards["glide_stance"] = R(func=sq.glide_stance, weight=1.0)
    cfg.rewards["posture"].weight = POSTURE_STAGES[0]["weight"]
    cfg.rewards["head_steady"].weight = HEAD_STEADY_STAGES[0]["weight"]
    cfg.rewards["action_rate_l2"].weight = ACTION_RATE_STAGES[0]["weight"]
    cfg.rewards["board_wobble"].weight = -0.2      # the chain's final stage (its curriculum is replaced below)
    if ENABLE_STEPFIX:
        cfg.rewards.pop("push_step_reach")
        cfg.rewards["push_step_path"] = R(func=sq.push_step_path, weight=STEP_PATH_W,
                                          params={"trunk_z_min": 0.112, "trunk_w": 1.0} if ENABLE_BRIDGE else {})
        cfg.metrics["seq_path_pass"] = MetricsTermCfg(func=sq.m_path_pass)
        if ENABLE_BRIDGE_ADAPT:
            cfg.metrics["seq_bridge_frontier"] = MetricsTermCfg(func=sq.m_bridge_frontier)
    if ENABLE_P9:
        p["seq_overrides"].update(push_ramp=P9_RAMP_STAGES[0][1], stroke_sweep=P9_SWEEP)
        hs = (sq.ST_PUSH,)
        cfg.rewards["push_foot_height"] = R(func=sq.push_foot_height, weight=P9_W["foot_height"])
        cfg.rewards["push_trunk_floor"] = R(func=sq.push_trunk_floor, weight=P9_W["trunk_floor"])
        cfg.rewards["push_head_level"] = R(func=sq.head_level, weight=P9_HEAD_LEVEL_STAGES[0][1],
                                           params={"free_deg": P9_HEAD["free_deg"], "stages": hs})
        cfg.rewards["push_head_rate"] = R(func=sq.head_rate, weight=P9_HEAD_RATE_STAGES[0][1],
                                          params={"free": P9_HEAD["rate_free"], "stages": hs})
        cfg.rewards["step_lurch"] = R(func=sq.step_lurch, weight=P9_W["lurch"])
        cfg.rewards["step_clearance"] = R(func=sq.step_clearance, weight=P9_W["clearance"])
        cfg.metrics["seq_cmd_start_n"] = MetricsTermCfg(func=sq.m_cmd_start_n)
        cfg.metrics["seq_cmd_start_t"] = MetricsTermCfg(func=sq.m_cmd_start_t)
    if ENABLE_P9B:
        p["seq_overrides"].update(stance_tol=P9B_TOL_STAGES[0][1], stance_hold_s=P9B_HOLD_S)
        cfg.rewards["settle_stance_path"] = R(func=sq.settle_stance_path, weight=P9B_PATH_W)
        for k, f in (("seq_settle", sq.m_settle), ("seq_stance_ok", sq.m_stance_ok), ("seq_stance_d", sq.m_stance_d),
                     ("seq_stance_gate_p", sq.m_stance_gate_p)):
            cfg.metrics[k] = MetricsTermCfg(func=f)
    if ENABLE_P10:
        p["seq_overrides"].update(p10_track=True)
        f0 = P10_FRAC_STAGES[0][1]
        cfg.rewards["push_cadence"] = R(func=sq.push_cadence, weight=f0 * P10_W["cadence"])
        cfg.rewards["push_stroke_shape"] = R(func=sq.push_stroke_shape, weight=f0 * P10_W["shape"])
        cfg.rewards["push_overspeed"] = R(func=sq.push_overspeed, weight=f0 * P10_W["overspeed"])
        cfg.rewards["push_swing_foot"] = R(func=sq.push_swing_foot, weight=f0 * P10_W["swing"])
        cfg.rewards["push_head_bob"] = R(func=sq.head_bob, weight=f0 * P10_W["head_bob"],
                                         params={"stages": (sq.ST_PUSH,)})
        cfg.rewards["settle_head_pose"] = R(func=sq.settle_head_pose, weight=f0 * P10_W["settle_head"])
        for k, f in (("seq_p10_td", sq.m_p10_td), ("seq_p10_gap", sq.m_p10_gap), ("seq_p10_lo", sq.m_p10_lo),
                     ("seq_p10_contact", sq.m_p10_contact), ("seq_p10_sweep", sq.m_p10_sweep)):
            cfg.metrics[k] = MetricsTermCfg(func=f)
    if ENABLE_HEADING:
        cfg.rewards["heading_line"] = R(func=sq.heading_line_cost, weight=HEAD_FRAC_STAGES[0][1] * HEAD_W,
                                        params=dict(HEAD_PARAMS))
        p["line_noise"] = _head_noise(HEAD_NOISE_STAGES[0][1])
        p["seq_overrides"].update(HEAD_GEOMETRY)           # headrev: the chain's start / target (see HEAD_GEOMETRY)
        cfg.metrics["seq_heading_err"] = MetricsTermCfg(func=sq.m_heading_err)
    if ENABLE_P12:
        p["seq_overrides"].update(P12_TRACK, v_t=P12_VT_STAGES[0][1])
        cfg.rewards["push_overspeed"].params = dict(cfg.rewards["push_overspeed"].params or {},
                                                    margin=P12_OVERSPEED_MARGIN)
        f0 = P12_FRAC_STAGES[0][1]
        for key, fn in (("head_steady", sq.p12_head_steady), ("trunk_steady", sq.p12_trunk_steady),
                        ("stepon_head", sq.p12_stepon_head), ("swing_sole", sq.p12_swing_sole)):
            cfg.rewards[P12_TERMS[key]] = R(func=fn, weight=f0 * P12_W[key])
        cfg.rewards["p12_cadence"] = R(func=sq.push_cadence, weight=f0 * P12_W["cadence"], params=dict(P12_CADENCE))
        for k, f in (("seq_p12_win", sq.m_p12_win), ("seq_p12_rise", sq.m_p12_rise),
                     ("seq_p12_headrate", sq.m_p12_headrate)):
            cfg.metrics[k] = MetricsTermCfg(func=f)
    if ENABLE_P13:
        p["seq_overrides"].update(P13_TRACK)
        if "push_step_path" in cfg.rewards and ENABLE_BRIDGE:
            cfg.rewards["push_step_path"].params["trunk_z_min"] = P13_PATH_TRUNK_Z_MIN
        f0 = P13_FRAC_STAGES[0][1]
        for key, fn in (("trunk_hold", sq.p13_trunk_hold), ("stepon_pose", sq.p13_stepon_pose),
                        ("swing_tilt", sq.p13_swing_tilt), ("hover", sq.p13_hover)):
            cfg.rewards[P13_TERMS[key]] = R(func=fn, weight=f0 * P13_W[key])
        for k, f in (("seq_p13_hover", sq.m_p13_hover), ("seq_p13_swing", sq.m_p13_swing), ("seq_p13_tilt", sq.m_p13_tilt),
                     ("seq_p13_win_tz", sq.m_p13_win_tz)):
            cfg.metrics[k] = MetricsTermCfg(func=f)
    if ENABLE_P14:
        f0 = P14_FRAC_STAGES[0][1]
        for key, fn in (("stepon_pop", sq.p14_stepon_pop), ("sole_roll", sq.p14_sole_roll), ("head_yaw", sq.p14_head_yaw)):
            cfg.rewards[P14_TERMS[key]] = R(func=fn, weight=f0 * P14_W[key])
        for k, f in (("seq_p14_vz", sq.m_p14_vz), ("seq_p14_win", sq.m_p14_win), ("seq_p14_roll", sq.m_p14_roll),
                     ("seq_p14_hyaw", sq.m_p14_hyaw)):
            cfg.metrics[k] = MetricsTermCfg(func=f)
    if ENABLE_P15:
        for key, fn in (("speed", sq.p15_speed_deficit), ("land_roll", sq.p15_land_roll)):
            cfg.rewards[P15_TERMS[key]] = R(func=fn, weight=P15_FRAC_STAGES[key][0][1] * P15_W[key])
        for name in P15_DROP:
            cfg.rewards[name].weight = 0.0
        cfg.rewards["p14_head_yaw"].params = dict(cfg.rewards["p14_head_yaw"].params or {}, **P15_HEAD_YAW)
        for k, f in (("seq_p15_deficit", sq.m_p15_deficit), ("seq_p15_pre", sq.m_p15_pre),
                     ("seq_p15_land_roll", sq.m_p15_land_roll)):
            cfg.metrics[k] = MetricsTermCfg(func=f)
    if ENABLE_P16:
        for key, fn in (("knee", sq.p16_knee_side), ("stance", sq.p16_stance_time), ("lift", sq.p16_foot_lift),
                        ("neck", sq.p16_neck), ("limits", sq.p16_leg_limits)):
            cfg.rewards[P16_TERMS[key]] = R(func=fn, weight=0.5 * P16_W[key])
        cfg.rewards["push_swing_foot"].params = dict(cfg.rewards["push_swing_foot"].params or {}, clear=P16_SWING_CLEAR)
        if ENABLE_P17E:   # p17e: gradient past the stop (margin 1 deg, cap 40) + the past-the-stop metric
            cfg.rewards[P16_TERMS["limits"]].params = dict(cfg.rewards[P16_TERMS["limits"]].params or {},
                                                           cap=P17E_LIMITS_CAP, margin_deg=P17E_LIMITS_MARGIN_DEG)
            cfg.metrics["seq_p17e_past"] = MetricsTermCfg(func=m_p17e_past)
        for k, f in (("seq_p16_knee", sq.m_p16_knee), ("seq_p16_contact", sq.m_p16_contact),
                     ("seq_p16_amps", sq.m_p16_amps), ("seq_p16_heat_w", sq.m_p16_heat_w),
                     ("seq_p16_gate_p", sq.m_p16_gate_p), ("seq_p16_limits", sq.m_p16_limits)):
            cfg.metrics[k] = MetricsTermCfg(func=f)
    if ENABLE_P17:
        from mjlab.managers import EventTermCfg
        cfg.events["reset_p17"] = EventTermCfg(func=sq.reset_p17, mode="reset", params=dict(
            bank_path=P17_BANK, glide_frac=P17_GLIDE_FRAC, **P17_RESET))
        if ENABLE_P17B:   # p17b: the stance-bridge bank, by alpha rung
            cfg.events["reset_p17"].params["bridge"] = dict(bank=P17_BRIDGE_BANK, low_share=P17B_LOW_SHARE)
            cfg.metrics["seq_p17b_alpha"] = MetricsTermCfg(func=sq.m_p17b_alpha)
            cfg.metrics["seq_p17b_gate_p"] = MetricsTermCfg(func=sq.m_p17b_gate_p)
            for k_ in range(11):
                cfg.metrics[f"seq_p17b_s{k_:02d}"] = MetricsTermCfg(func=sq.m_p17b_rung_success, params=dict(k=k_))
                cfg.metrics[f"seq_p17b_n{k_:02d}"] = MetricsTermCfg(func=sq.m_p17b_rung_n, params=dict(k=k_))
        if ENABLE_P17C:   # p17c: floor-equivalent placement obs on pipe placements (actor), more pipe spawns
            cfg.events["reset_p17"].params.update(floor_obs=True, floor_share=P17C_FLOOR_SHARE,
                                                  v_range_pipe=P17C_V_PIPE)
            cfg.observations["actor"].terms["seq"].func = sq.seq_obs_p17c
            for k_ in range(11):
                cfg.metrics[f"seq_p17b_p{k_:02d}"] = MetricsTermCfg(func=sq.m_p17b_rung_pipe, params=dict(k=k_))
        if ENABLE_P17D:   # p17d: chain hand-over states at their true speeds, 0.25-0.75 m/s on the pipe
            cfg.events["reset_p17"].params.update(v_range_pipe=P17D_V_PIPE, true_speed=dict(P17D_TRUE))
        if "heading_line" in cfg.rewards:
            cfg.rewards["heading_line"].func = sq.p17_heading_line_cost
        cfg.rewards["p17_glide_tax"] = R(func=sq.p17_glide_tax, weight=P17_GLIDE_TAX_W)
        cfg.rewards["fall"].func = sq.p17_fall_push        # same params as seq_fall_push + attempt_scale
        cfg.rewards["fall"].params = dict(cfg.rewards["fall"].params or {}, attempt_scale=P17_ATTEMPT_SCALE,
                                            deadline_s=P17_FAIL["deadline_s"], wall_rise=P17_FAIL["wall_rise"])
        if ENABLE_P17D:   # p17d: the pipe placements' deadline (= p17_fail's)
            cfg.rewards["fall"].params["deadline_pipe"] = P17D_DEADLINE_PIPE
        if "settle_stance_path" in cfg.rewards:   # p17 envs: the potential toward their OWN riding stance instead
            cfg.rewards["settle_stance_path"].func = sq.p17_settle_stance_path
            cfg.rewards["p17_stance_path"] = R(func=sq.p17_stance_path, weight=P9B_PATH_W)
        for k, f in (("seq_p17_success", sq.m_p17_success), ("seq_p17_success_pipe", sq.m_p17_success_pipe),
                     ("seq_p17_share", sq.m_p17_share), ("seq_p17_ok", sq.m_p17_ok), ("seq_p17_brake", sq.m_p17_brake),
                     ("seq_p17_deadline", sq.m_p17_deadline)):
            cfg.metrics[k] = MetricsTermCfg(func=f)

    # ── curriculum: wholesale replacement, stages in iterations of this run ──
    def rw(name, stages):
        return CurriculumTermCfg(func=microduck_mdp.reward_weight, params={"reward_name": name, "weight_stages": stages})

    cfg.curriculum = {
        "push_stroke": rw("push_stroke", PUSH_STROKE_STAGES),
        "posture": rw("posture", POSTURE_STAGES),
        "head_steady": rw("head_steady", HEAD_STEADY_STAGES),
        "action_rate": rw("action_rate_l2", ACTION_RATE_STAGES),
    }
    if ENABLE_STEPFIX:
        cfg.curriculum["spawns"] = CurriculumTermCfg(func=microduck_mdp.event_param_curriculum, params={
            "event_name": "reset_sequence",
            "param_stages": [{"step": it * IT, "params": {"spawn_probs": dict(sp)}}
                             for it, sp in (BRIDGE_SPAWN_STAGES if ENABLE_BRIDGE else STEPFIX_SPAWN_STAGES)]})
        if ENABLE_P17:   # p17: the riding-spawn mix from the resume point (appended: the earlier stages are unchanged)
            cfg.curriculum["spawns"].params["param_stages"].append(
                {"step": P17_IT0 * IT, "params": {"spawn_probs": dict(P17_SPAWN_PROBS)}})
    if ENABLE_P17:
        cfg.curriculum["p17_deadline"] = CurriculumTermCfg(func=sq.p17_deadline_curriculum, params=dict(P17_DEADLINE))
    if ENABLE_P17B:   # logged as Curriculum/p17b_alpha
        cfg.curriculum["p17b_alpha"] = CurriculumTermCfg(func=sq.p17b_alpha_curriculum,
                                                     params=dict(P17B_ALPHA, bank=P17_BRIDGE_BANK))
    if ENABLE_P9:
        # p9 stages count from the resume point P9_IT0; stage 0 sits at step 0 so a --curriculum-step 0 eval and the
        # resumed run's first iterations see the same p9 values
        p9 = lambda pairs: [{"step": 0 if k == 0 else (P9_IT0 + it) * IT, "weight": w} for k, (it, w) in enumerate(pairs)]
        cfg.curriculum["push_head_level"] = rw("push_head_level", p9(P9_HEAD_LEVEL_STAGES))
        cfg.curriculum["push_head_rate"] = rw("push_head_rate", p9(P9_HEAD_RATE_STAGES))
        cfg.curriculum["push_ramp"] = CurriculumTermCfg(func=sq.seq_cfg_curriculum, params={"stages": [
            {"step": 0 if k == 0 else (P9_IT0 + it) * IT, "cfg": {"push_ramp": r}}
            for k, (it, r) in enumerate(P9_RAMP_STAGES)]})
    if ENABLE_P9B:
        cfg.curriculum["stance_tol"] = CurriculumTermCfg(func=sq.stance_tol_curriculum, params={"stages": [
            {"step": 0 if k == 0 else (P9_IT0 + it) * IT, "cfg": {"stance_tol": tol}}
            for k, (it, tol) in enumerate(P9B_TOL_STAGES)], "gate": P9B_TOL_GATE})
    if ENABLE_P10:
        # stages count from the resume point P10_IT0; stage 0 at step 0 (a --curriculum-step 0 eval = the first stage)
        for key, name in P10_TERMS.items():
            cfg.curriculum[name] = rw(name, [{"step": 0 if k == 0 else (P10_IT0 + it) * IT, "weight": fr * P10_W[key]}
                                             for k, (it, fr) in enumerate(P10_FRAC_STAGES)])
        if ENABLE_P12:   # p12: push_stroke_shape -0.5 -> -1.0 at P12_IT0 + 100 (appended after the p10 stages)
            cfg.curriculum["push_stroke_shape"].params["weight_stages"].append(
                {"step": (P12_IT0 + 100) * IT, "weight": P12_SHAPE_W})
    if ENABLE_P12:
        # stages count from the resume point P12_IT0; stage 0 at step 0 (a --curriculum-step 0 eval = the first stage)
        s12 = lambda it, k: 0 if k == 0 else (P12_IT0 + it) * IT
        for key, name in P12_TERMS.items():
            cfg.curriculum[name] = rw(name, [{"step": s12(it, k), "weight": fr * P12_W[key]}
                                             for k, (it, fr) in enumerate(P12_FRAC_STAGES)])
        cfg.curriculum["p12_vt"] = CurriculumTermCfg(func=sq.seq_cfg_curriculum, params={"stages": [
            {"step": s12(it, k), "cfg": {"v_t": vt}} for k, (it, vt) in enumerate(P12_VT_STAGES)]})
    if ENABLE_P13:
        # stages count from the resume point P13_IT0; stage 0 at step 0 (a --curriculum-step 0 eval = the first stage)
        s13 = lambda it, k: 0 if k == 0 else (P13_IT0 + it) * IT
        for key, name in P13_TERMS.items():
            cfg.curriculum[name] = rw(name, [{"step": s13(it, k), "weight": fr * P13_W[key]}
                                             for k, (it, fr) in enumerate(P13_FRAC_STAGES)])
    if ENABLE_P14:
        # stages count from the resume point P14_IT0; stage 0 at step 0 (a --curriculum-step 0 eval = the first stage)
        s14 = lambda it, k: 0 if k == 0 else (P14_IT0 + it) * IT
        for key, name in P14_TERMS.items():
            cfg.curriculum[name] = rw(name, [{"step": s14(it, k), "weight": fr * P14_W[key]}
                                             for k, (it, fr) in enumerate(P14_FRAC_STAGES)])
    if ENABLE_P15:
        # stages count from the resume point P15_IT0; stage 0 at step 0 (a --curriculum-step 0 eval = the first stage)
        s15 = lambda it, k: 0 if k == 0 else (P15_IT0 + it) * IT
        for key, name in P15_TERMS.items():
            cfg.curriculum[name] = rw(name, [{"step": s15(it, k), "weight": fr * P15_W[key]}
                                             for k, (it, fr) in enumerate(P15_FRAC_STAGES[key])])
        for name in P15_DROP:          # replaces the p13 / p14 schedules: 0 from step 0 (the resume starts past them)
            cfg.curriculum[name] = rw(name, [{"step": 0, "weight": 0.0}])
    if ENABLE_P16:
        # stages count from the resume point P16_IT0; stage 0 at step 0 (a --curriculum-step 0 eval = the first stage)
        for key, name in P16_TERMS.items():
            cfg.curriculum[name] = rw(name, [{"step": 0, "weight": 0.5 * P16_W[key]},
                                             {"step": (P16_IT0 + P16_FULL_AT) * IT, "weight": P16_W[key]}])
    if ENABLE_P16_AMPS:
        amp_st = [{"amps": a, "step": (P16_IT0 + lo) * IT if k else 0, "deadline": (P16_IT0 + dl) * IT if k else 0}
                  for k, (a, lo, dl) in enumerate(P16_AMPS_STAGES)]
        heat = None
        if "skate_heat" in cfg.rewards:   # MICRODUCK_SKATE_HEAT=1: its weight is owned by this curriculum from here on
            heat = dict(term="skate_heat", weight=cfg.rewards["skate_heat"].weight, delay_steps=P16_HEAT["delay"] * IT,
                        deadline=(P16_IT0 + P16_HEAT["deadline"]) * IT, half_steps=P16_HEAT["half"] * IT)
            cfg.rewards["skate_heat"].weight = 0.0
        cfg.curriculum["p16_amps"] = CurriculumTermCfg(func=sq.p16_current_curriculum, params={
            "stages": amp_st, "gate": P16_AMPS_GATE, "heat": heat, "k_min": P16_AMPS_K0, "heat_on_at_start": P16_HEAT_ON})
    if ENABLE_HEADING:
        hs = lambda it, k: 0 if k == 0 else (HEAD_IT0 + it) * IT
        cfg.curriculum["heading_line"] = rw("heading_line", [{"step": hs(it, k), "weight": fr * HEAD_W}
                                                             for k, (it, fr) in enumerate(HEAD_FRAC_STAGES)])
        # line_noise rides on its own params curriculum (the "spawns" one only sets spawn_probs: no conflict)
        cfg.curriculum["heading_noise"] = CurriculumTermCfg(func=microduck_mdp.event_param_curriculum, params={
            "event_name": "reset_sequence",
            "param_stages": [{"step": hs(it, k), "params": {"line_noise": _head_noise(fr)}}
                             for k, (it, fr) in enumerate(HEAD_NOISE_STAGES)]})

    # ── terminations: success (time-out), overtime + stricter board_lost (failures, charged by seq_fall) ──
    T = TerminationTermCfg
    cfg.terminations["success"] = T(func=sq.seq_success, time_out=True, params=dict(SUCCESS))
    limits = OVERTIME_LIMITS
    if ENABLE_P9B:
        cfg.terminations["success"].params["stance"] = True
        cfg.rewards["fall"].params["settle_overtime_scale"] = P9B_SETTLE_OVERTIME_SCALE
        # p9brev: a SETTLE fall / board_lost never undercuts the discounted overtime (else stepping off ~0.3 s before
        # the overtime saved ~2 discounted on every missed stance)
        cfg.rewards["fall"].params["settle_fall_match"] = (P9B_SETTLE_OVERTIME_S, 0.99)
        limits = tuple((s, P9B_SETTLE_OVERTIME_S if s == sq.ST_SETTLE else lim) for s, lim in OVERTIME_LIMITS)
    cfg.terminations["overtime"] = T(func=sq.seq_overtime, params={"limits": limits, "cmd_max_s": CMD_MAX_S})
    cfg.terminations["board_lost"] = T(func=sq.seq_board_lost, params={"ride_off_s": RIDE_OFF_S})
    if ENABLE_LANDRULE:
        cfg.terminations["cmd_landing"] = T(func=sq.seq_cmd_landing, params={"reach_s": LAND_REACH_S})
        if ENABLE_P12:   # p12rev: step_on spawns failed cmd_landing at 0.04 s on the sole settling (see seq_cmd_landing)
            cfg.terminations["cmd_landing"].params["spawn_grace_s"] = P12_LAND_GRACE_S
    if ENABLE_P17:   # p17: success also needs the legal stroke on p17 envs; the p17 deadline / wall failure
        cfg.terminations["success"].func = sq.p17_success
        cfg.terminations["p17_fail"] = T(func=sq.p17_fail, params=dict(
            P17_FAIL, success=dict(cfg.terminations["success"].params)))
        if ENABLE_P17D:   # p17d: pipe deadline = the chain STROKE deadline; low-speed wall = turned back on the far wall
            cfg.terminations["p17_fail"].params.update(deadline_pipe=P17D_DEADLINE_PIPE, wall_turn=P17D_WALL_TURN)
    return cfg


def make_microduck_skate_seq_push_ppo_runner_cfg():
    rl = make_microduck_skate_sequence_ppo_runner_cfg()
    rl.experiment_name = "skate_seq_push"
    rl.max_iterations = 1200
    rl.save_interval = 100
    rl.algorithm.entropy_coef = 0.002          # review (push p0): weaker std inflation on unused joints
    return rl
