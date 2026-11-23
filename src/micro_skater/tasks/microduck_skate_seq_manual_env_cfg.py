# micro-skater: part of the author's skateboard add-on for microduck_rl. Apache-2.0 (see LICENSE).
"""Skate SEQUENCE manual SPECIALIST (2026-11-20). Task Mjlab-Skate-SeqManual-MicroDuck.

One of three specialists (push + step-on /
manual / approach + ramp) with the SAME 114D obs and actions as Mjlab-Skate-Sequence-MicroDuck (action centre fwd
`tail`, scale 1.0: the chaining contract), chained by scheduler stage at playback. Built on
make_microduck_skate_sequence_env_cfg; the scheduler runs SETTLE -> MANUAL -> EXIT and is capped at EXIT (seq_overrides
last_stage). Play the chain with manual_end="hold" too.

  - MANUAL ends on a HOLD (seq_cfg manual_end="hold"): leaky band_t (+dt in 8-15 deg, front up / back down / tail
    clear / soles on, else -leak*dt) reaching T_hold ~ manual_hold_s, inside seq_update.
  - success (time_out=True: the value bootstraps to the riding value): EXIT, >= 0.5 s in it (exit_min_s 0.5) and all
    four wheels down >= 0.5 s. Failures: the base set + seq_overtime(MANUAL 4 s, EXIT 2 s), charged by seq_fall (-10,
    x0.6 in MANUAL).
  - clock_offset U(3, 7) s on the trick-obs clock: in the chain the manual starts at t ~ 3-7 s.
  - start_gap 5 -> 14 m: every spawn is on the floor before the pipe; at 1.2 m/s an 8 s episode coasts ~7 m, and with the
    chain's 5 m the manual / glide spawns (|x| = 0.45-0.75 m) would ride into the pipe wall's end face.

Warm start from (r2b in the 114D layout, std ~0.185) with MICRODUCK_WARM_START=1 (iteration
counter restarts at 0); entropy 0.005 -> 0.002 keeps the std low. review (mrev): model_0_ax015 = model_0 with the
obs-normalizer mean of the board block's trunk / sole deck-frame x shifted +15 mm.
r2b's soles always sat at deck x -0.055 (normalizer std ~5 mm); this board's stances put them at
-0.040, which r2b reads as "feet far forward": it throws its weight back, the nose hits the 27 deg tail strike within
0.16 s, the tail drags the board to a stop and a foot goes down (~1 s). Plain model_0 must not be used here. Curriculum stages in iterations of THIS run (x 24 env steps).

m2 (2026-11-20, video polish, RESUMES m1 model_700 - no MICRODUCK_WARM_START): pitch shape 11-13 deg + manual_over
above 14 deg, dead-band pitch-rate cost on the pop and the set-down, setdown_rate x2, level-referenced head_level +
head_steady x4, stance_width, manual_speed_loss, T_hold 1.2-1.8 s, settle 0.8-1.2 s, hold band 9-14 deg at it 1000 (m2rev; was 900). m2rev also gates pitch_rate in SETTLE (pre-pop loophole).

m3 (2026-11-20, video polish 3, RESUMES m2 model_1000 - no MICRODUCK_WARM_START; stored common_step_counter 24048 =
1002 x 24, so M3_IT stages are live from the first resumed step): head_rate (all phases) + head_rate_pop (pop window),
head_level dead band 10 -> 5 deg, pop_rate (pop-only pitch-rate cost, dead band 0.6 rad/s, gated through the whole first
rise to 11 deg and SETTLE), T_hold 1.2-1.8 -> 1.5-2.5 s, MANUAL overtime 4 -> 5 s. Stance stagger: INFEASIBLE, not rewarded.

m5 (2026-11-20, video polish 5, DEFAULT OFF: MICRODUCK_SEQMANUAL_M5=1, RESUMES m4 model_1499 with the m4 env vars):
bounded allowances against the pop / catch head throw (neck +60 deg, face to the sky) and back-lean - head off level
beyond 20 deg, head joints beyond 0.35 rad of the HANDOFF head pose, trunk pitch outside -15..+12 deg - and
head_rate_pop -2 -> -3; optional nose-up band 9-14 -> 11-16 deg (MICRODUCK_SEQMANUAL_M5_BAND=1). See the m5 block below. Pre-m5 files:
 
"""

from __future__ import annotations

import os

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import CurriculumTermCfg, MetricsTermCfg, RewardTermCfg, TerminationTermCfg

from micro_skater.tasks import mdp as microduck_mdp
from micro_skater.tasks import skate_sequence as sq
from micro_skater.tasks.microduck_skate_sequence_env_cfg import (make_microduck_skate_sequence_env_cfg,
                                                                    make_microduck_skate_sequence_ppo_runner_cfg)

EPISODE_LENGTH_S = 8.0
IT = 24                                   # env steps per iteration (NUM_STEPS_PER_ENV)
HARDEN_IT = 500                           # T_hold / leak hardening
# review (mrev): the other hardenings staggered after it (a metric stepping down at a boundary must be
# attributable; four hardenings in one iteration are not)
SPAWN_IT = 600                            # spawn mix -> glide-heavy
RATE_IT = 700                             # action_rate -0.1 -> -0.2
# review (mrev): speeds 0.3-0.7 -> 0.4-0.9 (it 200) -> 0.6-1.2 (it 400), clock offset (0, 7) -> (3, 7) at it 300. The
# iteration-0 battery of the warm start (model_0_ax015, deterministic, tail stance): 0.3-0.6 m/s 84 % success / 0 % foot
# down; 0.8-1.2 m/s 9 % / 30 %; clock (3, 7) vs none: success 48 vs 84 % (manual), 44 vs 73 % (glide)
SPEED_IT = 200
SPEED2_IT = 400
CLOCK_IT = 300

# ── m2 (2026-11-20, video polish; resumes m1 model_700, whose stored common_step_counter is 16824 = 701 x 24, so a
# stage at M2_IT is live from the first resumed step). Measured on model_700 (
# measure_700*.json, 64 envs deterministic + stochastic): pop 1 -> 10 deg in 0.06 s (peak 205-225 deg/s), set-down
# 10 -> 1 deg in 0.10 s (peak ~110 deg/s), hold at 14.2 deg median (28 % above 15, 18 % in 11-13) for ~3 s, settle
# 0.64 s, head 13-16 deg off level (p90 32-50 deg) in the hold / set-down, soles 104 mm apart in the hold (85 mm on four
# wheels), board speed -15 % over the manual (~coasting + the long hold). ──
M2_IT = 700                               # m2 rewards, T_hold 1.2-1.8 s (set directly / live on resume)
# m2rev: 900 -> 1000. model_700 under the 9-14 band (audit, 64 envs, m2 cfg): success det 86 -> 39 %, stoch
# 52 -> 16 %, nearly all MANUAL overtime (hold median 14.2 deg); per-episode return fail 21.2 vs success 25.4 (det).
# Check the det hold median on model_900 / model_1000: if it is still > 13.5 deg, delay this stage again.
BAND_IT = 1000                            # hold band 8-15 -> 9-14 deg, once the new pitch shape has moved the hold
M3_IT = 1001                              # m3: T_hold 1.5-2.5 s (live on the model_1000 resume: 1001 x 24 <= 24048)
# manual_hold_s / manual_leak / manual_band_deg are NOT here: the seq_cfg curriculum owns them (overrides apply after the
# curriculum's first call and would reset them). m2: settle_s 0.4-1.0 -> 0.8-1.2 (the m1 video settled 0.45 s).
SEQ_OVERRIDES = dict(manual_end="hold", settle_s=(0.8, 1.2), exit_min_s=(0.5, 0.5),
                     setdown_s=0.6, start_gap=14.0, last_stage=sq.ST_EXIT)
HOLD_STAGES = [(0, dict(manual_hold_s=(0.8, 1.2), manual_leak=1.0, manual_band_deg=(8.0, 15.0))),
               (HARDEN_IT, dict(manual_hold_s=(1.0, 2.0), manual_leak=2.0, manual_band_deg=(8.0, 15.0))),
               # m2: a ~1.5 s hold (m1's 1-2 s plus the leak at 15-16 deg dragged to 3-4 s on video)
               (M2_IT, dict(manual_hold_s=(1.2, 1.8), manual_leak=2.0, manual_band_deg=(8.0, 15.0))),
               (BAND_IT, dict(manual_hold_s=(1.2, 1.8), manual_leak=2.0, manual_band_deg=(9.0, 14.0))),
               # m3: reviewers want ~2 s on video (1.2 s typical read short; 3-4 s dragged); playback can override T_hold
               (M3_IT, dict(manual_hold_s=(1.5, 2.5), manual_leak=2.0, manual_band_deg=(9.0, 14.0)))]

SPAWN_PROBS_0 = dict(glide=0.40, manual=0.45, exit=0.15)
SPAWN_PROBS_1 = dict(glide=0.55, manual=0.35, exit=0.10)
SPAWN_STANCES = dict(glide=["glide_step", "glide_step", "glide_step", "glide"],
                     manual=["tail", "tail", "tail_back", "glide_step"],   # mrev: tail_back x2 -> x1 (tilted r2b: tail
                                                                           # 88 % success, tail_back 52 %, 100 % tail touch)
                     exit=["tail", "tail_back", "glide_step"])
FLAT_SPEEDS_0 = dict(push=(0.1, 0.8), glide=(0.3, 0.7), manual=(0.3, 0.7), exit=(0.3, 0.7))
FLAT_SPEEDS_1 = dict(push=(0.1, 0.8), glide=(0.4, 0.9), manual=(0.4, 0.9), exit=(0.4, 0.9))
FLAT_SPEEDS_2 = dict(push=(0.1, 0.8), glide=(0.6, 1.2), manual=(0.6, 1.2), exit=(0.6, 1.2))
SPAWN_NOISE = dict(glide=dict(joint=0.04, rpy=0.04))
MANUAL_TILT_PROB = 0.6
MANUAL_TILT_DEG = (6.0, 14.0)
EXIT_TILT_DEG = (5.0, 14.0)
MANUAL_PREFILL = (0.0, 0.9)               # tilted MANUAL spawns: band_t = U(0, 0.9) * T_hold
CLOCK_OFFSET = (3.0, 7.0)                 # the chain's clock at the manual (final; the event's own param value)
CLOCK_OFFSET_0 = (0.0, 7.0)               # mrev: until CLOCK_IT, half the draws near r2b's own clock
# handoff2 (2026-11-20, default OFF): MICRODUCK_SEQMANUAL_HANDOFF=<bank.npz>
# makes HANDOFF_PROB of the glide (SETTLE) spawns restore a recorded push-policy hand-off
# state (skate_sequence.reset_sequence_state handoff_path / handoff_prob). Read when the cfg is built.
# m5rev (2026-11-21): now defaults to --gate stance = chain_play --stance-gate's moment (stance held
# 0.2 s / 1.5 s in SETTLE / MANUAL entry, settle_s 1.2-1.5), not SETTLE entry; bank stage_t ~0.5-0.7 s. m4 model_1499
# from such spawns (p13 model_3600 bank, 32 det): neck to 62 deg, head 121 deg off level in MANUAL - the chain's throw.
HANDOFF_ENV = "MICRODUCK_SEQMANUAL_HANDOFF"
HANDOFF_PROB = 0.5
# handoffrev (2026-11-20, only with the bank set): a SETTLE fall costs 10 x 2 (default x1). m2 model_1000 under the m3
# cfg from hand-off spawns (192 eps, det / stoch 0.18): 66-73 % fall in SETTLE at ~0.5 s,
# return -11..-13; 27-34 % reach MANUAL and fall at ~1.5 s, return -26 (head_rate_pop -11.5 + head_rate -7: the fall's
# head whip inside the pop window), 0 % success. Falling early was 2x cheaper than surviving into MANUAL (a "bail out
# in SETTLE" gradient while MANUAL survival is ~0); x2 makes the two ~equal (-21 vs -26), riding on (MANUAL overtime
# ~-2) stays far better. Plain-glide SETTLE falls are ~3 % of glides.
HANDOFF_SETTLE_FALL_SCALE = 2.0

# + stage_t >= T_exit = 0.5 s. review (mrev): on_s 0 -> 0.2 (both_on >= 0 was always true: a set-down with a sole
# hovering off the deck counted as the demo's clean exit)
SUCCESS = dict(stage=sq.ST_EXIT, on_s=0.2, wheels_s=0.5)
# m3: MANUAL 4 -> 5 s for the 1.5-2.5 s hold + a ~0.3 s pop (with leak 2, one 0.3 s out-of-band excursion costs ~0.9 s);
# refusing still nets 0.7 /s x 5 s - 6 = -2.5 (spec audit)
OVERTIME_LIMITS = ((sq.ST_MANUAL, 5.0), (sq.ST_EXIT, 2.0))
# review (mrev): a fall / timeout costs 10 x 0.6 in MANUAL AND in EXIT (seq_fall scaled only the MANUAL phase: a set-down
# fall or EXIT timeout cost 10, so completing the hold moved the risk into the dearer regime)
FALL_STAGE_SCALE = ((sq.ST_MANUAL, 0.6), (sq.ST_EXIT, 0.6))
# review (mrev): one-shot pay for completing the hold (MANUAL in band pays ~7.6 /s against ~4 /s in EXIT, so without it
# delaying the hold - hovering out of band, see-sawing a band edge so band_t leaks - out-earns completing it until the
# MANUAL timeout; ~3.6 /s x the ~2 s value horizon = ~7)
HOLD_DONE_W = 8.0

# m2 pitch shape (manual_quality): plateau 11-13 deg, 0 at 5 and 17; rate_std unchanged (rate is priced by pitch_rate)
M2_SHAPE = {"pts_deg": (5.0, 11.0, 13.0, 17.0), "rate_std": 2.0}
# m2 cost weights (per second). Raw values on model_700 (det / stochastic, per phase):
#  - pitch_rate (dead band 0.8 rad/s, MANUAL below 10 deg + set-down): snap pop ~0.34 raw, set-down ~0.09 raw per
#    episode -> -10 = -3.4 / -0.9 per manual; a <= 46 deg/s pop / set-down is free; hold excluded (noise there 0.06-0.15
#    raw/s = -1 /s at -10)
#  - manual_over (14 -> 18 deg): m1 hold 0.13-0.15 raw/s -> -3 = -0.4 /s; 16 deg costs -1.5 /s, saturates -3 /s at 18
#  - head_level (L1 beyond 10 deg, rad): 0.19-0.23 raw/s in MANUAL, 0.13 in SETTLE -> -2.5 = -0.5 / -0.3 /s; m1's
#    60 deg stare-down -2.2 /s; noise adds ~2 deg (~0.03 raw/s)
#  - head_steady ((rad/s)^2): raw means SETTLE 6.5, POP 18-21, HOLD 3.2-9.7 (det-stoch), EXIT 1-3.6 -> -0.01 -> -0.04
#    = -0.15..-0.4 /s in the hold (noise share ~+3 raw = -0.12 /s)
#  - stance_width (cm beyond 9 cm): HOLD 1.4, SETTLE 0.6-0.7, EXIT 0 -> -0.3 = -0.42 /s in the hold, cap -0.9 /s
#  - speed_loss (relative deficit vs entry - 0.04 m/s^2 coast): 0.02-0.08 raw/s (m1's loss is mostly coasting over a
#    3 s hold; the shorter hold is the main fix) -> -3 = -0.1..-0.25 /s, p90 episode -3 (tail-drag / braking holds)
M2_W = dict(pitch_rate=-10.0, manual_over=-3.0, head_level=-2.5, head_steady=-0.04, stance_width=-0.3, speed_loss=-3.0)

# m3 cost weights (per second). Raw values on m2 model_1000 (m1000.json, 64 envs,
# glide det / stoch; manual-spawn det / stoch in brackets). Head source: the NECK (neck_pitch); the trunk carries < 5 %
# of the head pitch-rate mean square (pop 0.26 vs 9.8-12.9 (rad/s)^2, hold 0.06-0.13 vs 1.7-2.8).
#  - head_rate (|w_xy| beyond 1.0 rad/s, squared): SETTLE 1.0-1.1 (mostly the spawn foot slide), POP 3.7-5.2 (over
#    0.135 s), HOLD 0.68-1.04 [2.8-3.1], SET-DOWN 0.2-0.7, four-wheel riding 0.02-0.06 (noise floor) -> -1.0 = -0.7..-1.0
#    /s in the hold now (head_steady prices it at -0.08..-0.14); a nod of std <= 3 deg at 1.2 Hz is ~free
#  - head_rate_pop (same raw, MANUAL until 0.3 s after the nose first reaches 11 deg, at most 1.2 s): the whip (+25-30 deg nose-down,
#    peak 260 deg/s, ~0.6-0.9 raw per pop) -> -2.0 = ~-1.5 per pop on top of head_rate's ~-0.7
#  - head_level dead band 10 -> 5 deg (rest posture 2-6 deg: escapable): HOLD 0.11-0.15 [0.26-0.30] raw/s, SETTLE
#    0.08-0.10, riding 0.02-0.05 -> -3.0 = -0.3..-0.46 /s in the hold (m2: -0.15..-0.23)
#  - pop_rate (deck pitch rate beyond 0.6 rad/s, squared, cap 2.25, MANUAL first rise to 11 deg + < 10 deg + SETTLE):
#    per pop 0.089 / 0.104 [0.06-0.13] -> -30 = -2.7..-3.1 per pop (+ pitch_rate's -0.6). A 1 -> 10 deg rise in 0.3 s
#    (mean 0.52 rad/s, peak ~0.8) costs ~0.004 x 30 = -0.12. Sizing: with gamma 0.99 a 0.2 s slower pop discounts the
#    post-pop value (~15-18) by ~10 % = ~-1.6, so the current snap must cost clearly more than that. Cap 2.25 (2.1 rad/s,
#    above the 1.9 rad/s current peak) bounds a snap (3 steps at cap = -4) below a fall's -6; pitch_rate still prices
#    beyond. SETTLE gated (the m2rev pre-pop loophole); its four-wheel |w| p90 0.03-0.31 rad/s is free.
# Checked under this cfg on model_1000, per episode, glide det / stoch [manual stoch]: head_rate
# -2.8 / -4.2 [-6.1], head_rate_pop -2.8 / -2.5 [-5.9], pop_rate -2.9 / -2.8 [-2.3], head_level -1.0 / -1.4 [-1.5];
# episode return stoch 23.6 [10.1] + the success bootstrap, vs refusing (MANUAL on four wheels ~1.6 /s x 5 s - 6 = +2,
# no bootstrap) and dying (-6): attempting still wins at the current (noisy-head) skill; a calm hold is worth ~8.6 /s.
# Settle jolt (first 0.5 s: board speed range 0.24 m/s, deck 54 deg/s, legs 1.2 rad/s vs 0.02 / 1 / 0.1 later): spawn-
# made - glide stance + no noise 0.12 m/s / 13 deg/s; glide_step + no noise 0.20 / 35 (the trained 28 mm foot slide
# out to the riding width); the rest is the 0.04 spawn noise. Not trained against; record the demo from a quiet spawn.
M3_W = dict(head_rate=-1.0, head_rate_pop=-2.0, head_level=-3.0, pop_rate=-30.0)

# heading (2026-11-20; DEFAULT OFF: MICRODUCK_SEQMANUAL_HEADING=1).
# m2 model_1000 from glide (32 det) rides straight: heading change median -3 deg (p10/p90 -8/+7) over SETTLE + MANUAL
# + EXIT, |x| drift < 0.5 m; the chain's sideways manual only carried the push's +70 deg. The term teaches CORRECTING
# moderate errors (line_noise: +-20 deg, +-0.5 m) mostly on four wheels: full weight in SETTLE / EXIT (EXIT's kappa
# obs steers on the same bearing error), a third in MANUAL (lean steering on two wheels is hard: only "don't turn
# further"). Stages count from MHEAD_IT0 (the resume checkpoint's iteration; m2 model_1000 -> 1000).
ENABLE_HEADING = os.environ.get("MICRODUCK_SEQMANUAL_HEADING", "0") == "1"
MHEAD_IT0 = int(os.environ.get("MICRODUCK_SEQMANUAL_HEAD_IT0", "1000"))
MHEAD_W = dict(ride=-0.6, manual=-0.2)
MHEAD_FRAC_STAGES = ((0, 0.5), (100, 1.0))
MHEAD_PARAMS = dict(free_deg=8.0, unit_deg=10.0, cap=3.0, v_full=0.3)
MHEAD_NOISE = {"glide": (0.5, 20.0, 0.5), "manual": (0.4, 15.0, 0.5)}   # spawn -> (prob, +-yaw deg, +-x m)
# headrev (2026-11-20, with the heading flag only): the chain's geometry. Bearing reference = the chain's EXIT pursuit
# target (ramp specialist target_x 0.45, target_in 0.6; EXIT's kappa obs here follows it too). start_gap 14 -> 12:
# in the chain (start 9 m out) SETTLE / MANUAL / EXIT run at y_open ~ -6.7 / -5.4 / -2.2; with 14 the latest MANUAL /
# EXIT states trained reached y_open ~ -3.8 / -2.8 (the y obs is unclipped only above y_open -3), with 12 the manual
# spawns' holds end at -7 .. -2.4 and the exit spawns reach -2.4. Glide spawns stay at y_open -10.8 .. -7.2 (straight
# bearing 5-8 deg: free); a ride-on overtime can still reach the pipe mouth only from the nearest manual spawns.
MHEAD_GEOMETRY = dict(start_gap=12.0, target_x=0.45, target_in=0.6)

# m5 (2026-11-20, video polish 5; DEFAULT OFF: MICRODUCK_SEQMANUAL_M5=1, RESUMES m4 model_1499 with the m4 env vars
# MICRODUCK_SEQMANUAL_HEADING=1 / HEAD_IT0=1000; m4's stored common_step_counter is 36048 = 1502 x 24, so stages at
# M5_IT0 = 1500 are live from the first resumed step). (
# m4_1499*.json / trace_*.npz). Measured on m4 model_1499 (32 envs, det, speeds 0.6-1.2):
#  - glide spawns (flat pop): calm. Neck max 19 deg (p90 27; HANDOFF 23), head off level max 7.5 (p90 15), trunk min
#    -4 (p10 -11; + = nose down), rise to 11 deg 0.37 s, head_rate_pop ~0.
#  - manual spawns, TILTED (60 %: the nose already 6-14 deg up at the spawn): the "catch" = the chain's head throw.
#    Neck +60 deg (p90 63), head 72 deg off level (p90 88: facing the sky), trunk leaning BACK -19 (p10 -31), all
#    within ~0.3 s of the spawn; every episode still reaches EXIT. Flat manual spawns: neck 25 / level 16 / trunk -7.
#  - the chain demo (chain_v1 frames 236-257): a snap pop from flat (1.5 -> 14 deg in 0.2 s, after a wind-up: trunk
#    +11 / neck -9) then the same catch: trunk -22, neck +60, head_pitch +32 -> gaze 56 deg up. Gaze ~ -neck_pitch +
#    head_pitch + trunk_pitch (fit on the traces, resid 0.45 deg): + neck_pitch tilts the face UP.
#  - counterweight? the throw appears only when the nose is up before the body is (catch), never in a controlled
#    flat pop, and it survives head_rate_pop's -5.3 per tilted episode (raw 2.66): the policy values it - a reaction
#    mass (the head is ~38 % of the mass). So: a bounded ALLOWANCE (free bands + caps), not a ban.
# Raw per episode (glide / manual-flat / manual-tilted): head off level beyond 20 deg 0.009 / 0.027 / 0.32
# rad s; head-joint L1 beyond 0.35 rad 0.10 / 0.13 / 0.19; trunk outside -15..+12 deg (units of 10 deg) 0.009 / 0.024
# / 0.23. At the M5_W weights a tilted catch pays ~-1.9 - 0.8 - 0.5 - 2.7 (head_rate_pop -2 -> -3) = ~-5.9 more, a
# flat pop ~-1.4, a glide ~-0.4: far below the success bootstrap, and refusing / falling earn no allowance back.
ENABLE_M5 = os.environ.get("MICRODUCK_SEQMANUAL_M5", "0") == "1"
M5_IT0 = int(os.environ.get("MICRODUCK_SEQMANUAL_M5_IT0", "1500"))
M5_FRAC_STAGES = ((0, 0.5), (50, 1.0))          # iterations after M5_IT0 -> fraction of M5_W
M5_W = dict(head_tilt=-6.0, head_dev=-4.0, trunk_band=-2.0, head_rate_pop=-3.0)
M5_HEAD_TILT = dict(free_deg=20.0, cap_deg=80.0)              # head off level (posture angle), all phases
M5_HEAD_DEV = dict(free=0.35, cap=3.0, stages=(sq.ST_SETTLE, sq.ST_MANUAL, sq.ST_EXIT))   # head joints vs HANDOFF
M5_TRUNK = dict(lo_deg=-15.0, hi_deg=12.0, unit_deg=10.0, cap=3.0)
# m5 nose-up +2 deg (SUB-FLAG, default off, only with ENABLE_M5: MICRODUCK_SEQMANUAL_M5_BAND=1): hold band 9-14 ->
# 11-16 (seq_cfg curriculum stage at M5_IT0), pitch shape plateau 11-13 -> 13-15 (0 at 7 / 19), manual_over 14 -> 16.
# Tail geometry: rigid strike 27.4 deg about the rear-wheel contact (roll 0-8 deg: 27.4-28.5; lowest corner 7-8 mm up
# at 16 deg); m4 rollouts (manual det): tail_floor contacts only at 25-29 deg nose-up, none below 20 deg
# (roll p90 3-4 deg). The chain_v1 HUD "tail gap 0.2 cm at 13-14 deg" disagrees with the contact sensor - read it as a
# HUD ground-reference offset, not a strike. Control: m1 held at 14.2 deg median (28 % above 15) without tail contact.
# Chain consistency: chain_play does NOT read manual_band_deg from the run - its scheduler uses the SeqRamp chain cfg's
# _chain_overrides = man.HOLD_STAGES[-1] 9-14 (m5rev: checked, the printed seq_overrides; this flag appends to a local
# copy, never to HOLD_STAGES), so play an m5-band checkpoint with --seq-override 'manual_band_deg=(11.0, 16.0)' (else
# 14-16 deg holds stop counting).
ENABLE_M5_BAND = ENABLE_M5 and os.environ.get("MICRODUCK_SEQMANUAL_M5_BAND", "0") == "1"
M5_BAND = dict(manual_band_deg=(11.0, 16.0), shape_pts=(7.0, 13.0, 15.0, 19.0), over_lo=16.0)

# m6 (2026-11-21, real servo limits; DEFAULT OFF: MICRODUCK_SEQMANUAL_M6=1, needs M5=1; RESUMES m5 model_1998 with the
# m5 env vars + every realism var: MICRODUCK_BAM_MAX_CURRENT=1.0, _DATASHEET_SPEED=1, MICRODUCK_SKATE_HEAT=1 / _W -0.3,
# MICRODUCK_SKATE_MASS_RANGE=0.98,1.10)..
# m5's stored common_step_counter is 48048 = 2002 x 24 (the run's iteration label is 1998: the m4 / m5 resumes left a
# +4 offset), so M6_IT0 = 2002 and every m6 stage counts from the first resumed step.
#  - current: 1.0 A directly from the startup env var (no runtime staging; m5 at 1.0 A zero-shot keeps the pop / hold).
#  - heat (skate_heat, the base cfg's MICRODUCK_SKATE_HEAT term): half weight until M6_IT0 + M6_HEAT_HALF, then full.
#  - costs reused from the p16 push block (skate_sequence.py, NEGATIVE weights, half weight until M6_IT0 + M6_FULL_AT):
#    p16_knee_side (wrong-side knee, every stage after a 0.4 s spawn grace), p16_leg_limits (leg joints within 3 deg of
#    their MJCF stops; left hip yaw exempt in SETTLE = the hand-off stance sits on it), and m6_neck = p16_neck's shape
#    without its PUSH-stroking gate (which never opens in this task): neck_pitch below -0.35 rad, squared per 0.1 rad.
#  - speed_loss -3 -> -4 at M6_IT0 + M6_FULL_AT (exit speed, see M6_SPEED_LOSS_W; m6rev: not on the 1.0 A step).
ENABLE_M6 = ENABLE_M5 and os.environ.get("MICRODUCK_SEQMANUAL_M6", "0") == "1"
M6_IT0 = int(os.environ.get("MICRODUCK_SEQMANUAL_M6_IT0", "2002"))
M6_W = dict(knee=-0.27, limits=-0.5, neck=-0.25)                 # = P16_W (push cfg): same robot, same per-s prices
M6_TERMS = dict(knee="m6_knee_side", limits="m6_leg_limits", neck="m6_neck")
M6_FULL_AT = 100                                                 # iterations after M6_IT0: half weight before
M6_HEAT_HALF = 50                                                # iterations after M6_IT0: heat at half weight before
M6_NECK = dict(lo=-0.35, unit=0.10, cap=4.0)
# m6 speed_loss -3 -> -4 at M6_IT0 + M6_FULL_AT: m5 at 1.0 A from MANUAL spawns enters EXIT at p10 0.31 m/s
# (MANUAL entry p10 0.70; glide / bank spawns p10 0.53 / 0.69), and the chain at 1.0 A (out/chain_real1A_manual) times
# out in EXIT 16 / 128 at 0.1-0.4 m/s - the board stops before the pipe. Raw on those spawns -1.92 / episode at -3.
M6_SPEED_LOSS_W = -4.0


def m5_trunk_band(env, lo_deg: float = -15.0, hi_deg: float = 12.0, unit_deg: float = 10.0, cap: float = 3.0):
    """Cost (m5, >= 0, NEGATIVE weight): world trunk pitch (deg, + = nose down; skate_sequence._p12_trunk) outside
    [lo_deg, hi_deg], in units of unit_deg, capped. m4 rides at -4..+4 (hold median -1..-3), leans back to -19 (p10
    -31) in the tilted-spawn catch / chain pop. Local to this task (skate_sequence.py is shared)."""
    tp = sq._p12_trunk(env)[1]
    return ((lo_deg - tp).clamp(min=0.0) + (tp - hi_deg).clamp(min=0.0)).div(unit_deg).clamp(max=cap)


def m6_neck(env, lo: float = -0.35, unit: float = 0.10, cap: float = 4.0):
    """Cost (m6, >= 0, NEGATIVE weight): sq.p16_neck's shape in every stage (p16_neck is gated on PUSH stroking, never
    true here): neck_pitch below `lo` rad (the forward crane), squared per `unit`, capped. Local to this task."""
    q = sq._robot(env).data.joint_pos[:, sq._p16_ids(env)["neck"]].nan_to_num()
    return ((lo - q) / unit).clamp(min=0.0).pow(2).clamp(max=cap)


PUSH_RAMP_TERMS = ("push_progress", "push_time", "push_speed", "push_stroke", "push_step_on", "push_step_reach",
                   "ramp_lateral", "ramp_apex")
PUSH_RAMP_METRICS = ("seq_push", "seq_liftoff", "seq_stroke_q", "seq_stepon", "seq_ramp", "seq_apex")


def _stages(*pairs) -> list[dict]:
    """(iteration, weight) pairs -> reward_weight stages in env steps."""
    return [{"step": it * IT, "weight": w} for it, w in pairs]


ACTION_RATE_STAGES = _stages((0, -0.1), (RATE_IT, -0.2))


def make_microduck_skate_seq_manual_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_skate_sequence_env_cfg(play=play)
    cfg.episode_length_s = EPISODE_LENGTH_S

    # ── spawns + scheduler overrides (hold-ended MANUAL, cap at EXIT) ──
    p = cfg.events["reset_sequence"].params
    p["spawn_probs"] = dict(SPAWN_PROBS_0)
    p["spawn_stances"] = {k: list(v) for k, v in SPAWN_STANCES.items()}
    p["flat_speeds"] = dict(FLAT_SPEEDS_0)
    p["step_blend"] = None
    p["seq_overrides"] = dict(SEQ_OVERRIDES)
    p["manual_tilt_prob"] = MANUAL_TILT_PROB
    p["manual_tilt_deg"] = MANUAL_TILT_DEG
    p["exit_tilt_deg"] = EXIT_TILT_DEG
    p["manual_prefill"] = MANUAL_PREFILL
    p["spawn_noise"] = {k: dict(v) for k, v in SPAWN_NOISE.items()}
    p["clock_offset"] = CLOCK_OFFSET
    bank = os.environ.get(HANDOFF_ENV, "").strip()
    fall_scale = FALL_STAGE_SCALE
    if bank:                                  # handoff2 (default off)
        fall_scale = FALL_STAGE_SCALE + ((sq.ST_SETTLE, HANDOFF_SETTLE_FALL_SCALE),)   # handoffrev
        assert os.path.isfile(bank), f"{HANDOFF_ENV}: no such bank file {bank}"
        p["handoff_path"] = bank
        p["handoff_prob"] = HANDOFF_PROB
        print(f"[seq_manual] hand-off spawns: {HANDOFF_PROB} of glide from {bank}", flush=True)

    # ── rewards (per second). Kept from the chain task: upright 1, on_deck 0.5, head_pose -0.5, body_on_board -1,
    # fall -10 (x0.6 in MANUAL), ride 3, ride_wheels 1, posture 0.75, dof_pos_limits. Every PUSH / RAMP term dropped ──
    R = RewardTermCfg
    for k in PUSH_RAMP_TERMS:
        cfg.rewards.pop(k)
    cfg.rewards["manual"].weight = 4.0
    cfg.rewards["manual"].params = {"lo_deg": 5.0, "hi_deg": 22.0}
    # m2: plateau 8-15 -> 11-13 deg, 0 at 5 / 17 (m1 (4, 8, 15, 22) paid ~0.9 of full at the 14-16 deg hold; m1 hold
    # scores 0.66 on the new shape)
    cfg.rewards["manual_shape"] = R(func=sq.manual_quality, weight=4.0, params=dict(M2_SHAPE))
    cfg.rewards["fall"] = R(func=sq.seq_fall_stage, weight=-10.0, params={"stage_scale": fall_scale})
    cfg.rewards["manual_hold_done"] = R(func=sq.manual_hold_done, weight=HOLD_DONE_W)
    # m2 -1 -> -2: raw ~0.35 per m1 set-down (-0.7); a 0.3 s descent from 12 deg (0.7 rad/s) costs 2 x 0.15 = -0.3
    # (deck rate^2 integrated over a fixed angle = angle^2 / duration: slower is cheaper)
    cfg.rewards["setdown_rate"] = R(func=sq.setdown_rate, weight=-2.0, params={"window_s": 0.6, "cap": 25.0})
    cfg.rewards["tail_strike"].weight = -3.0
    cfg.rewards["posture"].weight = 0.75
    cfg.rewards["head_pose"].weight = -0.5
    cfg.rewards["head_steady"].weight = M2_W["head_steady"]
    cfg.rewards["body_on_board"].weight = -1.0
    cfg.rewards["board_wobble"].weight = -0.2
    cfg.rewards["action_rate_l2"].weight = ACTION_RATE_STAGES[0]["weight"]
    # m2 costs (all functions return >= 0; negative weights). Sizes from the measured raw values, see M2_W
    cfg.rewards["manual_over"] = R(func=sq.manual_over, weight=M2_W["manual_over"],
                                   params={"lo_deg": 14.0, "span_deg": 4.0})
    cfg.rewards["pitch_rate"] = R(func=sq.manual_pitch_rate, weight=M2_W["pitch_rate"],
                                  params={"free": 0.8, "window_s": 0.6, "cap": 6.25, "below_deg": 10.0,
                                          "until_wheels_s": 0.2, "settle": True})   # m2rev: gate SETTLE too
    cfg.rewards["head_level"] = R(func=sq.head_level, weight=M3_W["head_level"],      # m3: 10 -> 5 deg, -2.5 -> -3
                                  params={"free_deg": 5.0, "cap_deg": 60.0})
    # m3: head steadiness in the world (the neck nods / the pop whip) and a slow pop (see M3_W)
    cfg.rewards["head_rate"] = R(func=sq.head_rate, weight=M3_W["head_rate"], params={"free": 1.0, "cap": 25.0})
    cfg.rewards["head_rate_pop"] = R(func=sq.head_rate, weight=M3_W["head_rate_pop"],
                                     params={"free": 1.0, "cap": 25.0, "pop_only": True, "until_deg": 11.0,
                                             "post_s": 0.3, "max_s": 1.2})
    cfg.rewards["pop_rate"] = R(func=sq.manual_pitch_rate, weight=M3_W["pop_rate"],
                                params={"free": 0.6, "cap": 2.25, "below_deg": 10.0, "until_deg": 11.0,
                                        "settle": True, "exit": False})
    cfg.rewards["stance_width"] = R(func=sq.stance_width, weight=M2_W["stance_width"],
                                    params={"free_m": 0.090, "unit_m": 0.01, "cap": 3.0})
    cfg.rewards["speed_loss"] = R(func=sq.manual_speed_loss, weight=M2_W["speed_loss"],
                                  params={"decel": 0.04, "window_s": 0.6, "cap": 0.5})
    if ENABLE_HEADING:
        f0 = MHEAD_FRAC_STAGES[0][1]
        cfg.rewards["heading_line"] = R(func=sq.heading_line_cost, weight=f0 * MHEAD_W["ride"],
                                        params=dict(MHEAD_PARAMS, stages=(sq.ST_SETTLE, sq.ST_EXIT)))
        cfg.rewards["heading_line_manual"] = R(func=sq.heading_line_cost, weight=f0 * MHEAD_W["manual"],
                                               params=dict(MHEAD_PARAMS, stages=(sq.ST_MANUAL,)))
        p["line_noise"] = {k: tuple(v) for k, v in MHEAD_NOISE.items()}
        p["seq_overrides"].update(MHEAD_GEOMETRY)          # headrev: the chain's target, nearer floor (see above)
        cfg.metrics["seq_heading_err"] = MetricsTermCfg(func=sq.m_heading_err)
    if ENABLE_M5:                                         # m5 (default off): head / trunk allowance costs, see M5_W
        f0 = M5_FRAC_STAGES[0][1]
        cfg.rewards["m5_head_tilt"] = R(func=sq.head_level, weight=f0 * M5_W["head_tilt"], params=dict(M5_HEAD_TILT))
        cfg.rewards["m5_head_dev"] = R(func=sq.settle_head_pose, weight=f0 * M5_W["head_dev"], params=dict(M5_HEAD_DEV))
        cfg.rewards["m5_trunk_band"] = R(func=m5_trunk_band, weight=f0 * M5_W["trunk_band"], params=dict(M5_TRUNK))
        if ENABLE_M5_BAND:                                # resume-only: these params apply from the first step
            cfg.rewards["manual_shape"].params["pts_deg"] = M5_BAND["shape_pts"]
            cfg.rewards["manual_over"].params["lo_deg"] = M5_BAND["over_lo"]
    if ENABLE_M6:                                         # m6 (default off): real-limit costs, see M6_W
        for key, fn, prm in (("knee", sq.p16_knee_side, None), ("limits", sq.p16_leg_limits, None),
                             ("neck", m6_neck, dict(M6_NECK))):
            cfg.rewards[M6_TERMS[key]] = R(func=fn, weight=0.5 * M6_W[key], params=prm or {})
        for k, f in (("seq_m6_limits", sq.m_p16_limits), ("seq_m6_knee", sq.m_p16_knee), ("seq_m6_amps", sq.m_p16_amps)):
            cfg.metrics[k] = MetricsTermCfg(func=f)
        amps = os.environ.get("MICRODUCK_BAM_MAX_CURRENT", "")
        if amps != "1.0" or "skate_heat" not in cfg.rewards:
            print(f"[seq_manual] WARNING m6 without the real limits: MICRODUCK_BAM_MAX_CURRENT={amps!r}, "
                  f"skate_heat {'on' if 'skate_heat' in cfg.rewards else 'OFF'}", flush=True)

    # ── curriculum: wholesale replacement, stages in iterations of this run ──
    hold_stages = list(HOLD_STAGES)
    if ENABLE_M5_BAND:
        hold_stages.append((M5_IT0, dict(HOLD_STAGES[-1][1], manual_band_deg=M5_BAND["manual_band_deg"])))
    cfg.curriculum = {
        "action_rate": CurriculumTermCfg(func=microduck_mdp.reward_weight,
                                         params={"reward_name": "action_rate_l2", "weight_stages": ACTION_RATE_STAGES}),
        "manual_hold": CurriculumTermCfg(func=sq.seq_cfg_curriculum,
                                         params={"stages": [{"step": it * IT, "cfg": dict(c)} for it, c in hold_stages]}),
        "spawns": CurriculumTermCfg(func=microduck_mdp.event_param_curriculum, params={
            "event_name": "reset_sequence", "param_stages": [
                {"step": it * IT, "params": {"spawn_probs": dict(sp), "flat_speeds": dict(fs), "clock_offset": co}}
                for it, sp, fs, co in ((0, SPAWN_PROBS_0, FLAT_SPEEDS_0, CLOCK_OFFSET_0),
                                       (SPEED_IT, SPAWN_PROBS_0, FLAT_SPEEDS_1, CLOCK_OFFSET_0),
                                       (CLOCK_IT, SPAWN_PROBS_0, FLAT_SPEEDS_1, CLOCK_OFFSET),
                                       (SPEED2_IT, SPAWN_PROBS_0, FLAT_SPEEDS_2, CLOCK_OFFSET),
                                       (SPAWN_IT, SPAWN_PROBS_1, FLAT_SPEEDS_2, CLOCK_OFFSET))]}),
    }
    if ENABLE_HEADING:
        for name, key in (("heading_line", "ride"), ("heading_line_manual", "manual")):
            cfg.curriculum[name] = CurriculumTermCfg(func=microduck_mdp.reward_weight, params={
                "reward_name": name, "weight_stages": [{"step": 0 if k == 0 else (MHEAD_IT0 + it) * IT,
                                                        "weight": fr * MHEAD_W[key]}
                                                       for k, (it, fr) in enumerate(MHEAD_FRAC_STAGES)]})
    if ENABLE_M5:
        # step 0 = the build value (half weight); resumed at >= M5_IT0 x 24 the latest stage applies from the first step
        for name, key in (("m5_head_tilt", "head_tilt"), ("m5_head_dev", "head_dev"), ("m5_trunk_band", "trunk_band")):
            cfg.curriculum[name] = CurriculumTermCfg(func=microduck_mdp.reward_weight, params={
                "reward_name": name, "weight_stages": [{"step": 0 if k == 0 else (M5_IT0 + it) * IT,
                                                        "weight": fr * M5_W[key]}
                                                       for k, (it, fr) in enumerate(M5_FRAC_STAGES)]})
        # head_rate_pop -2 (m3) -> -3 at M5_IT0 (a run below M5_IT0 keeps m3's -2)
        cfg.curriculum["m5_head_rate_pop"] = CurriculumTermCfg(func=microduck_mdp.reward_weight, params={
            "reward_name": "head_rate_pop", "weight_stages": [{"step": 0, "weight": M3_W["head_rate_pop"]},
                                                              {"step": M5_IT0 * IT, "weight": M5_W["head_rate_pop"]}]})
    if ENABLE_M6:
        # step 0 = the build value (half weight); full from (M6_IT0 + M6_FULL_AT) x 24 (resume: the counter continues)
        for key, name in M6_TERMS.items():
            cfg.curriculum[name] = CurriculumTermCfg(func=microduck_mdp.reward_weight, params={
                "reward_name": name, "weight_stages": [{"step": 0, "weight": 0.5 * M6_W[key]},
                                                       {"step": (M6_IT0 + M6_FULL_AT) * IT, "weight": M6_W[key]}]})
        cfg.curriculum["m6_speed_loss"] = CurriculumTermCfg(func=microduck_mdp.reward_weight, params={
            "reward_name": "speed_loss", "weight_stages": [{"step": 0, "weight": M2_W["speed_loss"]},
                                                           {"step": (M6_IT0 + M6_FULL_AT) * IT,
                                                            "weight": M6_SPEED_LOSS_W}]})
        if "skate_heat" in cfg.rewards:                   # MICRODUCK_SKATE_HEAT=1: half weight for M6_HEAT_HALF it
            hw = cfg.rewards["skate_heat"].weight
            cfg.rewards["skate_heat"].weight = 0.5 * hw
            cfg.curriculum["m6_heat"] = CurriculumTermCfg(func=microduck_mdp.reward_weight, params={
                "reward_name": "skate_heat", "weight_stages": [{"step": 0, "weight": 0.5 * hw},
                                                               {"step": (M6_IT0 + M6_HEAT_HALF) * IT, "weight": hw}]})

    # ── metrics: drop the PUSH / RAMP ones, add the hold / set-down / tail ones ──
    for k in PUSH_RAMP_METRICS:
        cfg.metrics.pop(k, None)
    for k, f in (("seq_band_t_done", sq.m_band_t_done), ("seq_setdown", sq.m_setdown), ("seq_setdown_n", sq.m_setdown_n),
                 ("seq_tail_manual", sq.m_tail_manual)):
        cfg.metrics[k] = MetricsTermCfg(func=f)

    # ── terminations: success (time-out), stage timeouts (failures), riding_tilt after 0.5 s ──
    T = TerminationTermCfg
    cfg.terminations["success"] = T(func=sq.seq_success, time_out=True, params=dict(SUCCESS))
    cfg.terminations["overtime"] = T(func=sq.seq_overtime, params={"limits": OVERTIME_LIMITS})
    cfg.terminations["riding_tilt"].params = {"after_s": 0.5}
    return cfg


def make_microduck_skate_seq_manual_ppo_runner_cfg():
    rl = make_microduck_skate_sequence_ppo_runner_cfg()
    rl.experiment_name = "skate_seq_manual"
    rl.max_iterations = 1500
    rl.save_interval = 100
    rl.algorithm.entropy_coef = 0.002
    return rl
