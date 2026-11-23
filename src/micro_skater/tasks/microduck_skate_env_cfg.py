# micro-skater: part of the author's skateboard add-on for microduck_rl. Apache-2.0 (see LICENSE).
"""Microduck Skate task: kick-push a cruiser skateboard, scooter style, then glide with both feet on the deck.

The robot faces forward on the board (its hips turn only 25-30 degrees, so a sideways human stance is out). The
LEFT foot rides the deck; the RIGHT foot pushes on the floor, then joins the left on the deck to glide. The board
(robot/assets/skateboard.xml) is a separate floating entity with kingpin trucks that steer toward the lean and
wheels with measured rolling resistance (coasting decel about 0.07 m/s^2).

Command: twist[0] = cruise speed v_on (m/s), twist[2] = board yaw rate, twist[1] is padding. twist[0] = 0 is STOP.

v3 (after skate_v2, which tracked speed with constant 1.4-3.3 Hz micro-pushes and almost never glided):
  - Per-env mode machine (mdp.py SKATE section). The mode speed is the robot+board centre-of-mass forward speed,
    low-passed with tau 0.1 s (internal motions cannot change it). PUSH -> GLIDE at v_on, GLIDE -> PUSH at
    v_off = v_on - gap, min dwell 0.3 s; twist[0] < 0.05 forces STOP. The actor sees mode, time in mode, the
    filtered speed, v_off and both soles on the deck.
  - PUSH and GLIDE credit the signed potential d min(v_sys, v_board, v_on) (verifier fix 2026-11-02; per 0.1 m/s;
    gains only with the board foot on the deck)
    and costs time spent in PUSH, so strong, few strokes win and hovering below v_on loses.
  - GLIDE (after a grace window) pays a product: each sole carries >= 25 % of the weight on the deck, push foot off
    the floor, trunk yaw on the board heading, head yaw / neck pitch at HOME, hip yaws at 0, trunk level. A
    potential pulls both soles to their glide spots (from the GLIDE keyframe) during the mount. Glide pays more
    per second than pushing; push-foot floor contact and braking (decel > 0.2 m/s^2) are costed.
  - A 2 s mean-speed tracking term (target (v_on + v_off) / 2) and STOP (board at rest, both feet planted).
  - Resets are a reverse curriculum: standstill starts, push poses below v_on, settled glides, end-of-stroke
    poses already in GLIDE mode at v_on (the mount), and MOUNT / DISMOUNT keyframes when skate_keyframes.py has
    them. Spawns whose keyframe is missing are dropped.
  - Curricula: v_on fixed near 0.20 / gap 0.08 / grace 1 s first, then the cruise range widens to 0.20-0.35 with
    gap 0.10 and grace 0.5 s; action-rate cost and looser trucks only after glides should exist.
  - Actor obs: the 61D family layout + 14D board/mode block (mdp.SKATE_ACTOR_OBS_STATS) = 75D. Sim/video policy,
    not hot-swappable with the 61D family.

v4 (steering; warm start from skate_v3g, which pushed and glided well but turned only ~1 deg / 4 s on a yaw command):
  - twist[2] is now path CURVATURE kappa (1/m): target board yaw rate = kappa * board speed
    (mdp.SkateCurvatureCommand: own 2-6 s hold timer, 30 % exactly straight, |kappa| >= 0.25 x range max otherwise).
    kappa range staged +-0.3 -> +-0.6 -> +-1.0 1/m.
  - Softer trucks: (rigid rider, GLIDE stance, sideways weight shift) measured a 4-8 m turn
    radius on the built 3.0 N m/rad trucks, 1.7 m at 1.0 for the same shift. Truck stiffness is set in absolute N m/rad,
    staged 2.4-3.0 -> ... -> 0.8-1.2 (TRUCK_STIFFNESS_FINAL = 1.0, design choice 2026-11-02).
  - Rewards: curvature tracking on the low-passed board yaw rate (weight 2), leaky heading-error cost, the wiggle cost
    charges only the high-pass yaw rate (the v3f form charged every commanded turn), and the glide look factor splits
    trunk pitch (10 deg) from roll (20 deg: leaning is the steering input).
  - V4_WARM_FROM_V3: a warm start restarts the step counter, so v3's curricula (mode gap, cruise range, action rate,
    CoM / push DR) start at their final stage; only the curvature range and the trucks are staged.

v5 (step on from a stand beside the board, step off on request; warm start from skate_v4):
  - STEP-ON spawns (STEP_ON_PROB of resets): robot standing in HOME on the floor to the RIGHT of the board (board on
    its left), trunk at deck-frame x +-4 cm, y -13..-16 cm (left sole just clear of the deck edge), yaw +-0.3 rad,
    board at rest. A per-env "mounted" latch (board foot carrying 25 % of the weight on the deck) switches the
    episode to riding. Before it: left-sole reach potential, a board-kick cost, board_lost relaxed (trunk 0.3 m, mount
    within 4 s). The riding terms do the rest (on_board, board_foot_on_floor, push_progress needs the board foot).
  - STEP-OFF: twist[1] = mdp.SKATE_DISMOUNT_FLAG (0.04) from a random time on, in DISMOUNT_PROB of episodes, held to
    the end, at any speed or in STOP. While it is up, every riding term is gated off (mdp.skate_riding_gated) and the
    goal is standing on the floor next to the board (no sole on the deck, both on the floor, trunk at rest and upright,
    HOME-like legs), with a +1-per-foot-off potential and a per-foot-on-deck cost. board_lost / board_flipped ignore
    it. Falls at speed just end the episode (author: "if it falls, it falls").
  - V5_WARM_FROM_V4: v4's curvature / truck curricula start at their final stage.
  - v5b (skate_v5 iteration 1000: step-on 89-97 % at cruise commands, but EVERY step-off ended on purpose with
    trunk_low, the time cost per foot on the deck beat the -3 termination): dense per-sole floor term, -15 for a
    termination under the flag, pending cost eased; warm start from skate_v5 model_1000.
  - v5c (skate_v5b iteration 750: no step-off falls any more, but only 6-22 % ever left the deck and none stood: it
    parked in the push stance, push foot on the floor, which the per-sole term paid): the floor term pays only once
    the BOARD foot is down beside the board; reverse-curriculum step-off spawns with the flag up from t = 0
    (STEP_OFF_SPAWNS: push stance at rest and slow, and standing beside the board = nearly done); pending -0.5.
  - v5d (skate_v5c iteration 500 still sat in the push stance 96 % of the time after the flag: unlike the mount,
    the step-off had no dense path): dismount_reach, a potential on both soles' deck-frame y toward the STAND spawn's
    sole spots right of the board (mdp step_off_left_y / step_off_right_y); warm start from skate_v5c model_500.

v6a (riding fine-tune; warm start from skate_v5d model_1500 extended to 79D obs):
  - ENABLE_BOX_SOLES: flat box soles (robot.microduck_constants.SKATE_SOLE_BOXES) take the deck contact, the mesh
    soles everything else (bitmask in SKATE_COLLISION / SKATEBOARD_V6_COLLISION). The ~5k-vertex hulls chattered on
    the deck boxes (contact set changing on 41-55 % of steps, 1.75 mm sinking in a lean): the "collision jitter".
    v5d on the box soles lost 19-38 / 64 riders in 20 s (mesh: 0-3): it balanced on a 6 x 4 mm rocking patch.
  - Throttle / coast: twist[0] = mdp.SKATE_COAST_CMD (0.10) is COAST (GLIDE, never back to PUSH, no speed target,
    natural decel free, faster decel charged at the push-progress rate); the range max = "W held" (VMAX_PROB);
    cruise range opens to 0.45 m/s (V6_MODE_STAGES).
  - Tighter turns: curvature tracked in 1/m (std 0.4, weight 3), glide load gate 12 % per sole, roll std 30 deg,
    MAX_CURVATURE_PROB of turn holds at exactly +-1.
  - Jitter: riding-only action-rate tax staged to -2 (ACTION_RATE_RIDING_STAGES); skate_v5d dithered 0.39-0.56
    rad/step.
  - Approach obs block (mdp.skate_approach_obs, 4D, appended last; zero-weight at the start) for v6b, and a wider
    near step-on zone (STEP_ON_XY_WIDE).
  - v6a2 (verifier 2026-11-03, v6a stopped at iteration 90 with trunk_low rising): COAST only once mounted, the COAST
    charge on the planar system COM speed, curvature pay scaled by min(v / v_min, 1), deck-sole collision bit 4.

v6b (walk-up step-on from anywhere within 0.5 m of the board, either side; warm start from skate_v6a2):
  - Canonical deck frame (mdp._skate_board_quat): while the per-episode side latch is set (robot on the board's
    left), every deck-frame quantity uses the deck turned by pi. The board is symmetric under that turn, so stepping
    on from the left is the same task; FLIP_PROB of ALL spawns are mirrored. The latch is frozen per episode.
  - Ring spawns: trunk uniform in area within [APPROACH_RING_CLEAR, r_max] of the board footprint, any yaw by the end
    (APPROACH_RING_STAGES); core / wide zones kept for the near field.
  - Before the mount: approach_progress / approach_heading potentials, premount_time cost, mount_reach saturated at
    20 cm L1, no GLIDE; board_foot_on_floor, push_time, glide_shaping, on_board and brake are riding-only. board_lost
    gives each env 4 s + d0 / 6 cm/s and a wander limit d0 + 15 cm; the step-off flag comes after that budget.
  - Board-block soles / dyaw (and the critic's trunk / push sole) clamped: the warm-started normalizer would see
    far spawns at 10-20 std.

v6c riding fixes, trained in the same run as v6b (skate_v6a2 it500 eval + turn-physics probe 2026-11-03):
  - Steering: v6a2 turned left only (+0.42 at kappa +1, -0.03 at kappa -1, straight +0.16). A scripted rider reaches
    the same |kappa| both ways, so it was the reward: the std 0.4 Gaussian has no slope at a 1 / m error. Curvature
    now pays half a std 0.3 and half a std 1.0 Gaussian (V6C_CURVATURE).
  - Tighter turns: at 1.0 N m/rad the hip-roll range caps a scripted rider at kappa ~0.7; trucks staged to 0.6-0.85
    (V6C_TRUCK_STAGES, nominal 0.7, kappa ~0.9 both ways). 0.5 is near the roll-over edge.
  - Jitter: |da| 0.43-0.48 rad/step with obs noise on or off (policy dither, not sensing; the contact audit found no
    wheel-deck contact and ~1 mm sole penetration). The riding tax goes -2, then -3 from it600 (it charges the
    exploration noise too: at -1.2 it was already ~33 % of the riding reward, so a heavier tax makes not riding win).
  - speed / cruise_speed / curvature / push_progress / stop are latch-gated (riding only) like the tax.

v6d walk-up distillation (warm start from skate_v6c): v6c it250 fell from 125 of 128 ring spawns within ~0.4 s (it
crouched into its one-step step-on wherever it stood; no gait to build the approach potentials on). The walker,
fed the skater's obs[:61] and a planner twist (skate_distill.walkup_command, the play runtime's Approach ported to
the approach obs), reached the step-on zone from 127 / 128 ring spawns (r 0.5, any yaw, both sides, median 3.2 s, no
falls). After each PPO update the skater is regressed onto that walker on its own walk-up frames (before the mount,
outside a 5 cm / 0.3 rad zone at the staging spot); the step-on, riding and step-off stay PPO's. Trucks start at
v6c's final stage.
  - v6d-1 (no anchor) wrecked the warm start's step-on in one pass; v6d2 (anchor to v6c it500, latched step-on box,
    lr 1e-4, coef 0.3 boosted to a 9 % reference mass) kept riding but at it350 the skater still could not walk
    (action error 0.1 rad^2, 0 / 128 ring spawns mounted).

v6e walker-executed walk-up (resume from skate_v6d2 it350): the env drives every unmounted, flag-down env that has not
yet entered the step-on box this episode with the walker + planner (skate_distill.SkateWalkerApproachAction); the
skater acts from the first frame in the box and trains its step-on from the walker's real arrival states from every
ring spawn (0.5 m, any yaw, both sides). Walker-driven frames get zero advantage. No BC. skate_play.Director does the
same hand-off.

v6f gated hand-off + steering (resume from skate_v6e, V6F_START_IT). Walk-up diagnosis of v6e it500 (480 walk-ups):
54 % riding at the end; the losses were step-ons from bad hand-off states (outer box edge, turned, board foot in
swing), not the walker or the budget. The walker now keeps the robot, inside the wide box too, until the trunk is in
y [-0.17, -0.13], |dyaw| <= 0.25 with the left sole on the floor (76 % riding untrained); the advantage mask reads the
frames the env term executed. Steering diagnosis: the glide stance parks both hip rolls near their +-0.38 stop as a
brace, leaving no lean range (33-50 % turn authority): riding-only L1 on |hip_roll| > 0.25 staged -1 -> -4, curvature
std 0.5 / wide share 0.6, heading-drift clamp 1.5 rad.

v6g (resume from skate_v6f, V6G_START_IT): the qpos-side cost alone did not unpark the right hip roll (its target was
pushed past the stop instead, to +2.3 rad on a +-0.38 joint): riding-only action_over_limit_penalty (overshoot 0.3)
staged -0.25 -> -1.
"""

import math
from copy import deepcopy

ENABLE_SYMMETRY = False                    # the stance is one-sided
ENABLE_BOX_SOLES = True                    # v6: flat box soles for the deck contact (mesh-sole chatter fix)

ENABLE_COM_RANDOMIZATION             = True
ENABLE_HEAD_COM_RANDOMIZATION        = True
ENABLE_MASS_INERTIA_RANDOMIZATION    = True
ENABLE_JOINT_FRICTION_RANDOMIZATION  = True
ENABLE_ARMATURE_RANDOMIZATION        = True
ENABLE_VELOCITY_PUSHES               = True
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True
ENABLE_ENCODER_BIAS                  = True
ENABLE_BOARD_RANDOMIZATION           = True

COM_RANDOMIZATION_RANGE             = 0.003
HEAD_COM_RANDOMIZATION_RANGE        = 0.003
MASS_INERTIA_RANDOMIZATION_RANGE    = (0.95, 1.05)
ARMATURE_RANDOMIZATION_RANGE        = (0.9, 1.1)
JOINT_FRICTION_RANDOMIZATION_RANGE  = (0.9, 1.1)
ENCODER_BIAS_RANGE                  = (-0.015, 0.015)
VELOCITY_PUSH_INTERVAL_S            = (4.0, 8.0)
VELOCITY_PUSH_RANGE                 = (-0.15, 0.15)   # ramped in late
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0

# Board randomization: trucks only ever LOOSER than built (HUSKY: board stiffness is the critical parameter; tight
# trucks are the easy end), rolling resistance x0.5..2, deck grip. Truck looseness is staged in (curriculum below).
BOARD_STIFFNESS_STAGES = [(0.9, 1.0), (0.75, 1.0), (0.6, 1.0)]
BOARD_ROLLING_SCALE   = (0.5, 2.0)
DECK_FRICTION_RANGE   = (0.9, 1.4)

EPISODE_LENGTH_S = 20.0
STANDING_FRACTION = 0.1           # exact-zero commands = STOP

# v4 steering (see the module docstring).
V4_WARM_FROM_V3 = True
CURVATURE_STAGES = [                                   # twist[2] = kappa (1/m)
    {"step": 0,         "range": (-0.3, 0.3)},
    {"step": 300 * 24,  "range": (-0.6, 0.6)},
    {"step": 800 * 24,  "range": (-1.0, 1.0)},
]
TRUCK_STIFFNESS_STAGES = [                             # absolute N m/rad per truck (built: 3.0)
    {"step": 0,         "value": (2.4, 3.0)},
    {"step": 300 * 24,  "value": (1.6, 2.2)},
    {"step": 700 * 24,  "value": (1.1, 1.5)},
    {"step": 1100 * 24, "value": (0.8, 1.2)},
]
TRUCK_STIFFNESS_FINAL = 1.0                            # nominal for eval / play / renders

# v5 step on / step off (see the module docstring).
V5_WARM_FROM_V4 = True
STEP_ON_PROB = 0.22                                    # share of resets that start standing beside the board
STEP_ON_XY = ((-0.04, 0.04), (-0.16, -0.13))           # trunk in the deck frame (left sole outer edge = y + 0.062)
STEP_ON_YAW = 0.3                                      # rad
STAND_Z = 0.119                                        # HOME trunk height with the soles on the floor (measured)
DISMOUNT_PROB = 0.35
DISMOUNT_TIME_RANGE = (1.0, 10.0)                      # s into the episode
V5_ACTION_RATE = -0.15                                 # v4 ended at -0.3; eased while the new steps are discovered
# v5c reverse curriculum for the step-off: episodes that start with the flag already up ("dismount": True).
STEP_OFF_SPAWNS = [
    dict(key="PUSH_MID", prob=0.05, speed="zero", mode="push", dismount=True),   # push stance at rest
    dict(key="PUSH_MID", prob=0.03, speed="push", mode="push", dismount=True),   # push stance, rolling slowly
    dict(key="STAND",    prob=0.03, speed="zero", mode="push", xy=STEP_ON_XY, yaw_noise=STEP_ON_YAW,
         mounted=False, dismount=True),                                          # already off: stay there
]

# v6 riding fine-tune (see the module docstring), warm start from skate_v5d model_1500 (obs extended to 79D).
V6_WARM_FROM_V5 = True
V6_MODE_STAGES = [                                     # gap / grace as v5's final stage; the cruise range opens up
    {"step": 0,         "gap": 0.10, "grace": 0.5, "cmd_range": (0.20, 0.35)},
    {"step": 300 * 24,  "gap": 0.10, "grace": 0.5, "cmd_range": (0.20, 0.40)},
    {"step": 700 * 24,  "gap": 0.10, "grace": 0.5, "cmd_range": (0.20, 0.45)},
]
COAST_PROB = 0.30                                      # share of speed resamples at mdp.SKATE_COAST_CMD (no key)
VMAX_PROB = 0.35                                       # share at exactly the range max (W held)
MAX_CURVATURE_PROB = 0.35                              # share of turn holds at exactly +-kappa max (A / D held)
SPEED_RESAMPLE_S = (3.0, 8.0)
ACTION_RATE_RIDING_STAGES = [                          # riding-only smoothness tax (the jitter fix), on top of -0.15
    {"step": 0,         "weight": -0.25},
    {"step": 200 * 24,  "weight": -0.6},
    {"step": 500 * 24,  "weight": -1.2},
    {"step": 900 * 24,  "weight": -2.0},
]
# v6 near step-on zone (walk-up hand-off lands here). y <= -0.13: at -0.12 with yaw 0.5 the left box sole spawned up
# to ~5 mm under the deck lip (verifier 2026-11-03).
STEP_ON_XY_WIDE = ((-0.06, 0.06), (-0.19, -0.13))
STEP_ON_YAW_WIDE = 0.5
STEP_ON_CORE_SHARE = 0.3                               # share of step-on spawns kept in the v5 zone

# v6b walk-up step-on from anywhere within 0.5 m of the board (warm start from skate_v6a2; see the docstring).
V6B_WARM_FROM_V6A = __import__("os").environ.get("MICRODUCK_SKATE_V6B", "1") == "1"   # =0: evaluate v6a2 checkpoints
V6B_STEP_ON_PROB = 0.35                                # share of resets that start standing near the board
V6B_STEP_ON_SHARES = dict(core=0.15, wide=0.25, ring=0.60)
if __import__("os").environ.get("MICRODUCK_SKATE_V6D", "1") == "1":
    # v6d (it160): more walk-up data for the walker BC (ring spawns ~1 % of the frames); core / wide mount 64 / 60 of 64
    V6B_STEP_ON_PROB = 0.45
    V6B_STEP_ON_SHARES = dict(core=0.10, wide=0.15, ring=0.75)
APPROACH_RING_CLEAR = 0.085                            # trunk >= this from the board footprint (any yaw: feet clear)
APPROACH_RING_STAGES = [                               # r_max (m from the footprint), trunk yaw range (rad)
    {"step": 0,         "r_max": 0.20, "yaw": 0.8},
    {"step": 150 * 24,  "r_max": 0.30, "yaw": 1.6},
    {"step": 300 * 24,  "r_max": 0.40, "yaw": 2.4},
    {"step": 450 * 24,  "r_max": 0.50, "yaw": math.pi},  # the author's goal from iteration 450 on (of 1500)
]
FLIP_PROB = 0.5                                        # spawns mirrored to the board's left (canonical frame)

# v6c riding fixes trained together with v6b (see the docstring), warm start from skate_v6a2.
V6C_RIDING_FIX = __import__("os").environ.get("MICRODUCK_SKATE_V6C", "1") == "1"   # =0: evaluate older checkpoints
V6C_TRUCK_STAGES = [                                   # softer trucks: kappa ~0.9 reachable both ways (0.7 at 1.0);
    {"step": 0,         "value": (0.8, 1.2)},      # offset from the approach-ring stages (150 / 300 / 450)
    {"step": 75 * 24,   "value": (0.7, 1.0)},
    {"step": 225 * 24,  "value": (0.6, 0.85)},
]
# Riding tax: the sampled actions' exploration noise (2 sum sigma^2 ~ 1.4 per step at it750) is ~87 % of what it
# charges, so at -1.2 it already took ~33 % of the riding reward (verifier 2026-11-03); -4 / -6 would make falling
# off or never mounting win. One mild step, after the ring stages; dither is judged on the deterministic JITTER battery.
V6C_ACTION_RATE_RIDING_STAGES = [
    {"step": 0,         "weight": -2.0},
    {"step": 600 * 24,  "weight": -3.0},
]
V6C_CURVATURE = dict(std=0.3, std_wide=1.0, wide_share=0.5, v_min=0.15)
if V6C_RIDING_FIX:
    TRUCK_STIFFNESS_FINAL = 0.7                        # nominal for eval / play / renders (v6a2 and older: 1.0)

# v6d walk-up distillation from the walker (skate_distill.py, see the docstring), warm start from skate_v6c.
V6D_WALKUP_BC = __import__("os").environ.get("MICRODUCK_SKATE_V6D", "1") == "1"
# v6e: the walker EXECUTES the walk-up (skate_distill.SkateWalkerApproachAction) until the step-on box, the skater
# trains the step-on from its arrival states (resume from skate_v6d2 it350; see the docstring).
V6E_WALKER_APPROACH = V6D_WALKUP_BC and __import__("os").environ.get("MICRODUCK_SKATE_V6E", "1") == "1"
# hand-off inside the trained wide step-on zone itself (the 4 cm margin was for the BC, it handed over 9 cm out);
# the env term and the PPO advantage mask must use the same dict
V6E_PLANNER = dict(__import__("micro_skater.tasks.skate_distill", fromlist=["x"]).default_walkup_bc_cfg(),
                   box_margin=0.0)

# v6f (resume from skate_v6e; see the docstring): gated walk-up hand-off and freed hip-roll range for steering.
V6F_GATE_AND_STEER = V6E_WALKER_APPROACH and __import__("os").environ.get("MICRODUCK_SKATE_V6F", "1") == "1"
V6F_START_IT = 1200                                    # the v6e iteration v6f resumes from (its curricula continue)
if V6F_GATE_AND_STEER:
    V6E_PLANNER = dict(V6E_PLANNER, handoff_gate=True)
# skate_v6e it500 steered at 33-50 % of the command: it glided with both hip rolls parked at +-0.30-0.35 of their
# +-0.38 stop (a brace it falls without), leaving no range for the lean; curvature's narrow Gaussian was dead at the
# 0.6 / m error and heading_drift's 0.5 rad clamp saturated (steering diagnosis 2026-11-03). Riding-only L1 on
# |hip_roll| beyond 0.25 rad, staged in (the brace is load-bearing), a wider curvature centre, a 1.5 rad drift clamp.
V6F_HIP_ROLL_MARGIN = 0.25
V6F_HIP_ROLL_STAGES = [
    {"step": 0,                          "weight": -1.0},
    {"step": (V6F_START_IT + 100) * 24,  "weight": -2.5},
    {"step": (V6F_START_IT + 250) * 24,  "weight": -4.0},
]
V6F_CURVATURE = dict(std=0.5, std_wide=1.0, wide_share=0.6, v_min=0.15)
V6F_HEAD_CLAMP = 1.5

# v6g (resume from skate_v6f): the qpos-side hip-roll cost did not free the lean (v6f it1350 / it1500: right hip roll
# still at +0.34-0.37 of its 0.38 stop, its TARGET pushed from +1.7 to +2.3 rad, turn authority 39-56 %). The brace
# is a commanded slam into the stop (mean 1.44 rad beyond limit + 0.3, ~99 % of all over-limit command, right_hip_roll),
# the sim-only trick action_over_limit_penalty exists for (it does not transfer).
# Riding-only, staged in.
V6G_ACTION_LIMIT = V6F_GATE_AND_STEER and __import__("os").environ.get("MICRODUCK_SKATE_V6G", "1") == "1"
V6G_START_IT = 1650                                    # the v6f iteration v6g resumes from
V6G_OVER_LIMIT_STAGES = [
    {"step": 0,                          "weight": -0.25},
    {"step": (V6G_START_IT + 100) * 24,  "weight": -0.5},
    {"step": (V6G_START_IT + 250) * 24,  "weight": -1.0},
]

# v7 (resume from skate_v6g model_2249; see the docstring): hold-W pumps up to the physical top speed, and slopes.
# Baseline (v6g it2000): at v_on 0.70 it held 0.68 (extrapolating), at 1.0 it
# collapsed near 0.8 m/s; a 2 deg downhill coast fell at ~1-1.3 m/s (12 / 16), uphill coasting rolled back and fell.
V7_SPEED_SLOPE = V6G_ACTION_LIMIT and __import__("os").environ.get("MICRODUCK_SKATE_V7", "1") == "1"
V7_START_IT = 2250                                     # the v6g iteration v7 resumes from
V7_MODE_STAGES = [                                     # W held = the range max: PUSH until v >= v_on
    {"step": 0,                         "gap": 0.10, "grace": 0.5, "cmd_range": (0.20, 0.60)},
    {"step": (V7_START_IT + 100) * 24,  "gap": 0.10, "grace": 0.5, "cmd_range": (0.20, 0.75)},
    {"step": (V7_START_IT + 250) * 24,  "gap": 0.10, "grace": 0.5, "cmd_range": (0.20, 0.90)},
    {"step": (V7_START_IT + 400) * 24,  "gap": 0.10, "grace": 0.5, "cmd_range": (0.20, 1.00)},
    # v7b it2600 reached 1.0 from rest within ~1 s and then glided (PUSH 29 %): 1.0 is below the push limit, so W
    # (the range max) opens further; above the limit W simply pushes all the time (the author's "physical max")
    {"step": (V7_START_IT + 550) * 24,  "gap": 0.10, "grace": 0.5, "cmd_range": (0.20, 1.20)},
    {"step": (V7_START_IT + 700) * 24,  "gap": 0.10, "grace": 0.5, "cmd_range": (0.20, 1.40)},
]
V7_SLOPE_STAGES = [                                    # mdp.SKATE_SLOPE_DEFAULTS keys; offset from the speed stages
    {"step": 0,                         "max_deg": 3.0, "up_max_deg": 1.5, "drop_m": (0.02, 0.06)},
    {"step": (V7_START_IT + 175) * 24,  "max_deg": 5.0, "up_max_deg": 2.5, "drop_m": (0.03, 0.10)},
    {"step": (V7_START_IT + 325) * 24,  "max_deg": 7.0, "up_max_deg": 3.0, "drop_m": (0.04, 0.15)},
    {"step": (V7_START_IT + 475) * 24,  "max_deg": 9.0, "up_max_deg": 3.0, "drop_m": (0.05, 0.20)},
]
V7_FAST_SPAWN_PROB = 0.06                              # GLIDE spawns above the push range (relative to SKATE_SPAWNS)

# v7d (resume from skate_v7c model_3449): speed-aware steering. v7c it3449 turned
# at kappa +0.46 / -0.30 at 0.45 m/s, +0.17 / -0.03 with W (1.4 m/s) and ~0 at 2.2 m/s downhill: the curvature target
# kappa * v asked for a_lat = kappa v^2 = 4 m/s^2 at 2 m/s, unreachable, so steering at speed was given up. The target
# is now clamped to +-V7D_A_LAT / v^2 (mdp.skate_feasible_kappa; full kappa 1 below ~1.1 m/s, 0.3 at 2 m/s), and more
# uphill riding (W up 2 / 3 deg survived 5/9, 3/9; turning uphill fell mostly for that).
V7D_TURN = V7_SPEED_SLOPE and __import__("os").environ.get("MICRODUCK_SKATE_V7D", "1") == "1"
V7D_START_IT = 3450
V7D_A_LAT = 1.2                                        # m/s^2 (~7 deg lean), final (evals / play)
# staged up (pre-launch review: v7c reached ~0.33 m/s^2 at speed, so at 1.2 from the start the heading-drift cost
# stays saturated and only the curvature Gaussian has a gradient): mode stages that keep v7's last speed stage
V7D_MODE_STAGES = [{**V7_MODE_STAGES[-1], "step": (V7D_START_IT + d) * 24, "a_lat": a}
                   for d, a in ((0, 0.7), (150, 1.0), (300, V7D_A_LAT))] if V7D_TURN else []
V7D_UP_PROB = 0.25                                     # mdp.SKATE_SLOPE_DEFAULTS up_prob 0.15 before
if V7D_TURN:
    V7_SLOPE_STAGES = V7_SLOPE_STAGES + [{**V7_SLOPE_STAGES[-1], "step": V7D_START_IT * 24, "up_prob": V7D_UP_PROB}]
# v7e (resume from skate_v7d model_3899): practice turning at speed. v7d it3750 still turned ~0 on downhills (2.1-2.2
# m/s) and weakly at W on the flat (+0.21 / -0.04 vs +-0.6 allowed); gliding fast without pushing it turned right
# (-0.31) but not left (+0.11). Fast riding was ~6 % fast spawns + short downhill bursts (drop <= 0.2 m, ~1 s).
V7E_SPEED_TURN = V7D_TURN and __import__("os").environ.get("MICRODUCK_SKATE_V7E", "1") == "1"
V7E_START_IT = 3900
V7E_FAST_SHARE = 0.18          # share of all reset spawns (probs are relative weights: v7's 0.06 was ~2.4 %)
V7E_DROP_M = (0.05, 0.30)
if V7E_SPEED_TURN:
    V7_SLOPE_STAGES = V7_SLOPE_STAGES + [{**V7_SLOPE_STAGES[-1], "step": V7E_START_IT * 24, "drop_m": V7E_DROP_M}]
# v7f (resume from skate_v7e model_4200): un-brace the hips. v7e it4150 glided with
# the hip rolls splayed at -0.23 / +0.21 (keyframe +-0.087), parked just under hip_roll_limit's 0.25 margin, 0.15 rad
# from the +-0.38 stops: no lean range (deck roll +-1 deg at W, ~0 downhill). A riding-only cost on the common splay
# beyond V7F_SPLAY_MAX (a lean is the differential part, not charged); the per-joint margin moves out to 0.32 so a
# lean is not charged before the stops are near.
V7F_HIP_UNBRACE = V7E_SPEED_TURN and __import__("os").environ.get("MICRODUCK_SKATE_V7F", "1") == "1"
V7F_START_IT = 4200
V7F_SPLAY_MAX = 0.12
V7F_HIP_ROLL_MARGIN = 0.32
V7F_SPLAY_STAGES = [
    {"step": 0,                           "weight": -2.0},
    {"step": (V7F_START_IT + 100) * 24,   "weight": -4.0},
    # v7f2 (resume from v7f model_4350): at -4 the splay stayed at -0.24 / +0.23 (cost ~0.4 / s, paid): -15
    {"step": (V7F_START_IT + 150) * 24,   "weight": -15.0},
]


def final_mode_stage() -> dict:
    """The mode-machine stage evals and the play export use (cruise range max = W held)."""
    if V7_SPEED_SLOPE:
        return V7_MODE_STAGES[-1]
    return (V6_MODE_STAGES if V6_WARM_FROM_V5 else MODE_STAGES)[-1]

# Mode machine stages (steps = iteration x 24). cmd_range = cruise speed v_on, v_off = v_on - gap.
MODE_STAGES = [
    {"step": 0,         "gap": 0.08, "grace": 1.0,  "cmd_range": (0.19, 0.21)},
    {"step": 800 * 24,  "gap": 0.09, "grace": 0.75, "cmd_range": (0.19, 0.28)},
    {"step": 1600 * 24, "gap": 0.10, "grace": 0.5,  "cmd_range": (0.20, 0.35)},
]

# Reset mix (reverse curriculum). speed: a kind relative to the cruise range (mdp.skate_spawn_speed_range) or an
# explicit (lo, hi). Spawns whose keyframe is not in SKATE_KEYFRAMES are dropped and the rest renormalized.
SKATE_SPAWNS = [
    dict(key="PUSH_MID",   prob=0.30, speed="zero",     mode="push"),                  # standstill start
    dict(key="PUSH_MID",   prob=0.07, speed="push",     mode="push"),
    dict(key="PUSH_FRONT", prob=0.07, speed="push",     mode="push"),
    dict(key="PUSH_BACK",  prob=0.05, speed="push",     mode="push"),
    dict(key="PUSH_BACK",  prob=0.08, speed="mount",    mode="glide"),                 # end of the last stroke
    dict(key="GLIDE",      prob=0.13, speed="glide",    mode="glide", settled=True),
    # v3e (skate_v3d iteration 750 glided on two feet but never stepped off to push again: push/s 0, coasting at
    # 0.11-0.13 m/s in PUSH mode at a 0.20-0.25 command): two-foot stance below v_off in PUSH mode, so the
    # glide -> push transition (the dismount) gets on-policy data.
    dict(key="GLIDE",      prob=0.15, speed="dismount", mode="push", settled=True),
    dict(key="GLIDE",      prob=0.05, speed="push",     mode="push", settled=True),
    dict(key="MOUNT",      prob=0.08, speed="mount",    mode="glide"),                 # if solved
    dict(key="DISMOUNT",   prob=0.07, speed="dismount", mode="push"),                  # if solved
]
if V7_SPEED_SLOPE:   # v7: high-speed glides (a downhill's runout), above the push range
    SKATE_SPAWNS = SKATE_SPAWNS + [dict(key="GLIDE", prob=V7_FAST_SPAWN_PROB, speed="fast", mode="glide",
                                        settled=True)]

_NECK_JOINTS = [5, 6, 7, 8]

import mujoco

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from micro_skater.robot.microduck_constants import (
    MICRODUCK_SKATE_ROBOT_CFG,
    MICRODUCK_SKATEBOARD_CFG,
    MICRODUCK_SKATEBOARD_V6_CFG,
    MICRODUCK_STANDUP_ROBOT_CFG,
    SKATEBOARD_DECK_CENTER_Z,
    SKATEBOARD_DECK_HALF_LEN,
    SKATEBOARD_DECK_HALF_WID,
    SKATEBOARD_WHEEL_RADIUS,
    get_standup_spec,
)
from micro_skater.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import HEAD_BODY_NAMES
from micro_skater.tasks.skate_keyframes import SKATE_KEYFRAMES
from micro_skater.tasks.skate_distill import PpoWithWalkupBcCfg, SkateWalkerApproachActionCfg, default_walkup_bc_cfg
from mjlab_microduck.tasks.symmetry import SYMMETRY_CFG, PpoWithSymmetryCfg

_ROBOT_BODY_NO_FEET = r"^(trunk_base|hip_l|leg|hip_l_2|leg_2|jaw_soft)$"
_DECK_GEOMS = r"^deck.*_collision$"   # standing platform deck_collision + other deck parts deck_<x>_collision
# Sole geoms that touch the deck: v6 box deck soles (ENABLE_BOX_SOLES), else the mesh soles.
_LEFT_DECK_SOLE = r"^left_sole_deck_collision$" if ENABLE_BOX_SOLES else r"^left_foot_collision$"
_RIGHT_DECK_SOLE = r"^right_sole_deck_collision$" if ENABLE_BOX_SOLES else r"^right_foot_collision$"


def active_spawns(keyframes: dict = SKATE_KEYFRAMES, spawns: list = SKATE_SPAWNS) -> list:
    """The spawns whose keyframe exists, probabilities renormalized to 1."""
    valid = [dict(s) for s in spawns if s["key"] in keyframes and s["prob"] > 0]
    assert valid, f"no skate spawn keyframe present in {list(keyframes)}"
    total = sum(s["prob"] for s in valid)
    for s in valid:
        s["prob"] = s["prob"] / total
    return valid


# HOME leg joints (microduck_constants.HOME_FRAME): the stance the walking policy stands in.
STAND_JOINTS = {
    "left_hip_yaw": 0.0, "left_hip_roll": -0.0873, "left_hip_pitch": -0.4579, "left_knee": -0.0049,
    "left_ankle": 0.453, "right_hip_yaw": 0.0, "right_hip_roll": 0.0873, "right_hip_pitch": 0.4579,
    "right_knee": 0.0049, "right_ankle": -0.453,
}


def v5_keyframes(keyframes: dict = SKATE_KEYFRAMES) -> dict:
    """The skate keyframes plus STAND: HOME on the floor beside the board (xy is replaced by the spawn's range)."""
    kf = dict(keyframes)
    ref = next(iter(keyframes.values()))["joints"]
    kf["STAND"] = dict(pos=(0.0, -0.145, STAND_Z), rpy=(0.0, 0.0, 0.0), joints={j: STAND_JOINTS[j] for j in ref})
    return kf


def _v7e_fast_share(spawns: list) -> list:
    """v7e: reweight the fast GLIDE spawn so it is V7E_FAST_SHARE of all resets (spawn probs are relative)."""
    if not V7E_SPEED_TURN:
        return spawns
    rest = sum(sp["prob"] for sp in spawns if sp.get("speed") != "fast")
    out = [dict(sp, prob=V7E_FAST_SHARE / (1.0 - V7E_FAST_SHARE) * rest) if sp.get("speed") == "fast" else dict(sp)
           for sp in spawns]
    tot = sum(sp["prob"] for sp in out)
    return [dict(sp, prob=sp["prob"] / tot) for sp in out]


def v5_spawns(keyframes: dict = SKATE_KEYFRAMES, spawns: list = SKATE_SPAWNS) -> list:
    """Riding spawns (renormalized) scaled to 1 - STEP_ON_PROB - the step-off spawns, plus the step-on spawn and the
    v5c step-off spawns (flag up from t = 0)."""
    out = active_spawns(keyframes, spawns)
    off = [dict(s) for s in STEP_OFF_SPAWNS if s["key"] in keyframes or s["key"] == "STAND"]
    step_on = V6B_STEP_ON_PROB if V6B_WARM_FROM_V6A else STEP_ON_PROB
    rest = 1.0 - step_on - sum(s["prob"] for s in off)
    for s in out:
        s["prob"] *= rest
    if V6B_WARM_FROM_V6A:
        # v6b: v5 core zone, v6 wide zone, and the walk-up ring (anywhere within r_max of the board, staged by
        # APPROACH_RING_STAGES); every spawn mirrored to the board's left with FLIP_PROB (canonical deck frame)
        sh = V6B_STEP_ON_SHARES
        out.append(dict(key="STAND", prob=step_on * sh["core"], speed="zero", mode="push", xy=STEP_ON_XY,
                        yaw_noise=STEP_ON_YAW, mounted=False))
        out.append(dict(key="STAND", prob=step_on * sh["wide"], speed="zero", mode="push", xy=STEP_ON_XY_WIDE,
                        yaw_noise=STEP_ON_YAW_WIDE, mounted=False))
        r0 = APPROACH_RING_STAGES[0]
        out.append(dict(key="STAND", prob=step_on * sh["ring"], speed="zero", mode="push",
                        ring=(APPROACH_RING_CLEAR, r0["r_max"]), yaw_noise=r0["yaw"], mounted=False))
        for s in out + off:
            s["flip_prob"] = FLIP_PROB
    elif V6_WARM_FROM_V5:   # v6: most step-on spawns from the wider near zone, a share kept in the v5 zone
        out.append(dict(key="STAND", prob=STEP_ON_PROB * STEP_ON_CORE_SHARE, speed="zero", mode="push",
                        xy=STEP_ON_XY, yaw_noise=STEP_ON_YAW, mounted=False))
        out.append(dict(key="STAND", prob=STEP_ON_PROB * (1.0 - STEP_ON_CORE_SHARE), speed="zero", mode="push",
                        xy=STEP_ON_XY_WIDE, yaw_noise=STEP_ON_YAW_WIDE, mounted=False))
    else:
        out.append(dict(key="STAND", prob=STEP_ON_PROB, speed="zero", mode="push", xy=STEP_ON_XY,
                        yaw_noise=STEP_ON_YAW, mounted=False))
    return out + off


def glide_sole_targets(keyframes: dict = SKATE_KEYFRAMES, key: str = "GLIDE",
                       fallback: tuple = ((0.0, 0.04), (0.0, -0.04))) -> tuple[tuple, tuple]:
    """Left / right sole xy on the deck in keyframe ``key`` (forward kinematics of the robot model; keyframe
    frame = deck centre at the origin). Falls back to ``fallback`` if the keyframe is missing."""
    kf = keyframes.get(key)
    if kf is None:
        return fallback
    m = get_standup_spec().compile()
    d = mujoco.MjData(m)
    r, p, y = kf["rpy"]
    cr, sr, cp, sp, cy, sy = (math.cos(r / 2), math.sin(r / 2), math.cos(p / 2), math.sin(p / 2), math.cos(y / 2),
                              math.sin(y / 2))
    d.qpos[0:3] = kf["pos"]
    d.qpos[3:7] = [cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
                   cr * cp * sy - sr * sp * cy]
    for j, v in kf["joints"].items():
        d.qpos[m.jnt_qposadr[m.joint(j).id]] = v
    mujoco.mj_kinematics(m, d)
    left = d.geom_xpos[m.geom("left_foot_collision").id]
    right = d.geom_xpos[m.geom("right_foot_collision").id]
    return (round(float(left[0]), 4), round(float(left[1]), 4)), (round(float(right[0]), 4), round(float(right[1]), 4))


def apply_final_curriculum_stages(cfg: ManagerBasedRlEnvCfg, slopes: bool = False) -> None:
    """For evaluation: set the curriculum-driven parameters to their final stage, then drop the curricula. v7 slopes
    stay off (flat, comparable batteries) unless ``slopes``: then their final stage."""
    final = final_mode_stage()
    if "skate_slope_setup" in cfg.events:
        p = cfg.events["skate_slope_setup"].params
        if slopes:
            p.update({k: v for k, v in V7_SLOPE_STAGES[-1].items() if k != "step"})
        else:
            p.update(max_deg=0.0, up_max_deg=0.0)
    cfg.events["skate_mode_setup"].params["gap"] = final["gap"]
    cfg.events["skate_mode_setup"].params["grace"] = final["grace"]
    cfg.commands["twist"].ranges.lin_vel_x = final["cmd_range"]
    cfg.commands["twist"].ranges.ang_vel_z = CURVATURE_STAGES[-1]["range"]
    if "board_truck_stiffness" in cfg.events:
        cfg.events["board_truck_stiffness"].params["ranges"] = (V6C_TRUCK_STAGES if V6C_RIDING_FIX
                                                                else TRUCK_STIFFNESS_STAGES)[-1]["value"]
    if hasattr(cfg.commands["twist"], "dismount_prob"):
        cfg.commands["twist"].dismount_prob = 0.0       # evals request the step-off themselves
    for name, stages_key in (("action_rate_weight", "weight_stages"), ("action_rate_riding_weight", "weight_stages"),
                             ("hip_roll_limit_weight", "weight_stages"), ("action_over_limit_weight", "weight_stages"),
                             ("hip_splay_weight", "weight_stages")):
        term = cfg.curriculum.get(name)
        if term is not None:
            cfg.rewards[term.params["reward_name"]].weight = term.params[stages_key][-1]["weight"]
    if "approach_ring" in cfg.curriculum:
        last = APPROACH_RING_STAGES[-1]
        for s in cfg.events["reset_skate"].params["spawns"]:
            if "ring" in s:
                s["ring"] = (s["ring"][0], last["r_max"])
                s["yaw_noise"] = last["yaw"]
    cfg.curriculum = {}


def make_microduck_skate_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="geom", pattern=r"^(left_foot_collision|right_foot_collision)$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    board_foot_ground_cfg = ContactSensorCfg(
        name="board_foot_ground",
        primary=ContactMatch(mode="geom", pattern=r"^left_foot_collision$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="netforce",
        num_slots=1,
    )
    # Deck sensors: one primary per deck geom (a geom secondary resolves to a single name, so the deck parts must
    # be the primaries); "force" gives the sole's load on the deck.
    board_foot_deck_cfg = ContactSensorCfg(
        name="board_foot_deck",
        primary=ContactMatch(mode="geom", pattern=_DECK_GEOMS, entity="board"),
        secondary=ContactMatch(mode="geom", pattern=_LEFT_DECK_SOLE, entity="robot"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    push_foot_ground_cfg = ContactSensorCfg(
        name="push_foot_ground",
        primary=ContactMatch(mode="geom", pattern=r"^right_foot_collision$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    push_foot_deck_cfg = ContactSensorCfg(
        name="push_foot_deck",
        primary=ContactMatch(mode="geom", pattern=_DECK_GEOMS, entity="board"),
        secondary=ContactMatch(mode="geom", pattern=_RIGHT_DECK_SOLE, entity="robot"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,   # v5: debounced step-off potential
    )
    body_board_cfg = ContactSensorCfg(
        name="body_board",
        primary=ContactMatch(mode="body", pattern=_ROBOT_BODY_NO_FEET, entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="skateboard", entity="board"),
        fields=("found",),
        reduce="netforce",
        num_slots=1,
    )
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    cfg = make_velocity_env_cfg()

    def from_final(stages: list[dict]) -> list[dict]:
        """v4 warm start: a v3 curriculum starts at (and stays on) its final stage."""
        return [dict(stages[-1], step=0)] if V4_WARM_FROM_V3 else stages

    def from_final_v4(stages: list[dict]) -> list[dict]:
        """v5 warm start: a v4 curriculum starts at (and stays on) its final stage."""
        return [dict(stages[-1], step=0)] if V5_WARM_FROM_V4 else stages

    mode_stages = V6_MODE_STAGES if V6_WARM_FROM_V5 else from_final(MODE_STAGES)
    riding_rate_stages = ACTION_RATE_RIDING_STAGES
    if V6B_WARM_FROM_V6A:   # v6b warm start: v6a's riding curricula start at (and stay on) their final stage
        mode_stages = [dict(V6_MODE_STAGES[-1], step=0)]
        riding_rate_stages = [dict(ACTION_RATE_RIDING_STAGES[-1], step=0)]
    curvature_stages = from_final_v4(CURVATURE_STAGES)
    truck_stages = from_final_v4(TRUCK_STIFFNESS_STAGES)
    if V6C_RIDING_FIX:      # v6c: riding tax and truck stages of their own (warm start from skate_v6a2)
        riding_rate_stages = V6C_ACTION_RATE_RIDING_STAGES
        truck_stages = V6C_TRUCK_STAGES
    if V6C_RIDING_FIX and V6D_WALKUP_BC:   # v6d warm start from v6c: its softened trucks from the start
        truck_stages = [dict(V6C_TRUCK_STAGES[-1], step=0)]
    if V7_SPEED_SLOPE:      # v7 resume: the cruise range opens toward the top speed
        mode_stages = V7_MODE_STAGES + V7D_MODE_STAGES

    # Robot MUST stay the first entity (base reset events write robot root state at qpos[:, 0:7]).
    cfg.scene.entities = {
        "robot": MICRODUCK_SKATE_ROBOT_CFG if ENABLE_BOX_SOLES else MICRODUCK_STANDUP_ROBOT_CFG,
        "board": MICRODUCK_SKATEBOARD_V6_CFG if ENABLE_BOX_SOLES else MICRODUCK_SKATEBOARD_CFG,
    }
    cfg.scene.sensors = (feet_ground_cfg, board_foot_ground_cfg, board_foot_deck_cfg, push_foot_ground_cfg,
                         push_foot_deck_cfg, body_board_cfg, self_collision_cfg)
    cfg.viewer.body_name = "trunk_base"
    cfg.episode_length_s = EPISODE_LENGTH_S
    # Soles on the deck (v6: box-box, up to 8 points per pair) plus four condim-6 wheels on the floor.
    cfg.sim.nconmax = 160 if ENABLE_BOX_SOLES else 140

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0
    if V6E_WALKER_APPROACH:
        import dataclasses

        cfg.actions["joint_pos"] = SkateWalkerApproachActionCfg(
            **{f.name: getattr(joint_pos_action, f.name) for f in dataclasses.fields(joint_pos_action) if f.init},
            planner=V6E_PLANNER)

    # â”€â”€ Rewards â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    for name in ["track_linear_velocity", "track_angular_velocity", "air_time", "foot_clearance",
                 "foot_swing_height", "foot_slip", "pose", "soft_landing"]:
        cfg.rewards.pop(name, None)

    # Signed potential on min(v_sys, v_board, v_on), per 0.1 m/s, credited in PUSH and GLIDE (weight 2 ->
    # standstill to 0.2 m/s pays 4 once; a push/glide cycle nets 0, the glide term is what pays per cycle).
    cfg.rewards["push_progress"] = RewardTermCfg(func=microduck_mdp.skate_push_progress, weight=2.0,
                                                 params={"scale": 0.1})
    # GLIDE: 3.5 per second of a loaded, look-forward, two-foot glide (more than pushing pays per second).
    # v3d (skate_v3c iteration 500: two-foot coasting holds speed, -0.000 m/s^2, yet the push foot dabbed the floor
    # 1/3 of GLIDE time for balance; the look factor averaged 0.25, held down by hip yaw and tilt): wider hip-yaw and
    # tilt stds so a decent glide scores visibly.
    # v4: roll gets its own 20 deg std (leaning is the steering input); pitch keeps 10 deg.
    cfg.rewards["glide"] = RewardTermCfg(func=microduck_mdp.skate_glide, weight=3.5,
                                         params={"platform_half_len": SKATEBOARD_DECK_HALF_LEN,
                                                 "hip_yaw_std": 0.25, "tilt_std": math.radians(10.0),
                                                 "roll_std": math.radians(20.0)})
    # 0.5 (v3g, was 1.0): now per-mode (push stance in PUSH, glide stance in GLIDE), so a cycle pays both steps.
    cfg.rewards["glide_shaping"] = RewardTermCfg(func=microduck_mdp.skate_glide_shaping, weight=0.5)
    # v3b (skate_v3 iteration 250 stood still in PUSH mode: speed 0.02 m/s at every command, 2/320 standstill
    # starts reached v_on): cruise is the dense drive to get moving, as speed tracking was in v1/v2. Wide std so a
    # standing robot still sees a gradient (err 0.14 -> 0.42); its centre is mid-band, so a push/glide cycle scores
    # full and pushing to hover just under v_on scores about 0.9 â€” glide (3.5) is still what separates them.
    # v3c (skate_v3b iteration 250 still stood: 0/320 standstill starts reached v_on, mean speed 0.03 m/s, and
    # GLIDE spawns were braked to a stop by dragging the push foot inside the grace window): the 2 s mean gives a
    # single stroke almost no credit. speed = instantaneous min(mode speed, board speed), linear below the band
    # centre (constant gradient from rest), Gaussian above; cruise keeps a small weight on the mean.
    cfg.rewards["speed"] = RewardTermCfg(func=microduck_mdp.skate_speed_track, weight=3.0, params={"std": 0.15})
    cfg.rewards["cruise_speed"] = RewardTermCfg(func=microduck_mdp.skate_cruise_track, weight=1.0,
                                                params={"std": 0.15})
    cfg.rewards["stop"] = RewardTermCfg(func=microduck_mdp.skate_stop, weight=3.5, params={"std": 0.05})
    # v4: curvature tracking (target yaw rate kappa * v on the 0.3 s low-passed board yaw rate; at kappa 1 and 0.3 m/s
    # the target is 0.3 rad/s) and a leaky heading-error cost (0.2 rad of drift costs 0.4 / s).
    cfg.rewards["curvature"] = RewardTermCfg(func=microduck_mdp.skate_curvature_track, weight=2.0,
                                             params={"std": 0.15})
    if V6_WARM_FROM_V5:
        # v6 (skate_v5d turned at ~40 % of the commanded curvature: a rigid rider reaches kappa 0.88 at these trucks,
        # but each sole had to carry 25 % of the weight and a 20 deg roll std taxed the lean): load gate 12 %, roll
        # std 30 deg, curvature tracked in 1/m (std 0.4) at weight 3.
        cfg.rewards["glide"].params["min_load_share"] = 0.12
        cfg.rewards["glide"].params["roll_std"] = math.radians(30.0)
        cfg.rewards["curvature"] = RewardTermCfg(func=microduck_mdp.skate_curvature_track_k, weight=3.0,
                                                 params={"std": 0.4, "v_min": 0.15})
        if V6C_RIDING_FIX:  # v6c: a wide Gaussian half so a far-off turn (v6a2 right turns, err ~1 / m) has a gradient
            cfg.rewards["curvature"].params = dict(V6C_CURVATURE)
    cfg.rewards["heading_drift"] = RewardTermCfg(func=microduck_mdp.skate_heading_drift, weight=-2.0)
    cfg.rewards["on_board"] = RewardTermCfg(func=microduck_mdp.skate_on_board, weight=1.0)

    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["upright"].weight = 1.5
    cfg.rewards["upright"].params["std"] = math.sqrt(0.05)

    cfg.rewards["pose_neck"] = RewardTermCfg(
        func=microduck_mdp.pose_target_match, weight=0.5,
        params={"std": 0.3, "joint_indices": _NECK_JOINTS, "target_overrides": None},
    )

    # Costs: mjlab-style (return >= 0, negative weight).
    cfg.rewards["push_time"] = RewardTermCfg(func=microduck_mdp.skate_in_mode, weight=-1.0,
                                             params={"mode": microduck_mdp.SKATE_PUSH})
    # -4 (v3d, was -2): balance dabs in GLIDE must cost more than a clean glide's pay is worth to skip.
    # v3f: tic-tac propulsion (board yaw-rate rms 0.33 rad/s, speed from rest with no floor contact) costs ~1/s;
    # a straight glide (~0.05 rad/s) costs ~0.02/s.
    # v4: high-pass only, so a held turn is free.
    cfg.rewards["board_wiggle"] = RewardTermCfg(func=microduck_mdp.skate_board_wiggle_hp, weight=-8.0)
    cfg.rewards["glide_floor"] = RewardTermCfg(func=microduck_mdp.skate_glide_floor, weight=-4.0,
                                               params={"push_floor_sensor": push_foot_ground_cfg.name})
    # Pump-and-brake (brake in the GLIDE dwell, re-earn push_progress) must lose to an honest coast.
    # Threshold 0.1 (v3c): measured coasting is 0.001-0.031 m/s^2, and a dragged push foot shed 0.2 m/s in about a
    # second (~0.2 m/s^2), which the old 0.2 threshold left free.
    cfg.rewards["brake"] = RewardTermCfg(func=microduck_mdp.skate_brake, weight=-4.0, params={"threshold": 0.1})
    cfg.rewards["action_rate_l2"].weight = -0.02     # curriculum below (after glides exist)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost, weight=-1.0, params={"sensor_name": self_collision_cfg.name},
    )
    cfg.rewards["board_foot_slip"] = RewardTermCfg(func=microduck_mdp.skate_board_foot_slip, weight=-5.0)
    cfg.rewards["body_on_board"] = RewardTermCfg(
        func=microduck_mdp.skate_contact_cost, weight=-1.0, params={"sensor_name": body_board_cfg.name},
    )
    cfg.rewards["board_foot_on_floor"] = RewardTermCfg(
        func=microduck_mdp.skate_contact_cost, weight=-1.0, params={"sensor_name": board_foot_ground_cfg.name},
    )
    cfg.rewards["terminated"] = RewardTermCfg(func=mdp.is_terminated, weight=-3.0)

    # â”€â”€ v5 step on / step off â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Riding terms pay / cost nothing while the step-off flag is up (on_board and board_foot_on_floor would fight it).
    for name in ("push_progress", "glide", "glide_shaping", "speed", "cruise_speed", "stop", "curvature",
                 "heading_drift", "on_board", "push_time", "board_wiggle", "glide_floor", "brake", "board_foot_slip",
                 "board_foot_on_floor"):
        cfg.rewards[name].func = microduck_mdp.skate_riding_gated(cfg.rewards[name].func)
    # Step-on: left-sole reach potential (12 cm to the deck spot ~ +2.4 once) on top of on_board (+1/s) and
    # board_foot_on_floor (-1/s); shoving the resting board away is costed.
    cfg.rewards["mount_reach"] = RewardTermCfg(func=microduck_mdp.skate_mount_reach, weight=1.0)
    cfg.rewards["board_kick"] = RewardTermCfg(func=microduck_mdp.skate_board_kick, weight=-5.0)
    # Step-off: +1 per foot leaving the deck (potential; falling off both ways nets 2 - 3 for the termination < 0),
    # 4/s standing beside the board at rest (about what riding pays per second), a HOME-like hand-off posture, -1/s
    # per foot still on the deck.
    cfg.rewards["dismount_progress"] = RewardTermCfg(func=microduck_mdp.skate_dismount_progress, weight=1.0)
    cfg.rewards["dismount_stand"] = RewardTermCfg(
        func=microduck_mdp.skate_dismount_stand, weight=4.0,
        params={"feet_sensor": feet_ground_cfg.name, "speed_std": 0.15, "tilt_std": math.radians(15.0)},
    )
    cfg.rewards["dismount_pose"] = RewardTermCfg(func=microduck_mdp.skate_dismount_pose, weight=1.0,
                                                 params={"std": 0.3})
    # v5b (skate_v5 iteration 1000 ended every step-off on purpose, trunk_low: -1/s per foot on the deck for up to
    # 19 s beat the one-off -3 termination): pending eased to -0.25, a dense +1/s per sole on the floor beside the
    # board, and -15 for any termination while the flag is up.
    cfg.rewards["dismount_pending"] = RewardTermCfg(func=microduck_mdp.skate_dismount_pending, weight=-0.5)
    cfg.rewards["dismount_feet"] = RewardTermCfg(func=microduck_mdp.skate_dismount_feet, weight=1.0,
                                                 params={"feet_sensor": feet_ground_cfg.name})
    cfg.rewards["dismount_failed"] = RewardTermCfg(func=microduck_mdp.skate_dismount_failed, weight=-15.0)
    # v5d: dense reach potential toward the step-off sole spots right of the board (about +5 in all from the push
    # stance; a fall still nets well below 0 with the -15).
    cfg.rewards["dismount_reach"] = RewardTermCfg(func=microduck_mdp.skate_dismount_reach, weight=1.0)
    if V6_WARM_FROM_V5:
        # v6 jitter: skate_v5d dithered its targets 0.39 rad/step straight / 0.56 turning (15-38 % of the action
        # power above 5 Hz); at -0.15 the action-rate cost was ~0.2 % of the reward. A riding-only tax, staged
        # (ACTION_RATE_RIDING_STAGES), on top of the global -0.15 that keeps step-on / step-off untaxed.
        cfg.rewards["action_rate_riding"] = RewardTermCfg(
            func=microduck_mdp.skate_riding_only(cfg.rewards["action_rate_l2"].func),
            weight=riding_rate_stages[0]["weight"], params=dict(cfg.rewards["action_rate_l2"].params),
        )
    if V6B_WARM_FROM_V6A:
        # v6b walk-up: trunk progress to the staging spot (0.5 m ~ +5 at 0.5), lining up with the deck near it, and a
        # pre-mount time cost in place of board_foot_on_floor / push_time (riding only now: walking needs the board
        # foot on the floor, and -2/s over a 10 s walk-up would beat a -3 fall). mount_reach saturates at 20 cm L1
        # (mode setup reach_sat) so it pays the last step, not the walk.
        cfg.rewards["approach_progress"] = RewardTermCfg(func=microduck_mdp.skate_approach_progress, weight=0.5)
        cfg.rewards["approach_heading"] = RewardTermCfg(func=microduck_mdp.skate_approach_heading, weight=0.5)
        cfg.rewards["premount_time"] = RewardTermCfg(func=microduck_mdp.skate_premount_time, weight=-0.5)
        # Riding only as well (verifier 2026-11-03): glide_shaping's push-stance potential paid the walk-up a second
        # time (+4.9 for 0.5 m), on_board paid +1/s for a foot parked on the deck before the mount, brake charged
        # the walking robot's gait decelerations (v_mode is the system speed).
        for name in ("board_foot_on_floor", "push_time", "glide_shaping", "on_board", "brake"):
            cfg.rewards[name].func = microduck_mdp.skate_riding_only(cfg.rewards[name].func)
    if V6B_WARM_FROM_V6A and V6C_RIDING_FIX:
        # v6c (verifier 2026-11-03): the riding pays were only flag-gated (board-foot contact), the tax is latch-gated:
        # a light left foot on the deck (< 25 % load) or scootering beside the board collected them untaxed.
        for name in ("speed", "cruise_speed", "curvature", "push_progress", "stop"):
            cfg.rewards[name].func = microduck_mdp.skate_riding_only(cfg.rewards[name].func)
    if V6F_GATE_AND_STEER:
        cfg.rewards["curvature"].params = dict(V6F_CURVATURE)
        cfg.rewards["hip_roll_limit"] = RewardTermCfg(
            func=microduck_mdp.skate_riding_only(microduck_mdp.skate_joint_limit_proximity),
            weight=V6F_HIP_ROLL_STAGES[0]["weight"],
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(r"^(left|right)_hip_roll$",)),
                    "margin": V7F_HIP_ROLL_MARGIN if V7F_HIP_UNBRACE else V6F_HIP_ROLL_MARGIN},
        )
        if V7F_HIP_UNBRACE:
            cfg.rewards["hip_splay"] = RewardTermCfg(
                func=microduck_mdp.skate_riding_only(microduck_mdp.skate_hip_splay),
                weight=V7F_SPLAY_STAGES[0]["weight"], params={"s_max": V7F_SPLAY_MAX},
            )
    if V6G_ACTION_LIMIT:
        cfg.rewards["action_over_limit"] = RewardTermCfg(
            func=microduck_mdp.skate_riding_only(microduck_mdp.action_over_limit_penalty),
            weight=V6G_OVER_LIMIT_STAGES[0]["weight"], params={"action_name": "joint_pos", "overshoot": 0.3},
        )

    # â”€â”€ Observations: 61D family layout + 14D board/mode block â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    del cfg.observations["actor"].terms["base_lin_vel"]
    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(func=mdp.base_lin_vel, scale=1.0)
    del cfg.observations["critic"].terms["foot_height"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    gravity_term_name = "projected_gravity"
    cfg.observations["actor"].terms[gravity_term_name] = deepcopy(cfg.observations["actor"].terms[gravity_term_name])
    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(cfg.observations["actor"].terms["base_ang_vel"])
    for term in ("base_ang_vel", gravity_term_name):
        cfg.observations["actor"].terms[term].delay_min_lag = 0
        cfg.observations["actor"].terms[term].delay_max_lag = 1
        cfg.observations["actor"].terms[term].delay_update_period = 64
    cfg.observations["actor"].terms["base_ang_vel"].noise    = Unoise(n_min=-0.03, n_max=0.03)
    cfg.observations["actor"].terms[gravity_term_name].noise = Unoise(n_min=-0.01, n_max=0.01)
    cfg.observations["actor"].terms["joint_pos"].noise       = Unoise(n_min=-0.001, n_max=0.001)
    cfg.observations["actor"].terms["joint_vel"].noise       = Unoise(n_min=-0.25, n_max=0.25)

    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        av = cfg.observations["actor"].terms["base_ang_vel"]
        av.func = microduck_mdp.base_ang_vel_imu_misaligned
        av.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
        g = cfg.observations["actor"].terms[gravity_term_name]
        g.func = microduck_mdp.projected_gravity_imu_misaligned
        g.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}

    cfg.observations["actor"].terms["joint_vel"] = deepcopy(cfg.observations["actor"].terms["joint_vel"])
    cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_update_period = 0

    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    for grp in ("actor", "critic"):
        for term in ("joint_pos", "joint_vel"):
            cfg.observations[grp].terms[term] = deepcopy(cfg.observations[grp].terms[term])
            cfg.observations[grp].terms[term].params["asset_cfg"] = deepcopy(passive_excluded)

    if ENABLE_ENCODER_BIAS:
        cfg.events["encoder_bias"].params["bias_range"] = ENCODER_BIAS_RANGE
        cfg.observations["actor"].terms["joint_pos"].params["biased"] = True
        cfg.observations["critic"].terms["joint_pos"].params["biased"] = False
    else:
        cfg.events.pop("encoder_bias", None)

    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 4},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 6},
        )
        # Board block LAST so a 61D checkpoint warm-starts by zero-padding the first layer's trailing inputs.
        cfg.observations[group].terms["board"] = ObservationTermCfg(func=microduck_mdp.skate_actor_obs)
    cfg.observations["critic"].terms["board_privileged"] = ObservationTermCfg(func=microduck_mdp.skate_critic_obs)
    if V6_WARM_FROM_V5:
        # v6 approach block LAST in both groups (75 -> 79 actor): a v5 checkpoint warm-starts through
        # (zero first-layer columns, seeded normalizer).
        for group in ("actor", "critic"):
            cfg.observations[group].terms["approach"] = ObservationTermCfg(
                func=microduck_mdp.skate_approach_obs, params={"clip": 0.8 if V6B_WARM_FROM_V6A else 0.6})

    # â”€â”€ Command â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    command = cfg.commands["twist"]
    command.rel_standing_envs = STANDING_FRACTION
    command.rel_heading_envs = 0.0
    command.rel_forward_envs = 0.0
    command.heading_command = False
    command.ranges.heading = None
    command.resampling_time_range = (6.0, 10.0)
    command.debug_vis = False
    command.ranges.lin_vel_x = mode_stages[0]["cmd_range"]
    command.ranges.lin_vel_y = (-0.01, 0.01)
    command.ranges.ang_vel_z = curvature_stages[0]["range"]
    cfg.commands["twist"] = microduck_mdp.SkateCurvatureCommandCfg(
        **vars(command), dismount_prob=DISMOUNT_PROB, dismount_time_range=DISMOUNT_TIME_RANGE,
    )
    if V6_WARM_FROM_V5:
        tw = cfg.commands["twist"]
        tw.coast_prob, tw.vmax_prob, tw.max_curvature_prob = COAST_PROB, VMAX_PROB, MAX_CURVATURE_PROB
        tw.resampling_time_range = SPEED_RESAMPLE_S

    # â”€â”€ Terminations â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.terminations["fell_over"].params["limit_angle"] = math.radians(60.0)
    cfg.terminations["trunk_low"] = TerminationTermCfg(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.075},
    )
    cfg.terminations["board_lost"] = TerminationTermCfg(
        func=microduck_mdp.skate_board_lost, params={"max_dist": max(0.12, SKATEBOARD_DECK_HALF_WID + 0.065)},
    )
    if V6B_WARM_FROM_V6A:   # per-env walk-up budget: wander <= 15 cm beyond the spawn distance, 4 s + d0 / 6 cm/s
        cfg.terminations["board_lost"].params.update(approach_slack=0.15, approach_speed=0.06)
    cfg.terminations["board_flipped"] = TerminationTermCfg(func=microduck_mdp.skate_board_flipped)
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.skate_state_is_nan,
        params={"sensor_names": ("feet_ground_contact", "board_foot_deck", "push_foot_deck")},
    )

    # â”€â”€ Events â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    cfg.events["expand_bam_friction_fields"] = EventTermCfg(
        func=microduck_mdp.expand_bam_friction_fields, mode="startup",
    )
    left, right = glide_sole_targets()
    _, push_right = glide_sole_targets(key="PUSH_MID", fallback=((0.0, 0.04), (0.0, -0.09)))
    cfg.events["skate_mode_setup"] = EventTermCfg(
        func=microduck_mdp.skate_mode_setup, mode="startup",
        params={"gap": mode_stages[0]["gap"], "grace": mode_stages[0]["grace"], "glide_left": left,
                "glide_right": right, "push_right": push_right,
                "extra": ({"v6b": True, "reach_sat": 0.2,
                           **({"head_clamp": V6F_HEAD_CLAMP} if V6F_GATE_AND_STEER else {}),
                           **({"a_lat": V7D_A_LAT} if V7D_TURN else {})}
                          if V6B_WARM_FROM_V6A else None)},
    )
    if V7_SPEED_SLOPE:      # v7: per-env tilted gravity = slopes (mdp.skate_slope_setup)
        cfg.events["skate_slope_setup"] = EventTermCfg(
            func=microduck_mdp.skate_slope_setup, mode="startup",
            params={k: v for k, v in V7_SLOPE_STAGES[0].items() if k != "step"},
        )
    cfg.events["reset_action_history"] = EventTermCfg(func=microduck_mdp.reset_action_history, mode="reset")
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = (
        ("left_foot_collision", "right_foot_collision", "left_sole_deck_collision", "right_sole_deck_collision")
        if ENABLE_BOX_SOLES else ("left_foot_collision", "right_foot_collision"))
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.3)
    # MUST come after reset_base / reset_robot_joints (events run in insertion order).
    cfg.events["reset_skate"] = EventTermCfg(
        func=microduck_mdp.reset_skate_state,
        mode="reset",
        params={
            "keyframes": v5_keyframes(),
            "spawns": _v7e_fast_share(v5_spawns()),
            "deck_center_z": SKATEBOARD_DECK_CENTER_Z,
            "wheel_radius": SKATEBOARD_WHEEL_RADIUS,
        },
    )

    if ENABLE_BOARD_RANDOMIZATION:
        # v4: absolute N m/rad (TRUCK_STIFFNESS_STAGES), was a scale of the built 3.0 (BOARD_STIFFNESS_STAGES).
        cfg.events["board_truck_stiffness"] = EventTermCfg(
            func=dr.joint_stiffness, mode="reset",
            params={"asset_cfg": SceneEntityCfg("board", joint_names=(r"^passive_(front|rear)_truck$",)),
                    "operation": "abs", "ranges": truck_stages[0]["value"], "shared_random": True},
        )
        cfg.events["board_rolling"] = EventTermCfg(
            func=dr.geom_friction, mode="reset",
            params={"asset_cfg": SceneEntityCfg("board", geom_names=(r"^.*_wheel_[LR]_collision$",)),
                    "operation": "scale", "ranges": BOARD_ROLLING_SCALE, "axes": [2], "shared_random": True},
        )
        cfg.events["deck_friction"] = EventTermCfg(
            func=dr.geom_friction, mode="reset",
            params={"asset_cfg": SceneEntityCfg("board", geom_names=(_DECK_GEOMS,)),
                    "operation": "abs", "ranges": DECK_FRICTION_RANGE},
        )

    if ENABLE_VELOCITY_PUSHES:
        cfg.events["push_robot"] = EventTermCfg(
            func=mdp.push_by_setting_velocity, mode="interval",
            interval_range_s=(2.0, 4.0) if play else VELOCITY_PUSH_INTERVAL_S,
            params={"velocity_range": {"x": VELOCITY_PUSH_RANGE, "y": VELOCITY_PUSH_RANGE},
                    "asset_cfg": SceneEntityCfg("robot")},
        )
    else:
        cfg.events.pop("push_robot", None)

    if ENABLE_COM_RANDOMIZATION:
        cfg.events["randomize_com"] = EventTermCfg(
            func=dr.body_ipos, mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)), "operation": "add",
                    "ranges": (-COM_RANDOMIZATION_RANGE, COM_RANDOMIZATION_RANGE)},
        )
    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.events["randomize_head_com"] = EventTermCfg(
            func=dr.body_ipos, mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot", body_names=HEAD_BODY_NAMES), "operation": "add",
                    "ranges": (-HEAD_COM_RANDOMIZATION_RANGE, HEAD_COM_RANDOMIZATION_RANGE)},
        )
    if ENABLE_ARMATURE_RANDOMIZATION:
        cfg.events["randomize_armature"] = EventTermCfg(
            func=dr.joint_armature, mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",)), "operation": "scale",
                    "ranges": ARMATURE_RANDOMIZATION_RANGE},
        )
    if ENABLE_MASS_INERTIA_RANDOMIZATION:
        lo, hi = MASS_INERTIA_RANDOMIZATION_RANGE
        cfg.events["randomize_mass_inertia"] = EventTermCfg(
            func=dr.pseudo_inertia, mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                    "alpha_range": (math.log(lo) / 2.0, math.log(hi) / 2.0)},
        )
    if ENABLE_JOINT_FRICTION_RANDOMIZATION:
        cfg.events["randomize_joint_friction"] = EventTermCfg(
            func=microduck_mdp.randomize_bam_friction, mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot"), "scale_range": JOINT_FRICTION_RANDOMIZATION_RANGE},
        )

    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # â”€â”€ Curriculum (steps = iteration x 24) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    cfg.curriculum.pop("terrain_levels", None)
    cfg.curriculum.pop("command_vel", None)
    cfg.curriculum["skate_mode"] = CurriculumTermCfg(
        func=microduck_mdp.skate_mode_curriculum, params={"stages": mode_stages},
    )
    # Smoothness tax only once glides should exist (no attempt-tax during skill discovery).
    action_rate_stages = from_final([
        {"step": 0, "weight": -0.02},
        {"step": 1500 * 24, "weight": -0.1},
        {"step": 2000 * 24, "weight": -0.2},
        {"step": 2500 * 24, "weight": -0.3},
    ])
    if V5_WARM_FROM_V4:
        action_rate_stages = [{"step": 0, "weight": V5_ACTION_RATE}]
    cfg.rewards["action_rate_l2"].weight = action_rate_stages[0]["weight"]
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={"reward_name": "action_rate_l2", "weight_stages": action_rate_stages},
    )
    if V6_WARM_FROM_V5:
        cfg.curriculum["action_rate_riding_weight"] = CurriculumTermCfg(
            func=microduck_mdp.reward_weight,
            params={"reward_name": "action_rate_riding", "weight_stages": riding_rate_stages},
        )
    if V6F_GATE_AND_STEER:
        cfg.curriculum["hip_roll_limit_weight"] = CurriculumTermCfg(
            func=microduck_mdp.reward_weight,
            params={"reward_name": "hip_roll_limit", "weight_stages": V6F_HIP_ROLL_STAGES},
        )
    if V6G_ACTION_LIMIT:
        cfg.curriculum["action_over_limit_weight"] = CurriculumTermCfg(
            func=microduck_mdp.reward_weight,
            params={"reward_name": "action_over_limit", "weight_stages": V6G_OVER_LIMIT_STAGES},
        )
    if V7F_HIP_UNBRACE:
        cfg.curriculum["hip_splay_weight"] = CurriculumTermCfg(
            func=microduck_mdp.reward_weight,
            params={"reward_name": "hip_splay", "weight_stages": V7F_SPLAY_STAGES},
        )
    if V7_SPEED_SLOPE:
        cfg.curriculum["slope"] = CurriculumTermCfg(
            func=microduck_mdp.skate_slope_curriculum, params={"stages": V7_SLOPE_STAGES},
        )
    if V6B_WARM_FROM_V6A:
        cfg.curriculum["approach_ring"] = CurriculumTermCfg(
            func=microduck_mdp.skate_approach_ring_curriculum, params={"stages": APPROACH_RING_STAGES},
        )
    # v4 steering: curvature range and softer trucks, staged together (a wider turn needs the looser truck).
    cfg.curriculum["curvature_range"] = CurriculumTermCfg(
        func=microduck_mdp.skate_command_range_curriculum, params={"stages": curvature_stages, "attr": "ang_vel_z"},
    )
    if ENABLE_BOARD_RANDOMIZATION:
        cfg.curriculum["truck_stiffness"] = CurriculumTermCfg(
            func=microduck_mdp.skate_event_param_curriculum,
            params={"event_name": "board_truck_stiffness", "param": "ranges", "stages": truck_stages},
        )
    if ENABLE_COM_RANDOMIZATION:
        cfg.curriculum["com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={"event_name": "randomize_com", "range_stages": from_final([
                {"step": 0, "range": 0.003}, {"step": 1000 * 24, "range": 0.006}, {"step": 2000 * 24, "range": 0.01},
            ])},
        )
    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.curriculum["head_com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={"event_name": "randomize_head_com", "range_stages": from_final([
                {"step": 0, "range": 0.003}, {"step": 1000 * 24, "range": 0.006},
            ])},
        )
    if ENABLE_VELOCITY_PUSHES:
        cfg.curriculum["push_magnitude"] = CurriculumTermCfg(
            func=microduck_mdp.push_curriculum,
            params={"event_name": "push_robot", "push_stages": from_final([
                {"step": 0, "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}},
                {"step": 1500 * 24, "velocity_range": {"x": (-0.06, 0.06), "y": (-0.06, 0.06)}},
                {"step": 2500 * 24, "velocity_range": {"x": VELOCITY_PUSH_RANGE, "y": VELOCITY_PUSH_RANGE}},
            ])},
        )

    return cfg


MicroduckSkateRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
    algorithm=(PpoWithWalkupBcCfg if V6D_WALKUP_BC else PpoWithSymmetryCfg)(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None,
        **({"walkup_bc_cfg": dict(V6E_PLANNER, executed=True, coef=0.0, anchor_coef=0.0)}
           if V6E_WALKER_APPROACH else {}),
    ),
    wandb_project="mjlab_microduck",
    experiment_name="skate",
    run_name="skate",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=5000,
)
