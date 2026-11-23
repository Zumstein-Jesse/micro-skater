# micro-skater: part of the author's skateboard add-on for microduck_rl. Apache-2.0 (see LICENSE).
"""Skateboard SEQUENCE (2026-11-19): push from rest -> manual -> exit -> carve into a mini half-pipe -> up and down.

Task Mjlab-Skate-Sequence-MicroDuck. MDP terms + scheduler: skate_sequence.py. Board: sequence fishtail kicktail
(robot/assets/skateboard_sequence.xml).


Built on the forward trick cfg (make_microduck_skate_trick_env_cfg: teacher obs noise / delays / DR stack, sphere-sole
robot, stance-centred actions) with: the half-pipe terrain tile (box facets, plane floor), per-foot floor sensors,
the sequence spawn/scheduler, PUSH/RAMP rewards, gated foot/board terminations, a 13D sequence obs block appended LAST
(actor 61 + 28 + 12 + 13 = 114D) so the forward trick policy warm-starts through.
"""

from __future__ import annotations
from micro_skater.paths import data_path as _data_path

import math
import os
import sys

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import (CurriculumTermCfg, EventTermCfg, MetricsTermCfg, ObservationTermCfg, RewardTermCfg,
                            TerminationTermCfg)
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.tasks.velocity import mdp

from micro_skater.robot.skateboard_variants import MICRODUCK_SKATEBOARD_SEQUENCE_CFG
from micro_skater.tasks import mdp as microduck_mdp
from micro_skater.tasks import skate_sequence as sq
from micro_skater.tasks import skate_trick as tk
from micro_skater.tasks.microduck_skate_trick_env_cfg import (ALL_SENSORS, make_microduck_skate_trick_env_cfg,
                                                                 make_microduck_skate_trick_ppo_runner_cfg)

from micro_skater.tasks import halfpipe  # noqa: E402  (HalfPipeTerrainCfg: box facets + plane floor)

EPISODE_LENGTH_S = 22.0
ACTION_SCALE = 1.0                  # push poses sit up to ~0.7 rad from the tail stance
ACTION_STANCE = "tail"
ACTION_CENTER_JSON = str(_data_path("stance_poses_fwd.json"))
TILE_SIZE = (30.0, 30.0)
# real2 servo heat cost (2026-11-21, default off): MICRODUCK_SKATE_HEAT=1 adds rewards["skate_heat"] =
# MICRODUCK_SKATE_HEAT_W (default -0.3 per W) x mdp.skate_heat_cost = sum_j relu(2 s low-passed I_j^2 R0 - 0.65 W),
# capped at 8 W. r7 at 1.0 A holds both hip yaws at the limit (~0.99 A RMS, 95 % static stall, opposite signs): ~3 W of
# excess = ~-1.0 /s against an alive pay of ~4.9 /s; every other servo adds ~0.4 W (~-0.12 /s). Cap 8 W -> <= -2.4 /s.
SKATE_HEAT = os.environ.get("MICRODUCK_SKATE_HEAT", "0") == "1"
SKATE_HEAT_W = float(os.environ.get("MICRODUCK_SKATE_HEAT_W", "-0.3"))
SKATE_HEAT_P_CONT = float(os.environ.get("MICRODUCK_SKATE_HEAT_P_CONT", "0.65"))
SKATE_HEAT_CAP_W = float(os.environ.get("MICRODUCK_SKATE_HEAT_CAP_W", "8.0"))
SKATE_HEAT_IT0 = int(os.environ.get("MICRODUCK_SKATE_HEAT_IT0", "0"))
# audit_real item 7 (2026-11-21, default off): MICRODUCK_SKATE_MASS_RANGE="lo,hi" replaces the inherited trunk-only
# +-5 % startup pseudo_inertia DR with ONE whole-robot scale per env per reset on every robot body's mass + inertia
# (mdp.skate_robot_mass_scale, restore-then-apply). 0.98,1.10 x 0.737 kg = 0.72-0.81 kg (runtime 0.744, README ~0.80).
_MASS_RANGE_ENV = os.environ.get("MICRODUCK_SKATE_MASS_RANGE", "").strip()
SKATE_MASS_RANGE = tuple(float(v) for v in _MASS_RANGE_ENV.split(",")) if _MASS_RANGE_ENV else None
assert SKATE_MASS_RANGE is None or (len(SKATE_MASS_RANGE) == 2 and 0.0 < SKATE_MASS_RANGE[0] <= SKATE_MASS_RANGE[1]), \
    f"MICRODUCK_SKATE_MASS_RANGE must be 'lo,hi', got {_MASS_RANGE_ENV!r}"

# REV (2026-11-19): start (the demo's whole chain) 0.18 -> 0.20,
# step_on 0.12 -> 0.14 (the never-learned transition), manual 0.20 -> 0.22 (+ longer spawned holds, manual_spawn_s),
# exit 0.11 -> 0.08 (75-90 % already reach the pipe), ramp 0.26 -> 0.24.
SPAWN_PROBS = dict(start=0.18, push=0.04, step_on=0.18, glide=0.06, manual=0.22, exit=0.08, ramp_bottom=0.12,
                   ramp_wall=0.12)
# REV: manual spawns weight tail_back (the CoM-behind-the-axle lifting pose) and add glide_step
# (the stance the chain actually arrives in after the step-on: soles at x -0.040, 6 cm apart)
SPAWN_STANCES = dict(start=["mount"], push=["push", "push_back", "mount"], step_on=["push_back", "push"],
                     glide=["glide", "glide_step"],
                     manual=["tail", "tail_back", "tail_back", "glide_step"], exit=["tail", "glide", "glide_step"],
                     ramp_bottom=["glide", "glide_fwd"], ramp_wall=["glide", "glide_fwd", "tail"])
SEQ_SENSORS = ALL_SENSORS + ("left_foot_floor", "right_foot_floor")

# ── REV curriculum origin: smoothness taxes start AFTER the resume point (taxes after skill discovery; the
# step-on, the slow stroke and this board's manual are still being discovered at resume). s1e model_1750 stores
# common_step_counter 42120 (= 1755 x 24). For a fresh warm start (MICRODUCK_WARM_START=1) set
# MICRODUCK_SEQ_REV_STEP0=0. Stage steps are env steps (iteration x 24).
REV_STEP0 = int(os.environ.get("MICRODUCK_SEQ_REV_STEP0", "42120"))
ACTION_RATE_STAGES = [{"step": 0, "weight": -0.01}, {"step": 1000 * 24, "weight": -0.05},
                      {"step": REV_STEP0 + 1000 * 24, "weight": -0.08}, {"step": REV_STEP0 + 2000 * 24, "weight": -0.12}]
# Sized by measured mass (on s1e model_1750 with its exploration noise, diag/): raw
# head_steady 20-30 (rad/s)^2 riding, ~58 pushing (head action std 0.43-0.99: the head is task-irrelevant, so its noise
# never shrank -> the flailing head); raw board_wobble 0.4-0.6 riding; raw action_rate 11-19.
# head_steady: -0.005 = -0.1..-0.3 /s now (shrinks the useless head noise), x2 at +500, x4 at +1500 iterations.
HEAD_STEADY_STAGES = [{"step": 0, "weight": -0.005}, {"step": REV_STEP0 + 500 * 24, "weight": -0.01},
                      {"step": REV_STEP0 + 1500 * 24, "weight": -0.02}]
# board_wobble: ~-0.05 /s of calm riding, ~-0.4 per slammed manual set-down (|w| ~4-5 rad/s for 0.1 s)
BOARD_WOBBLE_STAGES = [{"step": 0, "weight": -0.1}, {"step": REV_STEP0 + 1000 * 24, "weight": -0.2}]
# push_stroke: the s1e tapping (4.5 touches/s, q 0.08) pays -4.2 /s at -1.0 against a PUSH stack of ~+2 /s; start at
# -0.6 so the first iterations of the resume are not a "fall early" incentive, full price after 400 iterations
# rev1 (review): -1.0 at +400 would make a fall (-10) as cheap as ~2 s of tapping PUSH; raise only late, to -0.8
PUSH_STROKE_STAGES = [{"step": 0, "weight": -0.6}, {"step": REV_STEP0 + 1200 * 24, "weight": -0.8}]


def _foot_sensor(name: str, geom: str) -> ContactSensorCfg:
    return ContactSensorCfg(name=name, primary=ContactMatch(mode="geom", pattern=geom, entity="robot"),
                            secondary=ContactMatch(mode="body", pattern="terrain"), fields=("found", "force"),
                            reduce="netforce", num_slots=1, track_air_time=True)


def make_microduck_skate_sequence_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_skate_trick_env_cfg(play=play, trick_probs=dict(ride=1.0),
                                             board_cfg=MICRODUCK_SKATEBOARD_SEQUENCE_CFG, stance_key=ACTION_STANCE,
                                             stance_path=sq.SEQ_STANCE_JSON, axle_x=sq.BOARD["axle_x"],
                                             dir_mode="plus")
    cfg.episode_length_s = EPISODE_LENGTH_S

    # ── terrain: one half-pipe tile; every env (= its own Warp world) sits at the pipe's flat-bottom centre ──
    P = sq.PIPE
    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
        size=TILE_SIZE, border_width=0.0, num_rows=1, num_cols=1, curriculum=False, add_lights=False,
        sub_terrains={"hp": halfpipe.HalfPipeTerrainCfg(proportion=1.0, h=P["h"], R=P["R"], L=P["L"], W=P["W"],
                                                        dphi_deg=P["dphi_deg"])})
    cfg.scene.sensors = tuple(cfg.scene.sensors) + (
        _foot_sensor("left_foot_floor", r"^left_foot_collision$"),
        _foot_sensor("right_foot_floor", r"^right_foot_collision$"))

    # ── actions: centred on the tail stance (as the forward trick policy), scale 1.0 ──
    cfg.actions["joint_pos"].scale = ACTION_SCALE
    # centre on the forward trick policy's tail stance (the warm start's action centre: identical targets at it 0; the
    # sequence board's own "tail" differs by up to 0.14 rad in hip pitch / ankle)
    cfg.actions["joint_pos"].stance_path = ACTION_CENTER_JSON

    # ── observations: trick block in the sequence's phase convention + the 13D sequence block LAST ──
    for grp in ("actor", "critic"):
        terms = cfg.observations[grp].terms
        terms["trick"] = ObservationTermCfg(func=sq.seq_trick_obs)
        terms["seq"] = ObservationTermCfg(func=sq.seq_obs)

    # ── events: the sequence spawn replaces the trick spawn ──
    cfg.events.pop("reset_trick")
    cfg.events["reset_sequence"] = EventTermCfg(
        func=sq.reset_sequence_state, mode="reset",
        params={"spawn_probs": dict(SPAWN_PROBS), "spawn_stances": {k: list(v) for k, v in SPAWN_STANCES.items()},
                "stance_path": sq.SEQ_STANCE_JSON, "wall_frac": (0.05, 0.2),
                "ramp_chain_prob": 0.5, "ramp_chain_deg": (20.0, 55.0),
                "step_blend": ("glide_step", (0.3, 0.9), 0.6)})

    # ── rewards (per second; mjlab multiplies by dt). Reward-mass audit per phase: ──
    # Kept from the trick base: upright 1, on_deck 0.5 (seq_on_deck below), fall -10 per event, body_on_board -1,
    # head_pose -0.5, dof_pos_limits, action_rate_l2 (re-staged below), ride 3; manual 5 -> 6 and fall rescaled below.
    R = RewardTermCfg
    for k in ("stop", "stop_tail", "kickturn", "kickturn_done", "tictac", "nose_lift", "manual_strike"):
        cfg.rewards.pop(k)
    cfg.rewards["on_deck"] = R(func=sq.seq_on_deck, weight=0.5)
    cfg.rewards["fall"] = R(func=sq.seq_fall, weight=-10.0)        # REV: a fall in MANUAL costs 6 (manual_scale 0.6)
    cfg.rewards["manual"].weight = 6.0                              # REV 5 -> 6: MANUAL pay vs the risk of attempting
    cfg.rewards["manual"].params = {"lo_deg": 5.0, "hi_deg": 25.0}   # rev1: a 3-4 deg hover paid nearly the full band
    # MANUAL: the demo band (8-15 deg) and a held pitch replace the trick env's 0-6-15-28 trapezoid (sq.manual_quality)
    cfg.rewards["manual_shape"] = R(func=sq.manual_quality, weight=2.0)
    cfg.rewards.update({
        # PUSH
        "push_progress": R(func=sq.push_progress, weight=10.0),
        "push_time": R(func=sq.phase_time, weight=-1.0, params={"phase": sq.PUSH}),
        "push_speed": R(func=sq.push_speed, weight=2.0),         # REV 3 -> 2: less premium on early speed = cadence
        "push_stroke": R(func=sq.push_stroke, weight=PUSH_STROKE_STAGES[0]["weight"]),  # REV: price of a q = 0 stroke
        "push_step_on": R(func=sq.push_step_on, weight=6.0),   # rev1c 2 -> 6 (PUSH ends after 0.12 s: not farmable)
        "push_step_reach": R(func=sq.push_step_reach, weight=20.0),  # REV: always-on potential, <= 20 x 0.075 = 1.5
        # RIDE (settle / exit / approach)
        "ride_wheels": R(func=sq.ride_wheels_down, weight=1.0),
        "tail_strike": R(func=sq.seq_tail_strike, weight=-1.0),
        "board_wobble": R(func=sq.board_wobble, weight=BOARD_WOBBLE_STAGES[0]["weight"]),
        # RAMP
        "ramp_lateral": R(func=sq.ramp_lateral, weight=3.0),
        "ramp_apex": R(func=sq.ramp_apex, weight=3.0),
        # all phases: posture (head level, facing along the deck) and the staged head-steadiness tax
        "posture": R(func=sq.posture, weight=0.75),
        "head_steady": R(func=sq.head_steady, weight=HEAD_STEADY_STAGES[0]["weight"]),
    })

    # ── curriculum: smoothness taxes staged after the resume point (REV_STEP0); replaces the trick base's
    # action_rate ramp, whose -0.1 stage would have fired at iteration 2000 = 250 iterations into the resume ──
    cfg.rewards["action_rate_l2"].weight = ACTION_RATE_STAGES[0]["weight"]
    cfg.curriculum = {
        "action_rate": CurriculumTermCfg(func=microduck_mdp.reward_weight,
                                         params={"reward_name": "action_rate_l2", "weight_stages": ACTION_RATE_STAGES}),
        "head_steady": CurriculumTermCfg(func=microduck_mdp.reward_weight,
                                         params={"reward_name": "head_steady", "weight_stages": HEAD_STEADY_STAGES}),
        "board_wobble": CurriculumTermCfg(func=microduck_mdp.reward_weight,
                                          params={"reward_name": "board_wobble", "weight_stages": BOARD_WOBBLE_STAGES}),
        "push_stroke": CurriculumTermCfg(func=microduck_mdp.reward_weight,
                                         params={"reward_name": "push_stroke", "weight_stages": PUSH_STROKE_STAGES}),
    }

    # ── metrics (per-step means; read as ratios, see skate_sequence.py "metrics") ──
    for k, f in (("seq_push", sq.m_push), ("seq_liftoff", sq.m_liftoff), ("seq_stroke_q", sq.m_stroke_q),
                 ("seq_stepon", sq.m_stepon), ("seq_manual", sq.m_manual), ("seq_manual_band", sq.m_manual_band),
                 ("seq_ramp", sq.m_ramp), ("seq_apex", sq.m_apex)):
        cfg.metrics[k] = MetricsTermCfg(func=f)

    # ── terminations: push foot may touch the floor in PUSH; the board must not climb to the coping ──
    T = TerminationTermCfg
    cfg.terminations["foot_on_floor"] = T(func=sq.seq_foot_on_floor)
    cfg.terminations["board_lost"] = T(func=sq.seq_board_lost)
    cfg.terminations["over_coping"] = T(func=sq.over_coping)
    cfg.terminations["riding_tilt"] = T(func=sq.riding_tilt_guard)    # REV: 50 deg guard, RIDE/RAMP after 1 s (fell_over 60)
    cfg.terminations["past_pipe_end"] = T(func=sq.past_pipe_end, time_out=True)
    cfg.terminations["nan_state"] = T(func=microduck_mdp.skate_state_is_nan, params={"sensor_names": SEQ_SENSORS})

    # real2 (2026-11-21, default off): servo heat cost for every sequence task built on this cfg (push / manual / ramp
    # specialists and the chain task). Plain reward term (no curriculum entry: the ramp task replaces cfg.curriculum);
    # optional start iteration via MICRODUCK_SKATE_HEAT_IT0.
    if SKATE_HEAT:
        cfg.rewards["skate_heat"] = R(func=microduck_mdp.skate_heat_cost, weight=SKATE_HEAT_W,
                                      params={"p_cont": SKATE_HEAT_P_CONT, "cap_w": SKATE_HEAT_CAP_W,
                                              "it0": SKATE_HEAT_IT0})
        cfg.metrics["seq_heat_w"] = MetricsTermCfg(func=microduck_mdp.skate_heat_cost,   # uncapped, ungated (W)
                                                   params={"p_cont": SKATE_HEAT_P_CONT, "cap_w": None, "it0": 0})
    if SKATE_MASS_RANGE is not None:
        cfg.events.pop("randomize_mass_inertia", None)   # trunk-only startup DR; the whole-robot scale supersedes it
        cfg.events["skate_mass_scale"] = EventTermCfg(func=microduck_mdp.skate_robot_mass_scale, mode="reset",
                                                      params={"scale_range": SKATE_MASS_RANGE})
    return cfg


def make_microduck_skate_sequence_ppo_runner_cfg():
    rl = make_microduck_skate_trick_ppo_runner_cfg("skate_sequence")
    rl.max_iterations = 4000
    return rl
