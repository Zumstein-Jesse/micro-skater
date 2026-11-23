# micro-skater: part of the author's skateboard add-on for microduck_rl. Apache-2.0 (see LICENSE).
"""Skateboard SEQUENCE on the sequence board (2026-11-19). MDP terms for microduck_skate_sequence_env_cfg.py.

One continuous run, the author's brief: "starts still with one foot on the board, one on the floor, pumps a few times to
gain speed, enters a manual and exits the manual, then with speed turns into a small half-pipe and goes up and down it".

Design (stance_poses_seq.json: LEFT = board foot, RIGHT = push foot, forward stance):
ONE policy, a phase one-hot, a state-gated SCHEDULER shared by training and play, reverse-curriculum spawns at every
phase).

The phase lives in the trick state (skate_trick._state()["trick"]) so the trick rewards (ride / manual) work unchanged:
  0 RIDE  (trick slot)   settle after pushing, manual exit, approach (curvature from the scheduler's pure pursuit)
  4 MANUAL (trick slot)
  5 PUSH  (new)          kick-push from rest / mid-stroke until the target speed, push foot back on the deck
  6 RAMP  (new)          in the half-pipe: ride up and down the walls
Scheduler stages (st["stage"]): PUSH -> SETTLE (RIDE) -> MANUAL -> EXIT (RIDE, until 4 wheels down and inside the pipe)
-> RAMP (to the end). Every spawn type enters the scheduler at its own stage, so downstream transitions always get data.

World layout (the 1x1 half-pipe tile; env origin = centre of the pipe's flat bottom, pipe axis = world y, walls at
x = +-(L/2 ...)): the chain starts on the floor at y = -(W/2 + start_gap) heading +y, rides into the open end of the
pipe, and the RAMP phase carves into the walls.

Costs are returned >= 0 (use NEGATIVE weights); per-event terms return 1/dt (mjlab multiplies every reward by dt).
"""

from __future__ import annotations
from micro_skater.paths import data_path as _data_path

import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import torch

from mjlab.entity import Entity
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, quat_mul, wrap_to_pi

from micro_skater.tasks import skate_trick as tk
from micro_skater.tasks.mdp import _quat_from_rpy_t, _skate_air_time, _yaw_from_quat

RIDE, MANUAL = tk.RIDE, tk.MANUAL
PUSH, RAMP = 5, 6
ST_PUSH, ST_SETTLE, ST_MANUAL, ST_EXIT, ST_RAMP = range(5)
STAGE_PHASE = (PUSH, RIDE, MANUAL, RIDE, RAMP)

SEQ_STANCE_JSON = str(_data_path("stance_poses_seq.json"))
SEQ_BOARD_XML = Path(__file__).resolve().parents[1] / "robot" / "microduck" / "skateboard_sequence.xml"


def board_geometry(xml: Path = SEQ_BOARD_XML) -> dict:
    """Rear axle |x|, axle z (deck frame), wheel radius from the board XML (fallbacks = the sequence board v1)."""
    out = dict(axle_x=0.045, axle_z=-0.014, wheel_r=0.008)
    try:
        for b in ET.parse(xml).iter("body"):
            if b.get("name") == "rear_truck":
                p = [float(v) for v in b.get("pos").split()]
                out["axle_x"], out["axle_z"] = abs(p[0]), p[2]
        for g in ET.parse(xml).iter("geom"):
            if g.get("name") == "rear_wheel_L_collision":
                out["wheel_r"] = float(g.get("size").split()[0])
    except (OSError, ET.ParseError):
        pass
    return out


BOARD = board_geometry()
DECK_Z = tk.DK_DECK_CENTER_Z           # deck origin height at rest on the flat (same on every double-kick derivative)

# Half-pipe (env origin = centre of the flat bottom; profile along x, constant along y). Sized from the
# feasibility probes: climbs of 3-6 cm at 0.8-1.0 m/s, the coping well above them.
PIPE = dict(h=0.12, R=0.40, L=1.0, W=6.0, deck=0.20, dphi_deg=4.0)
# pipe_h (2026-11-21, option a'): MICRODUCK_PIPE_H=<m> (e.g. 0.08) lowers the
# walls (same R / L / W). The terrain, rewards, spawns, terminations (over_coping: deck rise > h - 1 cm) and the render
# follow PIPE / st["pipe"] (the TRUE geometry). OBS_PIPE = the geometry every current policy trained on: seq_obs keeps
# its normalisation (deck height / OBS_PIPE h) and, off the pipe (|y| > W/2: the approach floor, where pipe_profile's
# x-only wall slope is a phantom), the trained phantom slope; inside the pipe the truck slopes come from the TRUE profile.
# Unset: PIPE == OBS_PIPE and seq_obs runs the original code path (bit-identical).
OBS_PIPE = dict(h=0.12, R=0.40, L=1.0)
if os.environ.get("MICRODUCK_PIPE_H"):
    PIPE["h"] = float(os.environ["MICRODUCK_PIPE_H"])

SEQ_DEFAULTS = dict(
    # REV (2026-11-19): lower bound 0.7 -> 0.8 (more speed into the pipe; s1e taps to 1.1-1.2 m/s, so
    # 0.7-0.9 exit thresholds are reachable with fewer, longer strokes). m/s push target (scheduler exit: v >= v_t - v_gap)
    v_t=(0.8, 1.0),
    v_gap=0.1,
    push_max_s=8.0,             # REV 7 -> 8: PUSH may exit below speed after this (still needs the step-on)
    push_late_s=4.5,            # REV 3 -> 4.5: 4-5 deliberate 1.3 Hz strokes take ~3.5 s; a 3 s deadline paid cadence
    both_on_s=0.12,             # both soles on the deck for this long ends PUSH (rev1c 0.3 -> 0.12: s2c reached 0.1-0.2 s)
    settle_s=(0.4, 1.0),
    manual_s=(1.2, 2.0),        # REV: chain manual (the demo's "held 1-2 s")
    manual_spawn_s=(2.0, 3.5),  # REV: MANUAL spawns hold longer (s1e: MANUAL was ~1.4 % of all experience)
    exit_min_s=(0.5, 1.0),
    wheels_down_s=0.3,
    start_gap=5.0,              # REV 7 -> 5 m of floor before the pipe's open end: s1e entered the pipe at 0.4-0.7 m/s
                                # after the long coast (0.06-0.09 m/s^2); 2 m less coasting is ~+0.15-0.2 m/s at entry
    start_off=0.6,              # REV 0.8 -> 0.6 start line |x| (still outside the 0.5 m opening: the run must turn in;
                                # with the shorter floor run the EXIT pursuit has ~1.5 m to clear the wall end face)
    flat_yaw_noise=0.08,        # REV 0.15 -> 0.08 rad heading noise of flat spawns (0.15 drifted +-0.5 m over the run)
    # push-stroke quality (push_stroke): air time before the touchdown and backward sweep of the planted sole (deck x)
    stroke_air=(0.10, 0.45),    # s: q_air ramps 0 -> 1 (0.45 s swing + ~0.25 s contact = 1.3-1.4 Hz cadence)
    stroke_sweep=(0.01, 0.06),  # m: q_sweep ramps 0 -> 1 (feasibility stroke: x +0.005 -> -0.075)
    target_x=0.35,              # pursuit target (|x|, on the far side of the axis = toward the wall it carves into)
    target_in=1.0,              # m inside the open end
    kappa_max=0.4,              # 1/m pursuit cap (measured carve ~0.25-0.5 1/m; saturation makes ride_track unpayable)
    stroke_window_s=0.6,        # push credit needs the push foot on the floor within this window
    apex_min_z=0.005,
    pipe_end_margin=0.3,        # RAMP episodes end (time_out, not a fall) this close to a pipe end
    # specialists (2026-11-20): the scheduler never advances past this stage (ST_RAMP = the full chain, no cap)
    last_stage=ST_RAMP,
    # step command (2026-11-20, push specialist): once PUSH is at speed (or past push_max_s) the phase obs switches to
    # RIDE while the stage stays PUSH until the step-on completes - the scheduler says "step on now", and the policy's
    # RIDE behaviour (both soles on the deck) pulls the push foot onto the deck. False = the old behaviour.
    step_cmd=False,
    # manual specialist (2026-11-20): how MANUAL ends. "timer" (default, the chain and push
    # tasks): after T_manual. "hold": when the leaky in-band counter band_t reaches T_hold (per env, from manual_hold_s);
    # band_t += dt in manual_band_deg with the front wheels up, the back wheels down, the tail clear and both soles on,
    # else -= manual_leak * dt (floored at 0). setdown_s: the EXIT set-down window of setdown_rate / m_setdown.
    manual_end="timer",
    manual_hold_s=(0.8, 1.2),
    manual_band_deg=(8.0, 15.0),
    manual_leak=1.0,
    setdown_s=0.6,
    # p9 (2026-11-20, push specialist; None = off, every other task): slewed push target. PUSH counts as "at speed" (step
    # command, push_done) only once the ramp v_ramp = spawn speed + push_ramp * time in PUSH (m/s per s) has also reached
    # v_tgt - v_gap, and push_progress / push_speed pay min(speed, v_ramp): being ahead of the ramp pays nothing and no
    # longer brings the step-on (the discounted terminal value) sooner, so violent early kicks stop paying.
    push_ramp=None,
    # p9b (2026-11-20, push specialist; None = off): hand-off stance. With stance_tol = (left sole m, right sole m, CoM
    # height m, trunk tilt along deck x deg, across deck y deg), SETTLE counts "in stance" (st["stance_t"], seconds,
    # reset when out) while every error is inside its tolerance (HANDOFF_STANCE targets) with both soles on the deck;
    # seq_success(stance=True) then also needs stance_t >= stance_hold_s.
    stance_tol=None,
    stance_hold_s=0.2,
    # p10 (2026-11-20, push specialist; False = off, every other task): per-stroke touchdown interval / contact time /
    # planted-sole sweep bookkeeping (_track_p10) for push_cadence / push_stroke_shape and the p10 metrics
    p10_track=False,
    # p12 (2026-11-20, push specialist; False = off, every other task): running means (EMA) of the head joints, trunk
    # height and trunk pitch, and the step-on window state (_track_p12) for the p12 posture costs
    p12_track=False,
    p12_tau=0.5,                # s, EMA time constant (~1 stroke)
    p12_win_s=0.4,              # s into SETTLE that the step-on window still covers
    # p13 (2026-11-20, push specialist; False = off, every other task): the swing-foot hover clock (_track_p13): per
    # swing (push foot off the floor and the deck in PUSH before the command), the time the sole has moved slower than
    # p13_v_still relative to the deck; p13_hov = still now and that time past p13_hover_s
    p13_track=False,
    p13_v_still=0.12,           # m/s, sole speed relative to the deck (3D, finite difference in the deck frame)
    p13_hover_s=0.25,           # s of still time per swing that are free
    # chainfix (2026-11-20, chain playback only; None = off, every task): EXIT -> RAMP also needs the board in front of /
    # inside the pipe's opening, |x| < ramp_gate_x (in_pipe checks y only: a chain that drifted 6 m beside the pipe
    # "entered" RAMP on the floor next to it and rode its length at rise 0)
    ramp_gate_x=None,
    # r4rev (2026-11-20, ramp specialist + chain playback; None = off): EXIT pursuit gate point before the open end,
    # (tx, y_in, switch, kappa_max) - see approach_kappa
    gate_pt=None,
)

# p9b hand-off target = the MANUAL specialist's own riding state 0.5 s after its glide spawn (medians of 64
# glide spawns): deck-frame soles
# and whole-robot CoM height, trunk z axis tilt in the deck frame (asin of its deck x / y components, deg), joints (rad).
# The manual's own spread (p90 of |error|): soles 13 / 11 mm, CoM 4 mm, tilt 2.6 / 3.2 deg, joints L2 0.64 rad.
HANDOFF_STANCE = dict(
    lsole=(-0.043, 0.042), rsole=(-0.035, -0.050), com_z=0.125, tilt_x=8.9, tilt_y=1.6,
    joints=dict(left_hip_yaw=-0.438, left_hip_roll=-0.131, left_hip_pitch=-0.254, left_knee=0.747, left_ankle=0.877,
                neck_pitch=0.409, head_pitch=0.281, head_yaw=-0.144, head_roll=0.0, right_hip_yaw=-0.464,
                right_hip_roll=-0.011, right_hip_pitch=0.068, right_knee=-0.589, right_ankle=-0.486),
)


def _st(env) -> dict:
    st = tk._state(env)
    if "stage" not in st:
        n, dev = env.num_envs, env.device
        z = lambda: torch.zeros(n, device=dev)
        st.update(
            stage=torch.zeros(n, dtype=torch.long, device=dev), t_stage=z(), T_settle=z(), T_manual=z(), T_exit=z(),
            v_tgt=z(), both_on=z(), wheels_on=z(), entry_s=z(), phi=z(), apex_side=z(), vz_prev=z(), step_d=z(),
            apex_pay=z(), stage_t=z(), seq_step=-1, seq_cfg=dict(SEQ_DEFAULTS), pipe=dict(PIPE), spawn=torch.zeros(n, dtype=torch.long, device=dev),
            # REV: push-stroke tracking (seq_update -> _track_strokes), step-on potential origin, events for metrics
            rf_prev=torch.zeros(n, dtype=torch.bool, device=dev), td_x=z(), td_air=z(),
            td_init=torch.zeros(n, dtype=torch.bool, device=dev), rx_prev=z(),
            lift_evt=torch.zeros(n, dtype=torch.bool, device=dev), lift_q=z(), step_d0=z(),
            stepon_evt=torch.zeros(n, dtype=torch.bool, device=dev), apex_evt=torch.zeros(n, dtype=torch.bool, device=dev),
            # manual specialist: hold counter + its target, obs clock offset, MANUAL -> EXIT by hold, set-down peak
            band_t=z(), T_hold=z(), t_off=z(), manual_evt=torch.zeros(n, dtype=torch.bool, device=dev), sd_peak=z(),
            sd_evt=torch.zeros(n, dtype=torch.bool, device=dev),
        )
    return st


def _robot(env) -> Entity:
    return env.scene["robot"]


def _origin(env) -> torch.Tensor:
    return env.scene.env_origins


def pipe_xyz(env) -> torch.Tensor:
    """(B, 3) board origin relative to the env origin (pipe frame = world axes)."""
    return tk._board(env).data.root_link_pos_w - _origin(env)


def pipe_profile(x: torch.Tensor, pipe: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Surface height z(x) and slope dz/dx of the half-pipe (exact arcs; the flat platform beyond)."""
    h, R, L = pipe["h"], pipe["R"], pipe["L"]
    phi_max = math.acos(1.0 - h / R)
    x_end = L / 2 + R * math.sin(phi_max)
    ax = x.abs()
    u = (ax - L / 2).clamp(min=0.0, max=x_end - L / 2)
    z = R - torch.sqrt((R * R - u * u).clamp(min=1e-9))
    s = u / torch.sqrt((R * R - u * u).clamp(min=1e-9))
    z = torch.where(ax > x_end, torch.full_like(z, h), z)
    s = torch.where(ax > x_end, torch.zeros_like(s), s)
    return z, s * torch.sign(x)


def in_pipe(env, margin: float = 0.2) -> torch.Tensor:
    p = pipe_xyz(env)
    W = _st(env)["pipe"]["W"]
    return (p[:, 1] > -W / 2 + margin) & (p[:, 1] < W / 2 - margin)


def _feet_floor(env) -> tuple[torch.Tensor, torch.Tensor]:
    return tk._contact_any(env, "left_foot_floor"), tk._contact_any(env, "right_foot_floor")


def _push_foot_recent(env, window: float) -> torch.Tensor:
    return tk._contact_any(env, "right_foot_floor") | (_skate_air_time(env, "right_foot_floor") < window)


def _wheels_all_down(env) -> torch.Tensor:
    return tk._contact_all(env, "front_wheels_floor") & tk._contact_all(env, "rear_wheels_floor")


def board_v_fwd(env) -> torch.Tensor:
    return tk.board_v_deck(env)[:, 0]


def robot_v_fwd(env) -> torch.Tensor:
    """Trunk velocity along the deck +x axis (world horizontal)."""
    q = tk._board(env).data.root_link_quat_w
    return torch.nan_to_num(quat_apply_inverse(q, _robot(env).data.root_link_lin_vel_w)[:, 0])


# ── scheduler ─────────────────────────────────────────────────────────────────────────────────────────────────────────


def seq_update(env) -> dict:
    """Once per env step: advance the scheduler on state gates, set the phase and the RIDE curvature (pure pursuit)."""
    st = tk.trick_update(env)
    st = _st(env)
    if st["seq_step"] == env.common_step_counter:
        return st
    st["seq_step"] = env.common_step_counter
    cfg = st["seq_cfg"]
    dt = env.step_dt
    t = tk._t(env)
    ts = t - st["t_stage"]
    stage = st["stage"]
    v = board_v_fwd(env)
    on = tk.soles_on_deck(env)
    lf, rf = _feet_floor(env)
    st["both_on"] = torch.where(on & ~rf & ~lf, st["both_on"] + dt, torch.zeros_like(st["both_on"]))
    down = _wheels_all_down(env)
    st["wheels_on"] = torch.where(down, st["wheels_on"] + dt, torch.zeros_like(st["wheels_on"]))
    _track_strokes(env, st, rf, t, stage)
    if cfg.get("p10_track"):                         # p10 stroke timing (default False: off)
        _track_p10(env, st, rf, t, stage)
    fast = v >= st["v_tgt"] - cfg["v_gap"]
    fast_cmd = fast
    if cfg.get("push_ramp"):                         # p9 slewed push target (default None: unchanged)
        st["v_ramp"] = st["v0"].clamp(min=0.0) + float(cfg["push_ramp"]) * st["stage_t"]
        fast = fast & (st["v_ramp"] >= st["v_tgt"] - cfg["v_gap"])
        # p9rev: the ramp makes the command time exogenous to the stroke phase (p8's speed crossing fell inside a
        # floor contact); a command latched mid-swing made the next landing a floor one (seq_cmd_landing: zero-shot
        # model_1200 from start 92 % at 0.8 -> 46 % at 0.55, 26/48 cmd_landing). Latch it only during a push-foot floor
        # contact (that contact is let finish), except where the ramp was met at the spawn (step_on / step_mid /
        # stroke_air / fast push spawns: immediate, as before).
        fast_cmd = fast & (rf | (st["v0"] >= st["v_tgt"] - cfg["v_gap"]))
    if "p17" in st:   # p17 (default off, see above): latch only on a contact whose min(board, trunk) speed is at the
        pv17 = _push_v(env, st)    # threshold AND has gained >= P17_GAIN_MIN since its touchdown (review p17 B: a
        fast_cmd = torch.where(st["p17"], fast_cmd & (pv17 >= st["v_tgt"] - cfg["v_gap"])      # board-only latch on
                               & ((pv17 - st["p17_tdv"]) >= P17_GAIN_MIN)                     # a lean-sped board;
                               & st["p17_rf"], fast_cmd)    # p17_rf: the contact is already logged, p17_tdv is ITS touchdown)
    if cfg.get("stance_tol") is not None:            # p9b hand-off stance tracker (default None: off)
        _track_stance(env, st, stage, dt)

    adv = torch.zeros_like(stage, dtype=torch.bool)
    push_done = (stage == ST_PUSH) & ((fast & (st["both_on"] >= cfg["both_on_s"]))
                                      | (ts > cfg["push_max_s"]) & (st["both_on"] >= cfg["both_on_s"]))
    if "p17" in st:   # p17 (push specialist; the state exists only with MICRODUCK_SEQPUSH_P17=1): no step-on without the
        push_done = push_done & (~st["p17"] | st["p17_ok"])     # legal stroke (zero-shot p16: 12 % reached SETTLE unpushed)
    settle_done = (stage == ST_SETTLE) & (ts >= st["T_settle"])
    hold = cfg.get("manual_end", "timer") == "hold"
    if hold:                                         # manual specialist: MANUAL ends on the held band (_track_hold)
        _track_hold(env, st, stage, dt)
        manual_done = (stage == ST_MANUAL) & (st["band_t"] >= st["T_hold"])
    else:
        manual_done = (stage == ST_MANUAL) & (ts >= st["T_manual"])
    exit_done = (stage == ST_EXIT) & (ts >= st["T_exit"]) & (st["wheels_on"] >= cfg["wheels_down_s"]) & in_pipe(env, 0.4)
    if cfg.get("ramp_gate_x") is not None:          # chainfix: in front of the opening too (default None: off)
        exit_done = exit_done & (pipe_xyz(env)[:, 0].abs() < float(cfg["ramp_gate_x"]))
    adv = push_done | settle_done | manual_done | exit_done
    adv = adv & (stage < cfg["last_stage"])          # specialist cap (default ST_RAMP: no change)
    push_done = push_done & adv
    st["stepon_evt"] = push_done & adv
    # time in the current stage, counted in env steps since the reset / last advance: = t - t_stage, except in a run's
    # first episode (rsl_rl init_at_random_ep_len randomizes episode_length_buf while t_stage stays 0). Read only by the
    # specialist terminations (seq_overtime / seq_success).
    st["stage_t"] = torch.where(adv, torch.zeros_like(st["stage_t"]), st["stage_t"] + dt)
    if bool(adv.any()):
        st["stage"] = torch.where(adv, stage + 1, stage)
        st["t_stage"] = torch.where(adv, t, st["t_stage"])
        st["phi"] = torch.where(push_done, torch.zeros_like(st["phi"]), st["phi"])
    st["trick"] = torch.tensor(STAGE_PHASE, device=env.device)[st["stage"]]
    if "cmd" not in st:
        st["cmd"] = torch.zeros_like(st["stage"], dtype=torch.bool)
    st["obs_ph"] = st["trick"]
    if cfg.get("step_cmd", False):
        # review (stepcmd): only the OBSERVED phase changes; rewards/terminations keep the PUSH stage pricing (relabelling
        # st["trick"] paid ride + wheels ~+5.5 /s for hovering under the command and dropped push_time / refunds)
        go = (st["stage"] == ST_PUSH) & (fast_cmd | (st["stage_t"] > cfg["push_max_s"]))
        st["cmd"] = (st["cmd"] | go) & (st["stage"] == ST_PUSH)
        st["obs_ph"] = torch.where(st["cmd"], torch.full_like(st["trick"], RIDE), st["trick"])
    if "cmd_t" not in st:
        st["cmd_t"] = torch.zeros_like(st["v_tgt"])
    st["cmd_t"] = torch.where(st["cmd"], st["cmd_t"] + dt, torch.zeros_like(st["cmd_t"]))
    st["manual_evt"] = manual_done & adv
    if hold:
        _track_setdown(env, st, adv)
    if cfg.get("p12_track"):                         # p12 posture trackers (default False: off), after cmd / stage
        _track_p12(env, st)
    if cfg.get("p13_track"):                         # p13 hover clock (default False: off), after cmd / stage
        _track_p13(env, st)

    # EXIT/approach curvature: pure pursuit onto the approach line (left turn = +kappa); SETTLE rides straight.
    k = approach_kappa(env, st)
    st["kappa"] = torch.where(st["stage"] == ST_EXIT, k, torch.zeros_like(k))
    st["kappa_next"][:] = 1.0e9                      # no random resampling in the sequence
    return st


def _track_strokes(env, st, rf: torch.Tensor, t: torch.Tensor, stage: torch.Tensor) -> None:
    """Push-foot stroke bookkeeping, once per env step (state lives here, not in a reward: mjlab skips weight-0 terms).

    Touchdown (floor contact edge): remember the push sole's deck-frame x and the air time that preceded it (sensor
    last_air_time, sub-step exact). A contact that starts in the first 0.1 s of the episode is the spawn stance (the
    "starts still, push foot on the floor" pose), not a swing: its air quality is 1.
    Lift-off: evaluate the stroke that just ended, q = q_air * q_sweep (SEQ_DEFAULTS stroke_air / stroke_sweep), with
    sweep = backward travel of the planted sole relative to the deck (the visible leg sweep). A contact shorter than one
    env step (both edges inside one step) is a tap: sweep 0, q = 0. lift_evt is raised only in the PUSH stage."""
    cfg = st["seq_cfg"]
    sens = env.scene.sensors["right_foot_floor"].data
    air_now = sens.current_air_time
    air_now = air_now.min(dim=-1).values if air_now.dim() > 1 else air_now
    last_air = sens.last_air_time
    last_air = last_air.max(dim=-1).values if last_air.dim() > 1 else last_air
    x = torch.nan_to_num(_right_sole_deck_xy(env)[:, 0])
    prev = st["rf_prev"]
    td = rf & ~prev
    tap = (~rf) & (~prev) & (air_now > 0.0) & (air_now < env.step_dt + 1e-6)
    tap &= t > env.step_dt + 1e-6          # review: the reset air timer makes step 1 look like a tap (fake bad stroke)
    lo = ((~rf) & prev) | tap
    st["td_x"] = torch.where(td, x, st["td_x"])
    st["td_air"] = torch.where(td, last_air, st["td_air"])
    st["td_init"] = torch.where(td, t < 0.1, st["td_init"])
    a_lo, a_hi = cfg["stroke_air"]
    s_lo, s_hi = cfg["stroke_sweep"]
    air = torch.where(prev, st["td_air"], last_air)
    q_air = torch.where(prev & st["td_init"], torch.ones_like(air), ((air - a_lo) / (a_hi - a_lo)).clamp(0.0, 1.0))
    sweep = torch.where(prev, st["td_x"] - st["rx_prev"], torch.zeros_like(x))
    q_sw = ((sweep - s_lo) / (s_hi - s_lo)).clamp(0.0, 1.0)
    st["lift_evt"] = lo & (stage == ST_PUSH)
    st["lift_q"] = torch.where(st["lift_evt"], q_air * q_sw, torch.zeros_like(x))
    st["rf_prev"] = rf
    st["rx_prev"] = torch.where(rf, x, st["rx_prev"])


def _track_p10(env, st, rf: torch.Tensor, t: torch.Tensor, stage: torch.Tensor) -> None:
    """p10 (push specialist, SEQ_DEFAULTS p10_track): push-foot floor stroke timing, once per env step, own state (the
    _track_strokes fields are left alone). A new episode is detected by the episode clock not advancing (t <= last t),
    so no reset hook is needed. Touchdown = floor-contact edge or a sub-step tap (both edges inside one env step, as
    _track_strokes); lift-off = contact end or the tap. Events count only in PUSH before the step command
    (p10_td_evt / p10_lo_evt); the spawn contact (touchdown in the first 0.1 s) is never a touchdown event but starts
    the interval clock, its lift-off is a stroke. Fields: p10_gap (s since the previous touchdown, at p10_td_evt),
    p10_contact (s, frame-based, 0 for a tap), p10_sweep (m, backward travel of the planted sole in the deck frame),
    p10_td_v (board speed at the touchdown, for the reach-limited contact target), at p10_lo_evt."""
    if "p10_tdt" not in st:
        z = torch.zeros_like(st["v_tgt"])
        b = torch.zeros_like(st["v_tgt"], dtype=torch.bool)
        st.update(p10_tdt=torch.full_like(z, -1.0), p10_td_x=z.clone(), p10_td_v=z.clone(), p10_rx=z.clone(),
                  p10_t_last=torch.full_like(z, 1e9), p10_prev=b.clone(), p10_td_evt=b.clone(), p10_lo_evt=b.clone(),
                  p10_gap=z.clone(), p10_contact=z.clone(), p10_sweep=z.clone(), p10_lo_v=z.clone())
    fresh = t <= st["p10_t_last"] + 1e-6
    st["p10_tdt"] = torch.where(fresh, torch.full_like(t, -1.0), st["p10_tdt"])
    st["p10_prev"] = st["p10_prev"] & ~fresh
    st["p10_t_last"] = t.clone()
    sens = env.scene.sensors["right_foot_floor"].data
    air_now = sens.current_air_time
    air_now = air_now.min(dim=-1).values if air_now.dim() > 1 else air_now
    x = torch.nan_to_num(_right_sole_deck_xy(env)[:, 0])
    v = torch.nan_to_num(board_v_fwd(env))
    prev = st["p10_prev"]
    tap = (~rf) & (~prev) & (air_now > 0.0) & (air_now < env.step_dt + 1e-6) & (t > env.step_dt + 1e-6)
    td = (rf & ~prev) | tap
    lo = ((~rf) & prev) | tap
    cmd = st.get("cmd", torch.zeros_like(rf))
    live = (stage == ST_PUSH) & ~cmd
    init = t < 0.1
    st["p10_td_evt"] = td & live & ~init & (st["p10_tdt"] >= 0.0)
    st["p10_gap"] = torch.where(td, t - st["p10_tdt"], st["p10_gap"])
    # lift-off of the stroke that started at the previous touchdown (a tap: contact 0, sweep 0)
    st["p10_lo_evt"] = lo & live & (st["p10_tdt"] >= 0.0) | (tap & live)
    st["p10_contact"] = torch.where(tap, torch.zeros_like(t), torch.where(lo, t - st["p10_tdt"], st["p10_contact"]))
    st["p10_sweep"] = torch.where(tap, torch.zeros_like(x), torch.where(lo, st["p10_td_x"] - st["p10_rx"], st["p10_sweep"]))
    st["p10_lo_v"] = torch.where(tap, v, torch.where(lo, st["p10_td_v"], st["p10_lo_v"]))
    st["p10_tdt"] = torch.where(td, t, st["p10_tdt"])
    st["p10_td_x"] = torch.where(td, x, st["p10_td_x"])
    st["p10_td_v"] = torch.where(td, v, st["p10_td_v"])
    st["p10_rx"] = torch.where(rf, x, st["p10_rx"])
    st["p10_prev"] = rf


def _stance_err(env) -> dict:
    """p9b: errors against HANDOFF_STANCE (deck frame): left / right sole xy distance (m), |CoM height - target| (m),
    |trunk tilt - target| along deck x / across deck y (deg), joint L2 distance (rad). Cached per env step."""
    c = tk._tc(env)
    if c.get("stance_step") == env.common_step_counter and "stance_err" in c:
        return c["stance_err"]
    H = HANDOFF_STANCE
    dev = env.device
    robot, b = _robot(env), tk._board(env)
    if "stance_jt" not in c:
        jn = list(robot.joint_names)
        ids = [jn.index(j) for j in H["joints"]]
        c["stance_jt"] = (torch.tensor(ids, device=dev), torch.tensor(list(H["joints"].values()), device=dev))
    ids, qt = c["stance_jt"]
    ls, rs = _left_sole_deck(env)[:, :2], _right_sole_deck_xy(env)[:, :2]
    rz = quat_apply(robot.data.root_link_quat_w, torch.tensor([0.0, 0.0, 1.0], device=dev).expand(env.num_envs, 3))
    rz_d = quat_apply_inverse(b.data.root_link_quat_w, rz)
    tx = torch.rad2deg(torch.asin(rz_d[:, 0].clamp(-1, 1)))
    ty = torch.rad2deg(torch.asin(rz_d[:, 1].clamp(-1, 1)))
    e = dict(dl=(ls - torch.tensor(H["lsole"], device=dev)).norm(dim=1),
             dr=(rs - torch.tensor(H["rsole"], device=dev)).norm(dim=1),
             dz=(_robot_com_deck(env)[:, 2] - H["com_z"]).abs(), tx=(tx - H["tilt_x"]).abs(), ty=(ty - H["tilt_y"]).abs(),
             dj=(robot.data.joint_pos[:, ids] - qt).norm(dim=1))
    e = {k: torch.nan_to_num(v, nan=1.0e3) for k, v in e.items()}
    c["stance_err"], c["stance_step"] = e, env.common_step_counter
    return e


def _stance_ok(env, tol) -> torch.Tensor:
    e = _stance_err(env)
    return ((e["dl"] < tol[0]) & (e["dr"] < tol[1]) & (e["dz"] < tol[2]) & (e["tx"] < tol[3]) & (e["ty"] < tol[4])
            & tk.soles_on_deck(env))


def _track_stance(env, st, stage: torch.Tensor, dt: float) -> None:
    """p9b (seq_update, before the advance; only with stance_tol set): stance_t += dt in SETTLE while _stance_ok, else 0."""
    if "stance_t" not in st:
        st["stance_t"] = torch.zeros_like(st["v_tgt"])
    ok = (stage == ST_SETTLE) & _stance_ok(env, st["seq_cfg"]["stance_tol"])
    st["stance_t"] = torch.where(ok, st["stance_t"] + dt, torch.zeros_like(st["stance_t"]))


def _in_manual_band(env, band_deg: tuple) -> torch.Tensor:
    """Front-up pitch inside band_deg, front wheels up, back wheels down, the tail clear, both soles on the deck."""
    th = torch.rad2deg(tk.front_up_pitch(env))
    front_down, back_down = tk._front_back_wheels(env)
    return ((~front_down) & back_down & ~tk._back_end_floor(env) & tk.soles_on_deck(env)
            & (th > band_deg[0]) & (th < band_deg[1]))


def _track_hold(env, st, stage: torch.Tensor, dt: float) -> None:
    """manual_end="hold" (seq_update, before the advance): leaky in-band counter band_t, only in MANUAL."""
    cfg = st["seq_cfg"]
    inb = _in_manual_band(env, cfg["manual_band_deg"])
    up = torch.where(inb, st["band_t"] + dt, (st["band_t"] - cfg["manual_leak"] * dt).clamp(min=0.0))
    st["band_t"] = torch.where(stage == ST_MANUAL, up, st["band_t"])


def _track_setdown(env, st, adv: torch.Tensor) -> None:
    """manual_end="hold" (seq_update, after the advance): peak |deck pitch rate| in the first min(setdown_s,
    exit_min_s[0]) s of EXIT, emitted once (sd_evt) when that window closes (m_setdown / m_setdown_n = mean peak). The
    window is capped at the success minimum so a fast success does not skip the event."""
    cfg = st["seq_cfg"]
    dt = env.step_dt
    win = min(cfg["setdown_s"], cfg["exit_min_s"][0])
    w = quat_apply_inverse(tk._board(env).data.root_link_quat_w, tk._board(env).data.root_link_ang_vel_w)
    wp = torch.nan_to_num(w[:, 1]).abs()
    ex = st["stage"] == ST_EXIT
    k = torch.round(st["stage_t"] / dt)
    pk = torch.where(adv, torch.zeros_like(st["sd_peak"]), st["sd_peak"])
    st["sd_peak"] = torch.where(ex & (k <= round(win / dt)), torch.maximum(pk, wp), pk)
    st["sd_evt"] = ex & (k == round(win / dt))


def approach_target(st) -> torch.Tensor:
    """(B, 2) pursuit target: just inside the pipe's open end, on the side of the wall the robot carves into."""
    cfg, W = st["seq_cfg"], st["pipe"]["W"]
    return torch.stack([st["entry_s"] * cfg["target_x"], torch.full_like(st["entry_s"], -W / 2 + cfg["target_in"])], dim=1)


def approach_kappa(env, st) -> torch.Tensor:
    """Pure pursuit to the target point: kappa = 2 sin(bearing error) / distance (left turn = +), capped.

    r4rev (2026-11-20; cfg gate_pt = (tx, y_in, switch, kappa_max), default None = off): while the board is more than
    `switch` m before the open end (y < -W/2 - switch), pursue the GATE point (entry_s * tx, -W/2 + y_in) with cap
    gate_pt[3] instead; then the approach target as before. Kinematic pursuit from the m4 hand-off bank (lag 0.4 s,
    0.85 m/s): with the old target (0.45, 0.6 m inside, cap 0.3) only 52 % cross the open end at |x| < 0.5 (|x| > 0.8:
    24 %; raising the cap alone: 54 %, the target 0.6 m inside converges too late); gate (0.2, 0, 0.3, 0.45): 75 % / 44 %
    (geometric bound at 0.45: 90 % / 75 %). The approach target itself (heading_line_err) is unchanged."""
    cfg = st["seq_cfg"]
    p = pipe_xyz(env)[:, :2]
    tgt = approach_target(st)
    kmax = torch.full_like(st["entry_s"], float(cfg["kappa_max"]))
    gp = cfg.get("gate_pt")
    if gp is not None:
        W = st["pipe"]["W"]
        before = p[:, 1] < -W / 2 - float(gp[2])
        gt = torch.stack([st["entry_s"] * float(gp[0]), torch.full_like(st["entry_s"], -W / 2 + float(gp[1]))], dim=1)
        tgt = torch.where(before.unsqueeze(1), gt, tgt)
        kmax = torch.where(before, torch.full_like(kmax, float(gp[3])), kmax)
    d = tgt - p
    dist = d.norm(dim=1).clamp(min=0.5)
    e = wrap_to_pi(torch.atan2(d[:, 1], d[:, 0]) - tk._board_yaw(env))
    k = 2.0 * torch.sin(e) / dist
    return torch.maximum(torch.minimum(k, kmax), -kmax)


# ── reset ─────────────────────────────────────────────────────────────────────────────────────────────────────────────

# step_mid (stepfix 2026-11-20, push specialist only; probability 0 everywhere else): the step-on reverse curriculum
# from VALID mid-step stances (whole-body IK: single support,
# robot CoM over the board sole, push sole 16-19 mm above its deck height outside / at / over the deck edge; keys
# step_lift / step_edge / step_over in stance_poses_seq_stepfix.json). Same stage / speed / placement as step_on. The
# step_blend joint-space blend it replaces sank the board sole 6-10 mm into the deck and jammed the push foot into the
# deck edge (--mode kin), so the old reverse curriculum trained on impossible states.
# stroke_air (bridge 2026-11-20, push specialist only): the policy's OWN stroke-recovery states under the step command
# (recorded deck-relative full states:), optionally blended toward step_lift with a validity
# check: rungs r0 (as recorded), r33, r67, r100 of the bank file `bridge_bank`.
SPAWNS = ("start", "push", "glide", "manual", "exit", "ramp_bottom", "ramp_wall", "step_on", "step_mid", "stroke_air")
STEP_SPOT = (-0.040, -0.041)     # deck-frame xy of the push (right) sole after the step-on (stance_poses_seq glide_step)
SEQ_STANCE_STEPFIX_JSON = str(_data_path("stance_poses_seq_stepfix.json"))
# bridge2 (2026-11-20): re-recorded from p7_bridge model_500, rungs r0 r10 r20 r30 r35 r40 r45 r50 r58 r67 r75 r83 r92
# r100 (the first bank, from p6 model_100: stepfix_bridge_bank.npz)
SEQ_BRIDGE_BANK = str(_data_path("stepfix_bridge_bank2.npz"))
BANK_FIELDS = ("rel_pos", "rel_quat", "jpos", "jvel", "rel_lin", "rel_ang", "vb")


def _bridge_bank(env, path: str) -> dict:
    """Cached {rung: {field: tensor}} from a bank file."""
    c = tk._tc(env)
    if c.get("bridge_path") != path:
        import numpy as np
        z = np.load(path)
        rungs = sorted({k.split("_", 1)[0] for k in z.files if k != "joint_names"})
        jn = [str(x) for x in z["joint_names"]]
        assert jn == list(_robot(env).joint_names), "bridge bank joint order differs from the robot"
        c["bridge"] = {r: {f: torch.tensor(z[f"{r}_{f}"], device=env.device, dtype=torch.float) for f in BANK_FIELDS}
                       for r in rungs}
        c["bridge_path"] = path
    return c["bridge"]


# handoff2 (2026-11-20, manual specialist; default off): whole recorded push -> manual hand-off states (the push policy
# alone under the SeqRamp chain cfg from `start`, at SETTLE entry + 0-0.3 s after it), recorded
# with the chain. Robot root pose / twist and board twist in the
# deck frame, board height above the env origin and tilt relative to its own yaw frame, all robot / board joints, and
# the scheduler fields restored at the spawn (stage_t, both_on, wheels_on); the rest is kept for analysis.
HANDOFF_FIELDS = ("rel_pos", "rel_quat", "jpos", "jvel", "rel_lin", "rel_ang", "b_z", "b_tilt", "b_lin", "b_ang", "bj",
                  "bjv", "vb", "stage_t", "both_on", "wheels_on")


def _handoff_bank(env, path: str) -> dict:
    """Cached {field: tensor (N, ...)} from a bank file (joint orders checked against the scene)."""
    c = tk._tc(env)
    if c.get("handoff_path") != path:
        import numpy as np
        z = np.load(path)
        assert [str(x) for x in z["joint_names"]] == list(_robot(env).joint_names), \
            "handoff bank: robot joint order differs from the scene"
        assert [str(x) for x in z["board_joint_names"]] == list(tk._board(env).joint_names), \
            "handoff bank: board joint order differs from the scene"
        kind = str(z["kind"]) if "kind" in z.files else "manual"       # rampbank: -> "ramp"
        fields = RAMP_HANDOFF_FIELDS if kind == "ramp" else HANDOFF_FIELDS
        c["handoff"] = {f: torch.tensor(z[f], device=env.device, dtype=torch.float) for f in fields}
        c["handoff_kind"] = kind
        c["handoff_path"] = path
    return c["handoff"]


# rampbank (2026-11-20, ramp specialist; default off): recorded manual -> ramp hand-off states of the full chain (the
# chain at the owner switch to the ramp policy + 0-0.2 s after it), recorded
# with the chain (bank kind "ramp"). Board and robot root pose /
# world twist in the PIPE frame (the place relative to the pipe is part of the state), all joints, and the scheduler
# fields restored at an `exit` spawn (stage, stage_t, both_on, wheels_on, entry_s, T_exit, v_tgt, clock, wz_lp).
RAMP_HANDOFF_FIELDS = ("b_pos", "b_quat", "b_lin", "b_ang", "bj", "bjv", "r_pos", "r_quat", "r_lin", "r_ang", "jpos",
                       "jvel", "vb", "stage", "stage_t", "both_on", "wheels_on", "entry_s", "T_exit", "v_tgt", "clock",
                       "wz_lp")
# dv (coordinator 2026-11-20): +-0.1 m/s speed offset (p12 may hand over slightly slower than p11)
RAMP_HANDOFF_NOISE = dict(xy=0.04, yaw_deg=3.0, joint=0.02, v_scale=0.0, dv=0.1)


def _handoff_kind(env, path: str) -> str:
    _handoff_bank(env, path)
    return tk._tc(env)["handoff_kind"]


def _ramp_handoff_reset(env, env_ids: torch.Tensor, sel: torch.Tensor, path: str, prob: float,
                        noise: dict | None = None) -> None:
    """rampbank: with probability `prob` an `exit` spawn (sel) restores a random recorded chain hand-off state
    (_handoff_bank kind "ramp") with small noise: xy shift (+-xy m), a yaw turn about the board origin (+-yaw_deg; the
    rider and both twists turn with it), joint noise (N(0, joint) rad), speed scale (1 +- v_scale, plus +- dv m/s on the
    board speed, as one scale factor (vb + dv) / vb on every twist / joint
    velocity). The scheduler takes the recorded stage (EXIT), time in it (t_stage = -stage_t), both_on / wheels_on,
    entry_s (pursuit side), T_exit, v_tgt, and t_off = the recorded clock (the trick-obs clock reads the chain's t + t_off
    at the switch). st["handoff"] flags those envs. Runs at the end of reset_sequence_state."""
    st = _st(env)
    n, dev = len(env_ids), env.device
    if "handoff" not in st:
        st["handoff"] = torch.zeros(env.num_envs, dtype=torch.bool, device=dev)
    ho = sel & (torch.rand(n, device=dev) < float(prob))
    st["handoff"][env_ids] = ho
    if not bool(ho.any()):
        return
    nz = {**RAMP_HANDOFF_NOISE, **(noise or {})}
    hb = _handoff_bank(env, path)
    m = int(ho.sum())
    pick = torch.randint(len(hb["vb"]), (m,), device=dev)       # ONE bank row per env for every field
    g = {f: v[pick] for f, v in hb.items()}
    ids = env_ids[ho]
    U = lambda s: (torch.rand(m, device=dev) * 2 - 1) * float(s)
    z0 = torch.zeros(m, device=dev)
    dq = _quat_from_rpy_t(z0, z0, torch.deg2rad(U(nz["yaw_deg"])))
    vs = (1.0 + U(nz["v_scale"]) + U(nz.get("dv", 0.0)) / g["vb"].abs().clamp(min=0.3)).clamp(0.5, 1.5).unsqueeze(1)
    org = _origin(env)[ids]
    bp = g["b_pos"] + torch.stack([U(nz["xy"]), U(nz["xy"]), z0], dim=1)
    nq = lambda q: q / q.norm(dim=1, keepdim=True)
    board = tk._board(env)
    robot = _robot(env)
    board.write_root_link_pose_to_sim(torch.cat([org + bp, nq(quat_mul(dq, g["b_quat"]))], dim=1), ids)
    board.write_root_link_velocity_to_sim(torch.cat([quat_apply(dq, g["b_lin"]), quat_apply(dq, g["b_ang"])], dim=1) * vs,
                                          ids)
    board.write_joint_state_to_sim(g["bj"], g["bjv"] * vs, env_ids=ids)
    rp_ = bp + quat_apply(dq, g["r_pos"] - g["b_pos"])
    robot.write_root_link_pose_to_sim(torch.cat([org + rp_, nq(quat_mul(dq, g["r_quat"]))], dim=1), ids)
    robot.write_root_link_velocity_to_sim(torch.cat([quat_apply(dq, g["r_lin"]), quat_apply(dq, g["r_ang"])], dim=1) * vs,
                                          ids)
    robot.write_joint_state_to_sim(g["jpos"] + torch.randn_like(g["jpos"]) * float(nz["joint"]), g["jvel"] * vs,
                                   env_ids=ids)
    stg = g["stage"].round().long()
    st["stage"][ids] = stg
    st["trick"][ids] = torch.tensor(STAGE_PHASE, device=dev)[stg]
    if "obs_ph" in st:
        st["obs_ph"][ids] = st["trick"][ids]
    st["stage_t"][ids] = g["stage_t"]
    st["t_stage"][ids] = -g["stage_t"]           # t = 0 after the reset: seq_obs / exit_done read t - t_stage
    st["both_on"][ids] = g["both_on"]
    st["wheels_on"][ids] = g["wheels_on"]
    st["entry_s"][ids] = g["entry_s"]
    st["T_exit"][ids] = g["T_exit"]
    st["v_tgt"][ids] = g["v_tgt"]
    st["t_off"][ids] = g["clock"]
    st["wz_lp"][ids] = g["wz_lp"] * vs.squeeze(1)    # r4rev: yaw rate scales with the speed like the twists
    st["v0"][ids] = g["vb"] * vs.squeeze(1)


def _bridge_adapt(env, st: dict, env_ids: torch.Tensor, names: list, decay: float) -> None:
    """bridgerev (default off): before env_ids are re-spawned, fold the episodes that just ended from a stroke_air spawn
    into per-rung decayed success counts (success = the `success` termination fired; read before termination_manager
    .reset). c["bridge_sn"][rung] = [S, N]; p = (S + 1) / (N + 2)."""
    c = tk._tc(env)
    sn = c.setdefault("bridge_sn", {})
    if "bridge_rung" not in st or "success" not in env.termination_manager.active_terms:
        return
    ri = st["bridge_rung"][env_ids]
    ok = env.termination_manager.get_term("success")[env_ids]
    for k_, r in enumerate(names):
        m = ri == k_
        s0, n0 = sn.get(r, (0.0, 0.0))
        sn[r] = (s0 * decay + float((ok & m).sum()), n0 * decay + float(m.sum()))
    st["bridge_rung"][env_ids] = -1


def bridge_rung_p(env, r: str) -> float:
    s, n = tk._tc(env).get("bridge_sn", {}).get(r, (0.0, 0.0))
    return (s + 1.0) / (n + 2.0)


def m_bridge_frontier(env) -> torch.Tensor:
    """bridgerev metric: blend alpha of the LOWEST rung whose success estimate is >= 0.5 (1.0 = none yet), broadcast."""
    sn = tk._tc(env).get("bridge_sn", {})
    rs = sorted((r for r in sn if sn[r][1] >= 5.0), key=lambda r: int(r[1:]))      # >= 5 effective episodes scored
    a = next((int(r[1:]) / 100.0 for r in rs if bridge_rung_p(env, r) >= 0.5), 1.0)
    return torch.full((env.num_envs,), a, device=env.device)


def _stance_tensors(env, key: str, path: str, jn: list) -> tuple:
    s = tk.load_stance(key, Path(path))
    dev = env.device
    return (torch.tensor(s["pos"], device=dev, dtype=torch.float), torch.tensor(s["rpy"], device=dev, dtype=torch.float),
            torch.tensor([s["joints"][j] for j in jn], device=dev, dtype=torch.float))


def reset_sequence_state(
    env,
    env_ids: torch.Tensor,
    spawn_probs: dict,
    spawn_stances: dict,
    stance_path: str = SEQ_STANCE_JSON,
    flat_speeds: dict | None = None,
    ramp_speed: tuple = (0.4, 1.0),
    ramp_heading_deg: float = 35.0,
    wall_frac: tuple = (0.05, 0.55),
    wall_speed: tuple = (0.0, 0.6),
    manual_tilt_prob: float = 0.5,
    manual_tilt_deg: tuple = (5.0, 14.0),
    pos_noise: float = 0.003,
    rpy_noise: float = 0.02,
    joint_noise: float = 0.02,
    z_lift: float = 0.001,
    ramp_chain_prob: float = 0.0,
    ramp_chain_deg: tuple = (20.0, 55.0),
    step_blend: tuple | None = None,
    seq_overrides: dict | None = None,
    clock_offset: tuple | None = None,
    manual_prefill: tuple = (0.0, 0.9),
    handoff_path: str | None = None,
    handoff_prob: float = 0.0,
    exit_tilt_deg: tuple | None = None,
    spawn_noise: dict | None = None,
    exit_tilt_prob: float = 1.0,
    bridge_bank: str | None = None,
    bridge_rungs: dict | None = None,
    bridge_adaptive: dict | None = None,
    start_pose: tuple | None = None,
    line_noise: dict | None = None,
    handoff_noise: dict | None = None,
):
    """Places board + rider for a spawn type (one per env), seeds the scheduler at that type's stage. MUST run after the
    base reset events. spawn_stances: spawn type -> list of stance keys (one drawn uniformly). A key may repeat in the
    list to weight it. seq_overrides: SEQ_DEFAULTS keys to replace in this env's st["seq_cfg"] (applied once, on the
    first reset call, before anything below reads the cfg; specialists, e.g. last_stage).

    Specialist params (defaults = no change): clock_offset (lo, hi) s: per-env offset added to the trick-obs clock
    (seq_trick_obs), so a specialist that starts mid-chain sees the chain's clock. manual_prefill (lo, hi): with
    manual_end="hold", tilted MANUAL spawns start with band_t = U(lo, hi) * T_hold (nearly done = reverse curriculum).
    exit_tilt_deg: EXIT spawn tilt range (None: manual_tilt_deg); exit_tilt_prob: share of tilted EXIT spawns (1.0 =
    all, the default; the rest start on four wheels). spawn_noise: spawn type -> dict(joint=, rpy=) noise
    overrides. handoff_path / handoff_prob (handoff2, manual specialist): with probability handoff_prob a glide
    (SETTLE) spawn restores a random recorded push-policy hand-off state (_handoff_bank; at the glide spawn's own x, y,
    yaw, so its floor length / clock offset / T_settle / T_hold draws stay), with stage_t / t_stage, both_on, wheels_on
    from the record and band_t = 0; st["handoff"] flags those envs. Off (None or 0): no RNG drawn, nothing changes.
    rampbank (ramp specialist): a bank of kind "ramp" is applied to `exit` spawns instead, in the
    pipe frame with handoff_noise (_ramp_handoff_reset).
    spawn_probs may change between calls (a params curriculum): the sampling weights follow.
    start_pose (chainfix, chain playback only; None = off, no change): (dx m, dy m, dyaw deg) added to the "start" spawns
    board x / y / yaw (the rider follows the board pose): a start aimed so the chain's push / manual drift ends on the
    approach line.
    line_noise (heading fix, push / manual specialists; None = off, no RNG drawn): spawn type -> (prob, yaw_deg, x_m):
    with probability prob that spawn's board yaw gets U(-yaw_deg, yaw_deg) and its x U(-x_m, x_m) added (flat spawns
    only; the rider / hand-off / bridge state follows the board pose), so heading_line_cost has errors to correct."""
    st = _st(env)
    c = tk._tc(env)
    if seq_overrides and not c.get("seq_overrides_applied", False):
        unknown = set(seq_overrides) - set(SEQ_DEFAULTS)
        assert not unknown, f"seq_overrides: unknown keys {sorted(unknown)}"
        st["seq_cfg"].update(seq_overrides)
        c["seq_overrides_applied"] = True
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device).long()
    if bridge_bank and bridge_adaptive:           # bridgerev: score the ending stroke_air episodes (default off)
        rw0 = bridge_rungs or {r: 1.0 for r in _bridge_bank(env, bridge_bank)}
        _bridge_adapt(env, st, env_ids, [r for r in rw0 if rw0[r] > 0 and r in _bridge_bank(env, bridge_bank)],
                      float(bridge_adaptive.get("decay", 0.995)))
    n, dev = len(env_ids), env.device
    robot, board = _robot(env), tk._board(env)
    cfg, pipe = st["seq_cfg"], st["pipe"]
    flat_speeds = flat_speeds or dict(push=(0.1, 0.8), glide=(0.5, 1.1), manual=(0.5, 1.0), exit=(0.5, 1.0))
    if "seq_kf" not in c:
        any_key = next(iter(spawn_stances.values()))[0]
        jn = list(tk.load_stance(any_key, Path(stance_path))["joints"].keys())
        keys = sorted({k for ks in spawn_stances.values() for k in ks})
        kt = {k: _stance_tensors(env, k, stance_path, jn) for k in keys}
        c["stance_yaw"] = 0.0
        c["seq_kf"] = dict(
            leg_ids=[list(robot.joint_names).index(j) for j in jn],
            wheel_ids=board.find_joints(r"^passive_.*_wheel_[LR]$")[0],
            pos=torch.stack([kt[k][0] for k in keys]), rpy=torch.stack([kt[k][1] for k in keys]),
            joints=torch.stack([kt[k][2] for k in keys]),
            choices=[[keys.index(k) for k in spawn_stances.get(s, [keys[0]])] for s in SPAWNS],
            keys=keys,
        )
        p = torch.tensor([float(spawn_probs.get(s, 0.0)) for s in SPAWNS], device=dev)
        assert p.sum() > 0
        c["seq_p"] = p / p.sum()
    pkey = tuple(sorted((k, float(v)) for k, v in spawn_probs.items()))
    if c.get("seq_p_key", pkey) != pkey:          # spawn_probs changed since the last call (curriculum): re-weight
        p = torch.tensor([float(spawn_probs.get(s, 0.0)) for s in SPAWNS], device=dev)
        assert p.sum() > 0
        c["seq_p"] = p / p.sum()
    c["seq_p_key"] = pkey
    kf = c["seq_kf"]

    sp = torch.multinomial(c["seq_p"], n, replacement=True)                          # spawn type per env
    kidx = torch.zeros(n, dtype=torch.long, device=dev)
    for si, ch in enumerate(kf["choices"]):
        m = sp == si
        if bool(m.any()):
            kidx[m] = torch.tensor(ch, device=dev)[torch.randint(len(ch), (int(m.sum()),), device=dev)]
    kp, kr, kj = kf["pos"][kidx].clone(), kf["rpy"][kidx].clone(), kf["joints"][kidx].clone()
    R_ = lambda lo, hi: lo + torch.rand(n, device=dev) * (hi - lo)
    # step_blend = (target_key, (a_lo, a_hi), prob): part of the step_on spawns start part-way through the step-on
    # (joint/root pose blended toward the post-step-on stance, push foot in the air over the deck) - the reverse
    # curriculum for the one transition the policy never discovered (s1 -> s2b: 0 step-ons in ~2000 iterations)
    if step_blend is not None and step_blend[0] in kf["keys"]:
        bm = (sp == SPAWNS.index("step_on")) & (torch.rand(n, device=dev) < step_blend[2])
        if bool(bm.any()):
            ti = kf["keys"].index(step_blend[0])
            a = (step_blend[1][0] + torch.rand(n, device=dev) * (step_blend[1][1] - step_blend[1][0])).unsqueeze(1)
            a = torch.where(bm.unsqueeze(1), a, torch.zeros_like(a))
            kp = kp + a * (kf["pos"][ti] - kp)
            kr = kr + a * (kf["rpy"][ti] - kr)
            kj = kj + a * (kf["joints"][ti] - kj)

    is_ = lambda name: sp == SPAWNS.index(name)
    flat = ~(is_("ramp_bottom") | is_("ramp_wall"))
    W = pipe["W"]
    # board placement in the pipe frame: x, y, yaw, speed (along deck +x), pitch about the deck y axis (nose up = +)
    bx, by = torch.zeros(n, device=dev), torch.zeros(n, device=dev)
    yaw = math.pi / 2 + (torch.rand(n, device=dev) * 2 - 1) * cfg.get("flat_yaw_noise", 0.15)
    v0 = torch.zeros(n, device=dev)
    pitch = torch.zeros(n, device=dev)
    bz = torch.full((n,), DECK_Z, device=dev)
    entry_s = torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)      # side of the wall it carves into
    x_start = -entry_s * cfg["start_off"]
    # flat spawns: on the floor before / at the open end, heading +y near the approach line
    y_start = -(W / 2 + cfg["start_gap"])
    along = torch.zeros(n, device=dev)
    # spawn distance along the run, as a fraction of start_gap (REV: scaled with the gap; s1e's absolute ranges were for
    # 7 m: push 0-1, glide 0.5-2.5, manual 1-3.5, exit 2-5.5 m)
    gap = cfg["start_gap"]
    for name, rng in (("push", (0.0, 0.15)), ("glide", (0.1, 0.4)), ("manual", (0.2, 0.55)), ("exit", (0.3, 0.8)),
                      ("step_on", (0.1, 0.3))):
        m = is_(name) | is_("step_mid") | is_("stroke_air") if name == "step_on" else is_(name)   # no extra RNG draw
        along = torch.where(m, R_(rng[0] * gap, rng[1] * gap), along)
        lo, hi = flat_speeds[name] if name != "step_on" else (0.0, 0.0)
        v0 = torch.where(m, R_(lo, hi), v0)
    bx = torch.where(flat, x_start + (torch.rand(n, device=dev) * 2 - 1) * 0.15, bx)
    by = torch.where(flat, y_start + along, by)
    # exit spawns: half are mid-turn onto the approach line (heading between +y and the line)
    turned = is_("exit") & (torch.rand(n, device=dev) < 0.5)
    yaw = torch.where(turned, yaw - entry_s * torch.rand(n, device=dev) * math.radians(30.0), yaw)
    if start_pose is not None:                    # chainfix: aimed start (default None: off, no RNG drawn)
        ms = is_("start")
        bx = torch.where(ms, bx + float(start_pose[0]), bx)
        by = torch.where(ms, by + float(start_pose[1]), by)
        yaw = torch.where(ms, yaw + math.radians(float(start_pose[2])), yaw)
    if line_noise:                                # heading fix: off-line / mis-aimed flat spawns (default None: off)
        for s_name, (pr, yd, xm) in line_noise.items():
            ml = is_(s_name) & (torch.rand(n, device=dev) < float(pr))
            yaw = torch.where(ml, yaw + torch.deg2rad(R_(-float(yd), float(yd))), yaw)
            bx = torch.where(ml, bx + R_(-float(xm), float(xm)), bx)
    # ramp_bottom: on the flat bottom heading toward a wall (+-x, +- ramp_heading_deg), any speed in ramp_speed
    side = torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)
    rb = is_("ramp_bottom")
    yaw = torch.where(rb, torch.where(side > 0, 0.0, math.pi) + (torch.rand(n, device=dev) * 2 - 1)
                      * math.radians(ramp_heading_deg), yaw)
    # REV: half the ramp_bottom spawns arrive the way the chain does: heading down the pipe (+y) and turned
    # ramp_chain_deg toward the wall on `side` (s1e's spawns were all across the pipe, the chain enters along it)
    chain = rb & (torch.rand(n, device=dev) < ramp_chain_prob)
    yaw = torch.where(chain, math.pi / 2 - side * torch.deg2rad(R_(*ramp_chain_deg)), yaw)
    bx = torch.where(rb, (torch.rand(n, device=dev) * 2 - 1) * pipe["L"] * 0.3, bx)
    by = torch.where(rb | is_("ramp_wall"), R_(-W / 2 + 1.0, W / 2 - 1.5), by)
    v0 = torch.where(rb, R_(*ramp_speed), v0)
    # ramp_wall: on a transition (circle arc), deck tangent to the chord between the trucks, facing the wall or away,
    # rolling up or down the surface
    rw = is_("ramp_wall")
    if bool(rw.any()):
        h, Rr, L = pipe["h"], pipe["R"], pipe["L"]
        phi_max = math.acos(1.0 - h / Rr)
        zf = R_(*wall_frac) * h
        phi = torch.acos((1.0 - zf / Rr).clamp(-1.0, 1.0)).clamp(max=phi_max * 0.9)
        rr = Rr - BOARD["wheel_r"]                               # wheel centres ride on this circle
        dl = math.asin(min(0.99, BOARD["axle_x"] / rr))
        dm = rr * math.cos(dl) + BOARD["axle_z"]                 # arc centre -> deck origin distance (axle_z < 0)
        face_wall = torch.rand(n, device=dev) < 0.5
        dsign = torch.where(face_wall, side, -side)              # deck +x along world +-x
        bx = torch.where(rw, side * (L / 2 + dm * torch.sin(phi)), bx)
        bz = torch.where(rw, Rr - dm * torch.cos(phi), bz)
        yaw = torch.where(rw, torch.where(dsign > 0, 0.0, math.pi), yaw)
        pitch = torch.where(rw, torch.where(face_wall, phi, -phi), pitch)   # nose up when facing the wall
        vw = R_(*wall_speed) * torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)
        v0 = torch.where(rw, vw, v0)
    # manual / exit: some spawns tilted nose-up about the back axle (exit: always tilted, then the RIDE command)
    # exit_tilt_prob < 1 (ramp specialist): the rest of the EXIT spawns start on four wheels (no RNG drawn at 1.0)
    ex_t = is_("exit") if exit_tilt_prob >= 1.0 else is_("exit") & (torch.rand(n, device=dev) < exit_tilt_prob)
    tilt = (is_("manual") & (torch.rand(n, device=dev) < manual_tilt_prob)) | ex_t
    pitch = torch.where(tilt, torch.deg2rad(R_(*manual_tilt_deg)), pitch)
    if exit_tilt_deg is not None:
        pitch = torch.where(ex_t, torch.deg2rad(R_(*exit_tilt_deg)), pitch)
    pivot_dz = torch.where(tilt, torch.ones(n, device=dev), torch.zeros(n, device=dev))

    # board quaternion: yaw then pitch about the deck y axis (nose up = rotation by -pitch about +y)
    qb = quat_mul(_quat_from_rpy_t(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw),
                  _quat_from_rpy_t(torch.zeros_like(yaw), -pitch, torch.zeros_like(yaw)))
    origin = _origin(env)[env_ids]
    bpos = origin + torch.stack([bx, by, bz], dim=1)
    # tilted manuals rotate about the back axle: shift the origin so the back axle stays on the floor
    if bool(tilt.any()):
        back = torch.stack([-torch.full_like(yaw, BOARD["axle_x"]), torch.zeros_like(yaw),
                            torch.full_like(yaw, BOARD["axle_z"])], dim=1)
        q_yaw = _quat_from_rpy_t(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
        shift = quat_apply(q_yaw, back) - quat_apply(qb, back)
        bpos = torch.where(tilt.unsqueeze(1), bpos + shift * pivot_dz.unsqueeze(1), bpos)

    rel = kp.clone()
    rel[:, :2] += (torch.rand(n, 2, device=dev) * 2 - 1) * pos_noise
    rel[:, 2] = rel[:, 2] - DECK_Z + z_lift                       # stance z is absolute with the board level at rest
    rpos = bpos + quat_apply(qb, rel)
    # a floor sole (mount / push stances) must stay on the floor: only flat spawns use those stances (not tilted)
    rn, jnz = rpy_noise, joint_noise
    if spawn_noise:                                               # per-spawn-type overrides, (n, 1) columns
        rn = torch.full((n, 1), float(rpy_noise), device=dev)
        jnz = torch.full((n, 1), float(joint_noise), device=dev)
        for s_name, d in spawn_noise.items():
            m = is_(s_name).unsqueeze(1)
            rn = torch.where(m, float(d.get("rpy", rpy_noise)), rn)
            jnz = torch.where(m, float(d.get("joint", joint_noise)), jnz)
    r = kr + (torch.rand(n, 3, device=dev) * 2 - 1) * rn
    rq = quat_mul(qb, _quat_from_rpy_t(r[:, 0], r[:, 1], r[:, 2]))

    vdir = quat_apply(qb, torch.tensor([1.0, 0.0, 0.0], device=dev).expand(n, 3))
    vel = torch.zeros(n, 6, device=dev)
    vel[:, :3] = vdir * v0.unsqueeze(1)
    bpose = torch.cat([bpos, qb], dim=1)
    board.write_root_link_pose_to_sim(bpose, env_ids)
    board.write_root_link_velocity_to_sim(vel, env_ids)
    bj = torch.zeros(n, board.num_joints, device=dev)
    bjv = torch.zeros(n, board.num_joints, device=dev)
    bjv[:, kf["wheel_ids"]] = (v0 / BOARD["wheel_r"]).unsqueeze(1)
    board.write_joint_state_to_sim(bj, bjv, env_ids=env_ids)
    rvel = vel.clone()
    rvel[:, :3] = torch.where(is_("start").unsqueeze(1), torch.zeros_like(rvel[:, :3]), rvel[:, :3])
    robot.write_root_link_pose_to_sim(torch.cat([rpos, rq], dim=1), env_ids)
    robot.write_root_link_velocity_to_sim(rvel, env_ids)
    jpos = robot.data.default_joint_pos[env_ids].clone()
    jpos[:, kf["leg_ids"]] = kj + torch.randn(n, len(kf["leg_ids"]), device=dev) * jnz
    robot.write_joint_state_to_sim(jpos, torch.zeros_like(jpos), env_ids=env_ids)

    # step_on spawns roll at (v_tgt - 0.05 .. v_tgt + 0.1): re-write their velocities now that v_tgt is drawn
    # (done below after v_tgt; see the second velocity write)
    # scheduler state
    stage0 = torch.tensor([ST_PUSH, ST_PUSH, ST_SETTLE, ST_MANUAL, ST_EXIT, ST_RAMP, ST_RAMP, ST_PUSH, ST_PUSH, ST_PUSH],
                          device=dev)[sp]
    st["spawn"][env_ids] = sp
    st["stage"][env_ids] = stage0
    st["trick"][env_ids] = torch.tensor(STAGE_PHASE, device=dev)[stage0]
    st["t_stage"][env_ids] = 0.0
    st["stage_t"][env_ids] = 0.0
    st["T_settle"][env_ids] = R_(*cfg["settle_s"])
    st["T_manual"][env_ids] = torch.where(is_("manual"), R_(*cfg.get("manual_spawn_s", cfg["manual_s"])),
                                          R_(*cfg["manual_s"]))
    st["T_exit"][env_ids] = R_(*cfg["exit_min_s"])
    st["v_tgt"][env_ids] = R_(*cfg["v_t"])
    so = is_("step_on") | is_("step_mid")
    if bool(so.any()):
        vs = st["v_tgt"][env_ids] + R_(-0.05, 0.1)
        vel2 = torch.zeros(n, 6, device=dev)
        vel2[:, :3] = vdir * vs.unsqueeze(1)
        ids = env_ids[so]
        board.write_root_link_velocity_to_sim(vel2[so], ids)
        bjv2 = torch.zeros(int(so.sum()), board.num_joints, device=dev)
        bjv2[:, kf["wheel_ids"]] = (vs[so] / BOARD["wheel_r"]).unsqueeze(1)
        board.write_joint_state_to_sim(torch.zeros_like(bjv2), bjv2, env_ids=ids)
        robot.write_root_link_velocity_to_sim(vel2[so], ids)
        v0 = torch.where(so, vs, v0)
    # bridge: stroke_air spawns take a whole recorded (or laddered) state relative to the board, at its own speed;
    # v_tgt is capped at that speed + 0.05 so the step command is latched from the first step, as when recorded
    sa = is_("stroke_air")
    if bridge_bank and bool(sa.any()):
        bk = _bridge_bank(env, bridge_bank)
        rw = bridge_rungs or {r: 1.0 for r in bk}
        names = [r for r in rw if r in bk and rw[r] > 0]
        ns = int(sa.sum())
        wts = [float(rw[r]) for r in names]
        if bridge_adaptive:     # bridgerev: frontier tracking, w = base * (floor + 4 p (1 - p)), p per-rung success est.
            fl = float(bridge_adaptive.get("floor", 0.1))
            wts = [w * (fl + 4.0 * bridge_rung_p(env, r) * (1.0 - bridge_rung_p(env, r))) for w, r in zip(wts, names)]
        ri = torch.multinomial(torch.tensor(wts, device=dev), ns, replacement=True)
        if bridge_adaptive:
            if "bridge_rung" not in st:
                st["bridge_rung"] = torch.full((env.num_envs,), -1, dtype=torch.long, device=dev)
            st["bridge_rung"][env_ids[sa]] = ri
        g = {f: torch.zeros((ns,) + bk[names[0]][f].shape[1:], device=dev) for f in BANK_FIELDS}
        for k_, r in enumerate(names):
            mk = ri == k_
            if bool(mk.any()):
                pick = torch.randint(len(bk[r]["vb"]), (int(mk.sum()),), device=dev)
                for f in BANK_FIELDS:
                    g[f][mk] = bk[r][f][pick]
        ids = env_ids[sa]
        qs, ps = qb[sa], bpos[sa]
        zl = torch.tensor([0.0, 0.0, z_lift], device=dev)
        robot.write_root_link_pose_to_sim(torch.cat([ps + quat_apply(qs, g["rel_pos"] + zl), quat_mul(qs, g["rel_quat"])],
                                                    dim=1), ids)
        robot.write_root_link_velocity_to_sim(torch.cat([quat_apply(qs, g["rel_lin"]), quat_apply(qs, g["rel_ang"])], dim=1),
                                              ids)
        robot.write_joint_state_to_sim(g["jpos"], g["jvel"], env_ids=ids)
        vb = g["vb"]
        bv = torch.zeros(ns, 6, device=dev)
        bv[:, :3] = vdir[sa] * vb.unsqueeze(1)
        board.write_root_link_velocity_to_sim(bv, ids)
        bjv3 = torch.zeros(ns, board.num_joints, device=dev)
        bjv3[:, kf["wheel_ids"]] = (vb / BOARD["wheel_r"]).unsqueeze(1)
        board.write_joint_state_to_sim(torch.zeros_like(bjv3), bjv3, env_ids=ids)
        vt = st["v_tgt"][env_ids]
        st["v_tgt"][ids] = torch.minimum(vt[sa], vb + 0.05)
        v0 = torch.where(sa, torch.zeros_like(v0).masked_scatter(sa, vb), v0)
    # handoff2 (default off: no RNG drawn): glide spawns -> a recorded push-policy hand-off state (_handoff_bank) at this
    # spawn's x, y, yaw: board height / tilt / twist / joints and the rider relative to the deck, as recorded
    ho, hg = None, None
    if handoff_path and handoff_prob > 0 and _handoff_kind(env, handoff_path) == "manual":
        ho = is_("glide") & (torch.rand(n, device=dev) < float(handoff_prob))
        if "handoff" not in st:
            st["handoff"] = torch.zeros(env.num_envs, dtype=torch.bool, device=dev)
        st["handoff"][env_ids] = ho
        if bool(ho.any()):
            hb = _handoff_bank(env, handoff_path)
            pick = torch.randint(len(hb["vb"]), (int(ho.sum()),), device=dev)
            hg = {f: v[pick] for f, v in hb.items()}
            ids = env_ids[ho]
            yh = yaw[ho]
            qh = quat_mul(_quat_from_rpy_t(torch.zeros_like(yh), torch.zeros_like(yh), yh), hg["b_tilt"])
            qh = qh / qh.norm(dim=1, keepdim=True)
            ph = origin[ho] + torch.stack([bx[ho], by[ho], hg["b_z"]], dim=1)
            board.write_root_link_pose_to_sim(torch.cat([ph, qh], dim=1), ids)
            board.write_root_link_velocity_to_sim(torch.cat([quat_apply(qh, hg["b_lin"]), quat_apply(qh, hg["b_ang"])],
                                                            dim=1), ids)
            board.write_joint_state_to_sim(hg["bj"], hg["bjv"], env_ids=ids)
            rq_h = quat_mul(qh, hg["rel_quat"])
            robot.write_root_link_pose_to_sim(torch.cat([ph + quat_apply(qh, hg["rel_pos"]),
                                                         rq_h / rq_h.norm(dim=1, keepdim=True)], dim=1), ids)
            robot.write_root_link_velocity_to_sim(torch.cat([quat_apply(qh, hg["rel_lin"]),
                                                             quat_apply(qh, hg["rel_ang"])], dim=1), ids)
            robot.write_joint_state_to_sim(hg["jpos"], hg["jvel"], env_ids=ids)
            v0 = torch.where(ho, torch.zeros_like(v0).masked_scatter(ho, hg["vb"]), v0)
    elif "handoff" in st:
        st["handoff"][env_ids] = False
    st["step_d"][env_ids] = -1.0
    st["step_d0"][env_ids] = 0.0
    st["rf_prev"][env_ids] = False
    st["td_init"][env_ids] = False
    st["td_x"][env_ids] = 0.0
    st["td_air"][env_ids] = 0.0
    st["rx_prev"][env_ids] = 0.0
    st["lift_evt"][env_ids] = False
    st["lift_q"][env_ids] = 0.0
    st["stepon_evt"][env_ids] = False
    if "cmd" in st:
        st["cmd"][env_ids] = False
    if "cmd_t" in st:
        st["cmd_t"][env_ids] = 0.0
    if bridge_bank and bool(sa.any()) and "cmd" in st:   # bridgerev: latch the command (3-8 % missed the v gate)
        st["cmd"][env_ids[sa]] = True
    if "cmd_air" in st:                          # landrule state (seq_cmd_landing)
        st["cmd_air"][env_ids] = False
        st["land_fail"][env_ids] = False
        st["land_deck"][env_ids] = False
    if "stance_t" in st:                         # p9b hand-off stance state (created on first use)
        st["stance_t"][env_ids] = 0.0
    if "sp_d" in st:
        st["sp_d"][env_ids] = -1.0
        st["sp_d0"][env_ids] = 0.0
    if "path_pass" in st:                        # stepfix push_step_path state (created by the reward on first use)
        st["path_pass"][env_ids] = False
        st["path_d"][env_ids] = -1.0
        st["path_d0"][env_ids] = 0.0
        st["path_evt"][env_ids] = False
    if "obs_ph" in st and "trick" in st:
        st["obs_ph"][env_ids] = st["trick"][env_ids]
        if bridge_bank and bool(sa.any()) and "cmd" in st:   # bridgerev: the reset obs already shows the command
            st["obs_ph"][env_ids[sa]] = RIDE
    if "man_ok" in st:                            # m2 manual_speed_loss state (created by the reward on first use)
        st["man_ok"][env_ids] = False
    if "pop_up" in st:                            # m3 manual_pitch_rate(until_deg) latch (created by the reward)
        st["pop_up"][env_ids] = False
    if "hr_pop_t" in st:                          # m3 head_rate(pop_only) latch (created by the reward)
        st["hr_pop_t"][env_ids] = float("inf")
    st["apex_evt"][env_ids] = False
    st["both_on"][env_ids] = 0.0
    st["wheels_on"][env_ids] = 0.0
    st["entry_s"][env_ids] = entry_s
    st["phi"][env_ids] = torch.where(stage0 == ST_PUSH, torch.minimum(v0, st["v_tgt"][env_ids]).clamp(min=0.0),
                                     torch.zeros_like(v0))
    st["apex_side"][env_ids] = 0.0
    st["vz_prev"][env_ids] = 0.0
    st["apex_pay"][env_ids] = 0.0
    st["dir"][env_ids] = 1.0
    st["v0"][env_ids] = v0
    st["kappa"][env_ids] = 0.0
    st["kappa_next"][env_ids] = 1.0e9
    st["wz_lp"][env_ids] = 0.0
    st["fresh"][env_ids] = True
    # manual specialist state (RNG drawn only when used: the default tasks' random streams are unchanged)
    st["manual_evt"][env_ids] = False
    st["sd_peak"][env_ids] = 0.0
    st["sd_evt"][env_ids] = False
    st["t_off"][env_ids] = R_(*clock_offset) if clock_offset is not None else 0.0
    if cfg.get("manual_end", "timer") == "hold":
        th = R_(*cfg["manual_hold_s"])
        st["T_hold"][env_ids] = th
        mt = tilt & is_("manual")
        st["band_t"][env_ids] = torch.where(mt, R_(*manual_prefill) * th, torch.zeros_like(th))
    else:
        st["T_hold"][env_ids] = 0.0
        st["band_t"][env_ids] = 0.0
    if hg is not None:                            # handoff2: the recorded scheduler state (stage SETTLE from glide)
        ids = env_ids[ho]
        st["stage_t"][ids] = hg["stage_t"]       # time already spent in SETTLE (0-0.3 s); T_settle = this task's draw
        st["t_stage"][ids] = -hg["stage_t"]      # t = 0 after the reset: seq_obs / settle_done read t - t_stage
        st["both_on"][ids] = hg["both_on"]
        st["wheels_on"][ids] = hg["wheels_on"]
        st["band_t"][ids] = 0.0
    # rampbank (default off: no RNG drawn): a bank of kind "ramp" restores recorded chain hand-off states at exit spawns
    if handoff_path and handoff_prob > 0 and _handoff_kind(env, handoff_path) == "ramp":
        _ramp_handoff_reset(env, env_ids, is_("exit"), handoff_path, handoff_prob, handoff_noise)


# ── observations ──────────────────────────────────────────────────────────────────────────────────────────────────────


def seq_trick_obs(env) -> torch.Tensor:
    """(B, 12) the trick block layout (one-hot ride/stop/kickturn/tictac/manual, dir, kappa, 0, 0, 1, 0, t/10); the one-hot
    is all-zero in PUSH / RAMP (their own slots are in seq_obs)."""
    st = seq_update(env)
    ph = st.get("obs_ph", st["trick"])
    oh = torch.nn.functional.one_hot(ph.clamp(max=tk.N_TRICK - 1), tk.N_TRICK).float() * (ph < tk.N_TRICK).float().unsqueeze(1)
    z = torch.zeros(env.num_envs, 1, device=env.device)
    t = tk._t(env) + st["t_off"]                     # t_off: reset_sequence_state clock_offset (default 0)
    return torch.cat([oh, st["dir"].unsqueeze(1), st["kappa"].unsqueeze(1), z, z, z + 1.0, z,
                      (t / 10.0).clamp(max=0.8).unsqueeze(1)], dim=1)


def seq_obs(env) -> torch.Tensor:
    """(B, 13): PUSH, RAMP one-hot (2), push target speed (PUSH only), time in stage / 5 (clipped 2), pipe-frame board
    x / 0.7, y / (W/2) (clipped +-2), sin / cos board yaw, deck height / h, surface slope under the front / rear truck,
    left / right foot on the floor."""
    st = seq_update(env)
    pipe = st["pipe"]
    ph = st.get("obs_ph", st["trick"])
    p = pipe_xyz(env)
    yaw = tk._board_yaw(env)
    q = tk._board(env).data.root_link_quat_w
    ex = quat_apply(q, torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(env.num_envs, 3))
    xf = p[:, 0] + ex[:, 0] * BOARD["axle_x"]
    xr = p[:, 0] - ex[:, 0] * BOARD["axle_x"]
    _, sf = pipe_profile(xf, pipe)
    _, sr = pipe_profile(xr, pipe)
    zn = pipe["h"]
    if any(pipe[k] != v for k, v in OBS_PIPE.items()):  # pipe_h: observe at the trained scale (see OBS_PIPE)
        zn = OBS_PIPE["h"]
        off = p[:, 1].abs() > pipe["W"] / 2
        sf = torch.where(off, pipe_profile(xf, OBS_PIPE)[1], sf)
        sr = torch.where(off, pipe_profile(xr, OBS_PIPE)[1], sr)
    lf, rf = _feet_floor(env)
    ts = ((tk._t(env) - st["t_stage"]) / 5.0).clamp(max=2.0)
    out = torch.stack([
        (ph == PUSH).float(), (ph == RAMP).float(), st["v_tgt"] * (ph == PUSH).float(), ts,
        (p[:, 0] / 0.7).clamp(-1.5, 1.5), (p[:, 1] / (pipe["W"] / 2)).clamp(-2.0, 2.0), torch.sin(yaw), torch.cos(yaw),
        p[:, 2] / zn, sf.clamp(-3, 3), sr.clamp(-3, 3), lf.float(), rf.float(),
    ], dim=1)
    return torch.nan_to_num(out)


SEQ_OBS_STATS = [  # (mean, std) seeds for the obs-normalizer extension of a warm-started checkpoint
    (0.25, 0.43), (0.3, 0.46), (0.25, 0.4), (0.4, 0.4), (0.0, 0.9), (-1.2, 0.8), (0.7, 0.4), (0.0, 0.6), (0.2, 0.3),
    (0.0, 0.4), (0.0, 0.4), (0.0, 0.2), (0.1, 0.3),
]


# ── rewards (per second; mjlab multiplies by dt) ──────────────────────────────────────────────────────────────────────


def _phase(env, k: int) -> torch.Tensor:
    return (seq_update(env)["trick"] == k).float()


def seq_on_deck(env) -> torch.Tensor:
    """Both soles on the deck; in PUSH only the board (left) sole counts."""
    st = seq_update(env)
    left = tk._contact_any(env, "left_deck")
    return torch.where(st["trick"] == PUSH, left, tk.soles_on_deck(env)).float()


def push_progress(env, cmd_gate: bool = False) -> torch.Tensor:
    """PUSH: signed potential on min(board speed, trunk speed, target) (cruiser skate_push_progress): gains are paid only
    with the board foot on the deck and a push-foot floor stroke within the window; losses are always charged; the held
    potential is charged at a termination inside PUSH. Returns dPhi/dt, so the episode sum = net speed gained (m/s)."""
    st = seq_update(env)
    cfg = st["seq_cfg"]
    push = st["trick"] == PUSH
    phi_new = torch.minimum(torch.minimum(board_v_fwd(env), robot_v_fwd(env)), st["v_tgt"]).clamp(min=0.0)
    if cfg.get("push_ramp") and "v_ramp" in st:     # p9: ahead of the slewed target pays nothing (default off)
        phi_new = torch.minimum(phi_new, st["v_ramp"])
    d = phi_new - st["phi"]
    gate = tk._contact_any(env, "left_deck") & _push_foot_recent(env, cfg["stroke_window_s"])
    pay = torch.where(d > 0, d * gate.float(), d)
    term = env.termination_manager.terminated
    banked = st["phi"] >= st["v_tgt"] - cfg["v_gap"]
    pay = torch.where(term & ~banked, -st["phi"], pay)
    if cmd_gate and "cmd" in st:      # landrule (default off): nothing under the step command - no pay for a stroke's
        pay = torch.where(st["cmd"], torch.zeros_like(pay), pay)   # gain, no 10x charge for the trunk slowing in an attempt
    out = torch.where(push, pay, torch.zeros_like(pay)) / env.step_dt
    st["phi"] = torch.where(push, phi_new, st["phi"])
    return out


def phase_time(env, phase: int = PUSH, late_mult: float = 3.0) -> torch.Tensor:
    """Cost per second in the phase; PUSH costs double once the target speed is reached or after push_max_s (parking in
    the push stance must net <= 0 against upright + on_deck)."""
    st = seq_update(env)
    m = _phase(env, phase)
    if phase != PUSH:
        return m
    cfg = st["seq_cfg"]
    # review (rev1): time-only doubling. The speed-triggered doubling + push_speed's at-speed gate made a -2.8 /s cliff
    # at v_tgt - v_gap, so hovering just under the threshold (where PUSH cannot end) was optimal.
    late = (tk._t(env) - st["t_stage"]) > cfg["push_late_s"]
    return m * (1.0 + (late_mult - 1.0) * late.float())     # rev1d: x3 when late (s2d: staying in PUSH 22 s beat the chain)


def _push_v(env, st) -> torch.Tensor:
    return torch.minimum(board_v_fwd(env), robot_v_fwd(env)).clamp(min=0.0)


def push_speed(env) -> torch.Tensor:
    """PUSH: dense speed pay min(board, trunk speed, target) / target with the board foot on the deck (cruiser `speed`:
    a constant gradient from standstill). Parking at speed in the push stance still loses to SETTLE (ride + wheels)."""
    st = seq_update(env)
    pv = _push_v(env, st)
    v = torch.minimum(pv, st["v_tgt"]) / st["v_tgt"].clamp(min=0.1)
    if st["seq_cfg"].get("push_ramp") and "v_ramp" in st:     # p9 (default off): pay up to the slewed target only
        v = torch.minimum(v, st["v_ramp"] / st["v_tgt"].clamp(min=0.1))
    # at the target speed, pay only with the push foot ON THE DECK (stepped on): s1b parked pushing with 6.6 Hz floor
    # taps; s1d it1000 kept tapping because "foot off the floor" still paid half the time
    _, rf = _feet_floor(env)
    at_speed = pv >= st["v_tgt"] - st["seq_cfg"]["v_gap"]
    if st["seq_cfg"].get("push_ramp") and "v_ramp" in st:     # p9: no 0.25 cliff before the step-on is allowed
        at_speed = at_speed & (st["v_ramp"] >= st["v_tgt"] - st["seq_cfg"]["v_gap"])
    stepped = tk._contact_any(env, "right_deck") & ~rf
    # review (rev1): no cliff at the threshold; once at speed an un-stepped push foot pays half (stepping on pays full)
    gate = torch.where(at_speed & ~stepped, torch.full_like(v, 0.25), torch.ones_like(v))   # rev1d 0.5 -> 0.25
    if "cmd" in st:      # p5 review: under the step command cmd_wait supplies the pressure; a 0.25 gate here paid the
        gate = torch.where(st["cmd"], torch.ones_like(gate), gate)   # policy to stay just under the threshold
    return _phase(env, PUSH) * tk._contact_any(env, "left_deck").float() * v * gate


def push_step_on(env, event: bool = False) -> torch.Tensor:
    """PUSH at the target speed: the push foot back on the deck (and off the floor) - the step-on that ends PUSH.
    event=True (push specialist): pays only the PUSH -> SETTLE transition itself (stepon_evt, 1/dt, once per episode:
    the stage never goes back), so brief taps onto the deck that do not complete the step-on earn nothing."""
    st = seq_update(env)
    if event:
        # review (push p0): scaled by speed / threshold, so "park until push_max_s, step on at rest" earns little
        vf = (_push_v(env, st) / (st["v_tgt"] - st["seq_cfg"]["v_gap"]).clamp(min=0.1)).clamp(0.0, 1.0)
        return st["stepon_evt"].float() * vf / env.step_dt
    at_speed = _push_v(env, st) >= st["v_tgt"] - st["seq_cfg"]["v_gap"]
    _, rf = _feet_floor(env)
    on = tk._contact_any(env, "right_deck") & ~rf & tk._contact_any(env, "left_deck")
    return _phase(env, PUSH) * (at_speed & on).float()


def _right_sole_deck_xy(env) -> torch.Tensor:
    robot, b = _robot(env), tk._board(env)
    c = tk._tc(env)
    if "rsole_id" not in c:
        c["rsole_id"] = robot.find_geoms(r"^right_sole_deck_collision$")[0][0]
    q = b.data.root_link_quat_w
    return quat_apply_inverse(q, robot.data.geom_pos_w[:, c["rsole_id"]] - b.data.root_link_pos_w)


def push_step_reach(env, d_max: float = 0.15) -> torch.Tensor:
    """Step-on shaping, exploit-free (REV). Potential Phi = -dist(push sole, its deck spot STEP_SPOT), TRACKED AND PAID
    ON EVERY PUSH STEP (no speed gate), so the PUSH-phase sum telescopes to dist(first PUSH step) - dist(now): at most
    the ~0.075 m the sole starts from the spot, whatever the foot does in between (a stroke out and back nets 0).
    A non-time-out termination inside PUSH charges the banked approach back (the episode sum is then exactly 0); the
    bank is kept only by leaving PUSH = completing the step-on.

    Fixes s1e: the old term was active only at speed and RESET its potential (no charge) whenever the speed dipped, so
    approach (paid) / dip / retreat (free) / speed up / approach again farmed +2.1 per 22 s logged (46 per episode,
    ~80 % of the return) with no step-on (push_step_on 0.001)."""
    st = seq_update(env)
    push = st["trick"] == PUSH
    p = _right_sole_deck_xy(env)
    tgt = torch.tensor([STEP_SPOT[0], STEP_SPOT[1], 0.004], device=env.device)
    dist = torch.nan_to_num((p - tgt).norm(dim=1), nan=d_max).clamp(max=d_max)
    init = push & (st["step_d"] < 0)
    live = push & ~init
    pay = torch.where(live, st["step_d"] - dist, torch.zeros_like(dist))
    st["step_d0"] = torch.where(init, dist, st["step_d0"])
    term = env.termination_manager.terminated
    pay = torch.where(live & term, pay - (st["step_d0"] - dist), pay)
    st["step_d"] = torch.where(push, dist, st["step_d"])
    return pay / env.step_dt


def _robot_com_deck(env) -> torch.Tensor:
    """(B, 3) whole-robot CoM in the deck frame."""
    robot, b = _robot(env), tk._board(env)
    com = robot.data.data.subtree_com[:, robot.data.indexing.root_body_id]
    return quat_apply_inverse(b.data.root_link_quat_w, com - b.data.root_link_pos_w)


def _left_sole_deck(env) -> torch.Tensor:
    robot, b = _robot(env), tk._board(env)
    c = tk._tc(env)
    if "lsole_id" not in c:
        c["lsole_id"] = robot.find_geoms(r"^left_sole_deck_collision$")[0][0]
    return quat_apply_inverse(b.data.root_link_quat_w, robot.data.geom_pos_w[:, c["lsole_id"]] - b.data.root_link_pos_w)


def push_step_path(env, via: tuple = (-0.050, -0.066, 0.020), pass_r: float = 0.015, inside_y: float = -0.055,
                   com_margin: float = 0.008, com_w: float = 1.0, d_max: float = 0.25, trunk_z_min: float = 0.0,
                   trunk_w: float = 0.0) -> torch.Tensor:
    """Step-on shaping UNDER THE STEP COMMAND (stepfix 2026-11-20, push specialist; needs step_cmd). Potential
    Phi = -(path length from the push sole to its deck spot + com_w * weight-shift deficit), paid as dPhi per step
    from the first command step, telescoping like push_step_reach (a non-time-out termination still in PUSH charges the
    whole banked sum back, so only a completed step-on keeps it: no jackpot, unfarmable).

    Path: until the via point (deck frame, over the deck edge at 16 mm clearance) is passed, length = |p - via| +
    |via - spot|; after (latched once per episode: within pass_r of the via point, or the sole above the deck surface
    inside y > inside_y), |p - spot|. A straight-line potential (push_step_reach) pulls the sole into the deck's side
    edge; this one leads up and over it. Weight shift: max(0, -(CoM_y - board_sole_y) - com_margin): the measured
    hover had the robot CoM 22 mm toward the push foot from the board sole (outside its 17 mm half-width) while every
    valid mid-step stance has it 7-15 mm.
    trunk_w / trunk_z_min (bridge, default off): + trunk_w * max(0, trunk_z_min - trunk height above the deck origin):
    p6's stroke-air states under the command crouch at 96 mm, every valid mid-step stance stands at 119-127 mm, and
    from the crouch the push sole cannot clear the deck."""
    st = seq_update(env)
    if "path_pass" not in st:
        n, dev = env.num_envs, env.device
        st.update(path_pass=torch.zeros(n, dtype=torch.bool, device=dev), path_d=torch.full((n,), -1.0, device=dev),
                  path_d0=torch.zeros(n, device=dev), path_evt=torch.zeros(n, dtype=torch.bool, device=dev))
    dev = env.device
    on = (st["stage"] == ST_PUSH) & st["cmd"]
    p = torch.nan_to_num(_right_sole_deck_xy(env))
    v = torch.tensor(via, device=dev)
    s = torch.tensor([STEP_SPOT[0], STEP_SPOT[1], 0.004], device=dev)
    near = (p - v).norm(dim=1) < pass_r
    above = (p[:, 1] > inside_y) & (p[:, 2] > 0.004) & (p[:, 1] < 0.0)
    newp = on & ~st["path_pass"] & (near | above)
    st["path_evt"] = newp
    st["path_pass"] = st["path_pass"] | newp
    d = torch.where(st["path_pass"], (p - s).norm(dim=1), (p - v).norm(dim=1) + (v - s).norm())
    dy = _robot_com_deck(env)[:, 1] - _left_sole_deck(env)[:, 1]
    d = d + com_w * (-dy - com_margin).clamp(min=0.0)
    if trunk_w > 0:
        b = tk._board(env)
        tz = quat_apply_inverse(b.data.root_link_quat_w, _robot(env).data.root_link_pos_w - b.data.root_link_pos_w)[:, 2]
        d = d + trunk_w * (trunk_z_min - tz).clamp(min=0.0)
    d = d.clamp(max=d_max)
    d = torch.nan_to_num(d, nan=d_max)
    init = on & (st["path_d"] < 0)
    live = on & ~init
    pay = torch.where(live, st["path_d"] - d, torch.zeros_like(d))
    st["path_d0"] = torch.where(init, d, st["path_d0"])
    term = env.termination_manager.terminated
    pay = torch.where(live & term, pay - (st["path_d0"] - d), pay)
    st["path_d"] = torch.where(on, d, st["path_d"])
    return pay / env.step_dt


def m_path_pass(env) -> torch.Tensor:
    """push_step_path via-point passes (once per episode): m_path_pass / (command episodes) = share over the edge."""
    st = seq_update(env)
    return st["path_evt"].float() if "path_evt" in st else torch.zeros(env.num_envs, device=env.device)


def push_stroke(env) -> torch.Tensor:
    """Cost per push-foot lift-off in PUSH: 1 - q, q = q_air * q_sweep in [0, 1] (_track_strokes). Returns 1/dt per
    event, so the weight is the price of one bad stroke; a stroke with a >= 0.45 s swing before it and a >= 6 cm
    backward sweep costs 0.

    Fixes the 4-7 Hz floor tapping (78 "strokes" in 18 s, air ~0.1 s): push_speed / push_progress pay
    speed however it is made, and push_time + push_speed put a premium on reaching speed early, i.e. on cadence. Pricing
    each touch by its quality makes FEW, LONG strokes with a glide (foot in the air ~0.45 s) the cheapest way to the
    same speed. It is a pure cost (q = 1 nets 0), so extra strokes never pay and stepping on is not delayed; a poor
    stroke (-1 x weight) still nets positive through push_progress (10 per m/s gained) once it adds >= 0.1 m/s, so
    pushing is never dearer than refusing to push."""
    st = seq_update(env)
    return st["lift_evt"].float() * (1.0 - st["lift_q"]) / env.step_dt


def glide_stance(env, std: float = 0.025) -> torch.Tensor:
    """RIDE: both soles on the deck, the push (right) sole at its post-step-on spot STEP_SPOT (deck-frame xy, Gaussian
    exp(-d^2 / std^2)): a clean glide stance after the step-on, not a foot perched on the tail or the deck edge."""
    p = _right_sole_deck_xy(env)[:, :2]
    tgt = torch.tensor(STEP_SPOT, device=env.device)
    g = torch.exp(-((p - tgt) ** 2).sum(dim=1) / std ** 2)
    return torch.nan_to_num(_phase(env, RIDE) * tk.soles_on_deck(env).float() * g)


def push_board_foot_floor(env) -> torch.Tensor:
    """Cost: in PUSH the board (left) foot touches the floor."""
    lf, _ = _feet_floor(env)
    return _phase(env, PUSH) * lf.float()


def ride_wheels_down(env, v_min: float = 0.05) -> torch.Tensor:
    """RIDE (settle / manual exit / approach): all four wheels on the ground while rolling."""
    v = board_v_fwd(env).abs()
    return _phase(env, RIDE) * (_wheels_all_down(env) & (v > v_min)).float()


def seq_tail_strike(env) -> torch.Tensor:
    """Cost: the tail kick on the ground in MANUAL / RIDE (scraping the tail on the manual exit)."""
    m = _phase(env, MANUAL) + _phase(env, RIDE)
    return m * tk._contact_any(env, "tail_floor").float()


def manual_quality(env, pts_deg: tuple = (2.0, 8.0, 15.0, 24.0), rate_std: float = 2.0) -> torch.Tensor:
    """MANUAL (replaces the trick env's manual_shape there): trapezoid on the nose-up pitch, 0 at 2 deg, 1 over the
    demo band 8-15 deg, 0 at 24 deg (6 deg short of the 30 deg tail strike), times exp(-(deck pitch rate / 2 rad/s)^2)
    (a held manual, not a see-saw), with both soles on the deck and the tail off the floor.

    Fixes: the s1e manual was in band 4-9 % of the stage on this 9 cm wheelbase board; manual_in pays the whole 3-25 deg
    band equally, so nothing pulled toward the 8-15 deg look or toward holding still. The ramp from 2 deg gives the lift
    a gradient from four-wheel riding (manual_in is a step at 3 deg)."""
    th = torch.rad2deg(tk.front_up_pitch(env))
    a, b, c_, d = pts_deg
    trap = torch.minimum(((th - a) / (b - a)).clamp(0, 1), ((d - th) / (d - c_)).clamp(0, 1))
    w = quat_apply_inverse(tk._board(env).data.root_link_quat_w, tk._board(env).data.root_link_ang_vel_w)
    steady = torch.exp(-(torch.nan_to_num(w[:, 1]) / rate_std) ** 2)
    ok = tk.soles_on_deck(env) & ~tk._back_end_floor(env)
    return _phase(env, MANUAL) * trap * steady * ok.float()


def _head_ref(env) -> dict:
    """Cached: robot head body id and the head-frame gravity direction when the trunk is level and the head joints are
    at HOME (forward kinematics on a CPU copy of the compiled model)."""
    c = tk._tc(env)
    if "head_ref" not in c:
        import mujoco
        robot = _robot(env)
        m = env.sim.mj_model
        d = mujoco.MjData(m)
        q0 = robot.data.default_joint_pos[0].detach().cpu().numpy()
        hit = 0
        for j, q in zip(robot.joint_names, q0):
            for nm in (f"robot/{j}", j):
                jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm)
                if jid >= 0:
                    d.qpos[m.jnt_qposadr[jid]] = float(q)
                    hit += 1
                    break
        assert hit >= 10, f"head ref: only {hit} robot joints resolved in the compiled model"
        mujoco.mj_kinematics(m, d)
        ids = []
        for b in ("trunk_base", "jaw_soft"):
            bid = next(i for i in (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nm) for nm in (f"robot/{b}", b))
                       if i >= 0)
            ids.append(bid)
        Rt, Rh = d.xmat[ids[0]].reshape(3, 3), d.xmat[ids[1]].reshape(3, 3)
        g_ref = (Rt.T @ Rh).T @ [0.0, 0.0, -1.0]
        c["head_ref"] = dict(g=torch.tensor(g_ref, device=env.device, dtype=torch.float),
                             id=robot.find_bodies(r"^jaw_soft$")[0][0])
    return c["head_ref"]


def posture(env, head_std: float = 0.6, yaw_std: float = 0.6) -> torch.Tensor:   # rev1: 0.35/0.4 gave ~0 gradient
    """All phases: the head is level in the world (angle between the head's gravity direction and its HOME-on-a-level-
    trunk direction, Gaussian, std 20 deg: the trunk may lean if the neck compensates = "eyes on the horizon"), times,
    once riding (RIDE / MANUAL / RAMP), the trunk facing along the deck (yaw std 23 deg).

    Fixes the look: hunched over the push with the head pitched toward the floor, and a head nodding/rolling
    on the ramp. The old stack only had head_pose (joint deviation, which a bent trunk escapes) and trunk upright."""
    ref = _head_ref(env)
    robot = _robot(env)
    qh = robot.data.body_link_quat_w[:, ref["id"]]
    g = quat_apply_inverse(qh, torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(env.num_envs, 3))
    ang = torch.acos((g * ref["g"]).sum(dim=1).clamp(-1.0, 1.0))
    level = torch.exp(-(ang / head_std) ** 2)
    ph = seq_update(env)["trick"]
    riding = (ph == RIDE) | (ph == MANUAL) | (ph == RAMP)
    dyaw = wrap_to_pi(_yaw_from_quat(robot.data.root_link_quat_w) - tk._board_yaw(env))
    face = torch.where(riding, torch.exp(-(dyaw / yaw_std) ** 2), torch.ones_like(dyaw))
    return torch.nan_to_num(level * face)


def head_steady(env, yaw_w: float = 0.25, cap: float = 100.0) -> torch.Tensor:
    """Cost: head angular velocity in the world, roll/pitch^2 + yaw_w * yaw^2 (rad/s), capped. "A steady head": the big
    head shell amplifies trunk bob and neck jitter. A SMOOTHNESS tax (it also charges exploration noise on the 4 neck
    actions), so it is staged in after REV_STEP0 (after skill discovery)."""
    ref = _head_ref(env)
    w = torch.nan_to_num(_robot(env).data.body_link_ang_vel_w[:, ref["id"]])
    return (w[:, 0] ** 2 + w[:, 1] ** 2 + yaw_w * w[:, 2] ** 2).clamp(max=cap)


def board_wobble(env, cap: float = 25.0) -> torch.Tensor:
    """Cost: deck angular rate^2 (deck frame), capped: roll + pitch in RIDE (settle / manual set-down / approach: a
    slammed nose and a rocking deck), roll only in RAMP (the transition itself pitches the board at ~v/R, which must
    stay free). Not in PUSH (strokes rock the deck) or MANUAL (the lift needs pitch rate)."""
    w = quat_apply_inverse(tk._board(env).data.root_link_quat_w, tk._board(env).data.root_link_ang_vel_w)
    w = torch.nan_to_num(w)
    ph = seq_update(env)["trick"]
    c = torch.where(ph == RIDE, w[:, 0] ** 2 + w[:, 1] ** 2, torch.where(ph == RAMP, w[:, 0] ** 2, torch.zeros_like(w[:, 0])))
    return c.clamp(max=cap)


def setdown_rate(env, window_s: float = 0.6, cap: float = 25.0) -> torch.Tensor:
    """Cost (manual specialist): deck pitch^2 + roll^2 rate (deck frame, (rad/s)^2, capped) in the first window_s of
    EXIT (scheduler stage clock stage_t): the manual set-down. On top of board_wobble (all RIDE): a 4.5 rad/s slam for
    0.1 s costs ~2.0 at weight -1, a 0.7 rad/s descent ~0.1."""
    st = seq_update(env)
    w = torch.nan_to_num(quat_apply_inverse(tk._board(env).data.root_link_quat_w, tk._board(env).data.root_link_ang_vel_w))
    gate = (st["stage"] == ST_EXIT) & (st["stage_t"] < window_s - 1e-6)
    return gate.float() * (w[:, 0] ** 2 + w[:, 1] ** 2).clamp(max=cap)


# ── manual specialist m2 (2026-11-20, polish for video): costs only, all >= 0 (NEGATIVE weights). Used only by
# microduck_skate_seq_manual_env_cfg; the other tasks never reference them. ──────────────────────────────────────────


def manual_pitch_rate(env, free: float = 0.8, window_s: float = 0.6, cap: float = 6.25,
                      below_deg: float = 10.0, until_wheels_s: float = 0.2, settle: bool = False,
                      until_deg: float | None = None, exit: bool = True) -> torch.Tensor:
    """Cost: deck pitch rate above a dead band, (max(|w_pitch| - free, 0))^2 ((rad/s)^2, capped), in MANUAL while the
    nose-up pitch is below below_deg (the pop, and any drop / re-pop through the low range) and in EXIT for the first
    window_s AND until all four wheels have been down for until_wheels_s (the set-down). A pop / set-down at <= free
    (0.8 rad/s = 46 deg/s) is free; the m1 snap (1 -> 10 deg in 0.06 s, peak 3.6-3.8 rad/s) costs ~0.35 raw per pop, its
    set-down ~0.09. State-gated: neither delaying the pop nor delaying the set-down buys a free snap. The hold
    (>= below_deg) is left out: there the rate is mostly exploration noise (stochastic |w| p90 ~1.0 rad/s, ~0.1 raw/s)
    and the slow 10 <-> 16 deg wobble is priced by the pitch shape, not by rate.
    settle=True (m2rev, default off): also the whole SETTLE stage, any pitch: a snap pop started in late SETTLE (where only
    board_wobble -0.2 prices it, ~20x cheaper) would otherwise skip the gate. Four-wheel SETTLE |w| p90 0.33-0.51 rad/s
    on model_700, so this is ~free for a ridden settle.
    until_deg (m3, default None = off): in MANUAL also gate the whole first rise, until the nose has first reached
    until_deg (latch pop_up, cleared at the reset and outside MANUAL): the m2 gate (< below_deg only) left the last
    degrees of the pop (10 -> 11 deg, the band entry) free. exit (m3, default True = unchanged): False drops the EXIT
    set-down gate (a pop-only instance with its own dead band / weight)."""
    st = seq_update(env)
    w = torch.nan_to_num(quat_apply_inverse(tk._board(env).data.root_link_quat_w, tk._board(env).data.root_link_ang_vel_w))
    th = torch.rad2deg(tk.front_up_pitch(env))
    low = th < below_deg
    if until_deg is not None:
        if "pop_up" not in st:
            st["pop_up"] = torch.zeros_like(st["stage"], dtype=torch.bool)
        man = st["stage"] == ST_MANUAL
        st["pop_up"] = (st["pop_up"] | (th >= until_deg)) & man
        low = low | ~st["pop_up"]
    sd = (st["stage_t"] < window_s - 1e-6) | (st["wheels_on"] < until_wheels_s)
    gate = ((st["stage"] == ST_MANUAL) & low) | ((st["stage"] == ST_EXIT) & sd & exit)
    if settle:
        gate = gate | (st["stage"] == ST_SETTLE)
    return gate.float() * ((w[:, 1].abs() - free).clamp(min=0.0) ** 2).clamp(max=cap)


def stance_width(env, free_m: float = 0.090, unit_m: float = 0.01, cap: float = 3.0) -> torch.Tensor:
    """Cost: lateral (deck y) distance between the two deck soles beyond free_m, in units of unit_m (cm), capped. All
    phases. m1 held the manual with the soles 104 mm apart (four-wheel riding 85 mm): the wide, bow-legged crouch."""
    dy = (_left_sole_deck(env)[:, 1] - _right_sole_deck_xy(env)[:, 1]).abs()
    return torch.nan_to_num(((dy - free_m) / unit_m).clamp(min=0.0, max=cap))


def manual_over(env, lo_deg: float = 14.0, span_deg: float = 4.0) -> torch.Tensor:
    """Cost: MANUAL nose-up pitch above lo_deg, linear, saturating at 1 at lo_deg + span_deg (the m1 hold sat at 14-16 deg,
    28 % of it above the 15 deg band edge, where manual_in + the old trapezoid still paid ~7.5 /s)."""
    th = torch.rad2deg(tk.front_up_pitch(env))
    return _phase(env, MANUAL) * ((th - lo_deg) / span_deg).clamp(0.0, 1.0)


def _stage_gate(env, stages: tuple | None) -> torch.Tensor:
    """p9 helper: 1 where the scheduler stage is in `stages` (None = everywhere)."""
    st = seq_update(env)
    if stages is None:
        return torch.ones_like(st["v_tgt"])
    m = torch.zeros_like(st["stage"], dtype=torch.bool)
    for s in stages:
        m |= st["stage"] == int(s)
    return m.float()


def head_level(env, free_deg: float = 10.0, cap_deg: float = 60.0, stages: tuple | None = None) -> torch.Tensor:
    """Cost: head tilt from level (the posture angle: head gravity direction vs its HOME-on-a-level-trunk direction), L1
    in rad beyond free_deg, capped at cap_deg. All phases. posture's Gaussian (std 0.6 rad) is nearly flat around the
    30-60 deg nods / stare-downs; this term keeps a constant gradient there.
    stages (p9, default None = all phases): only in these scheduler stages (e.g. (ST_PUSH,) = push + step-on)."""
    ref = _head_ref(env)
    qh = _robot(env).data.body_link_quat_w[:, ref["id"]]
    g = quat_apply_inverse(qh, torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(env.num_envs, 3))
    ang = torch.acos((g * ref["g"]).sum(dim=1).clamp(-1.0, 1.0))
    lo, hi = math.radians(free_deg), math.radians(cap_deg)
    out = torch.nan_to_num((ang - lo).clamp(min=0.0, max=hi - lo))
    return out if stages is None else out * _stage_gate(env, stages)


def head_rate(env, free: float = 1.0, cap: float = 25.0, pop_only: bool = False, until_deg: float = 11.0,
              post_s: float = 0.3, max_s: float = 1.2, stages: tuple | None = None) -> torch.Tensor:
    """Cost (m3): head (jaw_soft) angular velocity about the world horizontal axes (pitch + roll, heading-independent)
    above a dead band, (max(|w_xy| - free, 0))^2 ((rad/s)^2), capped. All phases, or with pop_only=True only in the pop
    window: MANUAL from its start until post_s after the nose first reaches until_deg (latch hr_pop_t, cleared at the
    reset and outside MANUAL), at most max_s into MANUAL (a hold parked at 9-11 deg / a low tilted spawn never latches:
    without the cap the pop weight would run through its whole hold).
    Measured on m2 model_1000: the head motion is the NECK (neck_pitch mostly; the
    trunk carries < 5 % of the head pitch-rate mean square): a +25-30 deg nose-down whip at the pop (neck +30, trunk
    -5 deg; head rate peak 260 deg/s) and ~1.2 Hz nods of std 8-9 deg through the hold. head_steady (w^2, -0.04) prices
    a 1.6 rad/s nod at ~-0.1 /s; the dead band keeps slow head adjustments and policy noise free (four-wheel riding
    0.02-0.06 raw/s at free 1.0)."""
    ref = _head_ref(env)
    w = torch.nan_to_num(_robot(env).data.body_link_ang_vel_w[:, ref["id"]])
    c = ((torch.sqrt(w[:, 0] ** 2 + w[:, 1] ** 2) - free).clamp(min=0.0) ** 2).clamp(max=cap)
    if not pop_only:
        return c if stages is None else c * _stage_gate(env, stages)   # p9 stages (default None: all phases)
    st = seq_update(env)
    if "hr_pop_t" not in st:
        st["hr_pop_t"] = torch.full_like(st["stage_t"], float("inf"))
    man = st["stage"] == ST_MANUAL
    hit = man & torch.isinf(st["hr_pop_t"]) & (torch.rad2deg(tk.front_up_pitch(env)) >= until_deg)
    st["hr_pop_t"] = torch.where(hit, st["stage_t"], st["hr_pop_t"])
    st["hr_pop_t"] = torch.where(man, st["hr_pop_t"], torch.full_like(st["hr_pop_t"], float("inf")))
    gate = man & (st["stage_t"] < torch.clamp(st["hr_pop_t"] + post_s, max=max_s))
    return gate.float() * c


# ── push specialist p9 (2026-11-20, demo polish): costs only, all >= 0 (NEGATIVE weights). Used only by
# microduck_skate_seq_push_env_cfg; the other tasks never reference them. Sized on p8 model_1200 from start
# ─────────────────────────────────────────────


def push_foot_height(env, free: float = 0.035, unit: float = 0.01, cap: float = 4.0, z_floor: float = 0.008
                     ) -> torch.Tensor:
    """Cost: PUSH before the step command, push foot in the air (off the floor and the deck): sole height above its
    on-floor height beyond `free`, in units of `unit` (cm), capped. The p8 "chicken kick": the recovery foot hung
    61-64 mm up (median peak of the main strokes) for ~0.3 s instead of a low front-to-back sweep. Under the command the
    foot must rise over the 24 mm deck: not charged there."""
    st = seq_update(env)
    _right_sole_deck_xy(env)                           # caches the sole geom id
    z = _robot(env).data.geom_pos_w[:, tk._tc(env)["rsole_id"], 2] - _origin(env)[:, 2]
    _, rf = _feet_floor(env)
    gate = (st["stage"] == ST_PUSH) & ~st["cmd"] & ~rf & ~tk._contact_any(env, "right_deck")
    return torch.nan_to_num(gate.float() * ((z - z_floor - free) / unit).clamp(min=0.0, max=cap))


def push_trunk_floor(env, z_min: float = 0.080, unit: float = 0.01, cap: float = 3.0) -> torch.Tensor:
    """Cost: PUSH (incl. the step command), trunk height above the deck origin (deck frame) below z_min, in cm, capped.
    p8 pushed at 86 mm median and dipped to 66-70 mm at every touchdown (the deep squat); the push leg was not yet at
    full length there (trunk-sole 124 mm of 134 mm), so >= 80 mm is reachable with a straighter leg."""
    st = seq_update(env)
    b = tk._board(env)
    tz = quat_apply_inverse(b.data.root_link_quat_w, _robot(env).data.root_link_pos_w - b.data.root_link_pos_w)[:, 2]
    gate = (st["stage"] == ST_PUSH).float()
    return torch.nan_to_num(gate * ((z_min - tz) / unit).clamp(min=0.0, max=cap))


def step_lurch(env, free: float = 0.10, unit: float = 0.1, cap: float = 9.0, settle_s: float = 0.2) -> torch.Tensor:
    """Cost: under the step command (and the first settle_s of SETTLE), trunk-vs-board forward speed difference beyond
    `free`, squared in units of `unit` (0.1 m/s), capped. p8's step-on shoved the board back with the board foot to
    throw the trunk forward (board 0.89 -> 0.55 -> 1.05 m/s while the trunk went 0.90 -> 1.07; |trunk - board| p99
    0.43 m/s under the command, 0.10 in SETTLE)."""
    st = seq_update(env)
    gate = ((st["stage"] == ST_PUSH) & st["cmd"]) | ((st["stage"] == ST_SETTLE) & (st["stage_t"] < settle_s - 1e-6))
    dv = (robot_v_fwd(env) - board_v_fwd(env)).abs()
    return torch.nan_to_num(gate.float() * (((dv - free) / unit).clamp(min=0.0) ** 2).clamp(max=cap))


def step_clearance(env, clear: float = 0.015, unit: float = 0.005, cap: float = 3.0, band_out: float = 0.012,
                   band_in: float = 0.004, sole_on_z: float = 0.004) -> torch.Tensor:
    """Cost: under the step command, push sole in the air within the deck-edge band (|y| from band_in inside to band_out
    outside the edge: 62 mm at the tail pad x < -12 mm, 46 mm elsewhere) and less than `clear` above its on-deck
    height (deck frame z 4 mm), in units of `unit`, capped. p8 crossed the edge 7.5 mm up (median; p10 2.5-4 mm):
    the swipe that barely clears. Only an attempt pays it (a refusal never crosses the edge): <= ~0.5 per step-on."""
    st = seq_update(env)
    p = torch.nan_to_num(_right_sole_deck_xy(env))
    edge = torch.where(p[:, 0] < -0.012, torch.full_like(p[:, 0], 0.062), torch.full_like(p[:, 0], 0.046))
    ay = -p[:, 1]
    band = (ay > edge - band_in) & (ay < edge + band_out)
    _, rf = _feet_floor(env)
    gate = (st["stage"] == ST_PUSH) & st["cmd"] & ~rf & ~tk._contact_any(env, "right_deck") & band
    return gate.float() * ((clear - (p[:, 2] - sole_on_z)) / unit).clamp(min=0.0, max=cap)


def m_cmd_start_n(env) -> torch.Tensor:
    """p9 pace metric: step-command latches of `start` spawns (m_cmd_start_t / m_cmd_start_n = mean push time from rest
    to the command)."""
    st = seq_update(env)
    if "cmd_t" not in st:
        return torch.zeros_like(st["v_tgt"])
    edge = st["cmd"] & (st["cmd_t"] <= env.step_dt + 1e-6) & (st["spawn"] == SPAWNS.index("start"))
    return edge.float()


def m_cmd_start_t(env) -> torch.Tensor:
    return m_cmd_start_n(env) * seq_update(env)["stage_t"]


def stance_distance(env, units: tuple = (0.01, 0.01, 0.01, 5.0, 5.0, 0.5), joint_w: float = 0.5, cap: float = 10.0
                    ) -> torch.Tensor:
    """p9b: hand-off stance distance in tolerance units, sum of min(err / unit, cap) over left sole, right sole (cm),
    CoM height (cm), trunk tilt x / y (5 deg) and joint_w x joint L2 (0.5 rad). p8 hand-off ~15.7, the manual's own
    riding state ~2.1."""
    e = _stance_err(env)
    parts = [e["dl"] / units[0], e["dr"] / units[1], e["dz"] / units[2], e["tx"] / units[3], e["ty"] / units[4],
             joint_w * e["dj"] / units[5]]
    return sum(p.clamp(max=cap) for p in parts)


def settle_stance_path(env, units: tuple = (0.01, 0.01, 0.01, 5.0, 5.0, 0.5), joint_w: float = 0.5,
                       cap: float = 10.0) -> torch.Tensor:
    """p9b shaping (POSITIVE weight; a signed potential, not a penalty): Phi = -stance_distance in SETTLE, paid as dPhi
    per step from the first SETTLE step (the step-on), so the SETTLE sum telescopes to D(step-on) - D(now): bounded by
    the distance at the step-on, unfarmable (out and back nets 0). A non-time-out termination in SETTLE (fall, board
    lost, SETTLE overtime) charges the banked sum back; only the success (a time-out term) keeps it."""
    st = seq_update(env)
    if "sp_d" not in st:
        st["sp_d"] = torch.full_like(st["v_tgt"], -1.0)
        st["sp_d0"] = torch.zeros_like(st["v_tgt"])
    on = st["stage"] == ST_SETTLE
    d = stance_distance(env, units, joint_w, cap)
    init = on & (st["sp_d"] < 0)
    live = on & ~init
    pay = torch.where(live, st["sp_d"] - d, torch.zeros_like(d))
    st["sp_d0"] = torch.where(init, d, st["sp_d0"])
    term = env.termination_manager.terminated
    pay = torch.where(live & term, pay - (st["sp_d0"] - d), pay)
    st["sp_d"] = torch.where(on, d, st["sp_d"])
    return pay / env.step_dt


def m_settle(env) -> torch.Tensor:
    """p9b: steps in SETTLE (m_stance_ok / m_settle = share of SETTLE time in the hand-off stance)."""
    return (seq_update(env)["stage"] == ST_SETTLE).float()


def m_stance_ok(env) -> torch.Tensor:
    st = seq_update(env)
    return ((st["stage"] == ST_SETTLE) & (st.get("stance_t", torch.zeros_like(st["v_tgt"])) > 0)).float()


def m_stance_d(env) -> torch.Tensor:
    """p9b: stance_distance summed over SETTLE steps (m_stance_d / m_settle = mean distance in SETTLE)."""
    return (seq_update(env)["stage"] == ST_SETTLE).float() * stance_distance(env)


# ── push specialist p10 (2026-11-20, "deliberate push" polish). Costs only,
# all >= 0 (NEGATIVE weights); only microduck_skate_seq_push_env_cfg uses them. push_cadence / push_stroke_shape and the
# m_p10_* metrics need SEQ_DEFAULTS p10_track=True (else they read 0). Sized on p9b model_2000 from start:
# touchdowns every 0.36 s, contacts 60 ms, sweep 47 mm, swing sole pitch 31 deg, head z bob 21 mm. ──


def push_cadence(env, t_ok: float = 0.65, t_bad: float = 0.30) -> torch.Tensor:
    """Cost per push-foot floor touchdown in PUSH before the step command (p10_td_evt): the touchdown interval below
    t_ok, linear to 1 at t_bad, i.e. a minimum touchdown interval. Returns 1/dt per event (weight = price of a tap
    stroke). The spawn contact and the first touchdown of a mid-stroke spawn are free."""
    st = seq_update(env)
    if "p10_td_evt" not in st:
        return torch.zeros_like(st["v_tgt"])
    c = ((t_ok - st["p10_gap"]) / (t_ok - t_bad)).clamp(0.0, 1.0)
    return st["p10_td_evt"].float() * c / env.step_dt


def push_stroke_shape(env, c_lo: float = 0.04, reach: float = 0.12, c_min: float = 0.12, c_max: float = 0.28,
                      s_lo: float = 0.03, s_hi: float = 0.10) -> torch.Tensor:
    """Cost per push-foot lift-off in PUSH before the step command (p10_lo_evt): 1 - (q_contact + q_sweep) / 2.
    q_contact ramps 0 -> 1 from c_lo to the reach-limited contact target clamp(reach / v_touchdown, c_min, c_max)
    (a planted sole travels back at the board speed, so at 1 m/s a 12 cm leg sweep is only 0.12 s of contact);
    q_sweep ramps 0 -> 1 over s_lo..s_hi of backward planted-sole travel in the deck frame. ADDITIVE (push_stroke's
    product q_air * q_sweep sat at ~0.1 = a flat per-stroke tax with no gradient). A tap (sub-step contact) costs 1."""
    st = seq_update(env)
    if "p10_lo_evt" not in st:
        return torch.zeros_like(st["v_tgt"])
    c_ref = (reach / st["p10_lo_v"].clamp(min=0.1)).clamp(c_min, c_max)
    qc = ((st["p10_contact"] - c_lo) / (c_ref - c_lo).clamp(min=1e-3)).clamp(0.0, 1.0)
    qs = ((st["p10_sweep"] - s_lo) / (s_hi - s_lo)).clamp(0.0, 1.0)
    return st["p10_lo_evt"].float() * (1.0 - 0.5 * (qc + qs)) / env.step_dt


def push_overspeed(env, margin: float = 0.15, unit: float = 0.1, cap: float = 5.0) -> torch.Tensor:
    """Cost: PUSH (incl. the step command), board speed above v_tgt + margin, in units of `unit` (0.1 m/s), capped. p9b
    reached the command at 0.7-0.9 m/s worth of ramp but rode at 1.46 m/s (strokes past the ramp pay nothing and were
    a free habit; the hand-off arrived at 1.31 m/s, above the manual's 0.5-1.0 training range). With the push_ramp
    slew (no jackpot) this only closes the window where extra strokes are free; it is not the pacing mechanism."""
    st = seq_update(env)
    v = torch.nan_to_num(board_v_fwd(env))
    gate = (st["stage"] == ST_PUSH).float()
    return gate * ((v - st["v_tgt"] - margin) / unit).clamp(min=0.0, max=cap)


def push_swing_foot(env, pitch_free: float = 15.0, yaw_free: float = 25.0, unit_deg: float = 10.0, cap: float = 3.0,
                    clear: float = 0.012, z_floor: float = 0.008, clear_unit: float = 0.005, air_min: float = 0.06,
                    half_len: float = 0.0228) -> torch.Tensor:
    """Cost: PUSH before the step command, push foot in the air for >= air_min (not on the deck): sole pitch along the
    foot beyond pitch_free (toe down / heel down) and sole heading vs the deck x beyond yaw_free (twist), each in
    unit_deg and capped, plus the lower sole end closer than `clear` to its floor height (z_floor) in clear_unit.
    Sole roll is NOT charged (no ankle roll: it follows hip roll + trunk roll, which the push stance needs)."""
    st = seq_update(env)
    robot = _robot(env)
    _right_sole_deck_xy(env)                           # caches the sole geom id
    gid = tk._tc(env)["rsole_id"]
    gq = robot.data.geom_quat_w[:, gid]
    ex = torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(env.num_envs, 3)
    sx_w = quat_apply(gq, ex)
    sx_d = quat_apply_inverse(tk._board(env).data.root_link_quat_w, sx_w)
    pitch = torch.rad2deg(torch.asin(sx_w[:, 2].clamp(-1.0, 1.0)).abs())
    yaw = torch.rad2deg(torch.atan2(sx_d[:, 1], sx_d[:, 0]).abs())
    z = robot.data.geom_pos_w[:, gid, 2] - _origin(env)[:, 2]
    z_low = z - half_len * torch.sin(torch.deg2rad(pitch))
    sens = env.scene.sensors["right_foot_floor"].data
    air = sens.current_air_time
    air = air.min(dim=-1).values if air.dim() > 1 else air
    _, rf = _feet_floor(env)
    gate = ((st["stage"] == ST_PUSH) & ~st.get("cmd", torch.zeros_like(rf)) & ~rf
            & ~tk._contact_any(env, "right_deck") & (air >= air_min))
    c = (((pitch - pitch_free) / unit_deg).clamp(0.0, cap) + ((yaw - yaw_free) / unit_deg).clamp(0.0, cap)
         + ((clear - (z_low - z_floor)) / clear_unit).clamp(0.0, cap))
    return torch.nan_to_num(gate.float() * c)


def head_bob(env, vz_free: float = 0.05, vz_unit: float = 0.1, roll_free: float = 0.5, roll_unit: float = 1.0,
             cap: float = 9.0, stages: tuple | None = None) -> torch.Tensor:
    """Cost: head (jaw_soft) vertical speed in the world beyond vz_free (units vz_unit) squared + head roll rate about
    the board's horizontal heading beyond roll_free (units roll_unit) squared, capped; only in `stages` (None = all).
    p9b bobbed the head 21 mm (p95-p5, 8 % of the height) with |vz| p90 0.20 m/s and rolled ~5 deg per stroke (roll rate
    p90 1.2 rad/s) - under head_rate's 2 rad/s dead band, which is sized for the pitch nod."""
    ref = _head_ref(env)
    robot = _robot(env)
    vz = torch.nan_to_num(robot.data.body_link_lin_vel_w[:, ref["id"], 2])
    w = torch.nan_to_num(robot.data.body_link_ang_vel_w[:, ref["id"]])
    yaw = tk._board_yaw(env)
    rr = w[:, 0] * torch.cos(yaw) + w[:, 1] * torch.sin(yaw)
    c = (((vz.abs() - vz_free) / vz_unit).clamp(min=0.0) ** 2 + ((rr.abs() - roll_free) / roll_unit).clamp(min=0.0) ** 2)
    c = c.clamp(max=cap)
    return c if stages is None else c * _stage_gate(env, stages)


P10_HEAD_JOINTS = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")


def settle_head_pose(env, free: float = 0.12, cap: float = 3.0, stages: tuple | None = (ST_SETTLE,)) -> torch.Tensor:
    """Cost: the four neck / head joints away from the HANDOFF_STANCE values (the manual specialist's own riding head
    pose: neck 0.41, head pitch 0.28, head yaw -0.14, roll 0 rad), L1 beyond `free` per joint (rad), capped; SETTLE by
    default. p9b settled with head pitch up to 1.2 rad (p90; the "pitched back" zig-zag) and head yaw +0.30 rad (the
    head turned ~24 deg off the travel direction). Toward the hand-off target, so it never fights settle_stance_path."""
    c = tk._tc(env)
    robot = _robot(env)
    if "p10_hj" not in c:
        ids = [robot.find_joints(rf"^{j}$")[0][0] for j in P10_HEAD_JOINTS]
        c["p10_hj"] = torch.tensor(ids, device=env.device, dtype=torch.long)
        c["p10_hq"] = torch.tensor([HANDOFF_STANCE["joints"][j] for j in P10_HEAD_JOINTS], device=env.device)
    e = (robot.data.joint_pos[:, c["p10_hj"]] - c["p10_hq"]).abs() - free
    out = torch.nan_to_num(e.clamp(min=0.0).sum(dim=1).clamp(max=cap))
    return out if stages is None else out * _stage_gate(env, stages)


def m_p10_td(env) -> torch.Tensor:
    """p10 metrics (PUSH before the command): touchdown events; m_p10_gap / m_p10_td = mean touchdown interval (s)."""
    st = seq_update(env)
    return st["p10_td_evt"].float() if "p10_td_evt" in st else torch.zeros_like(st["v_tgt"])


def m_p10_gap(env) -> torch.Tensor:
    st = seq_update(env)
    return (st["p10_td_evt"].float() * st["p10_gap"]) if "p10_td_evt" in st else torch.zeros_like(st["v_tgt"])


def m_p10_lo(env) -> torch.Tensor:
    """p10: lift-off events; m_p10_contact / m_p10_lo = mean contact (s), m_p10_sweep / m_p10_lo = mean sweep (m)."""
    st = seq_update(env)
    return st["p10_lo_evt"].float() if "p10_lo_evt" in st else torch.zeros_like(st["v_tgt"])


def m_p10_contact(env) -> torch.Tensor:
    st = seq_update(env)
    return (st["p10_lo_evt"].float() * st["p10_contact"]) if "p10_lo_evt" in st else torch.zeros_like(st["v_tgt"])


def m_p10_sweep(env) -> torch.Tensor:
    st = seq_update(env)
    return (st["p10_lo_evt"].float() * st["p10_sweep"]) if "p10_lo_evt" in st else torch.zeros_like(st["v_tgt"])


# ── push specialist p12 (2026-11-20, posture polish). Costs only, all >= 0
# (NEGATIVE weights); only microduck_skate_seq_push_env_cfg uses them; they need SEQ_DEFAULTS p12_track=True (else 0).
# Measured on p11 model_2799 from start (32 det / stoch): per stroke the head YAW joint swings 44 deg
# p2p (head heading in the deck frame 41 deg), head pitch 20, neck 16, head roll 15 deg, head-joint speed sum p50 4.6 /
# p90 14.5 rad/s - the head is flung as a reaction mass, which head_bob (head vz + roll RATE, free 0.05 m/s / 0.5 rad/s)
# barely saw (head z p2p only 8 mm, head vz rms 0.04 m/s). The trunk pitches 10.6 deg and rises / drops 10 mm per
# stroke (stance knee 10 deg, hip pitch 16 deg). The swing sole pitches toe-down 50 deg: push_swing_foot (free 15,
# unit 10, cap 3) is saturated at its cap = zero gradient. Step-on (command -> SETTLE + 0.4 s): the trunk rises 26 mm
# above the push crouch, head pitch peaks 0.97 rad and head yaw 0.71 rad (HANDOFF 0.28 / -0.14), head angular rate
# peak 9.7 rad/s - settle_head_pose only acts in SETTLE, so the command part of the step-on was free. ──


def _head_joint_ids(env) -> tuple[torch.Tensor, torch.Tensor]:
    """Cached (shared with settle_head_pose): the four neck / head joint ids and their HANDOFF_STANCE values."""
    c = tk._tc(env)
    if "p10_hj" not in c:
        robot = _robot(env)
        ids = [robot.find_joints(rf"^{j}$")[0][0] for j in P10_HEAD_JOINTS]
        c["p10_hj"] = torch.tensor(ids, device=env.device, dtype=torch.long)
        c["p10_hq"] = torch.tensor([HANDOFF_STANCE["joints"][j] for j in P10_HEAD_JOINTS], device=env.device)
    return c["p10_hj"], c["p10_hq"]


def _p12_trunk(env) -> tuple[torch.Tensor, torch.Tensor]:
    """Trunk (root) height above the env origin (m) and trunk pitch (deg, + = nose down), world frame."""
    robot = _robot(env)
    z = robot.data.root_link_pos_w[:, 2] - _origin(env)[:, 2]
    fx = quat_apply(robot.data.root_link_quat_w, torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(env.num_envs, 3))
    return torch.nan_to_num(z), torch.nan_to_num(torch.rad2deg(-torch.asin(fx[:, 2].clamp(-1.0, 1.0))))


def _track_p12(env, st) -> None:
    """p12 (seq_update, after the stage / command update; SEQ_DEFAULTS p12_track): exponential running means (time
    constant p12_tau s) of the head joints, trunk height and trunk pitch, reset to the current value on a new episode
    (episode clock not advancing, as _track_p10); at the step-command latch the means are frozen as the push-crouch
    reference (p12_zc / p12_qc); p12_win = the step-on window, from the command to p12_win_s into SETTLE."""
    cfg = st["seq_cfg"]
    dt, t = env.step_dt, tk._t(env)
    ids, _ = _head_joint_ids(env)
    q = torch.nan_to_num(_robot(env).data.joint_pos[:, ids])
    z, p = _p12_trunk(env)
    b = torch.zeros_like(st["v_tgt"], dtype=torch.bool)
    if "p12_t_last" not in st:
        st.update(p12_t_last=torch.full_like(z, 1e9), p12_q=q.clone(), p12_z=z.clone(), p12_p=p.clone(),
                  p12_zc=z.clone(), p12_qc=q.clone(), p12_cprev=b.clone(), p12_seen=b.clone(), p12_win=b.clone())
    fresh = t <= st["p12_t_last"] + 1e-6
    st["p12_t_last"] = t.clone()
    a = min(1.0, dt / float(cfg.get("p12_tau", 0.5)))
    st["p12_q"] = torch.where(fresh.unsqueeze(1), q, st["p12_q"] + a * (q - st["p12_q"]))
    st["p12_z"] = torch.where(fresh, z, st["p12_z"] + a * (z - st["p12_z"]))
    st["p12_p"] = torch.where(fresh, p, st["p12_p"] + a * (p - st["p12_p"]))
    cmd = st.get("cmd", b)
    rise = cmd & ~(st["p12_cprev"] & ~fresh)
    st["p12_zc"] = torch.where(rise, st["p12_z"], st["p12_zc"])
    st["p12_qc"] = torch.where(rise.unsqueeze(1), st["p12_q"], st["p12_qc"])
    st["p12_seen"] = (st["p12_seen"] & ~fresh) | rise
    st["p12_cprev"] = cmd.clone()
    stage = st["stage"]
    st["p12_win"] = st["p12_seen"] & (((stage == ST_PUSH) & cmd)
                                      | ((stage == ST_SETTLE) & (st["stage_t"] < float(cfg.get("p12_win_s", 0.4)))))


def _p12_stroking(st) -> torch.Tensor:
    """PUSH before the step command (the stroke cycle)."""
    return (st["stage"] == ST_PUSH) & ~st.get("cmd", torch.zeros_like(st["stage"], dtype=torch.bool))


def p12_head_steady(env, free: float = 0.05, rate_free: float = 2.0, rate_w: float = 0.25, cap: float = 6.0
                    ) -> torch.Tensor:
    """Cost, PUSH before the step command only: the four neck / head joints away from their own running mean (L1
    beyond `free` rad per joint: the per-stroke swing, not the pose) + rate_w x the head-joint speed sum beyond
    rate_free (rad/s), capped. Not a fixed pose (the push head pose, neck -0.30 / head pitch -0.30 rad medians, differs
    from the hand-off one) and not in the step-on, where the head must move to the hand-off pose (p12_stepon_head
    allows that move and prices the overshoot; settle_head_pose owns SETTLE)."""
    st = seq_update(env)
    if "p12_q" not in st:
        return torch.zeros_like(st["v_tgt"])
    ids, _ = _head_joint_ids(env)
    robot = _robot(env)
    dev = ((robot.data.joint_pos[:, ids] - st["p12_q"]).abs() - free).clamp(min=0.0).sum(dim=1)
    rate = rate_w * (robot.data.joint_vel[:, ids].abs().sum(dim=1) - rate_free).clamp(min=0.0)
    return torch.nan_to_num((dev + rate).clamp(max=cap) * _p12_stroking(st).float())


def p12_trunk_steady(env, z_free: float = 0.003, z_unit: float = 0.005, p_free: float = 2.0, p_unit: float = 4.0,
                     cap: float = 9.0, rise_max: float = 0.118, rise_unit: float = 0.005, rise_cap: float = 6.0
                     ) -> torch.Tensor:
    """Cost: in PUSH before the step command, trunk height / pitch away from their running means (squared, beyond
    z_free m / p_free deg in z_unit / p_unit, capped) - the per-stroke bob and rock; plus, in the step-on window, the
    trunk above rise_max over the deck origin (deck frame, linear in rise_unit, capped). The step-on itself NEEDS a
    rise: the push crouch is ~85 mm over the deck and push_step_path asks >= 112 mm for the sole to clear the deck
    (stepfix), which is the 26 mm stand-up the review saw; so only the overshoot past 118 mm is priced here (p11 peaks
    at ~the 112 mm floor). Lowering that floor is the lever for a smaller stand-up (not done: step-on risk)."""
    st = seq_update(env)
    if "p12_z" not in st:
        return torch.zeros_like(st["v_tgt"])
    z, p = _p12_trunk(env)
    osc = ((((z - st["p12_z"]).abs() - z_free) / z_unit).clamp(min=0.0) ** 2
           + (((p - st["p12_p"]).abs() - p_free) / p_unit).clamp(min=0.0) ** 2).clamp(max=cap)
    b = tk._board(env)
    tz = quat_apply_inverse(b.data.root_link_quat_w, _robot(env).data.root_link_pos_w - b.data.root_link_pos_w)[:, 2]
    rise = ((tz - rise_max) / rise_unit).clamp(0.0, rise_cap) * st["p12_win"].float()
    return torch.nan_to_num(osc * _p12_stroking(st).float() + rise)


def p12_stepon_head(env, free: float = 0.10, rate_free: float = 2.0, rate_w: float = 0.5, cap: float = 9.0
                    ) -> torch.Tensor:
    """Cost in the step-on window (p12_win): each head joint outside the interval between its push-crouch mean at the
    command and its HANDOFF_STANCE value (beyond `free` rad, L1 - a monotone move from the push head pose to the hand-off
    head pose is free, the overshoot / flail is not) + rate_w x the head's world angular speed beyond rate_free (rad/s),
    capped."""
    st = seq_update(env)
    if "p12_win" not in st:
        return torch.zeros_like(st["v_tgt"])
    ids, hq = _head_joint_ids(env)
    robot = _robot(env)
    q = robot.data.joint_pos[:, ids]
    lo, hi = torch.minimum(st["p12_qc"], hq), torch.maximum(st["p12_qc"], hq)
    out_box = ((lo - q).clamp(min=0.0) + (q - hi).clamp(min=0.0) - free).clamp(min=0.0).sum(dim=1)
    w = robot.data.body_link_ang_vel_w[:, _head_ref(env)["id"]].norm(dim=1)
    c = (out_box + rate_w * (w - rate_free).clamp(min=0.0)).clamp(max=cap)
    return torch.nan_to_num(c * st["p12_win"].float())


def p12_swing_sole(env, pitch_free: float = 15.0, unit_deg: float = 15.0, cap: float = 6.0, air_min: float = 0.06
                   ) -> torch.Tensor:
    """Cost: push_swing_foot's gate (PUSH before the command, push foot in the air >= air_min, not on the deck), sole
    pitch |toe down / up| beyond pitch_free in unit_deg, linear, capped at 6 (= 110 deg): p11 swings at 50 deg where
    push_swing_foot (cap 3 = 45 deg) is flat."""
    st = seq_update(env)
    robot = _robot(env)
    _right_sole_deck_xy(env)                           # caches the sole geom id
    gid = tk._tc(env)["rsole_id"]
    sx_w = quat_apply(robot.data.geom_quat_w[:, gid],
                      torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(env.num_envs, 3))
    pitch = torch.rad2deg(torch.asin(sx_w[:, 2].clamp(-1.0, 1.0)).abs())
    sens = env.scene.sensors["right_foot_floor"].data
    air = sens.current_air_time
    air = air.min(dim=-1).values if air.dim() > 1 else air
    _, rf = _feet_floor(env)
    gate = ((st["stage"] == ST_PUSH) & ~st.get("cmd", torch.zeros_like(rf)) & ~rf
            & ~tk._contact_any(env, "right_deck") & (air >= air_min))
    return torch.nan_to_num(gate.float() * ((pitch - pitch_free) / unit_deg).clamp(0.0, cap))


def m_p12_win(env) -> torch.Tensor:
    """p12 metrics: step-on window steps; m_p12_rise / m_p12_win = mean trunk height over the push crouch (m) there."""
    st = seq_update(env)
    return st["p12_win"].float() if "p12_win" in st else torch.zeros_like(st["v_tgt"])


def m_p12_rise(env) -> torch.Tensor:
    st = seq_update(env)
    if "p12_win" not in st:
        return torch.zeros_like(st["v_tgt"])
    return st["p12_win"].float() * (_p12_trunk(env)[0] - st["p12_zc"])


def m_p12_headrate(env) -> torch.Tensor:
    """p12: head-joint speed sum (rad/s) in PUSH before the command (per-step; / m_push-style step counts)."""
    st = seq_update(env)
    ids, _ = _head_joint_ids(env)
    g = ((st["stage"] == ST_PUSH) & ~st.get("cmd", torch.zeros_like(st["stage"], dtype=torch.bool))).float()
    return g * torch.nan_to_num(_robot(env).data.joint_vel[:, ids].abs().sum(dim=1))


# ── push specialist p13 (2026-11-20, posture polish 2). Costs only, all >= 0
# (NEGATIVE weights); only microduck_skate_seq_push_env_cfg uses them; the window terms need p12_track, p13_hover needs
# p13_track (else 0). Root causes measured on p12 model_3398 from start:
#   - "toe-down swing foot" = sole ROLL, not pitch: airborne sole roll -47 deg (p50), pitch +2 deg, total tilt 49.5 deg.
#     p12_swing_sole / push_swing_foot price pitch only (their gated pitch p50 +3.6 deg: already "flat"); the 50-deg
#     pitch an12 called pitch_swing_min is the toe-off transient (lift-off -37 deg, flat by +0.10 s), outside their
#     air >= 0.06 gate. The roll = right hip roll pinned at its -22 deg joint limit (FK: -27 deg rel. trunk) + ~20 deg
#     trunk roll toward the board: the foot is lifted by abduction, 44-48 mm up, 142 mm outboard of the deck centre -
#     never over the deck in PUSH, so no clearance needs it. Flat = (trunk roll + hip roll) ~ 0, lift by knee / hip pitch.
#   - step-on pop: trunk (deck frame) 87 mm at spawn, 82 / 81 / 77 mm per stroke (progressive crouch), 72 mm at the
#     command latch (a stroke touchdown), 110.7 mm peak at +0.2-0.34 s (push_step_path trunk_z_min 0.112 asks exactly
#     that; the p12_trunk_steady 118 mm rise cap never fired), the trunk pitched 34 deg nose-UP while the push sole is
#     already on the deck (+0.24 s), then 88-90 mm in SETTLE. Whole-body IK: the push sole passes
#     the deck-edge via point 16 mm up, level, board sole held, CoM over the board sole, down to a trunk of 81 mm.
#   - hover: per swing (0.56 s) the sole moves < 0.15 m/s relative to the deck for 0.28 s (p50, p90 0.40 s) - the
#     0.8 s cadence target filled with a parked foot. ──


def _track_p13(env, st) -> None:
    """p13 hover clock (seq_update after the stage / command update; SEQ_DEFAULTS p13_track): push-sole speed relative to
    the deck (finite difference of its deck-frame position, 0 on a new episode), the per-swing still time (reset on any
    floor / deck contact or outside PUSH-before-the-command) and p13_hov = still now with that time past p13_hover_s."""
    cfg = st["seq_cfg"]
    dt, t = env.step_dt, tk._t(env)
    p = torch.nan_to_num(_right_sole_deck_xy(env))
    if "p13_t_last" not in st:
        z = torch.zeros_like(st["v_tgt"])
        st.update(p13_t_last=torch.full_like(z, 1e9), p13_prev=p.clone(), p13_v=z.clone(), p13_still=z.clone(),
                  p13_hov=torch.zeros_like(z, dtype=torch.bool))
    fresh = t <= st["p13_t_last"] + 1e-6
    st["p13_t_last"] = t.clone()
    v = torch.where(fresh, torch.zeros_like(t), (p - st["p13_prev"]).norm(dim=1) / dt)
    st["p13_prev"] = p.clone()
    st["p13_v"] = v
    _, rf = _feet_floor(env)
    gate = (st["stage"] == ST_PUSH) & ~st.get("cmd", torch.zeros_like(rf)) & ~rf & ~tk._contact_any(env, "right_deck")
    gate = gate & ~fresh
    still = gate & (v < float(cfg.get("p13_v_still", 0.12)))
    st["p13_still"] = torch.where(gate, st["p13_still"] + dt * still.float(), torch.zeros_like(t))
    st["p13_hov"] = still & (st["p13_still"] > float(cfg.get("p13_hover_s", 0.25)))
    # p13rev: trunk deck-frame z at the episode's first step (p13_stepon_pose's rise reference: step_mid spawns start
    # mid-step at 120-128 mm, an unavoidable rise cost ~3 units for 0.3 s; start spawns ~87 mm and step_on <= 98 mm -> no-op)
    tz = torch.nan_to_num(_trunk_deck_z(env))
    st["p13_z0"] = torch.where(fresh, tz, st["p13_z0"]) if "p13_z0" in st else tz.clone()


def _trunk_deck_z(env) -> torch.Tensor:
    b = tk._board(env)
    return quat_apply_inverse(b.data.root_link_quat_w, _robot(env).data.root_link_pos_w - b.data.root_link_pos_w)[:, 2]


def p13_hover(env) -> torch.Tensor:
    """Cost 1 per step while the swing sole is parked (p13_hov): the weight is the price per second of hover past the
    free p13_hover_s per swing."""
    st = seq_update(env)
    return st["p13_hov"].float() if "p13_hov" in st else torch.zeros_like(st["v_tgt"])


def _swing_gate_tilt(env, air_min: float) -> tuple[torch.Tensor, torch.Tensor]:
    """(gate, tilt deg): PUSH before the command, push foot off the floor and the deck for >= air_min; total sole tilt =
    angle between the sole normal and the world vertical (roll and pitch together)."""
    st = seq_update(env)
    robot = _robot(env)
    _right_sole_deck_xy(env)                           # caches the sole geom id
    gid = tk._tc(env)["rsole_id"]
    nz = quat_apply(robot.data.geom_quat_w[:, gid],
                    torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(env.num_envs, 3))[:, 2]
    tilt = torch.rad2deg(torch.acos(nz.abs().clamp(max=1.0)))
    air = env.scene.sensors["right_foot_floor"].data.current_air_time
    air = air.min(dim=-1).values if air.dim() > 1 else air
    _, rf = _feet_floor(env)
    gate = ((st["stage"] == ST_PUSH) & ~st.get("cmd", torch.zeros_like(rf)) & ~rf
            & ~tk._contact_any(env, "right_deck") & (air >= air_min))
    return gate, torch.nan_to_num(tilt)


def p13_swing_tilt(env, free_deg: float = 20.0, unit_deg: float = 10.0, cap: float = 4.0, air_min: float = 0.02
                   ) -> torch.Tensor:
    """Cost: swing sole TILT (roll + pitch, vs the world vertical) beyond free_deg in unit_deg, linear, capped; gate as
    push_swing_foot but from air_min 0.02 s (the toe-off transient included). p12 swings at 49.5 deg (roll 47)."""
    gate, tilt = _swing_gate_tilt(env, air_min)
    return gate.float() * ((tilt - free_deg) / unit_deg).clamp(0.0, cap)


def p13_trunk_hold(env, z_lo: float = 0.082, z_hi: float = 0.096, unit: float = 0.005, cap: float = 4.0
                   ) -> torch.Tensor:
    """Cost: trunk height above the deck origin (deck frame) out of the push-stance band - below z_lo anywhere in PUSH
    (incl. the step command: the progressive crouch 87 -> 77 mm and the 72 mm launch dip), above z_hi before the command
    - linear in `unit`, capped. Holding the push height is what keeps the step-on rise small (p13_stepon_pose caps it)."""
    st = seq_update(env)
    tz = torch.nan_to_num(_trunk_deck_z(env), nan=z_lo)
    push = st["stage"] == ST_PUSH
    pre = push & ~st.get("cmd", torch.zeros_like(push))
    c = ((z_lo - tz) / unit).clamp(0.0, cap) * push.float() + ((tz - z_hi) / unit).clamp(0.0, cap) * pre.float()
    return c.clamp(max=cap)


def p13_stepon_pose(env, rise_hi: float = 0.098, unit: float = 0.005, rise_cap: float = 4.0, p_free: float = 12.0,
                    p_unit: float = 6.0, p_cap: float = 3.0, head_free: float = 0.03, head_unit: float = 0.1,
                    head_cap: float = 3.0, cap: float = 9.0) -> torch.Tensor:
    """Cost in the step-on window (p12_win: command -> SETTLE + p12_win_s): (a) trunk above rise_hi over the deck origin
    (= the 86 mm push target + 12 mm; SETTLE sits at 88-90 mm and the IK step-on passes at 81-91 mm) in `unit`, capped;
    (b) trunk pitch (world, either way) beyond p_free deg in p_unit (p12 leans back 34 deg nose-up while standing tall);
    (c) neck_pitch / head_pitch outside the interval between their push-crouch mean and HANDOFF_STANCE beyond head_free
    rad, in head_unit (the kinked neck), capped. Total capped."""
    st = seq_update(env)
    if "p12_win" not in st:
        return torch.zeros_like(st["v_tgt"])
    tz = torch.nan_to_num(_trunk_deck_z(env))
    _, p = _p12_trunk(env)
    ids, hq = _head_joint_ids(env)
    q = _robot(env).data.joint_pos[:, ids[:2]]
    lo, hi = torch.minimum(st["p12_qc"][:, :2], hq[:2]), torch.maximum(st["p12_qc"][:, :2], hq[:2])
    out = ((lo - q).clamp(min=0.0) + (q - hi).clamp(min=0.0) - head_free).clamp(min=0.0).sum(dim=1)
    # p13rev: never charge the spawn height itself (rise above max(rise_hi, trunk z at the episode start))
    r_hi = torch.clamp(st["p13_z0"], min=rise_hi) if "p13_z0" in st else torch.full_like(tz, rise_hi)
    c = (((tz - r_hi) / unit).clamp(0.0, rise_cap) + ((p.abs() - p_free) / p_unit).clamp(0.0, p_cap)
         + (out / head_unit).clamp(0.0, head_cap)).clamp(max=cap)
    return torch.nan_to_num(c * st["p12_win"].float())


def m_p13_hover(env) -> torch.Tensor:
    """p13 metrics: hover steps (x dt = hover seconds past the free time); m_p13_tilt / m_p13_swing = mean swing sole tilt
    (deg); m_p13_win_tz / seq_p12_win = mean trunk deck-frame z (m) in the step-on window."""
    st = seq_update(env)
    return st["p13_hov"].float() if "p13_hov" in st else torch.zeros_like(st["v_tgt"])


def m_p13_swing(env) -> torch.Tensor:
    return _swing_gate_tilt(env, 0.02)[0].float()


def m_p13_tilt(env) -> torch.Tensor:
    g, tilt = _swing_gate_tilt(env, 0.02)
    return g.float() * tilt


def m_p13_win_tz(env) -> torch.Tensor:
    st = seq_update(env)
    if "p12_win" not in st:
        return torch.zeros_like(st["v_tgt"])
    return st["p12_win"].float() * torch.nan_to_num(_trunk_deck_z(env))


# ── push specialist p14 (2026-11-21, posture polish 3). Costs only, all >= 0
# (NEGATIVE weights); only microduck_skate_seq_push_env_cfg uses them (MICRODUCK_SEQPUSH_P14=1, needs P13: p13_z0 /
# p12_seen). Measured on p13 model_3797 from start (det, 64 envs):
#   - step-on pop: trunk (deck frame) 75 mm at the command latch (a touchdown dip), peak 110 mm at +0.22 s, settled 91 mm
#     (overshoot p50 19 mm), world vz peak 0.27 m/s, trunk 33 deg nose-up at the peak. p13_stepon_pose priced it (linear,
#     -1.75 / episode) but the window ended at SETTLE + 0.4 s and nothing priced the speed.
#   - swing sole roll is REAL, not a frame artefact: sole tilt 2 deg standing (frame aligned), airborne roll -47 deg p50
#     (pitch +4), the planted sole lands on its edge (floor-contact roll p10 -34 deg); grids show the sole facing outboard.
#     p13_swing_tilt (-0.5, linear, 20 deg free) cost -1.8 / episode: too weak and flat for a gait change.
#   - head yaw: joint p2p 72 deg per PUSH (p50; rate mean 82 deg/s): the head is the reaction mass; p12_head_steady (L1 vs
#     its own running mean at -0.4) cost -0.86. Squared costs below: the price rises with the violation (gradient at the
#     current behaviour; caps above it). ──


def _p14_gate_pop(st) -> torch.Tensor:
    """Step-on window for p14_stepon_pop: after the step command was seen this episode, PUSH under the command or any
    SETTLE step (p12_win stops at SETTLE + 0.4 s: a late pop would be free)."""
    b = torch.zeros_like(st["stage"], dtype=torch.bool)
    seen = st.get("p12_seen", b)
    return seen & (((st["stage"] == ST_PUSH) & st.get("cmd", b)) | (st["stage"] == ST_SETTLE))


def p14_stepon_pop(env, v_free: float = 0.12, v_unit: float = 0.05, z_hi: float = 0.098, z_unit: float = 0.005,
                   cap: float = 25.0, z0_slew: float = 0.10) -> torch.Tensor:
    """Cost in the step-on window (_p14_gate_pop): ((trunk vz rel. the board - v_free) / v_unit)^2 + ((trunk deck-frame z
    - max(z_hi, spawn z)) / z_unit)^2, each capped. z_hi = the ~90 mm settled height + 8 mm; the spawn z (p13_z0) keeps
    the step_mid spawns (120-128 mm) free of their own height. p13 model_3797: 0.27 m/s, 110 mm -> ~1.7 raw / episode."""
    st = seq_update(env)
    robot, b = _robot(env), tk._board(env)
    vz = torch.nan_to_num(robot.data.root_link_lin_vel_w[:, 2] - b.data.root_link_lin_vel_w[:, 2])
    tz = torch.nan_to_num(_trunk_deck_z(env))
    # p14rev: the spawn-height allowance slews down at z0_slew m/s (below v_free) to z_hi (step_mid 128 mm: 0.3 s), so a
    # later pop in a step_mid episode is priced; caps 25 (vz 0.37 m/s, 123 mm) leave a gradient above p13's 0.27 / 110
    z0 = st["p13_z0"] - z0_slew * tk._t(env) if "p13_z0" in st else torch.full_like(tz, z_hi)
    zh = torch.clamp(z0, min=z_hi)
    c = (((vz - v_free) / v_unit).clamp(min=0.0) ** 2).clamp(max=cap) + (((tz - zh) / z_unit).clamp(min=0.0) ** 2).clamp(max=cap)
    return torch.nan_to_num(c * _p14_gate_pop(st).float())


def _push_sole_roll(env) -> torch.Tensor:
    """Push (right) sole roll, deg: atan2(sole y axis . world z, sole z axis . world z) - yaw- and pitch-independent
    (p14rev: asin(y_z) = cos(pitch) sin(roll) read ~18 deg low at |pitch| > 30 deg, so toe-down hid the roll)."""
    robot = _robot(env)
    _right_sole_deck_xy(env)                           # caches the sole geom id
    gid = tk._tc(env)["rsole_id"]
    q = robot.data.geom_quat_w[:, gid]
    n, dev = env.num_envs, env.device
    sy = quat_apply(q, torch.tensor([0.0, 1.0, 0.0], device=dev).expand(n, 3))
    sz = quat_apply(q, torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3))
    return torch.nan_to_num(torch.rad2deg(torch.atan2(sy[:, 2], sz[:, 2])))


def p14_sole_roll(env, free_deg: float = 10.0, unit_deg: float = 15.0, cap: float = 12.0) -> torch.Tensor:
    """Cost, the whole PUSH stage (swing, touchdown, planted, and the step-on swing; any contact state): push-sole ROLL
    beyond free_deg, ((|roll| - free) / unit)^2, capped. Pitch is free (toe-off is natural; p12_swing_sole owns it). A
    planted sole is flat (free), so only the rolled swing and the edge landing pay. p13: 47 deg -> 6.1 / step."""
    st = seq_update(env)
    r = _push_sole_roll(env).abs()
    return ((r - free_deg) / unit_deg).clamp(min=0.0).pow(2).clamp(max=cap) * (st["stage"] == ST_PUSH).float()


def p14_head_yaw(env, free: float = 0.10, unit: float = 0.20, cap: float = 9.0, rate_free: float = 1.0,
                 rate_unit: float = 2.0, rate_cap: float = 6.0) -> torch.Tensor:
    """Cost, PUSH before the step command (the step-on head move is p12_stepon_head's): head_yaw joint away from neutral
    (0; HANDOFF -0.144 sits inside free + unit/2) ((|q| - free) / unit)^2 capped + ((|qdot| - rate_free) / rate_unit)^2
    capped. p13: 72 deg p2p, 1.4 rad/s mean -> ~3.6 raw / episode; +-12 deg at 0.8 rad/s -> ~0.1."""
    st = seq_update(env)
    ids, _ = _head_joint_ids(env)
    robot = _robot(env)
    q = torch.nan_to_num(robot.data.joint_pos[:, ids[2]])
    qd = torch.nan_to_num(robot.data.joint_vel[:, ids[2]])
    c = (((q.abs() - free) / unit).clamp(min=0.0).pow(2).clamp(max=cap)
         + ((qd.abs() - rate_free) / rate_unit).clamp(min=0.0).pow(2).clamp(max=rate_cap))
    return c * _p12_stroking(st).float()


def m_p14_vz(env) -> torch.Tensor:
    """p14 metrics: trunk vz rel. the board (m/s) in the p14 pop window (/ seq_p14_win = mean); m_p14_roll = |push sole
    roll| (deg) in PUSH (/ m_push-style step counts); m_p14_hyaw = |head_yaw| (rad) in PUSH before the command."""
    st = seq_update(env)
    robot, b = _robot(env), tk._board(env)
    vz = torch.nan_to_num(robot.data.root_link_lin_vel_w[:, 2] - b.data.root_link_lin_vel_w[:, 2])
    return vz.clamp(min=0.0) * _p14_gate_pop(st).float()


def m_p14_win(env) -> torch.Tensor:
    return _p14_gate_pop(seq_update(env)).float()


def m_p14_roll(env) -> torch.Tensor:
    st = seq_update(env)
    return _push_sole_roll(env).abs() * (st["stage"] == ST_PUSH).float()


def m_p14_hyaw(env) -> torch.Tensor:
    st = seq_update(env)
    ids, _ = _head_joint_ids(env)
    return torch.nan_to_num(_robot(env).data.joint_pos[:, ids[2]].abs()) * _p12_stroking(st).float()


# ── push specialist p15 (2026-11-21, speed back). Costs only, all >= 0
# (NEGATIVE weights); only microduck_skate_seq_push_env_cfg uses them (MICRODUCK_SEQPUSH_P15=1, needs P14: p10_td_evt,
# v_ramp). Measured on p14 model_4246 vs p13 model_3797 (start, det, 64 envs):
#   - p14 STALLS: v >= 0.2 m/s at 1.92 s (p13 0.70), the push foot stays planted. Why: the step command is exogenous
#     (v_ramp = 0.35 m/s^2 x t must reach v_tgt), push_progress is a potential (its sum = the speed gained, path-free),
#     so the only price of a late push is push_speed (<= ~0.9 / episode), while every stroke pays ~10 stroke-shaped costs:
#     p13 behaviour costs ~12 / episode in PUSH pre-command (sole_roll 1.5, swing_tilt 1.7, push_stroke 1.4, shape 1.2,
#     head_pose 1.2, head_yaw 1.0, ...), p14's planted stall ~3.8 -> the stall nets ~+6 more.
#   - airborne sole roll is NOT fixable at a reasonable price: the leg is hip_yaw / hip_roll (+-22 deg) / hip_pitch /
#     knee / ankle (pitch) - no ankle roll. In the swing hip roll sits at its -22 deg limit and the trunk rolls -20..-25
#     deg (planted: -7); a bounded IK holding the recorded trunk and sole position reaches only ~-8 deg at a 14-20 mm
#     sole miss with hip roll driven to its +22 deg limit (0 % of frames within 10 deg / 5 mm). The planted sole is flat.
#     So the swing roll is left free; only the LANDING is priced (p13 touchdown |roll| p50 33 deg: lands on the edge). ──


def p15_speed_deficit(env, free: float = 0.07, unit: float = 0.08, cap: float = 2.5, grace_s: float = 0.3
                      ) -> torch.Tensor:
    """Cost, PUSH before the step command: speed (min(board, trunk) forward) below the slewed push target
    min(v_ramp, v_tgt) by more than free, in unit, linear, capped. v_ramp = spawn speed + push_ramp x time in PUSH, so
    from rest at 0.35 m/s^2 the free band asks for 0.2 m/s by ~0.8 s and v_tgt - 0.07 at the ramp's end; being ahead of
    the ramp costs nothing (no jackpot). Cap x weight (2.5 x 1.5 = 3.75 / s) stays below the discounted fall price.
    p15rev: free 0.07 < v_gap 0.10, so hovering just under the command latch (v_tgt - v_gap) is priced; the first
    grace_s of PUSH are free (a push spawn at v0 with the sole on the floor drags min(board, trunk) below v0 at once).
    p14 model_4246 (stall) ~4.3 raw / episode, p13 model_3797 ~0.2."""
    st = seq_update(env)
    if "v_ramp" not in st:
        return torch.zeros_like(st["v_tgt"])
    ref = torch.minimum(st["v_ramp"], st["v_tgt"])
    c = ((ref - _push_v(env, st) - free) / unit).clamp(0.0, cap)
    return torch.nan_to_num(c * (_p12_stroking(st) & (st["stage_t"] >= grace_s)).float())


def p15_land_roll(env, free_deg: float = 10.0, unit_deg: float = 10.0, cap: float = 9.0) -> torch.Tensor:
    """Event cost per push-foot FLOOR touchdown in PUSH before the command (p10_td_evt; the spawn contact never counts):
    ((|sole roll| - free) / unit)^2 capped, at the touchdown step, / dt (the weight is the price per landing). The swing
    roll itself is free (see the block comment). p13: 33 deg -> ~5.3 per landing; <= 10 deg -> 0."""
    st = seq_update(env)
    if "p10_td_evt" not in st:
        return torch.zeros_like(st["v_tgt"])
    r = _push_sole_roll(env).abs()
    c = ((r - free_deg) / unit_deg).clamp(min=0.0).pow(2).clamp(max=cap)
    return torch.nan_to_num(c * st["p10_td_evt"].float()) / env.step_dt


def m_p15_deficit(env) -> torch.Tensor:
    """p15 metrics: speed deficit below min(v_ramp, v_tgt) (m/s, PUSH pre-command; / seq_p15_pre = mean); |roll| at the
    touchdowns (deg; / seq_p10_td = mean landing roll); PUSH-pre-command step counts."""
    st = seq_update(env)
    if "v_ramp" not in st:
        return torch.zeros_like(st["v_tgt"])
    d = (torch.minimum(st["v_ramp"], st["v_tgt"]) - _push_v(env, st)).clamp(min=0.0)
    return torch.nan_to_num(d * _p12_stroking(st).float())


def m_p15_pre(env) -> torch.Tensor:
    return _p12_stroking(seq_update(env)).float()


def m_p15_land_roll(env) -> torch.Tensor:
    st = seq_update(env)
    if "p10_td_evt" not in st:
        return torch.zeros_like(st["v_tgt"])
    return _push_sole_roll(env).abs() * st["p10_td_evt"].float()


# ── push specialist p16 (2026-11-21, real servo limits). Costs only, all
# >= 0 (NEGATIVE weights) + one curriculum; only microduck_skate_seq_push_env_cfg uses them (MICRODUCK_SEQPUSH_P16=1,
# needs P15: p10_lo_evt / p10_contact / p10_lo_v). Measured on p15 model_4645 at the real 1.0 A firmware limit with the
# datasheet speed (n 32, diag_1A.json): from start 72 % fall (fell_over, median 1.38 s, all during the
# pre-command strokes), 25 % cmd_landing, 3 % success; from step_on 22 % success, 53 % cmd_landing, 22 % foot_on_floor.
# Saturation (|I| >= 0.97 A, share of substeps): pre-command PUSH right hip yaw 49 %, right hip roll 33 %; under the
# command right hip yaw 91 %, right knee 68 %; SETTLE right hip yaw 99.6 % (a static squeeze); the last 0.5 s before a
# fall: right hip yaw 75, right ankle 67, left hip yaw 67, right knee 66, right hip roll 63 %. Right knee > +0.1 rad
# 41-50 % of the pre-command PUSH (p90 +0.62 / +1.13 rad), floor contacts p50 0.08 s, swing sole peak 2.0 / 2.9 cm
# (p50 / p90) above its rest height, neck pitch p10 -0.77 rad. ──


def _p16_ids(env) -> dict:
    c = tk._tc(env)
    if "p16_ids" not in c:
        robot = _robot(env)
        kl = robot.find_joints(r"^left_knee$")[0]
        kr = robot.find_joints(r"^right_knee$")[0]
        nk = robot.find_joints(r"^neck_pitch$")[0]
        assert len(kl) == 1 and len(kr) == 1 and len(nk) == 1, f"p16 joints {kl} {kr} {nk}"
        c["p16_ids"] = dict(kl=kl[0], kr=kr[0], neck=nk[0])
    return c["p16_ids"]


def p16_knee_side(env, free: float = 0.10, unit: float = 0.10, cap: float = 11.0, grace_s: float = 0.4
                  ) -> torch.Tensor:
    """Cost (every stage after the first grace_s of the episode): right knee above +free / left knee below -free (rad;
    the normal sides are right <= 0, left >= 0 - HANDOFF_STANCE right -0.59 / left +0.75), LINEAR per unit, capped at
    1.2 rad (review p16: a squared cost capped at 4 flattened at 0.3 rad, while p15's bird knee sits at +0.6..+1.4 rad -
    no gradient). The grace: the push_back / glide_step / step_mid spawn stances themselves have the right knee at
    +0.67..+1.11 rad (stance_poses_seq_stepfix.json); the first 0.4 s let the policy leave them free of charge."""
    ids = _p16_ids(env)
    q = _robot(env).data.joint_pos
    c = ((q[:, ids["kr"]] - free) / unit).clamp(min=0.0) + ((-free - q[:, ids["kl"]]) / unit).clamp(min=0.0)
    return torch.nan_to_num(c).clamp(max=cap) * (tk._t(env) >= grace_s).float()


def p16_stance_time(env, t_min: float = 0.15, reach: float = 0.12, t_floor: float = 0.10, unit: float = 0.04,
                    cap: float = 4.0, grace_s: float = 0.3) -> torch.Tensor:
    """Event cost per push-foot lift-off in PUSH before the command (p10_lo_evt; a sub-step tap has contact 0):
    ((target - contact) / unit)^2 capped, / dt (the weight = price per short stroke). target = clamp(reach / v_touchdown,
    t_floor, t_min): a planted sole travels back at the board speed, so the contact is reach-limited (~12 cm of sweep:
    0.15 s up to 0.8 m/s, 0.13 s at 0.9). p15: contact p50 0.08 s -> ~3 per stroke; >= target -> 0. Lift-offs in the
    first grace_s of the episode are free (the spawn contact of a push spawn at speed: p15_speed_deficit's grace)."""
    st = seq_update(env)
    if "p10_lo_evt" not in st:
        return torch.zeros_like(st["v_tgt"])
    tgt = (reach / st["p10_lo_v"].clamp(min=0.1)).clamp(t_floor, t_min)
    c = ((tgt - st["p10_contact"]) / unit).clamp(min=0.0).pow(2).clamp(max=cap)
    evt = st["p10_lo_evt"] & (tk._t(env) >= grace_s)
    return torch.nan_to_num(evt.float() * c) / env.step_dt


def p16_foot_lift(env, free: float = 0.020, unit: float = 0.01, cap: float = 4.0, z_floor: float = 0.008
                  ) -> torch.Tensor:
    """Cost: PUSH before the command, push foot in the air (off the floor and the deck; push_foot_height's gate): sole
    height above its on-floor height beyond `free` (2 cm), squared per `unit` (1 cm), capped. push_foot_height (free
    3.5 cm, linear) stays; this is the low-sweep cap (p15 swing peaks 2.0 / 2.9 cm p50 / p90)."""
    st = seq_update(env)
    _right_sole_deck_xy(env)                           # caches the sole geom id
    z = _robot(env).data.geom_pos_w[:, tk._tc(env)["rsole_id"], 2] - _origin(env)[:, 2]
    _, rf = _feet_floor(env)
    gate = _p12_stroking(st) & ~rf & ~tk._contact_any(env, "right_deck")
    return torch.nan_to_num(gate.float() * ((z - z_floor - free) / unit).clamp(min=0.0).pow(2).clamp(max=cap))


def p16_neck(env, lo: float = -0.35, unit: float = 0.10, cap: float = 4.0) -> torch.Tensor:
    """Cost: PUSH before the command, neck_pitch below `lo` (rad; the forward crane, p15 at 1.0 A p10 -0.77 rad; the
    push median ~0 and HANDOFF +0.41 are free), squared per unit, capped."""
    st = seq_update(env)
    q = torch.nan_to_num(_robot(env).data.joint_pos[:, _p16_ids(env)["neck"]])
    return _p12_stroking(st).float() * ((lo - q) / unit).clamp(min=0.0).pow(2).clamp(max=cap)


def _p16_leg_ids(env) -> list:
    c = tk._tc(env)
    if "p16_legs" not in c:
        legs = _robot(env).find_joints(r"^(left|right)_(hip_yaw|hip_roll|hip_pitch|knee|ankle)$")[0]
        assert len(legs) == 10, f"p16 leg joints {legs}"
        c["p16_legs"] = list(legs)
    return c["p16_legs"]


def _p16_limit_excess(env, margin: float) -> torch.Tensor:
    """(N, 10) rad by which each leg joint is inside `margin` of its MJCF range limit (0 outside the band)."""
    robot = _robot(env)
    ids = _p16_leg_ids(env)
    q = robot.data.joint_pos[:, ids]
    lim = robot.data.joint_pos_limits[:, ids]
    return torch.nan_to_num((q - (lim[..., 1] - margin)).clamp(min=0.0) + ((lim[..., 0] + margin) - q).clamp(min=0.0))


def p16_leg_limits(env, margin_deg: float = 3.0, cap: float = 4.0) -> torch.Tensor:
    """Cost (every stage): the 10 leg joints within margin_deg of their MJCF range limits, (excess / margin)^2 summed
    (1 per joint sitting ON a stop), capped. The real runtime does not enforce the MJCF stops (hip
    yaws within 2 deg of a stop 9-15 % of the chain frames), so parking on them must not be a trained habit."""
    m = math.radians(margin_deg)
    e = _p16_limit_excess(env, m)
    # review p16: HANDOFF_STANCE left_hip_yaw (-0.438) sits on its stop (-0.436) and settle_stance_path pulls there -
    # exempt that joint in SETTLE (the hand-off contract with the manual specialist is not this run's to change)
    c = tk._tc(env)
    if "p16_lhy" not in c:
        c["p16_lhy"] = _p16_leg_ids(env).index(_robot(env).find_joints(r"^left_hip_yaw$")[0][0])
    st = seq_update(env)
    keep = torch.ones_like(e)
    keep[:, c["p16_lhy"]] = (st["stage"] != ST_SETTLE).float()
    return (((e * keep) / m) ** 2).sum(dim=1).clamp(max=cap)


def m_p16_limits(env) -> torch.Tensor:
    """p16 metric: any leg joint within 3 deg of its range limit (0/1 per step)."""
    return (_p16_limit_excess(env, math.radians(3.0)) > 0).any(dim=1).float()


def m_p16_knee(env) -> torch.Tensor:
    """p16 metrics: right knee past +0.1 rad (0/1 per step, every stage); contact time at each pre-command lift-off (s,
    / seq_p10_lo-style counts = mean); the BAM current limit in force (A)."""
    q = _robot(env).data.joint_pos[:, _p16_ids(env)["kr"]]
    return (torch.nan_to_num(q) > 0.1).float()


def m_p16_contact(env) -> torch.Tensor:
    st = seq_update(env)
    if "p10_lo_evt" not in st:
        return torch.zeros_like(st["v_tgt"])
    return st["p10_lo_evt"].float() * st["p10_contact"]


def m_p16_amps(env) -> torch.Tensor:
    from micro_skater.tasks.mdp import _bam_heat_actuators

    acts = _bam_heat_actuators(env)
    a = float(acts[0]._bam_model.actuator.max_current or 0.0) if acts else 0.0
    return torch.full((env.num_envs,), a, device=env.device)


def p16_current_curriculum(env, env_ids, stages: list[dict], gate: float = 0.7, decay: float = 0.995,
                           min_n: float = 300.0, heat: dict | None = None, k_min: int = 0,
                           heat_on_at_start: bool = False) -> torch.Tensor:
    """p16: the BAM firmware current limit (max_current, read by bam's compute_control on every physics substep, so a
    runtime change applies at once to every env) stepped down over training, + the servo-heat weight switched on at
    the final current. stages = [{"step", "deadline", "amps"}] (env steps, absolute: the resume keeps the counter).
    Stage k + 1 starts when counter >= its step AND the decayed `success` share over the ended start-spawn episodes (since the last
    advance, >= min_n of them) >= gate, or at its deadline regardless. A resume rebuilds the stage from the deadlines
    (the gate state is cached, not checkpointed). heat (optional, None = untouched): {"term", "weight", "delay_steps",
    "deadline", "half_steps"}: the reward term's weight is 0 until the final stage has run delay_steps AND the gate
    holds (or the absolute heat deadline), then weight / 2 for half_steps, then weight. Returns the amps in force."""
    from micro_skater.tasks.mdp import _bam_heat_actuators

    c = tk._tc(env)
    n_now = env.common_step_counter
    if "p16_k" not in c:
        # review p16: k_min / heat_on_at_start (MICRODUCK_SEQPUSH_P16_AMPS_K0 / _HEAT_ON) for a resume of p16 itself,
        # whose stage the deadlines alone would rebuild too low (the current could go back up, the heat off)
        k0 = min(int(k_min), len(stages) - 1)
        for k, s in enumerate(stages):
            if k > 0 and n_now >= s["deadline"]:
                k0 = max(k0, k)
        c.update(p16_k=k0, p16_kstep=n_now, p16_s=0.0, p16_n=0.0, p16_heat_on=None, p16_amps=None)
        if heat is not None and (n_now >= heat["deadline"] or heat_on_at_start):
            c["p16_heat_on"] = min(n_now, heat["deadline"]) - heat["half_steps"]   # full weight at once
    tm = env.termination_manager
    if env_ids is not None and not isinstance(env_ids, slice) and len(env_ids) > 0 and "success" in tm.active_terms:
        ids = env_ids.to(env.device).long()
        # review p16: the gate counts START-spawn episodes only (bridge_adaptive keeps the stroke_air rungs near 50 %)
        ended = (env.episode_length_buf[ids] > 0) & (_st(env)["spawn"][ids] == SPAWNS.index("start"))
        ok = tm.get_term("success")[ids] & ended
        c["p16_s"] = c["p16_s"] * decay + float(ok.sum())
        c["p16_n"] = c["p16_n"] * decay + float(ended.sum())
    p = (c["p16_s"] + 1.0) / (c["p16_n"] + 2.0)
    c["p16_p"] = p
    k = c["p16_k"]
    k_dl = max([j for j, s in enumerate(stages) if j > 0 and n_now >= s["deadline"]], default=0)
    if k_dl > k:                       # deadline(s) passed (also: an eval that sets the step counter after the build)
        k = c["p16_k"] = k_dl
        c["p16_kstep"], c["p16_s"], c["p16_n"] = n_now, 0.0, 0.0
    elif k + 1 < len(stages):
        nxt = stages[k + 1]
        if n_now >= nxt["step"] and c["p16_n"] >= min_n and p >= gate:
            k = c["p16_k"] = k + 1
            c["p16_kstep"], c["p16_s"], c["p16_n"] = n_now, 0.0, 0.0
    amps = float(stages[k]["amps"])
    if c["p16_amps"] != amps:
        for a in _bam_heat_actuators(env):
            a._bam_model.actuator.max_current = amps
        print(f"[p16] max_current {c['p16_amps']} -> {amps} A at step {n_now} (it {n_now // 24}, success p {p:.2f})",
              flush=True)
        c["p16_amps"] = amps
    if heat is not None:
        last = k == len(stages) - 1
        if c["p16_heat_on"] is None and n_now >= heat["deadline"]:
            c["p16_heat_on"] = heat["deadline"]
            print(f"[p16] heat cost on (deadline {heat['deadline']}) at step {n_now}", flush=True)
        elif c["p16_heat_on"] is None and last and (
                n_now >= c["p16_kstep"] + heat["delay_steps"] and c["p16_n"] >= min_n and p >= gate):
            c["p16_heat_on"] = n_now
            print(f"[p16] heat cost on at step {n_now} (it {n_now // 24}, success p {p:.2f})", flush=True)
        on = c["p16_heat_on"]
        w = 0.0 if on is None else (0.5 if n_now < on + heat["half_steps"] else 1.0) * heat["weight"]
        if heat["term"] in env.reward_manager.active_terms:
            env.reward_manager.get_term_cfg(heat["term"]).weight = w
        c["p16_heat_w"] = w
    return torch.tensor([amps])


def m_p16_heat_w(env) -> torch.Tensor:
    return torch.full((env.num_envs,), float(tk._tc(env).get("p16_heat_w", 0.0)), device=env.device)


def m_p16_gate_p(env) -> torch.Tensor:
    return torch.full((env.num_envs,), float(tk._tc(env).get("p16_p", 0.0)), device=env.device)


def manual_speed_loss(env, decel: float = 0.04, window_s: float = 0.6, cap: float = 0.5) -> torch.Tensor:
    """Cost: relative board-speed deficit against a coasting reference from the MANUAL entry, max(v_ref - v, 0) / v_entry
    with v_ref = v_entry - decel * (t - t_entry) (decel = the coast allowance; 4-wheel coasting measured ~0.03 m/s^2),
    capped, in MANUAL and the first window_s of EXIT. Gaining speed pays nothing. State (v_man, t_man, man_ok) is kept
    here (the term runs every step at a non-zero weight) and cleared by reset_sequence_state when present."""
    st = seq_update(env)
    if "man_ok" not in st:
        st["man_ok"] = torch.zeros_like(st["stage"], dtype=torch.bool)
        st["v_man"] = torch.zeros_like(st["v_tgt"])
        st["t_man"] = torch.zeros_like(st["v_tgt"])
    v, t = board_v_fwd(env), tk._t(env)
    new = (st["stage"] == ST_MANUAL) & ~st["man_ok"]
    st["v_man"] = torch.where(new, v, st["v_man"])
    st["t_man"] = torch.where(new, t, st["t_man"])
    st["man_ok"] = st["man_ok"] | new
    gate = st["man_ok"] & ((st["stage"] == ST_MANUAL) | ((st["stage"] == ST_EXIT) & (st["stage_t"] < window_s - 1e-6)))
    ref = st["v_man"] - decel * (t - st["t_man"]).clamp(min=0.0)
    d = ((ref - v) / st["v_man"].clamp(min=0.2)).clamp(min=0.0, max=cap)
    return torch.nan_to_num(gate.float() * d)


def heading_line_err(env, lookahead: float = 1.5) -> torch.Tensor:
    """(B,) rad, + = the reference is to the LEFT of the board heading: bearing from the board to the scheduler's
    approach target (approach_target: the point EXIT's pure pursuit / st["kappa"] steers to), its y floored at
    board y + lookahead (past the target the reference runs along the pipe axis instead of flipping behind). The same
    error approach_kappa turns on, so the EXIT kappa obs and this term agree. Observable: seq_obs carries x, y, yaw."""
    st = seq_update(env)
    p = pipe_xyz(env)[:, :2]
    tgt = approach_target(st)
    ty = torch.maximum(tgt[:, 1], p[:, 1] + lookahead)
    e = wrap_to_pi(torch.atan2(ty - p[:, 1], tgt[:, 0] - p[:, 0]) - tk._board_yaw(env))
    return torch.nan_to_num(e)


def heading_line_cost(env, free_deg: float = 8.0, unit_deg: float = 10.0, cap: float = 3.0, v_full: float = 0.3,
                      stages: tuple | None = (ST_PUSH, ST_SETTLE)) -> torch.Tensor:
    """Cost (heading fix 2026-11-20, push / manual specialists): |heading_line_err| beyond free_deg in units of
    unit_deg, linear, capped; x min(|v| / v_full, 1) (a board at rest cannot steer: no cost at the start / a refusal
    gets no relief from it either, it loses the stage pay) and only in `stages`. p9b model_2100 from start turned
    +61 deg left in PUSH (yaw rate +32 deg/s with the push foot up, deck rolled ~1.5 deg left: the board-foot lean
    between strokes) and nothing priced it: raw episode cost 9 (drifting) vs 0.4 (straight on the start heading)."""
    e = torch.rad2deg(heading_line_err(env).abs())
    c = ((e - free_deg) / unit_deg).clamp(min=0.0, max=cap)
    g = (board_v_fwd(env).abs() / v_full).clamp(0.0, 1.0)
    return torch.nan_to_num(_stage_gate(env, stages) * g * c)


def m_heading_err(env) -> torch.Tensor:
    """metric: |heading_line_err| (deg), every env every step."""
    return torch.rad2deg(heading_line_err(env).abs())


def ramp_lateral(env, v_cap: float = 0.6) -> torch.Tensor:
    """RAMP: speed across the pipe (world x) inside the pipe footprint, soles on the deck: going up and down the walls
    (rolling along the axis pays nothing)."""
    st = seq_update(env)
    vx = tk._board(env).data.root_link_lin_vel_w[:, 0].abs().clamp(max=v_cap) / v_cap
    ok = in_pipe(env) & tk.soles_on_deck(env)
    return _phase(env, RAMP) * ok.float() * torch.nan_to_num(vx)


def ramp_apex(env, z_cap: float = 0.05) -> torch.Tensor:
    """RAMP: an apex on a wall (deck vertical velocity + -> -) pays min(rise, z_cap) / z_cap as an event, only when the
    previous apex was on the OTHER wall (or the first one): alternation keeps it physics-bounded and unfarmable."""
    st = seq_update(env)
    cfg, pipe = st["seq_cfg"], st["pipe"]
    p = pipe_xyz(env)
    vz = torch.nan_to_num(tk._board(env).data.root_link_lin_vel_w[:, 2])
    rise = (p[:, 2] - DECK_Z).clamp(min=0.0)
    on_wall = p[:, 0].abs() > pipe["L"] / 2
    side = torch.sign(p[:, 0])
    apex = (st["vz_prev"] > 0.02) & (vz <= 0.0) & on_wall & (rise > cfg["apex_min_z"]) & in_pipe(env)
    new = apex & (side != st["apex_side"]) & (st["trick"] == RAMP) & tk.soles_on_deck(env)
    pay = new.float() * (rise.clamp(max=z_cap) / z_cap)
    st["apex_evt"] = new
    st["apex_side"] = torch.where(new, side, st["apex_side"])
    st["vz_prev"] = vz
    return pay / env.step_dt


# ── metrics (REV; per-step means, read them as RATIOS: e.g. liftoff / push * 50 = strokes per PUSH-second) ─────────────


def m_push(env) -> torch.Tensor:
    return _phase(env, PUSH)


def m_liftoff(env) -> torch.Tensor:
    return seq_update(env)["lift_evt"].float()


def m_stroke_q(env) -> torch.Tensor:
    """Stroke quality summed at lift-offs: m_stroke_q / m_liftoff = mean q."""
    return seq_update(env)["lift_q"]


def m_stepon(env) -> torch.Tensor:
    """PUSH -> SETTLE transitions (the step-on)."""
    return seq_update(env)["stepon_evt"].float()


def m_manual(env) -> torch.Tensor:
    return _phase(env, MANUAL)


def m_manual_band(env) -> torch.Tensor:
    """In the 8-15 deg demo band (MANUAL, back wheels down, tail up, soles on): m_manual_band / m_manual."""
    th = torch.rad2deg(tk.front_up_pitch(env))
    front_down, back_down = tk._front_back_wheels(env)
    ok = (~front_down) & back_down & ~tk._back_end_floor(env) & tk.soles_on_deck(env) & (th > 8.0) & (th < 15.0)
    return _phase(env, MANUAL) * ok.float()


def m_ramp(env) -> torch.Tensor:
    return _phase(env, RAMP)


def m_apex(env) -> torch.Tensor:
    """Alternating-wall apexes (needs ramp_apex at a non-zero weight): m_apex / m_ramp * 50 = passes per RAMP-second."""
    return seq_update(env)["apex_evt"].float()


def m_band_t_done(env) -> torch.Tensor:
    """MANUAL -> EXIT transitions (manual_end="hold": the hold completed; "timer": the timer ran out)."""
    return seq_update(env)["manual_evt"].float()


def m_setdown(env) -> torch.Tensor:
    """Peak |deck pitch rate| (rad/s) of the set-down window, summed at its close: m_setdown / m_setdown_n = mean peak
    (manual_end="hold" only)."""
    st = seq_update(env)
    return st["sd_evt"].float() * st["sd_peak"]


def m_setdown_n(env) -> torch.Tensor:
    return seq_update(env)["sd_evt"].float()


def m_tail_manual(env) -> torch.Tensor:
    """Tail on the floor in MANUAL: m_tail_manual / m_manual = share of MANUAL time with a tail touch."""
    return _phase(env, MANUAL) * tk._back_end_floor(env).float()


# ── terminations ──────────────────────────────────────────────────────────────────────────────────────────────────────


def seq_foot_on_floor(env) -> torch.Tensor:
    """Any foot on the ground ends the episode, except the push (right) foot during PUSH."""
    st = seq_update(env)
    lf, rf = _feet_floor(env)
    push_ok = (st["trick"] == PUSH) | st.get("cmd", torch.zeros_like(lf))     # step command: still the PUSH stage
    return lf | (rf & ~push_ok)


def seq_board_lost(env, max_dist: float = 0.18, max_off_s: float = 0.6, ride_off_s: float | None = None) -> torch.Tensor:
    """Trunk too far from the board, the board (left) foot off the deck > max_off_s, or the push (right) foot off the
    deck outside PUSH for > ride_off_s (default None: max_off_s)."""
    st = seq_update(env)
    robot: Entity = _robot(env)
    d = torch.nan_to_num((robot.data.root_link_pos_w[:, :2] - tk._board(env).data.root_link_pos_w[:, :2]).norm(dim=1),
                         nan=1.0)
    l_off = _skate_air_time(env, "left_deck") > max_off_s
    r_lim = max_off_s if ride_off_s is None else ride_off_s
    push_ok = (st["trick"] == PUSH) | st.get("cmd", torch.zeros_like(d, dtype=torch.bool))
    r_off = (_skate_air_time(env, "right_deck") > r_lim) & ~push_ok
    return (d > max_dist) | l_off | r_off


def seq_success(env, stage: int = ST_SETTLE, on_s: float = 0.3, wheels_s: float = 0.2, stance: bool = False
                ) -> torch.Tensor:
    """Specialist success (use as a time_out=True termination): in scheduler `stage`, its minimum time done (SETTLE:
    T_settle), both soles on the deck (feet off the floor) for >= on_s and all four wheels down for >= wheels_s.
    stance (p9b, default False; needs seq_cfg stance_tol): also in the hand-off stance for >= stance_hold_s."""
    st = seq_update(env)
    ts = st["stage_t"]
    t_min = {ST_SETTLE: st["T_settle"], ST_MANUAL: st["T_manual"], ST_EXIT: st["T_exit"]}.get(stage,
                                                                                            torch.zeros_like(ts))
    out = (st["stage"] == stage) & (ts >= t_min) & (st["both_on"] >= on_s) & (st["wheels_on"] >= wheels_s)
    if stance:
        assert st["seq_cfg"].get("stance_tol") is not None, "seq_success(stance=True) needs seq_cfg stance_tol"
        out = out & (st.get("stance_t", torch.zeros_like(ts)) >= st["seq_cfg"]["stance_hold_s"] - 1e-6)
    return out


def seq_overtime(env, limits: tuple = ((ST_PUSH, 8.0), (ST_SETTLE, 2.5)), cmd_max_s: float | None = None
                 ) -> torch.Tensor:
    """Failure: more than the limit (s) spent in a scheduler stage (refusing the transition has a terminal price).
    cmd_max_s (push specialist, step command): also fails when the step-on command has been ignored this long - the
    PUSH-stage limit alone was 6-8 s away and discounted to ~0 (gamma 0.99 at 50 Hz), so refusing to step on was free."""
    st = seq_update(env)
    ts = st["stage_t"]
    out = torch.zeros_like(st["stage"], dtype=torch.bool)
    for s, lim in limits:
        out |= (st["stage"] == s) & (ts > lim)
    if cmd_max_s is not None and "cmd_t" in st:
        out |= st["cmd"] & (st["cmd_t"] > cmd_max_s)
    return out


def cmd_wait(env, base: float = 1.0, slope: float = 1.0, cap: float = 4.0) -> torch.Tensor:
    """Cost per second while the step-on command is set and not yet obeyed (stage still PUSH): base + slope * time
    since the command, capped. Makes ignoring the command expensive NOW rather than at a distant overtime."""
    st = seq_update(env)
    if "cmd_t" not in st:
        return torch.zeros_like(st["v_tgt"])
    return st["cmd"].float() * (base + slope * st["cmd_t"]).clamp(max=cap)


def seq_fall_push(env, cmd_scale: float = 0.5, cmd_max_s: float = 2.5, manual_scale: float = 0.6,
                  land_scale: float | None = None, settle_overtime_scale: float | None = None,
                  settle_fall_match: tuple | None = None) -> torch.Tensor:
    """seq_fall, but a fall while the step-on command is active costs cmd_scale (an attempted step-on that fails must
    be cheaper than ignoring the command); the command timeout itself (cmd_t > cmd_max_s) keeps the full price.
    settle_overtime_scale (p9b, default None = off): the `overtime` termination in SETTLE (the hand-off stance never
    held) costs this scale. Measured on p8 model_1200: at 1.0 the dense SETTLE pay (ride, wheels,
    upright, ~3 /s) collected until the overtime made a missed stance worth as much as a success (step_mid +10.8 vs
    +10.9 discounted).
    settle_fall_match (p9brev, (overtime limit s, gamma), default None = off; needs settle_overtime_scale): any other
    failure in SETTLE (fall, board_lost) at stage time tau costs max(1, scale * gamma^((limit - tau) / dt)) = at least
    the overtime's price discounted back to tau. Without it a stance that will be missed was cheaper to end by
    stepping off / falling after ~0.7 s (-10 at tau = 1.2 s ~ -5.5 discounted vs -15 at 1.5 s ~ -7.0)."""
    st = seq_update(env)
    s = torch.where(st["trick"] == MANUAL, torch.full_like(st["v_tgt"], manual_scale), torch.ones_like(st["v_tgt"]))
    if "cmd_t" in st:
        attempt = st["cmd"] & (st["cmd_t"] <= cmd_max_s)
        ramp = cmd_scale + (1.0 - cmd_scale) * (st["cmd_t"] / cmd_max_s).clamp(0.0, 1.0)   # p5: no jump at the timeout
        s = torch.where(attempt, ramp, s)
    if land_scale is not None and "cmd_landing" in env.termination_manager.active_terms:   # landrule (default off)
        s = torch.where(env.termination_manager.get_term("cmd_landing"), torch.full_like(s, land_scale), s)
    if settle_overtime_scale is not None and "overtime" in env.termination_manager.active_terms:   # p9b (default off)
        ot = env.termination_manager.get_term("overtime") & (st["stage"] == ST_SETTLE)
        if settle_fall_match is not None:          # p9brev: a SETTLE fall never undercuts the discounted overtime
            lim, gam = settle_fall_match
            k = ((float(lim) - st["stage_t"]).clamp(min=0.0) / env.step_dt)
            fm = (settle_overtime_scale * torch.pow(torch.full_like(k, float(gam)), k)).clamp(min=1.0)
            s = torch.where(st["stage"] == ST_SETTLE, torch.maximum(s, fm), s)
        s = torch.where(ot, torch.full_like(s, settle_overtime_scale), s)
    return env.termination_manager.terminated.float() * s / env.step_dt


def seq_cmd_landing(env, reach_s: float | None = None, spawn_grace_s: float | None = None) -> torch.Tensor:
    """landrule (2026-11-20, push specialist; a termination, not in the default tasks): under the step command the push
    foot's NEXT landing must be on the deck. A floor contact already under way when the command latches is let finish;
    once the push foot has been off the floor under the command, a new floor touchdown ends the episode (charged by
    seq_fall_push land_scale). Turns the 2.5 s command timeout (discounted to ~0.3 at the decision) into an immediate
    failure at the moment the policy chooses another stroke over the step-on.
    reach_s (landrev review, default None = off): also fails once the command is older than reach_s while the push
    sole has not yet touched the deck under it. Without it the two refusals the rule cannot see - foot kept planted
    (the latch grace never ends) and foot held up beside the board - still ran to the 2.5 s overtime, whose price is
    discounted to ~0.28 and, with push_progress gated off, came out near or above an attempt that falls. Successful
    step-ons (p7 m700, every spawn) touch the deck 0.10-0.24 s after the command.
    spawn_grace_s (p12rev review, default None = off): step_on (and fast push) spawns latch the command on the first
    step with the push sole in a floor stance 1-5 mm up (z_lift); in ~half of them the floor sensor has not registered
    it yet (air 5-20 ms), cmd_air armed, and the sole settling onto the floor at 0.04 s ended 62-66 % of step_on
    spawns (p11 m2799). A no-contact reading in the first spawn_grace_s of the stage (= since the reset for a PUSH
    spawn) no longer arms the rule, except for the airborne-foot spawns (stroke_air / step_mid)."""
    st = seq_update(env)
    _, rf = _feet_floor(env)
    on = st["cmd"] & (st["stage"] == ST_PUSH)
    if "cmd_air" not in st:
        st["cmd_air"] = torch.zeros_like(rf)
        st["land_fail"] = torch.zeros_like(rf)
        st["land_deck"] = torch.zeros_like(rf)
    if st.get("land_step") != env.common_step_counter:
        st["land_step"] = env.common_step_counter
        fail = on & st["cmd_air"] & rf
        lift = ~rf
        if spawn_grace_s is not None:   # p12rev: a floor-stance spawn's foot settling onto the floor is not a lift
            air_sp = (st["spawn"] == SPAWNS.index("stroke_air")) | (st["spawn"] == SPAWNS.index("step_mid"))
            lift = lift & ((st["stage_t"] >= spawn_grace_s - 1e-6) | air_sp)
        st["cmd_air"] = on & (st["cmd_air"] | lift)
        st["land_deck"] = on & (st["land_deck"] | tk._contact_any(env, "right_deck"))
        if reach_s is not None:
            fail = fail | (on & ~st["land_deck"] & (st["cmd_t"] > reach_s))
        st["land_fail"] = fail
    return st["land_fail"]


def riding_tilt_guard(env, limit_deg: float = 50.0, after_s: float = 1.0) -> torch.Tensor:
    """REV guard: riding with both feet on (RIDE / RAMP), more than after_s into the episode, a trunk tilt beyond
    limit_deg ends the episode (the base fell_over is 60 deg). Between 50 and 60 deg the rider is visibly going down
    (s1b ramp frames: folded sideways over the deck); for a demo that is a failure and the fall cost prices it as one.
    Not in PUSH (stroke stances roll the trunk ~0.2 rad), not in MANUAL (a manual attempt must not get riskier than
    refusing it), not in the first second (tilted manual / exit / wall spawns pass 45 deg in transients the policy
    recovers from: the first 45-deg version ended 14/64 ramp_wall spawns at 0.2-0.7 s, diag/)."""
    ph = seq_update(env)["trick"]
    g = _robot(env).data.projected_gravity_b
    tilt = torch.acos((-g[:, 2]).clamp(-1.0, 1.0))
    riding = ((ph == RIDE) | (ph == RAMP)) & (tk._t(env) > after_s)
    return riding & (torch.nan_to_num(tilt, nan=0.0) > math.radians(limit_deg))


def seq_fall(env, manual_scale: float = 0.6) -> torch.Tensor:
    """1/dt on a non-time-out termination (the weight is the cost of one fall), scaled by manual_scale in MANUAL.
    REV (refusing a phase must not be cheaper than attempting it): at s1e's skill a manual attempt fell every
    ~6 s of MANUAL (-1.6 /s at -10) against 0.3 /s of manual pay, so riding the MANUAL window on four wheels was the
    cheaper choice; a manual fall costs 6 instead of 10 (the lost downstream chain still makes falling expensive)."""
    st = seq_update(env)
    s = torch.where(st["trick"] == MANUAL, torch.full_like(st["v_tgt"], manual_scale), torch.ones_like(st["v_tgt"]))
    return env.termination_manager.terminated.float() * s / env.step_dt


def seq_fall_stage(env, stage_scale: tuple = ((ST_MANUAL, 0.6), (ST_EXIT, 0.6))) -> torch.Tensor:
    """Manual specialist (review 2026-11-20): seq_fall with the scale picked by SCHEDULER STAGE (seq_fall scales only
    the MANUAL phase, so a set-down fall or an EXIT timeout cost 10 against 6 for a MANUAL fall / MANUAL timeout: completing
    the hold moved the risk into the dearer regime, and stalling in MANUAL until its timeout was the cheaper ending)."""
    st = seq_update(env)
    s = torch.ones_like(st["v_tgt"])
    for stg, sc in stage_scale:
        s = torch.where(st["stage"] == stg, torch.full_like(s, float(sc)), s)
    return env.termination_manager.terminated.float() * s / env.step_dt


def manual_hold_done(env) -> torch.Tensor:
    """Manual specialist (review 2026-11-20): 1/dt on the MANUAL -> EXIT transition (manual_evt: the leaky hold band_t
    reached T_hold), once per episode (the stage never goes back), so the weight is the price of completing the hold.
    Needed because MANUAL pays ~7.6 /s in band against ~4 /s in EXIT: without it, delaying the hold (hovering out of
    band, or see-sawing across a band edge so band_t leaks) out-earns completing it until the MANUAL timeout."""
    return seq_update(env)["manual_evt"].float() / env.step_dt


def past_pipe_end(env) -> torch.Tensor:
    """RAMP and near a pipe end: end the episode as a time-out (riding along the axis has no more to give)."""
    st = seq_update(env)
    p = pipe_xyz(env)
    return (st["trick"] == RAMP) & (p[:, 1].abs() > st["pipe"]["W"] / 2 - st["seq_cfg"]["pipe_end_margin"])


def over_coping(env, margin: float = 0.01) -> torch.Tensor:
    """The board climbed to the coping (top platform edge): out of the trained envelope."""
    st = _st(env)
    return (pipe_xyz(env)[:, 2] - DECK_Z) > st["pipe"]["h"] - margin


# ── curricula ─────────────────────────────────────────────────────────────────────────────────────────────────────────


def seq_cfg_curriculum(env, env_ids, stages: list[dict]) -> torch.Tensor:
    """Step-staged st["seq_cfg"] values: stages = [{"step": env steps, "cfg": {SEQ_DEFAULTS key: value}}], the latest
    stage whose step has been reached is merged in (per-env draws such as T_hold pick it up at their next reset)."""
    del env_ids
    st = _st(env)
    cur = stages[0]["cfg"]
    for s in stages:
        if env.common_step_counter >= s["step"]:
            cur = s["cfg"]
    unknown = set(cur) - set(SEQ_DEFAULTS)
    assert not unknown, f"seq_cfg_curriculum: unknown keys {sorted(unknown)}"
    st["seq_cfg"].update(cur)
    v = next(iter(cur.values()))
    return torch.tensor([float(v[0] if isinstance(v, (tuple, list)) else v)])


def stance_tol_curriculum(env, env_ids, stages: list[dict], gate: float = 0.6, decay: float = 0.995,
                          min_n: float = 200.0) -> dict:
    """p9brev (push specialist): seq_cfg_curriculum for stance_tol, gated by performance. Stage k + 1 needs BOTH its
    step (env steps, as seq_cfg_curriculum) AND a decayed success share >= gate over the episodes that ended in SETTLE
    (the `success` termination fired; read here, before the reset event / termination reset) with >= min_n of them since
    the last advance. Stages never go back. A time-only schedule could tighten past what the policy reaches; with the
    potential charged back on the overtime a 0 % stage then has no learning signal and success stays at 0. State lives
    in the per-env cache (not in the checkpoint): a resume re-climbs from stage 0 at the gate's pace. A single stage
    (--stance-tol) is a fixed tolerance."""
    st = _st(env)
    c = tk._tc(env)
    if "stol_k" not in c:
        c["stol_k"], c["stol_s"], c["stol_n"] = 0, 0.0, 0.0
    tm = env.termination_manager
    if env_ids is not None and not isinstance(env_ids, slice) and len(env_ids) > 0 and "success" in tm.active_terms:
        ids = env_ids.to(env.device).long()
        ended = (st["stage"][ids] == ST_SETTLE) & (env.episode_length_buf[ids] > 0)
        if "p17" in st:   # review p17 C (default off): p17 episodes are judged on their own stance, not stance_tol
            ended = ended & ~st["p17"][ids]
        ok = tm.get_term("success")[ids] & ended
        c["stol_s"] = c["stol_s"] * decay + float(ok.sum())
        c["stol_n"] = c["stol_n"] * decay + float(ended.sum())
    k = c["stol_k"]
    p = (c["stol_s"] + 1.0) / (c["stol_n"] + 2.0)
    if (k + 1 < len(stages) and env.common_step_counter >= stages[k + 1]["step"] and c["stol_n"] >= min_n
            and p >= gate):
        k = c["stol_k"] = k + 1
        c["stol_s"], c["stol_n"] = 0.0, 0.0
    cur = stages[min(k, len(stages) - 1)]["cfg"]
    unknown = set(cur) - set(SEQ_DEFAULTS)
    assert not unknown, f"stance_tol_curriculum: unknown keys {sorted(unknown)}"
    st["seq_cfg"].update(cur)
    c["stol_p"] = p
    return torch.tensor([float(k)])        # a scalar, not a dict (mjlab get_active_iterable_terms breaks on dicts)


def m_stance_gate_p(env) -> torch.Tensor:
    """p9brev metric: stance_tol_curriculum's success share estimate (since its last advance), broadcast."""
    return torch.full((env.num_envs,), float(tk._tc(env).get("stol_p", 0.0)), device=env.device)


# ── p17 one-stroke push from the RIDING stance (2026-11-21, push specialist only; DEFAULT OFF: every function below is
# used only by microduck_skate_seq_push_env_cfg with MICRODUCK_SEQPUSH_P17=1; nothing above reads the p17 state).
# A share of the push task's `glide` spawns is
# re-placed (reset_p17, a reset event AFTER reset_sequence) into a REAL riding state of the ramp specialist at the start
# of a forward half-pipe flat crossing (r9 model_4500, both soles on the deck, four wheels down,
# 0.3+ m/s nose-first), on the pipe's flat bottom (the chain's geometry, MICRODUCK_PIPE_H must match the bank) or
# transplanted deck-relative onto the approach floor (the push policy's usual obs geometry). The scheduler starts in
# PUSH with a ONE-STROKE target: v_tgt - v_gap = spawn speed + dv_req (0.04-0.10 m/s), v0 = the spawn speed, so the
# slewed ramp meets the threshold within ~0.3 s and the step command latches during the first push-foot floor contact
# that has raised the speed by dv_req (the p9rev floor-contact latch). Success = the usual seq_success (SETTLE, both
# soles on >= 0.3 s, four wheels >= 0.2 s, hand-off stance held) with a short T_settle AND a legal stroke (p17_ok:
# a floor contact with the board foot on the deck whose lift-off speed beats its touchdown speed by >= P17_GAIN_MIN);
# p17_fail ends it as a failure past the deadline (spawn + deadline_s) or once the board has climbed wall_rise up the
# far wall (pipe placement).
P17_BANK = str(_data_path("bank_ride.npz"))
P17_BANK_FIELDS = ("b_pos", "b_quat", "b_lin", "b_ang", "bj", "bjv", "r_pos", "r_quat", "r_lin", "r_ang", "jpos", "jvel",
                   "vb")
P17_GAIN_MIN = 0.02          # m/s: board speed gained over one push-foot floor contact for it to count as a stroke
P17_BRAKE = 0.02             # m/s: a floor contact that LOSES more than this is a brake (metric seq_p17_brake)
# p17 riding-stance target = the spawn's OWN recorded riding stance (the ramp specialist rides ~10 deg
# more upright than HANDOFF_STANCE, tilt_x err p50 10.7 deg > the final p9b 8 deg; soles within ~1 cm): deck-frame left
# sole xy, right sole xy, CoM height, trunk tilt x / y (deg). Tolerances (left m, right m, CoM m, tilt x deg, tilt y deg)
# ~2x the ramp policy's own spread at the crossing (soles +-1 cm, CoM +-0.5 cm, tilt x +-4.6 deg).
P17_REF_FIELDS = ("ls_x", "ls_y", "rs_x", "rs_y", "com_z", "tilt_x", "tilt_y")
P17_STANCE_TOL = (0.03, 0.025, 0.015, 8.0, 8.0)
P17_PATH_UNITS = (0.01, 0.01, 0.01, 5.0, 5.0)


def _p17_bank(env, path: str) -> dict:
    """Cached {field: tensor (N, ...)} of a bank (joint orders checked) + its pipe height."""
    import numpy as np
    c = tk._tc(env)
    if c.get("p17_bank_path") != path:
        z = np.load(path)
        assert [str(x) for x in z["joint_names"]] == list(_robot(env).joint_names), "p17 bank: robot joint order"
        assert [str(x) for x in z["board_joint_names"]] == list(tk._board(env).joint_names), "p17 bank: board joints"
        c["p17_bank"] = {f: torch.tensor(z[f], device=env.device, dtype=torch.float)
                         for f in P17_BANK_FIELDS + P17_REF_FIELDS}
        c["p17_bank_h"] = float(z["pipe"][0]) if "pipe" in z.files else None
        c["p17_bank_path"] = path
    return c["p17_bank"]


def _p17_state(env) -> dict:
    st = _st(env)
    if "p17" not in st:
        n, dev = env.num_envs, env.device
        b = lambda: torch.zeros(n, dtype=torch.bool, device=dev)
        z = lambda: torch.zeros(n, device=dev)
        st.update(p17=b(), p17_t=z(), p17_vs=z(), p17_rf=b(), p17_tdv=z(), p17_ok=b(), p17_gain=z(), p17_ntd=z(),
                  p17_brake=z(), p17_place=torch.zeros(n, dtype=torch.long, device=dev), p17_step=-1,
                  p17_ref=torch.zeros(n, len(P17_REF_FIELDS), device=dev), p17_st=z(), p17_sp_d=z() - 1.0,
                  p17_sp_d0=z(), p17_pos=b(), p17b_rung=z() - 1.0)
    return st


def _p17_score(env, st: dict, ids: torch.Tensor, decay: float) -> None:
    """Fold the ending p17 episodes of ids into decayed success counts (read before the termination reset), overall
    and per placement (0 pipe / 1 floor): tk._tc(env)["p17_sn"] = {key: [S, N]}."""
    tm = env.termination_manager
    if "success" not in tm.active_terms:
        return
    c = tk._tc(env)
    sn = c.setdefault("p17_sn", {})
    ended = st["p17"][ids] & (env.episode_length_buf[ids] > 0)
    ok = tm.get_term("success")[ids] & ended
    for key, m in (("all", ended), ("pipe", ended & (st["p17_place"][ids] == 0)),
                   ("floor", ended & (st["p17_place"][ids] == 1))):
        s0, n0 = sn.get(key, (0.0, 0.0))
        sn[key] = (s0 * decay + float((ok & m).sum()), n0 * decay + float(m.sum()))


def reset_p17(env, env_ids: torch.Tensor, bank_path: str = P17_BANK, glide_frac: float = 0.85,
              floor_share: float = 0.3, v_range: tuple = (0.4, 0.9), dv_req: tuple = (0.04, 0.10),
              t_settle: tuple = (0.2, 0.3), xy: float = 0.02, y_shift: float = 1.0, yaw_deg: float = 3.0,
              joint: float = 0.02, pick_tol: float = 0.03, decay: float = 0.995, v_scale: tuple = (0.7, 1.7),
              v_range_pipe: tuple | None = None, bridge: dict | None = None, floor_obs: bool = False,
              true_speed: dict | None = None) -> None:
    """p17 reset event (after reset_sequence): glide spawns become p17 riding spawns with probability glide_frac.
    Speed u ~ U(v_range); the bank row is a random one among those within ~pick_tol of u (nearest + jitter), and every
    twist / joint velocity of that state is scaled by u / vb (clamped to v_scale: the r9 crossings run 0.32-0.59 m/s
    p10-p90, so 0.6-0.9 m/s comes from the scaling). st["p17_ref"] = that state's riding stance (P17_REF_FIELDS).
    v_range_pipe (None = v_range): the pipe placements' speed range (on the 1.0 m flat a board above ~0.75 m/s reaches
    the far wall before any stroke + step-on can finish; the r9 crossings themselves run 0.32-0.59 m/s). Placement: with probability
    floor_share on the approach floor (the glide spawn's floor band, heading +y; the recorded state turned about the
    board origin), else in the pipe as recorded (y shifted U(-y_shift, y_shift) along the pipe, x U(-xy, xy), yaw
    U(-yaw_deg, yaw_deg) about the board origin). Joints + N(0, joint). Scheduler: PUSH, v_tgt = u + v_gap +
    U(dv_req), v0 = u, T_settle U(t_settle), clock offset 0. st["p17"] flags those envs.
    bridge (p17b, default None = p17 unchanged): dict(bank=< bank>, low_share=0.25): rows come from the
    stance-bridge bank instead, from the alpha rung tk._tc(env)["p17b_k"] (p17b_alpha_curriculum) or, with probability
    low_share, a uniformly drawn LOWER rung; st["p17b_rung"] = the rung (-1 = not a bridge spawn); p17_vs = the
    row's min(board, trunk) speed x the scale (the stroke must raise THAT); for bridge spawns that same speed is also
    the scheduler's v0 / phi / v_tgt base.
    floor_obs (p17c, default False): the pipe placements get a p17_floor_ref reference (st["p17c_ref"]) and the flag
    st["p17c_on"] read by seq_obs_p17c.
    true_speed (p17d, default None = unchanged; needs bridge): dict(share, src): with probability share a PIPE placement
    takes a uniformly drawn row of its rung whose ride source is src (bank field src; p17d: 0 = bank_chain, the chain's
    real STROKE hand-overs) and whose board speed lies inside v_range_pipe, written at its TRUE recorded speed (scale 1);
    the other placements draw u and scale as above."""
    st = _p17_state(env)
    if env_ids is None or len(env_ids) == 0:
        return
    ids_all = env_ids.to(env.device).long()
    _p17_score(env, st, ids_all, decay)
    for k in ("p17", "p17_rf", "p17_ok", "p17_pos"):
        st[k][ids_all] = False
    for k in ("p17_t", "p17_vs", "p17_tdv", "p17_gain", "p17_ntd", "p17_brake", "p17_st", "p17_sp_d0"):
        st[k][ids_all] = 0.0
    st["p17_sp_d"][ids_all] = -1.0
    st["p17_place"][ids_all] = 0
    st["p17b_rung"][ids_all] = -1.0
    if floor_obs:
        if "p17c_on" not in st:
            st["p17c_on"] = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            st["p17c_ref"] = torch.zeros(env.num_envs, 6, device=env.device)
        st["p17c_on"][ids_all] = False
    dev = env.device
    sel = (st["spawn"][ids_all] == SPAWNS.index("glide")) & (torch.rand(len(ids_all), device=dev) < float(glide_frac))
    if not bool(sel.any()):
        return
    ids = ids_all[sel]
    m = len(ids)
    bk = _p17b_bank(env, bridge["bank"]) if bridge else _p17_bank(env, bank_path)
    U = lambda lo, hi: lo + torch.rand(m, device=dev) * (hi - lo)
    floor = torch.rand(m, device=dev) < float(floor_share)
    vr_p = v_range_pipe or v_range
    u = torch.where(floor, U(*v_range), U(*vr_p))
    d = (bk["vb"].unsqueeze(0) - u.unsqueeze(1)).abs() + torch.rand(m, len(bk["vb"]), device=dev) * float(pick_tol)
    if bridge:   # p17b: rows of the current alpha rung, or (low_share) of a uniformly drawn lower rung
        k_cur = int(tk._tc(env).get("p17b_k", 0))
        low = (torch.rand(m, device=dev) < float(bridge.get("low_share", 0.25))) & (k_cur > 0)
        rr = torch.where(low, torch.randint(0, max(k_cur, 1), (m,), device=dev), torch.full((m,), k_cur, device=dev))
        d = d + 1.0e3 * (bk["rung"].unsqueeze(0) != rr.unsqueeze(1).float()).float()
    if true_speed:   # p17d: some pipe placements = a uniform row of the rung from source src, at its recorded speed
        assert bridge, "reset_p17 true_speed needs bridge"
        tsp = ~floor & (torch.rand(m, device=dev) < float(true_speed.get("share", 0.7)))
        okr = ((bk["src"] == float(true_speed.get("src", 0))) & (bk["vb"] >= float(vr_p[0]))
               & (bk["vb"] <= float(vr_p[1])))
        rung_ok = bk["rung"].unsqueeze(0) == rr.unsqueeze(1).float()                        # (m, N)
        tsp = tsp & (okr.unsqueeze(0) & rung_ok).any(dim=1)   # no qualifying row in its rung -> the u draw (review)
        d_t = (torch.rand(m, len(bk["vb"]), device=dev) + 1.0e3 * (~okr).float().unsqueeze(0)
               + 1.0e3 * (~rung_ok).float())
        d = torch.where(tsp.unsqueeze(1), d_t, d)
    pick = d.argmin(dim=1)
    g = {f: v[pick] for f, v in bk.items()}
    vs = (u / g["vb"].clamp(min=0.1)).clamp(float(v_scale[0]), float(v_scale[1])).unsqueeze(1)
    if true_speed:
        vs = torch.where(tsp.unsqueeze(1), torch.ones_like(vs), vs)
    u = g["vb"] * vs.squeeze(1)                    # the speed actually written (= u unless the scale clamped)
    if bridge:   # p17b review: ONE baseline = min(board, trunk) as written (v0, phi, v_tgt base and p17_vs)
        u = g["pv"] * vs.squeeze(1)
    if not bool(floor.all()):
        hb = tk._tc(env).get("p17b_bank_h" if bridge else "p17_bank_h")
        assert hb is None or abs(hb - st["pipe"]["h"]) < 1e-6, (
            f"p17 bank recorded on a {hb} m pipe, this env has {st['pipe']['h']} m: set MICRODUCK_PIPE_H={hb}")
    cfg, W = st["seq_cfg"], st["pipe"]["W"]
    z0 = torch.zeros(m, device=dev)
    yaw_r = _yaw_from_quat(g["b_quat"])
    gap = cfg["start_gap"]
    yaw_f = math.pi / 2 + (torch.rand(m, device=dev) * 2 - 1) * cfg.get("flat_yaw_noise", 0.08)
    dyaw = torch.where(floor, yaw_f - yaw_r, torch.deg2rad(U(-yaw_deg, yaw_deg)))
    dq = _quat_from_rpy_t(z0, z0, dyaw)
    y_pipe = (g["b_pos"][:, 1] + U(-y_shift, y_shift)).clamp(-W / 2 + 1.0, W / 2 - 1.0)
    bp_pipe = torch.stack([g["b_pos"][:, 0] + U(-xy, xy), y_pipe, g["b_pos"][:, 2]], dim=1)
    bp_floor = torch.stack([U(-0.6, 0.6), -(W / 2 + gap) + U(0.1 * gap, 0.4 * gap), g["b_pos"][:, 2]], dim=1)
    bp = torch.where(floor.unsqueeze(1), bp_floor, bp_pipe)
    nq = lambda q: q / q.norm(dim=1, keepdim=True)
    org = _origin(env)[ids]
    robot, board = _robot(env), tk._board(env)
    board.write_root_link_pose_to_sim(torch.cat([org + bp, nq(quat_mul(dq, g["b_quat"]))], dim=1), ids)
    board.write_root_link_velocity_to_sim(torch.cat([quat_apply(dq, g["b_lin"]), quat_apply(dq, g["b_ang"])], dim=1) * vs,
                                          ids)
    board.write_joint_state_to_sim(g["bj"], g["bjv"] * vs, env_ids=ids)
    rp_ = bp + quat_apply(dq, g["r_pos"] - g["b_pos"])
    robot.write_root_link_pose_to_sim(torch.cat([org + rp_, nq(quat_mul(dq, g["r_quat"]))], dim=1), ids)
    robot.write_root_link_velocity_to_sim(torch.cat([quat_apply(dq, g["r_lin"]), quat_apply(dq, g["r_ang"])], dim=1) * vs,
                                          ids)
    robot.write_joint_state_to_sim(g["jpos"] + torch.randn_like(g["jpos"]) * float(joint), g["jvel"] * vs, env_ids=ids)
    # scheduler: PUSH with a one-stroke target (see the block comment)
    st["stage"][ids] = ST_PUSH
    st["trick"][ids] = PUSH
    if "obs_ph" in st:
        st["obs_ph"][ids] = PUSH
    st["stage_t"][ids] = 0.0
    st["t_stage"][ids] = 0.0
    st["T_settle"][ids] = U(*t_settle)
    st["v_tgt"][ids] = u + float(cfg["v_gap"]) + U(*dv_req)
    st["v0"][ids] = u
    st["phi"][ids] = u
    st["both_on"][ids] = 0.0
    st["wheels_on"][ids] = 0.0
    st["t_off"][ids] = 0.0
    st["kappa"][ids] = 0.0
    st["dir"][ids] = 1.0
    if "cmd" in st:
        st["cmd"][ids] = False
        st["cmd_t"][ids] = 0.0
    st["p17"][ids] = True
    st["p17_vs"][ids] = u
    if bridge:
        st["p17b_rung"][ids] = rr.float()
    st["p17_ref"][ids] = torch.stack([g[f] for f in P17_REF_FIELDS], dim=1)
    st["p17_place"][ids] = floor.long()
    if floor_obs:   # p17c: the pipe placements' floor-equivalent obs reference (as written: bp, the board yaw)
        yaw_w = _yaw_from_quat(nq(quat_mul(dq, g["b_quat"])))
        st["p17c_ref"][ids] = p17_floor_ref(bp, yaw_w, W, float(gap))
        st["p17c_on"][ids] = ~floor


def _p17_update(env) -> dict:
    """Once per env step (called by the p17 termination / rewards / metrics): p17 clock, push-foot floor contacts
    (touchdown speed, lift-off gain), the legal-stroke latch p17_ok, touchdown and brake counts."""
    st = _p17_state(env)
    if st["p17_step"] == env.common_step_counter:
        return st
    st["p17_step"] = env.common_step_counter
    m = st["p17"]
    _, rf = _feet_floor(env)
    # min(board, trunk) forward speed (push_progress' measure): a light board sped up by the rider's own lean / internal
    # motion is not a stroke (zero-shot p16: board +0.04..+0.10 m/s with no floor contact)
    v = torch.nan_to_num(_push_v(env, st))
    td = rf & ~st["p17_rf"]
    lo = ~rf & st["p17_rf"]
    st["p17_tdv"] = torch.where(td, v, st["p17_tdv"])
    gain = v - st["p17_tdv"]
    left = tk._contact_any(env, "left_deck")
    st["p17_ntd"] = st["p17_ntd"] + (td & m).float()
    # review p17 A: the stroke must also leave min(board, trunk) above the SPAWN speed (system speed raised, not only
    # a touchdown timed at a trunk-lag dip)
    st["p17_ok"] = st["p17_ok"] | (lo & m & left & (gain >= P17_GAIN_MIN) & (v >= st["p17_vs"] + P17_GAIN_MIN))
    st["p17_gain"] = torch.where(lo & m, torch.maximum(st["p17_gain"], gain), st["p17_gain"])
    st["p17_pos"] = st["p17_pos"] | (lo & m & left & (gain >= 0.5 * P17_GAIN_MIN))      # p17_glide_tax: a lift-off that gained speed
    st["p17_brake"] = st["p17_brake"] + (lo & m & (gain < -P17_BRAKE)).float()
    st["p17_rf"] = rf
    st["p17_t"] = st["p17_t"] + env.step_dt * m.float()
    e = p17_stance_err(env)
    tol = P17_STANCE_TOL
    ok = (m & (st["stage"] == ST_SETTLE) & tk.soles_on_deck(env) & (e[:, 0] < tol[0]) & (e[:, 1] < tol[1])
          & (e[:, 2] < tol[2]) & (e[:, 3] < tol[3]) & (e[:, 4] < tol[4]))
    st["p17_st"] = torch.where(ok, st["p17_st"] + env.step_dt, torch.zeros_like(st["p17_st"]))
    return st


def p17_stance_feats(env) -> torch.Tensor:
    """(B, 7) the riding-stance features P17_REF_FIELDS (measures): deck-frame left sole xy, right
    sole xy (m), CoM height above the deck (m), trunk tilt along deck x / across deck y (deg). chain p17 (2026-11-21):
    shared by p17_stance_err (training) and chain_play's STROKE owner (its own stance, recorded at the hand-over)."""
    robot, b = _robot(env), tk._board(env)
    dev = env.device
    ls, rs = _left_sole_deck(env)[:, :2], _right_sole_deck_xy(env)[:, :2]
    rz = quat_apply(robot.data.root_link_quat_w, torch.tensor([0.0, 0.0, 1.0], device=dev).expand(env.num_envs, 3))
    rz_d = quat_apply_inverse(b.data.root_link_quat_w, rz)
    tx = torch.rad2deg(torch.asin(rz_d[:, 0].clamp(-1, 1)))
    ty = torch.rad2deg(torch.asin(rz_d[:, 1].clamp(-1, 1)))
    return torch.cat([ls, rs, _robot_com_deck(env)[:, 2:3], tx.unsqueeze(1), ty.unsqueeze(1)], dim=1)


def p17_stance_err_ref(env, ref: torch.Tensor) -> torch.Tensor:
    """(B, 5) p17_stance_err against an explicit (B, 7) reference (P17_REF_FIELDS order)."""
    f = p17_stance_feats(env)
    e = torch.stack([(f[:, 0:2] - ref[:, 0:2]).norm(dim=1), (f[:, 2:4] - ref[:, 2:4]).norm(dim=1),
                     (f[:, 4] - ref[:, 4]).abs(), (f[:, 5] - ref[:, 5]).abs(), (f[:, 6] - ref[:, 6]).abs()], dim=1)
    return torch.nan_to_num(e, nan=1.0e3)


def p17_stance_err(env) -> torch.Tensor:
    """(B, 5) errors against the env's own spawn riding stance st["p17_ref"]: left / right sole xy distance (m), |CoM
    height - ref| (m), |trunk tilt - ref| along deck x / across deck y (deg). Same measures as _stance_err.
    (chain p17: the body moved to p17_stance_feats / p17_stance_err_ref, unchanged)"""
    return p17_stance_err_ref(env, _p17_state(env)["p17_ref"])


def p17_success(env, stage: int = ST_SETTLE, on_s: float = 0.3, wheels_s: float = 0.2, stance: bool = False
                ) -> torch.Tensor:
    """Non-p17 envs: seq_success unchanged. p17 envs: seq_success WITHOUT the hand-off stance, AND the legal stroke
    (p17_ok) AND their own riding stance (p17_stance_err within P17_STANCE_TOL, both soles on) held for
    seq_cfg stance_hold_s."""
    st = _p17_update(env)
    base = seq_success(env, stage=stage, on_s=on_s, wheels_s=wheels_s, stance=stance)
    plain = seq_success(env, stage=stage, on_s=on_s, wheels_s=wheels_s, stance=False)
    hold = float(st["seq_cfg"].get("stance_hold_s", 0.2)) - 1e-6
    return torch.where(st["p17"], plain & st["p17_ok"] & (st["p17_st"] >= hold), base)


def p17_stance_path(env, cap: float = 10.0) -> torch.Tensor:
    """POSITIVE weight, p17 envs only: settle_stance_path's potential toward the env's OWN riding stance (distance =
    sum of min(p17_stance_err / P17_PATH_UNITS, cap)) in SETTLE, telescoping from the step-on; a non-time-out
    termination in SETTLE charges the banked sum back. The HANDOFF_STANCE potential is masked on p17 envs
    (p17_settle_stance_path)."""
    st = _p17_update(env)
    e = p17_stance_err(env)
    u = torch.tensor(P17_PATH_UNITS, device=env.device)
    d = (e / u).clamp(max=cap).sum(dim=1)
    on = st["p17"] & (st["stage"] == ST_SETTLE)
    init = on & (st["p17_sp_d"] < 0)
    live = on & ~init
    pay = torch.where(live, st["p17_sp_d"] - d, torch.zeros_like(d))
    st["p17_sp_d0"] = torch.where(init, d, st["p17_sp_d0"])
    term = env.termination_manager.terminated
    pay = torch.where(live & term, pay - (st["p17_sp_d0"] - d), pay)
    st["p17_sp_d"] = torch.where(on, d, st["p17_sp_d"])
    return pay / env.step_dt


def p17_settle_stance_path(env, **kw) -> torch.Tensor:
    """settle_stance_path (its state still updated every step) with the p17 envs masked out."""
    out = settle_stance_path(env, **kw)
    return out * (~_p17_state(env)["p17"]).float()


def p17_fail(env, deadline_s: float = 1.6, wall_rise: float = 0.02, success: dict | None = None,
             deadline_pipe: float | None = None, wall_turn: bool = False) -> torch.Tensor:
    """Failure of a p17 episode: not successful by deadline_s after the spawn, or (pipe placement) the board has
    climbed wall_rise up the far wall first. Never fires on a step where p17_success holds.
    p17d (defaults = unchanged): deadline_pipe = the pipe placements' deadline (s; the floor placements keep the
    curriculum; the larger of the two applies), wall_turn = a pipe placement also fails once the board has reached the
    far wall and turned back (p17_wall_turn: below ~0.63 m/s a board never climbs wall_rise = 2 cm, v^2 / 2g)."""
    st = _p17_update(env)
    p = pipe_xyz(env)
    rise = p[:, 2] - DECK_Z
    dl = max(float(deadline_s), float(tk._tc(env).get("p17_deadline", deadline_s)))   # p17_deadline_curriculum
    pipe_m = st["p17_place"] == 0
    wall = rise > wall_rise
    if wall_turn:
        assert "p17c_ref" in st, "p17_fail wall_turn needs reset_p17(floor_obs=True) (the travel sign)"
        wall = wall | p17_wall_turn(p, tk._board(env).data.root_link_lin_vel_w[:, 0], st["p17c_ref"][:, 2],
                                    float(st["pipe"]["L"]))
    if deadline_pipe is None:
        late = st["p17_t"] > dl
    else:
        late = st["p17_t"] > torch.where(pipe_m, torch.full_like(st["p17_t"], max(dl, float(deadline_pipe))),
                                         torch.full_like(st["p17_t"], dl))
    bad = st["p17"] & (late | (pipe_m & wall))
    return bad & ~p17_success(env, **(success or {}))


def p17_deadline_curriculum(env, env_ids, stages: tuple = (3.0, 2.4, 2.0, 1.6), gate: float = 0.5,
                            decay: float = 0.995, min_n: float = 200.0) -> torch.Tensor:
    """p17 (mini-run p17mini512: at the 1.6 s deadline from the start the riding spawns made ~0 successes, p16 first
    touches the floor at ~1.0-1.3 s): the p17 deadline (s) steps DOWN through `stages` once the decayed success share of
    the ended p17 episodes since the last step (>= min_n of them) reaches `gate`; never back up. Read by p17_fail as
    tk._tc(env)["p17_deadline"] (its deadline_s param is the floor). Cached, not checkpointed: a resume restarts at
    stages[0] and re-climbs at the gate's pace. Runs before the reset events, so the ending envs still carry st["p17"]."""
    c = tk._tc(env)
    if "p17_dk" not in c:
        c["p17_dk"], c["p17_ds"], c["p17_dn"] = 0, 0.0, 0.0
    tm = env.termination_manager
    st = _p17_state(env)
    if env_ids is not None and not isinstance(env_ids, slice) and len(env_ids) > 0 and "success" in tm.active_terms:
        ids = env_ids.to(env.device).long()
        ended = st["p17"][ids] & (env.episode_length_buf[ids] > 0)
        ok = tm.get_term("success")[ids] & ended
        c["p17_ds"] = c["p17_ds"] * decay + float(ok.sum())
        c["p17_dn"] = c["p17_dn"] * decay + float(ended.sum())
    p = (c["p17_ds"] + 1.0) / (c["p17_dn"] + 2.0)
    k = c["p17_dk"]
    if k + 1 < len(stages) and c["p17_dn"] >= min_n and p >= gate:
        k = c["p17_dk"] = k + 1
        c["p17_ds"], c["p17_dn"] = 0.0, 0.0
        print(f"[p17] deadline -> {stages[k]} s at step {env.common_step_counter} (success p {p:.2f})", flush=True)
    c["p17_deadline"] = float(stages[k])
    return torch.tensor([float(stages[k])])


def p17_glide_tax(env, after_s: float = 0.5) -> torch.Tensor:
    """Cost (NEGATIVE weight), p17 envs: 1 per second while still in PUSH and no push-foot lift-off has yet gained
    >= P17_GAIN_MIN / 2 (st["p17_pos"]), from after_s after the spawn. With the deadline curriculum starting at 3.0 s a glide to the deadline is discounted to
    ~0.22 at the spawn while an attempt that falls at ~0.8 s is not: without a dense price, refusing was as cheap as
    attempting (review p17). The escape is a lift-off that gained >= GAIN_MIN/2 (a bare tap does not; the deadline keeps full price,
    p17_fall_push)."""
    st = _p17_update(env)
    # re-review: the tax stops on a lift-off that GAINED speed (p17_pos), not on any touchdown (a 1-step tap bought a
    # free glide)
    return (st["p17"] & (st["stage"] == ST_PUSH) & ~st["p17_pos"] & (st["p17_t"] > after_s)).float()


def p17_fall_push(env, attempt_scale: float = 0.5, gamma: float = 0.99, stroke_window_s: float = 0.3,
                  deadline_s: float = 1.6, wall_rise: float = 0.02, deadline_pipe: float | None = None,
                  **kw) -> torch.Tensor:
    """seq_fall_push, and on a p17 env IN PUSH, not under the step command (seq_fall_push's own attempt ramp covers
    that), WHILE A STROKE IS UNDER WAY (push foot on the floor now or within stroke_window_s), a fall / board_lost /
    foot_on_floor costs max(attempt_scale, gamma^(T / dt)), T = time left to the deadline or, for pipe placements, to
    the far-wall gate if earlier: never below the p17_fail ending discounted back to now (re-review: an undiscounted 0.5 after any earlier tap made bailing just before the deadline / in SETTLE
    the cheapest refusal). The p17_fail termination (deadline / wall) and every other case keep seq_fall_push's price.
    Non-p17 envs: seq_fall_push unchanged. deadline_s = p17_fail's floor (the cached curriculum value wins if larger)."""
    out = seq_fall_push(env, **kw)
    st = _p17_update(env)
    tm = env.termination_manager
    dl = tm.get_term("p17_fail") if "p17_fail" in tm.active_terms else torch.zeros_like(st["p17"])
    cmd = st.get("cmd", torch.zeros_like(st["p17"]))
    att = (st["p17"] & (st["stage"] == ST_PUSH) & ~cmd & ~dl & (st["p17_ntd"] > 0.5)
           & _push_foot_recent(env, stroke_window_s))
    dls = max(float(deadline_s), float(tk._tc(env).get("p17_deadline", deadline_s)))
    T = dls - st["p17_t"]
    if deadline_pipe is not None:   # p17d: the pipe placements' own deadline (= p17_fail's)
        T = torch.where(st["p17_place"] == 0, max(dls, float(deadline_pipe)) - st["p17_t"], T)
    # re-review 2: pipe placements end at the far-wall gate (rise > wall_rise) long before a 2.4-3.0 s deadline: match
    # against the EARLIER of the deadline and the estimated wall arrival (board origin |x| where the arc has risen by
    # wall_rise, along the board heading, at the current speed floored at 0.2 m/s)
    pipe = st["pipe"]
    R, L = float(pipe["R"]), float(pipe["L"])
    x_gate = L / 2 + math.sqrt(max(2.0 * R * wall_rise - wall_rise * wall_rise, 0.0))
    q = tk._board(env).data.root_link_quat_w
    ex = quat_apply(q, torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(env.num_envs, 3))[:, 0]
    d_rem = x_gate - pipe_xyz(env)[:, 0] * torch.sign(ex)
    t_wall = d_rem.clamp(min=0.0) / board_v_fwd(env).clamp(min=0.2)
    T = torch.where(st["p17_place"] == 0, torch.minimum(T, t_wall), T)
    k = (torch.nan_to_num(T, nan=0.0).clamp(min=0.0) / env.step_dt)
    sc = torch.pow(torch.full_like(k, float(gamma)), k).clamp(min=float(attempt_scale), max=1.0)
    return torch.where(att, out * sc, out)


def m_p17_deadline(env) -> torch.Tensor:
    return torch.full((env.num_envs,), float(tk._tc(env).get("p17_deadline", 0.0)), device=env.device)


def p17_heading_line_cost(env, **kw) -> torch.Tensor:
    """heading_line_cost with the p17 envs masked out (they ride across the pipe / were not aimed by the start)."""
    st = _p17_state(env)
    return heading_line_cost(env, **kw) * (~st["p17"]).float()


def _p17_p(env, key: str) -> float:
    s, n = tk._tc(env).get("p17_sn", {}).get(key, (0.0, 0.0))
    return (s + 1.0) / (n + 2.0)


def m_p17_success(env) -> torch.Tensor:
    """metric: decayed success share of the ended p17 episodes (all placements), broadcast."""
    return torch.full((env.num_envs,), _p17_p(env, "all"), device=env.device)


def m_p17_success_pipe(env) -> torch.Tensor:
    return torch.full((env.num_envs,), _p17_p(env, "pipe"), device=env.device)


def m_p17_share(env) -> torch.Tensor:
    """metric: share of envs currently in a p17 episode."""
    return _p17_update(env)["p17"].float()


def _p17_mean(st: dict, x: torch.Tensor) -> torch.Tensor:
    """mean of x over the current p17 envs (0 if none), broadcast to every env (no nan in the metric means)."""
    m = st["p17"].float()
    return torch.full_like(x, float((x * m).sum() / m.sum().clamp(min=1.0)))


def m_p17_ok(env) -> torch.Tensor:
    """metric: among the current p17 envs, the share that has made its legal stroke so far."""
    st = _p17_update(env)
    return _p17_mean(st, st["p17_ok"].float())


def m_p17_brake(env) -> torch.Tensor:
    """metric: brake contacts so far, mean over the current p17 envs."""
    st = _p17_update(env)
    return _p17_mean(st, st["p17_brake"])


# ── p17b stance bridge (2026-11-21, DEFAULT OFF: MICRODUCK_SEQPUSH_P17B=1 on top of p17). p17 mini-runs: from the ramp
# policy's riding stance p16 never attempts a stroke (0 touchdowns at model_5800; zero-shot attempts fall ~1 s in).
# p17b spawns p17 envs from a bank of blend(alpha) states between p16's OWN pre-stroke glide states (
# successful episodes only) and the ramp riding states, made physically consistent offline
# (height fix + a static settle with ctrl = qpos; the velocities are the recorded ones scaled, never
# produced by the settle). Return-stance target = lerp(p16's settled stance after that stroke, the ride stance, alpha).
# alpha climbs 0.0 -> 1.0 in rungs of 0.1 (p17b_alpha_curriculum) when the success share at the current rung >= gate;
# reset_p17(bridge=...) keeps low_share of the bridge spawns on lower rungs.
P17B_BANK = str(_data_path("bank_bridge.npz"))


def _p17b_bank(env, path: str) -> dict:
    """Cached {field: tensor} of a bank (P17_BANK_FIELDS + P17_REF_FIELDS (the TARGET stance) + pv, rung,
    alpha); joint orders checked; every rung must have rows."""
    import numpy as np
    c = tk._tc(env)
    if c.get("p17b_bank_path") != path:
        z = np.load(path)
        assert [str(x) for x in z["joint_names"]] == list(_robot(env).joint_names), "p17b bank: robot joint order"
        assert [str(x) for x in z["board_joint_names"]] == list(tk._board(env).joint_names), "p17b bank: board joints"
        bk = {f: torch.tensor(z[f], device=env.device, dtype=torch.float)
              for f in P17_BANK_FIELDS + P17_REF_FIELDS + ("pv", "rung", "alpha")}
        # p17d: the ride source of each row (--ride A B ...; absent in single-source banks = 0)
        bk["src"] = (torch.tensor(z["src"], device=env.device, dtype=torch.float) if "src" in z.files
                     else torch.zeros_like(bk["vb"]))
        nr = len(z["alphas"])
        cnt = [int((bk["rung"] == k).sum()) for k in range(nr)]
        assert min(cnt) > 0, f"p17b bank: empty rung(s) {cnt}"
        c["p17b_bank"], c["p17b_alphas"] = bk, [float(x) for x in z["alphas"]]
        c["p17b_bank_h"] = float(z["pipe"][0]) if "pipe" in z.files else None
        c["p17b_bank_path"] = path
    return c["p17b_bank"]


def p17b_alpha_curriculum(env, env_ids, bank: str = P17B_BANK, gate: float = 0.6, min_n: float = 150.0,
                          decay: float = 0.995, start_k: int = 0, gate_place: str = "all") -> torch.Tensor:
    """p17b: the alpha rung tk._tc(env)["p17b_k"] (read by reset_p17(bridge=...)) steps UP by one when the decayed
    success share of the ended p17 episodes spawned AT the current rung (>= min_n of them) reaches gate; never down.
    Per-rung decayed counts c["p17b_sn"][k] = [S, N] (metrics seq_p17b_s<k>). Runs before the reset events, so the
    ending envs still carry st["p17b_rung"]. Cached, not checkpointed: a resume restarts at start_k. Returns alpha.
    gate_place: which placements the GATE counts: "all" (default), "floor" or "pipe" (separate decayed counts
    c["p17b_gn"]; metric seq_p17b_gate_p); the per-rung metrics always count all placements."""
    c = tk._tc(env)
    _p17b_bank(env, bank)
    al = c["p17b_alphas"]
    c.setdefault("p17b_k", int(start_k))
    c.setdefault("p17b_sn", [[0.0, 0.0] for _ in al])
    assert gate_place in ("all", "floor", "pipe"), f"p17b gate_place {gate_place!r}"
    c.setdefault("p17b_gn", [[0.0, 0.0] for _ in al])
    c.setdefault("p17b_pn", [[0.0, 0.0] for _ in al])
    tm = env.termination_manager
    st = _p17_state(env)
    if env_ids is not None and not isinstance(env_ids, slice) and len(env_ids) > 0 and "success" in tm.active_terms:
        ids = env_ids.to(env.device).long()
        ended = st["p17"][ids] & (env.episode_length_buf[ids] > 0) & (st["p17b_rung"][ids] >= 0)
        ok = tm.get_term("success")[ids] & ended
        rung = st["p17b_rung"][ids].long().clamp(min=0)
        ns_ = torch.bincount(rung[ended], minlength=len(al)).tolist()      # one sync each (review nit)
        ss_ = torch.bincount(rung[ok], minlength=len(al)).tolist()
        gm = ended if gate_place == "all" else ended & (st["p17_place"][ids] == (1 if gate_place == "floor" else 0))
        gn_ = torch.bincount(rung[gm], minlength=len(al)).tolist()
        gs_ = torch.bincount(rung[ok & gm], minlength=len(al)).tolist()
        pm = ended & (st["p17_place"][ids] == 0)                          # pipe-only counts (metrics)
        pn_ = torch.bincount(rung[pm], minlength=len(al)).tolist()
        ps_ = torch.bincount(rung[ok & pm], minlength=len(al)).tolist()
        for k in range(len(al)):
            s0, n0 = c["p17b_sn"][k]
            c["p17b_sn"][k] = [s0 * decay + float(ss_[k]), n0 * decay + float(ns_[k])]
            s0, n0 = c["p17b_gn"][k]
            c["p17b_gn"][k] = [s0 * decay + float(gs_[k]), n0 * decay + float(gn_[k])]
            s0, n0 = c["p17b_pn"][k]
            c["p17b_pn"][k] = [s0 * decay + float(ps_[k]), n0 * decay + float(pn_[k])]
    k = c["p17b_k"]
    s, nn = c["p17b_gn"][k]
    p = (s + 1.0) / (nn + 2.0)
    if k + 1 < len(al) and nn >= min_n and p >= gate:
        k = c["p17b_k"] = k + 1
        print(f"[p17b] alpha -> {al[k]:.2f} (rung {k}) at step {env.common_step_counter} (rung {k - 1} success p {p:.2f})",
              flush=True)
    return torch.tensor([float(al[k])])


def m_p17b_alpha(env) -> torch.Tensor:
    c = tk._tc(env)
    al = c.get("p17b_alphas", [0.0])
    return torch.full((env.num_envs,), float(al[int(c.get("p17b_k", 0))]), device=env.device)


def m_p17b_gate_p(env) -> torch.Tensor:
    """The share the p17b alpha gate reads at the current rung ((S+1)/(N+2) of its gate_place counts)."""
    c = tk._tc(env)
    gn = c.get("p17b_gn")
    s, nn = gn[int(c.get("p17b_k", 0))] if gn is not None else (0.0, 0.0)
    return torch.full((env.num_envs,), (s + 1.0) / (nn + 2.0), device=env.device)


def m_p17b_rung_success(env, k: int = 0) -> torch.Tensor:
    """Decayed success share (Laplace (S+1)/(N+2)) of the p17 episodes spawned at rung k (0.5 = no data)."""
    sn = tk._tc(env).get("p17b_sn")
    s, nn = sn[k] if sn is not None and k < len(sn) else (0.0, 0.0)
    return torch.full((env.num_envs,), (s + 1.0) / (nn + 2.0), device=env.device)


def m_p17b_rung_n(env, k: int = 0) -> torch.Tensor:
    sn = tk._tc(env).get("p17b_sn")
    nn = sn[k][1] if sn is not None and k < len(sn) else 0.0
    return torch.full((env.num_envs,), float(nn), device=env.device)


# ── p17c floor-equivalent placement obs (2026-11-21, DEFAULT OFF: MICRODUCK_SEQPUSH_P17C=1 on top of P17B). The pipe-flat
# diagnostic: p16 on pipe-flat placements mostly fails because the placement obs (seq slots
# -9..-3) are new to it (cmd_landing 39 % -> 6 % with them rewritten). The chain will feed the push policy those slots
# rewritten to a matching FLOOR placement; p17c trains pipe placements with the SAME rewrite. ONE implementation, used by
# the training obs term (seq_obs_p17c) and by chain playback: ref = p17_floor_ref(...) at the hand-over, then every step
# seq[:, -13:] = p17_floor_obs(seq[:, -13:], pipe_xyz, board yaw, ref, W).
P17C_XF0 = 0.4               # floor-equivalent start x ~ U(-0.4, 0.4) (|x| < OBS L/2: the approach floor's slopes are 0)
P17C_YF0 = (0.1, 0.4)        # floor-equivalent start y = -(W/2 + gap) + U(...) * gap (the floor placement's band)


def p17_floor_ref(p0: torch.Tensor, yaw0: torch.Tensor, W: float, gap: float, xf0: torch.Tensor | None = None,
                  yf0: torch.Tensor | None = None) -> torch.Tensor:
    """(B, 6) reference of a pipe-flat hand-over for p17_floor_obs: [x0, y0, s, xf0, yf0, z0n]: the pipe-frame board
    position at the hand-over (p0 (B, 3)), the travel sign s along the pipe x axis (+1 if cos yaw0 >= 0 else -1), the
    floor-equivalent start (xf0, yf0; drawn as the floor placement draws them when None) and the deck height obs at the
    hand-over (z / OBS_PIPE h)."""
    n, dev = p0.shape[0], p0.device
    s = torch.where(torch.cos(yaw0) >= 0, torch.ones_like(yaw0), -torch.ones_like(yaw0))
    if xf0 is None:
        xf0 = (torch.rand(n, device=dev) * 2 - 1) * P17C_XF0
    if yf0 is None:
        yf0 = -(W / 2 + gap) + (P17C_YF0[0] + (P17C_YF0[1] - P17C_YF0[0]) * torch.rand(n, device=dev)) * gap
    return torch.stack([p0[:, 0], p0[:, 1], s, xf0, yf0, p0[:, 2] / OBS_PIPE["h"]], dim=1)


def p17_floor_obs(seq: torch.Tensor, p: torch.Tensor, yaw: torch.Tensor, ref: torch.Tensor, W: float) -> torch.Tensor:
    """Pure: the (B, 13) seq_obs block with the placement slots (indices 4..10 = -9..-3: x / 0.7, y / (W/2), sin / cos
    yaw, deck z / h, front / rear truck slope) replaced by what a matching approach-floor placement shows: the travel
    distance d = (x - x0) s becomes +y from yf0, the lateral offset (y - y0) s becomes -x from xf0, the heading error
    from the travel axis is kept around +y (pi/2), deck z = the hand-over value, slopes 0 (the floor is flat). p = pipe
    frame board position (B, 3), yaw = board yaw (B,), ref = p17_floor_ref(...)."""
    s = ref[:, 2]
    d = (p[:, 0] - ref[:, 0]) * s
    lat = (p[:, 1] - ref[:, 1]) * s
    xf, yf = ref[:, 3] - lat, ref[:, 4] + d
    yw = yaw - torch.where(s > 0, torch.zeros_like(yaw), torch.full_like(yaw, math.pi)) + math.pi / 2
    out = seq.clone()
    out[:, 4] = (xf / 0.7).clamp(-1.5, 1.5)
    out[:, 5] = (yf / (W / 2)).clamp(-2.0, 2.0)
    out[:, 6] = torch.sin(yw)
    out[:, 7] = torch.cos(yw)
    out[:, 8] = ref[:, 5]
    out[:, 9] = 0.0
    out[:, 10] = 0.0
    return torch.nan_to_num(out)


def seq_obs_p17c(env) -> torch.Tensor:
    """seq_obs, with the p17 PIPE-placement envs (st["p17c_on"], set by reset_p17(floor_obs=True)) seeing
    p17_floor_obs (their reference st["p17c_ref"] from the spawn). Every other env: seq_obs unchanged."""
    out = seq_obs(env)
    st = _p17_state(env)
    if "p17c_on" not in st:
        return out
    m = st["p17c_on"] & st["p17"]
    rw = p17_floor_obs(out, pipe_xyz(env), tk._board_yaw(env), st["p17c_ref"], st["pipe"]["W"])
    return torch.where(m.unsqueeze(1), rw, out)


def m_p17b_rung_pipe(env, k: int = 0) -> torch.Tensor:
    """Decayed success share (S+1)/(N+2) of the PIPE-placement p17 episodes spawned at rung k (0.5 = no data)."""
    pn = tk._tc(env).get("p17b_pn")
    s, nn = pn[k] if pn is not None and k < len(pn) else (0.0, 0.0)
    return torch.full((env.num_envs,), (s + 1.0) / (nn + 2.0), device=env.device)


# -- p17d chain hand-over states (2026-11-21, DEFAULT OFF: MICRODUCK_SEQPUSH_P17D=1 on top of P17C). The chain's real
# STROKE hand-overs (-> bank_chain.npz: chain_play.StrokeGate entries of p16 + m6 + r9 at full
# realism) are nose-first flat crossings at ~0.23-0.40 m/s, while p17c's alpha-1 rung (bank_bridge.npz) sat at 0.50-0.65
# m/s (its ride rows were paired to the glide speeds). P17D_BANK = --ride bank_chain.npz bank_ride.npz
# --src-w ... --exact-top: alpha 1 = the recorded rows themselves (src 0 = chain, 1 = r9 ride bank); reset_p17
# true_speed draws pipe placements from src 0 at their recorded speed. Low speed: below ~0.63 m/s the board never
# climbs the 2 cm wall gate (v^2 / 2g), it turns back on the far transition -> p17_fail(wall_turn=True); the pipe
# placements' deadline = the chain's STROKE deadline (deadline_pipe).
P17D_BANK = str(_data_path("bank_bridge_chain.npz"))


def p17_wall_turn(p: torch.Tensor, vx: torch.Tensor, s: torch.Tensor, L: float) -> torch.Tensor:
    """Pure (B,) bool: the board centre is past the far flat end (pipe x * s > L / 2) and no longer moving along the
    travel sign s (world/pipe x velocity * s <= 0): it reached the far wall and turned back. p = pipe_xyz (B, 3), vx =
    board world x velocity (B,), s = +1 / -1 (p17_floor_ref's travel sign). Shared by p17_fail and chain_play."""
    return (p[:, 0] * s > 0.5 * L) & (vx * s <= 0.0)
