# micro-skater: part of the author's skateboard add-on for microduck_rl. Apache-2.0 (see LICENSE).
"""Approach + half-pipe SPECIALIST MDP terms (2026-11-20). Used by microduck_skate_seq_ramp_env_cfg.py.

Kept out of skate_sequence.py (shared with
the chain / push / manual tasks) on purpose; everything here builds on its scheduler state (sq._st / sq.seq_update).

Balance reference in RAMP (spec finding 1): trunk_upright / riding_tilt / fell_over measured tilt from world vertical,
so a rider square to the deck on an 18-29 deg wall scored 0.35-0.07 of upright and climbing cost ~0.6-0.9 /s. Here the
reference is up_eff = normalize(ez - (ez . ex_deck) ex_deck) (ex_deck = deck long axis): the deck normal on a straight
climb, world up on the flat, and a sideways slope (deck roll) is still charged. Outside RAMP (ramp_deck_ref=True) the
reference stays world up: a MANUAL / set-down pitch must not move it.

State added lazily to sq._st(env) (reset per env by reset_ramp_state, which must run AFTER reset_sequence):
  apex_n  alternating-wall apexes this episode (incremented by ramp_apex here; read by ramp_done)
  e_prev  ramp_energy potential, sentinel -1 = not started (it starts at the first RAMP step)

Costs are returned >= 0 (use NEGATIVE weights); per-event terms return 1/dt (mjlab multiplies every reward by dt).

r2 (2026-11-20, diagnosis). Everything here, nothing in skate_sequence.py:
  - apex_update: LATCHED pass detector. sq.ramp_apex needed vz_prev > 0.02 then vz <= 0 on consecutive 20 ms steps; near
    an apex vz changes only 0.005-0.02 per step, so 85-90 % of real apexes were missed. Now: on a wall (|x| > L/2 inside
    the pipe) track the max rise since the wall entry; a pass fires once the rise has dropped APEX_DROP below that max.
    Re-armed only back on the inner flat (|x| < APEX_REARM_X) or outside the pipe: alternating walls and same-wall
    returns (a turn-around on the flat) both count, a wobble on one wall does not.
  - ramp_carve: a pure-pursuit carve command in RAMP (observed kappa, seq_trick_obs below; paid by ramp_track), staged
    by carve_curriculum: stage 1 steers the crossing line on the flat only (toward the wall the board heads to), stage
    2 also asks for the turn back on the wall.
  - past_pipe_end split: the ENTRY end is a failure (fall-priced by seq_fall), the far end stays a time-out.
"""

from __future__ import annotations
from micro_skater.paths import data_path as _data_path

import math

import torch

from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, wrap_to_pi

from micro_skater.tasks import skate_sequence as sq
from micro_skater.tasks import skate_trick as tk
from micro_skater.tasks.mdp import _yaw_from_quat

G = 9.81
APEX_DROP = 0.003          # m below the max rise of this wall visit = past the apex (deck z jitter ~1 mm)
APEX_REARM_X = 0.30        # m: |x| below this (inner flat) re-arms the pass detector


def _rst(env) -> dict:
    """sq._st(env) with the ramp specialist's keys (added lazily, once)."""
    st = sq._st(env)
    if "apex_n" not in st:
        n, dev = env.num_envs, env.device
        st["apex_n"] = torch.zeros(n, device=dev)
        st["e_prev"] = torch.full((n,), -1.0, device=dev)
    if "ap_zmax" not in st:                                   # r2: latched pass detector + RAMP carve command
        n, dev = env.num_envs, env.device
        b = lambda v: torch.full((n,), v, dtype=torch.bool, device=dev)
        st.update(ap_zmax=torch.zeros(n, device=dev), ap_armed=b(False), ap_init=b(True),
                  ap_last=torch.zeros(n, device=dev), ap_rise=torch.zeros(n, device=dev), ap_step=-1,
                  rk_side=torch.zeros(n, device=dev), rk_on=b(False), rk_step=-1)
    if "ap_prev" not in st:                                   # r3: previous pass height (ramp_keep)
        n, dev = env.num_envs, env.device
        st.update(ap_prev=torch.zeros(n, device=dev), ap_ratio=torch.zeros(n, device=dev))
    return st


def reset_ramp_state(env, env_ids: torch.Tensor) -> None:
    """Reset event (after reset_sequence): apex counter 0, energy potential not started, pass detector re-initialised
    from the first post-reset state, carve target side cleared."""
    st = _rst(env)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = env_ids.to(env.device).long()
    st["apex_n"][env_ids] = 0.0
    st["e_prev"][env_ids] = -1.0
    st["ap_zmax"][env_ids] = 0.0
    st["ap_armed"][env_ids] = False
    st["ap_init"][env_ids] = True
    st["ap_last"][env_ids] = 0.0
    st["ap_rise"][env_ids] = 0.0
    st["rk_side"][env_ids] = 0.0
    st["rk_on"][env_ids] = False
    st["ap_prev"][env_ids] = 0.0
    st["ap_ratio"][env_ids] = 0.0


# ── balance reference ─────────────────────────────────────────────────────────────────────────────────────────────────


def _up_eff(env, ramp_deck_ref: bool = True) -> torch.Tensor:
    """(B, 3) world-frame balance reference: up_eff in RAMP (ramp_deck_ref), world up elsewhere."""
    n, dev = env.num_envs, env.device
    ez = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3)
    if not ramp_deck_ref:
        return ez
    ph = sq.seq_update(env)["trick"]
    q = tk._board(env).data.root_link_quat_w
    ex = quat_apply(q, torch.tensor([1.0, 0.0, 0.0], device=dev).expand(n, 3))
    u = ez - ex[:, 2:3] * ex
    nrm = u.norm(dim=1, keepdim=True)
    ok = (nrm > 1e-3) & torch.isfinite(nrm)
    u = torch.where(ok, u / nrm.clamp(min=1e-3), ez)
    return torch.where((ph == sq.RAMP).unsqueeze(1), u, ez)


def _up_body(env, ramp_deck_ref: bool = True) -> torch.Tensor:
    """(B, 3) the balance reference in the trunk frame (= -projected_gravity_b off the ramp)."""
    return quat_apply_inverse(sq._robot(env).data.root_link_quat_w, _up_eff(env, ramp_deck_ref))


def trunk_tilt(env, ramp_deck_ref: bool = True) -> torch.Tensor:
    """(B,) trunk tilt (rad) from the balance reference."""
    u = _up_body(env, ramp_deck_ref)
    return torch.nan_to_num(torch.acos(u[:, 2].clamp(-1.0, 1.0)), nan=0.0)


def seq_upright(env, std: float = 0.3, ramp_deck_ref: bool = True) -> torch.Tensor:
    """tk.trunk_upright (exp(-(g_x^2 + g_y^2) / std^2)) against the balance reference."""
    u = _up_body(env, ramp_deck_ref)
    return torch.nan_to_num(torch.exp(-(u[:, 0] ** 2 + u[:, 1] ** 2) / std ** 2))


def seq_fell_over(env, limit_angle: float = math.radians(60.0), ramp_deck_ref: bool = True) -> torch.Tensor:
    """mjlab bad_orientation against the balance reference."""
    return trunk_tilt(env, ramp_deck_ref) > limit_angle


def riding_tilt_guard(env, limit_deg: float = 50.0, after_s: float = 1.0, ramp_deck_ref: bool = True,
                      handoff_after_s: float | None = None) -> torch.Tensor:
    """sq.riding_tilt_guard (RIDE / RAMP, after after_s) with the tilt measured from the balance reference.
    rampbank: handoff_after_s (None = off) replaces after_s for envs flagged st["handoff"] (recorded chain hand-off
    spawns: in the chain the guard is already live at the switch, t ~ 7 s)."""
    st = sq.seq_update(env)
    ph = st["trick"]
    t = tk._t(env)
    if handoff_after_s is not None and "handoff" in st:
        live = torch.where(st["handoff"], t > handoff_after_s, t > after_s)
    else:
        live = t > after_s
    riding = ((ph == sq.RIDE) | (ph == sq.RAMP)) & live
    return riding & (trunk_tilt(env, ramp_deck_ref) > math.radians(limit_deg))


def posture(env, head_std: float = 0.6, yaw_std: float = 0.6, ramp_deck_ref: bool = True) -> torch.Tensor:
    """sq.posture with the head-level reference = -up_eff in RAMP ("eyes square to the wall" on a climb; world level
    elsewhere), times the trunk facing along the deck once riding."""
    ref = sq._head_ref(env)
    robot = sq._robot(env)
    qh = robot.data.body_link_quat_w[:, ref["id"]]
    g = quat_apply_inverse(qh, -_up_eff(env, ramp_deck_ref))
    ang = torch.acos((g * ref["g"]).sum(dim=1).clamp(-1.0, 1.0))
    level = torch.exp(-(ang / head_std) ** 2)
    ph = sq.seq_update(env)["trick"]
    riding = (ph == sq.RIDE) | (ph == sq.MANUAL) | (ph == sq.RAMP)
    dyaw = wrap_to_pi(_yaw_from_quat(robot.data.root_link_quat_w) - tk._board_yaw(env))
    face = torch.where(riding, torch.exp(-(dyaw / yaw_std) ** 2), torch.ones_like(dyaw))
    return torch.nan_to_num(level * face)


# ── RAMP rewards ──────────────────────────────────────────────────────────────────────────────────────────────────────


def _rise(env) -> torch.Tensor:
    return torch.nan_to_num(sq.pipe_xyz(env)[:, 2] - sq.DECK_Z)


def apex_update(env, alternate: bool = False) -> dict:
    """r2 latched pass detector, once per env step (called by ramp_apex / ramp_done / the metrics; the only writer of
    st["apex_evt"] in this task and in the chain cfg, so chain_play's passes and the seq_apex metric use it too).

    on_wall = |x| > L/2 inside the pipe's y range. While armed and on a wall, zmax = the max deck rise of this wall
    visit; a pass fires in RAMP when rise < zmax - APEX_DROP with zmax > seq_cfg apex_min_z, soles on the deck, inside
    in_pipe. A fire disarms; the detector re-arms on the inner flat (|x| < APEX_REARM_X) or outside the pipe (approach
    floor). First step after a reset: armed unless on a wall rolling down (a wall spawn's own height is never paid).
    alternate=True: a pass must also be on the other wall than the previous one (the r1 rule)."""
    st = _rst(env)
    if st["ap_step"] == env.common_step_counter:
        return st
    st["ap_step"] = env.common_step_counter
    sq.seq_update(env)
    cfg, pipe = st["seq_cfg"], st["pipe"]
    p = torch.nan_to_num(sq.pipe_xyz(env))
    rise = (p[:, 2] - sq.DECK_Z).clamp(min=0.0)
    vz = torch.nan_to_num(tk._board(env).data.root_link_lin_vel_w[:, 2])
    ax = p[:, 0].abs()
    inp = sq.in_pipe(env, 0.0)
    on_wall = (ax > pipe["L"] / 2) & inp
    armed = torch.where(st["ap_init"], ~on_wall | (vz > 0.0), st["ap_armed"])
    armed = armed | (ax < APEX_REARM_X) | ~inp
    zmax = torch.where(on_wall & armed, torch.maximum(st["ap_zmax"], rise), torch.zeros_like(rise))
    fire = ((st["stage"] == sq.ST_RAMP) & armed & on_wall & (zmax - rise > APEX_DROP)
            & (zmax > cfg["apex_min_z"]))
    side = torch.sign(p[:, 0])
    new = fire & sq.in_pipe(env) & tk.soles_on_deck(env)
    if alternate:
        new = new & (side != st["ap_last"])
    st["apex_evt"] = new
    st["ap_rise"] = torch.where(new, zmax, torch.zeros_like(zmax))
    # r3: height of this pass / the previous pass (0 for the first pass of the episode); read by ramp_keep
    st["ap_ratio"] = torch.where(new & (st["ap_prev"] > 0), zmax / st["ap_prev"].clamp(min=1e-4), torch.zeros_like(zmax))
    st["ap_pprev"] = torch.where(new, st["ap_prev"], torch.zeros_like(zmax))   # r3rev: previous pass height at the event
    st["ap_prev"] = torch.where(new, zmax, st["ap_prev"])
    st["ap_last"] = torch.where(new, side, st["ap_last"])
    st["apex_side"] = torch.where(new, side, st["apex_side"])
    st["apex_n"] = st["apex_n"] + new.float()
    st["ap_armed"] = armed & ~fire
    st["ap_zmax"] = torch.where(fire, torch.zeros_like(zmax), zmax)
    st["ap_init"] = torch.zeros_like(st["ap_init"])
    return st


def ramp_apex(env, z_cap: float = 0.04, alternate: bool = False) -> torch.Tensor:
    """Event: a pass (apex_update) pays min(apex rise, z_cap) / z_cap."""
    st = apex_update(env, alternate)
    return st["apex_evt"].float() * (st["ap_rise"].clamp(max=z_cap) / z_cap) / env.step_dt


def ramp_keep(env, floor: float = 0.2, cap: float = 0.8, z_ref: float = 0.02, den_cap: float = 0.04) -> torch.Tensor:
    """r3 event: pass n >= 2 pays  (clip(rise_n / rise_{n-1}, floor, cap) - floor) / (cap - floor) * min(rise_n / z_ref, 1)
    = the share of the previous pass height kept, priced only for passes that are still demo-visible (z_ref).

    r3: a stiff rider straight across keeps 0.62-0.67 of the
    pass height per wall visit (wheel rolling DR 0.65; 0.75 at 0.3, 0.5 at 1.0); a timed 2-3 cm crouch/extend pump adds
    only +0.01-0.03, so cap 0.8 is above anything reachable. The r2 policy keeps 0.29 (5.2 -> 1.5 cm at
    1.0 m/s straight across): the gap is the rider (absorbing legs / braking line), which is what this pays. floor 0.2:
    the current policy still scores (~0.15 of full pay) so the gradient is visible. z_ref: chains of tiny equal passes
    (energy carved in from the along-pipe speed) are not a retention jackpot. First pass: nothing here (ramp_apex).

    r3rev: the denominator is min(rise_{n-1}, den_cap) (den_cap = ramp_apex z_cap). r2 model_900's 2nd pass is ~1.2-2 cm
    whatever the 1st (5.1 -> 1.2, 7.4 -> 2.1), so with the raw ratio every cm of 1st pass above the
    apex cap (unpaid there) cut the pass-2 keep pay: a gradient toward a LOWER first pass. Height above the cap is now
    neutral here, and a stiff rider (0.65) from >= 5 cm scores full pass-2 keep."""
    st = apex_update(env)
    ratio = st["ap_rise"] / st["ap_pprev"].clamp(max=den_cap).clamp(min=1e-4)
    r = (ratio.clamp(min=floor, max=cap) - floor) / (cap - floor)
    vis = (st["ap_rise"] / z_ref).clamp(max=1.0)
    return (st["apex_evt"] & (st["ap_pprev"] > 0)).float() * r * vis / env.step_dt


# ── RAMP carve command (r2) ───────────────────────────────────────────────────────────────────────────────────────────
# Pure pursuit to (s * tx, y + lead), s = the wall it is sent to; s flips to the other wall once x * s > flip_x. Active
# (non-zero kappa, ramp_track pays) only rolling forward; wall=False: only on the flat (|x| <= L/2); toward=True: only
# while heading toward wall s (never "turn away from the wall you are climbing"). Off (kappa 0 in RAMP, the r1
# behaviour) until carve_curriculum sets st["rk_cfg"].
CARVE_FLAT = dict(kmax=0.5, tx=0.8, lead=0.6, flip_x=0.40, wall=False, toward=True)    # stage 1: line on the flat
CARVE_WALL = dict(kmax=0.5, tx=0.8, lead=0.6, flip_x=0.58, wall=True, toward=False)    # stage 2: + turn on the wall


def carve_curriculum(env, env_ids, stages: list[dict]) -> torch.Tensor:
    """stages = [{"step": env steps, "cfg": dict | None}]: the latest stage reached sets st["rk_cfg"] (None = off).
    Logged value: the stage index."""
    del env_ids
    st = _rst(env)
    cur, idx = stages[0]["cfg"], 0
    for i, s in enumerate(stages):
        if env.common_step_counter >= s["step"]:
            cur, idx = s["cfg"], i
    st["rk_cfg"] = dict(cur) if cur else None
    return torch.tensor([float(idx)])


def ramp_carve(env) -> dict:
    """Once per env step, AFTER sq.seq_update (whose first call each step resets kappa to the EXIT pursuit / 0): the RAMP
    carve kappa into st["kappa"] (what seq_trick_obs observes and ride_wz_target tracks), st["rk_on"] = active."""
    st = _rst(env)
    sq.seq_update(env)
    if st["rk_step"] == env.common_step_counter:
        return st
    st["rk_step"] = env.common_step_counter
    c = st.get("rk_cfg")
    ramp = st["stage"] == sq.ST_RAMP
    if not c:
        st["rk_on"] = torch.zeros_like(ramp)
        return st
    p = torch.nan_to_num(sq.pipe_xyz(env))
    yaw = tk._board_yaw(env)
    hx = torch.cos(yaw)
    s = st["rk_side"]
    s = torch.where(s == 0, torch.where(hx >= 0, 1.0, -1.0), s)
    s = torch.where(p[:, 0] * s > c["flip_x"], -s, s)
    s = torch.where(ramp, s, torch.zeros_like(s))
    st["rk_side"] = s
    dx = s * c["tx"] - p[:, 0]
    dy = torch.full_like(dx, float(c["lead"]))
    e = wrap_to_pi(torch.atan2(dy, dx) - yaw)
    k = (2.0 * torch.sin(e) / torch.sqrt(dx * dx + dy * dy).clamp(min=0.5)).clamp(-c["kmax"], c["kmax"])
    on = ramp & (sq.board_v_fwd(env) > 0.05)
    if not c.get("wall", False):
        on = on & (p[:, 0].abs() <= st["pipe"]["L"] / 2)
    if c.get("toward", False):
        on = on & (hx * s > 0.0)
    st["rk_on"] = on
    st["kappa"] = torch.where(ramp, torch.where(on, torch.nan_to_num(k), torch.zeros_like(k)), st["kappa"])
    return st


def seq_trick_obs(env) -> torch.Tensor:
    """sq.seq_trick_obs with the RAMP carve kappa (ramp_carve) in the kappa slot."""
    ramp_carve(env)
    return sq.seq_trick_obs(env)


def ramp_track(env, std: float = 0.12, v_full: float = 0.15) -> torch.Tensor:
    """RAMP: tk.ride_track (low-passed board yaw rate vs kappa * |v|, speed-gated) while the carve command is active."""
    st = ramp_carve(env)
    tk.trick_update(env)
    err = st["wz_lp"] - tk.ride_wz_target(env)
    v = tk.board_v_deck(env)[:, 0].abs()
    gate = ((v - 0.03) / (v_full - 0.03)).clamp(0.0, 1.0)
    return torch.nan_to_num(st["rk_on"].float() * gate * torch.exp(-(err ** 2) / std ** 2))


def ramp_park(env, v_min: float = 0.15, rise_max: float = 0.01, after_s: float = 1.5) -> torch.Tensor:
    """Cost: in RAMP for more than after_s, rolling slower than v_min across the pipe (world x) on the bottom (rise <
    rise_max): parked on / rolling along the flat bottom."""
    st = sq.seq_update(env)
    vx = torch.nan_to_num(tk._board(env).data.root_link_lin_vel_w[:, 0]).abs()
    m = (st["stage"] == sq.ST_RAMP) & (st["stage_t"] > after_s) & (vx < v_min) & (_rise(env) < rise_max)
    return m.float()


def exit_time(env) -> torch.Tensor:
    """Cost per second in the EXIT stage (review rrev). Measured on the warm start at the final weights
    (diag: ramp_battery, exit spawns): EXIT pays ~4.2 /s alive (ride 1.9 + ride_wheels 0.9 + upright / on_deck /
    posture) against ~2.3 /s in RAMP (no ride terms there), so delaying the RAMP entry (entering slower, keeping
    wheels_on < wheels_down_s) paid ~2 /s. This prices the EXIT stage down to the RAMP rate."""
    return (sq.seq_update(env)["stage"] == sq.ST_EXIT).float()


def ramp_energy(env) -> torch.Tensor:
    """RAMP: signed potential on the board's energy per mass e = v^2 / 2 + g * rise (rise = deck height above its rest
    height; v = full board speed), paid as de/dt from the first RAMP step (e_prev sentinel -1 -> init, pay 0). The sum
    telescopes to e(now) - e(RAMP entry): pumping (energy the rider adds) pays, friction / bad lines cost, parking
    gains nothing. Not charged at a termination (the fall cost prices that)."""
    st = _rst(env)
    sq.seq_update(env)
    ramp = st["stage"] == sq.ST_RAMP
    v = torch.nan_to_num(tk._board(env).data.root_link_lin_vel_w).norm(dim=1)
    e = 0.5 * v * v + G * _rise(env)
    init = ramp & (st["e_prev"] < -0.5)
    pay = torch.where(ramp & ~init, e - st["e_prev"], torch.zeros_like(e))
    st["e_prev"] = torch.where(ramp, e, st["e_prev"])
    return torch.nan_to_num(pay) / env.step_dt


# ── terminations / metrics ────────────────────────────────────────────────────────────────────────────────────────────


def ramp_done(env, ramp_s: float = 8.0, max_apex: int = 5) -> torch.Tensor:
    """Success (time_out=True): ramp_s in RAMP (scheduler stage clock) or max_apex passes (apex_update)."""
    st = apex_update(env)
    return (st["stage"] == sq.ST_RAMP) & ((st["stage_t"] >= ramp_s) | (st["apex_n"] >= max_apex))


def past_pipe_end_entry(env) -> torch.Tensor:
    """r2 failure (NOT a time-out; seq_fall prices it as a fall): in RAMP and rolled back out of the open end it came
    in by (y < -W/2 + pipe_end_margin). r1: 90-100 % of the exit-spawn endings were this fakie roll-back, for free."""
    st = sq.seq_update(env)
    p = sq.pipe_xyz(env)
    return (st["trick"] == sq.RAMP) & (p[:, 1] < -st["pipe"]["W"] / 2 + st["seq_cfg"]["pipe_end_margin"])


def past_pipe_end_far(env) -> torch.Tensor:
    """The far end (y > W/2 - pipe_end_margin) in RAMP: a time-out as before (riding the pipe's length is not a fault)."""
    st = sq.seq_update(env)
    p = sq.pipe_xyz(env)
    return (st["trick"] == sq.RAMP) & (p[:, 1] > st["pipe"]["W"] / 2 - st["seq_cfg"]["pipe_end_margin"])


def m_rise(env) -> torch.Tensor:
    """Deck rise (m) while in RAMP: m_rise / seq_ramp = mean rise."""
    return sq._phase(env, sq.RAMP) * _rise(env).clamp(min=0.0)


def m_pass_rise(env) -> torch.Tensor:
    """Rise (m) summed at the passes: m_pass_rise / seq_apex = mean pass height."""
    st = apex_update(env)
    return st["apex_evt"].float() * st["ap_rise"]


def m_carve_on(env) -> torch.Tensor:
    """Carve command active (RAMP): m_carve_on / seq_ramp = share of RAMP time with a carve command."""
    return ramp_carve(env)["rk_on"].float()


def m_keep(env) -> torch.Tensor:
    """r3: pass-height ratio (this / previous) summed at passes n >= 2: m_keep / m_pass2 = mean retention."""
    st = apex_update(env)
    return (st["apex_evt"] & (st["ap_ratio"] > 0)).float() * st["ap_ratio"]


def m_pass2(env) -> torch.Tensor:
    """r3: passes n >= 2 (those with a previous pass this episode)."""
    st = apex_update(env)
    return (st["apex_evt"] & (st["ap_ratio"] > 0)).float()


# ── pump (2026-11-20, default off: MICRODUCK_RAMP_PUMP=1 in microduck_skate_seq_ramp_env_cfg.py) ──────────────────────
# Pumping = leg work against the EXCESS
# normal load of the transition. The rider's legs add  m * N_eff * dh  to the board+rider energy (dh = CoM height above
# the deck along its normal, N_eff = specific normal load); the  g * dh  part telescopes (squat / stand on the flat nets
# 0), what is left is the pumping work
#     W = sum (N_eff - g) * dh,   N_eff = g n_z + v_perp^2 / R on the wall (geometric: no measured accelerations)
# (n_z = the pipe SURFACE normal's world z at the board's x, v_perp^2 = board vx^2 + vz^2 in the pipe's cross plane, R = transition radius).
# Positive for extending in the transition at speed (N_eff > g) and crouching near the apex (N_eff < g: n_z < 1, v ~ 0);
# exactly 0 on the flat (squatting at the bottom pays nothing), a squat held still on the wall telescopes, and a
# self-excited bounce is not counted (N_eff is geometric). Per step dh is clipped to PUMP_DH_MAX and the load excess to
# [-g, 2g]; the chain restarts whenever the gate (RAMP, soles on the deck, inside the pipe) closes.
PUMP_DH_MAX = 0.01            # m per 20 ms env step (0.5 m/s CoM speed along the normal, above the servo-limited rate)


def _pst(env) -> dict:
    st = sq._st(env)
    if "pm_h" not in st:
        n, dev = env.num_envs, env.device
        st.update(pm_h=torch.full((n,), float("nan"), device=dev), pm_pay=torch.zeros(n, device=dev),
                  pm_w=torch.zeros(n, device=dev), pm_step=-1)
    return st


def reset_pump_state(env, env_ids: torch.Tensor) -> None:
    """Reset event (pump on): CoM-height chain broken, episode pump work 0."""
    st = _pst(env)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = env_ids.to(env.device).long()
    st["pm_h"][env_ids] = float("nan")
    st["pm_w"][env_ids] = 0.0


def pump_com_h(env) -> tuple[torch.Tensor, torch.Tensor]:
    """(B,) whole-robot CoM height above the board origin along the deck normal (m), (B, 3) the deck normal."""
    robot = sq._robot(env)
    com = robot.data.data.subtree_com[:, robot.data.indexing.root_body_id]
    b = tk._board(env).data
    nz = quat_apply(b.root_link_quat_w, torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(env.num_envs, 3))
    return ((com - b.root_link_pos_w) * nz).sum(dim=1), nz


def pump_update(env, dh_max: float = PUMP_DH_MAX) -> dict:
    """Once per env step: st["pm_pay"] = this step's pumping work (J/kg), st["pm_w"] = episode sum."""
    st = _pst(env)
    if st["pm_step"] == env.common_step_counter:
        return st
    st["pm_step"] = env.common_step_counter
    sq.seq_update(env)
    pipe = st["pipe"]
    h, nz = pump_com_h(env)
    p = torch.nan_to_num(sq.pipe_xyz(env))
    bv = torch.nan_to_num(tk._board(env).data.root_link_lin_vel_w)
    # surface (not deck) normal for the g term: rolling the deck on its trucks must not modulate the load (farm)
    _, slope = sq.pipe_profile(p[:, 0], pipe)
    # r4rev: centripetal term only on the transition arcs (slope 0 on the top platform beyond x_end)
    wall = (p[:, 0].abs() > pipe["L"] / 2) & (slope.abs() > 1e-6)
    vp2 = bv[:, 0] ** 2 + bv[:, 2] ** 2
    neff = G / torch.sqrt(1.0 + slope * slope) + torch.where(wall, vp2 / pipe["R"], torch.zeros_like(vp2))
    # r4rev: the CoM chain runs through sole lift-offs (RAMP, inside the pipe); the pay gate needs both soles on the deck,
    # and a CoM move with a sole off is CHARGED -|N_eff - g| |dh| (else: squat on both soles where paid, stand up on one
    # leg unpaid, repeat = free work on any wall visit; with the old chain break at lift-off this was a farm)
    chain = (st["stage"] == sq.ST_RAMP) & sq.in_pipe(env, 0.0) & torch.isfinite(h)
    gate = chain & tk.soles_on_deck(env)
    dh = torch.nan_to_num((h - st["pm_h"]).clamp(-dh_max, dh_max))
    ok = chain & torch.isfinite(st["pm_h"])
    exc = (neff - G).clamp(-G, 2.0 * G)
    pay = torch.where(ok, torch.where(gate, exc * dh, -(exc * dh).abs()), torch.zeros_like(h))
    st["pm_h"] = torch.where(chain, h, torch.full_like(h, float("nan")))
    st["pm_pay"] = pay
    st["pm_w"] = st["pm_w"] + pay
    return st


def ramp_pump(env, dh_max: float = PUMP_DH_MAX) -> torch.Tensor:
    """Pumping work (J/kg per second; mjlab multiplies by dt, so the episode sum = weight * W)."""
    return torch.nan_to_num(pump_update(env, dh_max)["pm_pay"]) / env.step_dt


def m_pump(env) -> torch.Tensor:
    """Pumping work (J/kg) summed: seq_pump / episodes = mean W per episode."""
    return pump_update(env)["pm_pay"]


# ── r5 (2026-11-20, default off: MICRODUCK_RAMP_R5=1 in microduck_skate_seq_ramp_env_cfg.py) ────────────────────────────
# Chain v1 review: the head is a counterweight at
# every apex (thrown back ~60 deg / pitched 91 deg face-down), the rider is a frozen deep crouch (trunk 5-7 cm over the
# deck vs ~9 in the manual) and never pumps (ramp_pump ~0 in r4). Costs >= 0 (NEGATIVE weights); ramp_pump_dir >= / <= 0
# per step (POSITIVE weight). All gated to the EXIT / RAMP stages (stages tuple) so a chain playback is unaffected
# elsewhere.


def _stage_in(env, stages: tuple) -> torch.Tensor:
    st = sq.seq_update(env)["stage"]
    m = torch.zeros_like(st, dtype=torch.bool)
    for s in stages:
        m = m | (st == s)
    return m


def r5_head_hold(env, free: float = 0.12, rate_free: float = 2.0, rate_w: float = 0.25, cap: float = 4.0,
                 stages: tuple = (sq.ST_EXIT, sq.ST_RAMP)) -> torch.Tensor:
    """Cost (EXIT / RAMP): the four neck / head joints away from the hand-off head pose (sq.HANDOFF_STANCE: neck 0.41,
    head pitch 0.28, head yaw -0.14, roll 0 rad - the manual specialist's own riding head, what the chain hands over)
    as L1 beyond `free` rad per joint, + rate_w x the head's WORLD angular speed beyond rate_free rad/s; capped. The
    p12_stepon_head recipe. r4 model_2098 (bank spawns, rolling 0.3): head-joint |dev| sum up to 2.1-2.3
    rad on the walls, head level angle 50 deg mean at the apex / on the way down (13 deg on the flat), head angular
    speed p95 5 rad/s. The head is ~38 % of the mass: the world-rate part prices the counterweight swing itself, the
    pose part the parked nod; neither charges the trunk following the wall (that is the joints' job, not the head's).
    r5rev: the rate is the head's angular velocity RELATIVE TO THE BOARD (world rate minus board world rate): the board
    pitches at v / R following the transition (R 0.40: 2.1-3.0 rad/s at 0.85-1.2 m/s), which the world rate charged."""
    ids, hq = sq._head_joint_ids(env)
    robot = sq._robot(env)
    dev = ((robot.data.joint_pos[:, ids] - hq).abs() - free).clamp(min=0.0).sum(dim=1)
    w = (robot.data.body_link_ang_vel_w[:, sq._head_ref(env)["id"]]
         - tk._board(env).data.root_link_ang_vel_w).norm(dim=1)
    c = (dev + rate_w * (w - rate_free).clamp(min=0.0)).clamp(max=cap)
    return torch.nan_to_num(c * _stage_in(env, stages).float())


def r5_trunk_band(env, z_lo: float = 0.080, z_hi: float = 0.105, unit: float = 0.01, cap: float = 3.0,
                  stages: tuple = (sq.ST_RAMP,), z_lo_stages: list | None = None) -> torch.Tensor:
    """Cost (RAMP): trunk height above the deck origin in the deck frame (sq._trunk_deck_z, the p13 push-band measure)
    outside [z_lo, z_hi], linear in `unit`, capped. A BAND, not a target: inside it the CoM is free to move (the pump
    needs ~2-3 cm of travel). r4 model_2098: median 7.1 cm on every phase (a frozen deep squat with no room left to
    absorb / extend); the manual rides at ~9 (HANDOFF_STANCE com 12.5 cm).
    r5rev: z_lo_stages = [(env step, z_lo), ...] (the latest reached wins; None = the fixed z_lo): the lower edge is
    raised in steps (r4's squat is part of its balance)."""
    if z_lo_stages:
        for s, z in z_lo_stages:
            if env.common_step_counter > s:
                z_lo = z
    tz = torch.nan_to_num(sq._trunk_deck_z(env), nan=z_lo)
    c = ((z_lo - tz) / unit).clamp(min=0.0) + ((tz - z_hi) / unit).clamp(min=0.0)
    return c.clamp(max=cap) * _stage_in(env, stages).float()


def _r5st(env) -> dict:
    st = sq._st(env)
    if "r5_h" not in st:
        n, dev = env.num_envs, env.device
        st.update(r5_h=torch.full((n,), float("nan"), device=dev), r5_pay=torch.zeros(n, device=dev), r5_step=-1)
    return st


def reset_r5_state(env, env_ids: torch.Tensor) -> None:
    """Reset event (r5 on): the ramp_pump_dir CoM chain is broken."""
    st = _r5st(env)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    st["r5_h"][env_ids.to(env.device).long()] = float("nan")


def pump_dir_update(env, e0: float = 1.5, dh_unit: float = 0.01, dh_max: float = PUMP_DH_MAX) -> dict:
    """Once per env step: st["r5_pay"] = f(N_eff - g) * dh / dh_unit with f = clamp((N_eff - g) / e0, -1, 1).

    The pump_update geometry (surface-normal g term, centripetal v_perp^2 / R only on the transition arcs, CoM height
    along the deck normal, dh clipped to dh_max per step) with the load excess SQUASHED to its sign beyond e0 (m/s^2):
    ramp_pump pays real work, ~0.005 J/kg per visit for the r4 rider (x weight 15 = 0.07 per visit, invisible); here
    1 cm of CoM travel in the right direction pays 1 unit wherever the load excess is >= e0 (the loaded transition
    at speed: v^2 / R = 1.8 m/s^2 at 0.85 m/s; the unloaded upper wall: g (n_z - 1) down to -2.9 m/s^2), less near
    the crossover. Not farmable: 0 on the flat (N_eff = g), a held CoM pays 0, an oscillation at a fixed load excess
    telescopes (f constant); f is monotone in the excess, so a two-level cycle pays with the sign of the true work.
    Gate and lift-off rule as pump_update: chain in RAMP inside the pipe, pay with both soles on the deck, a CoM move
    with a sole off is charged -|f dh|."""
    st = _r5st(env)
    if st["r5_step"] == env.common_step_counter:
        return st
    st["r5_step"] = env.common_step_counter
    sq.seq_update(env)
    pipe = st["pipe"]
    h, _ = pump_com_h(env)
    p = torch.nan_to_num(sq.pipe_xyz(env))
    bv = torch.nan_to_num(tk._board(env).data.root_link_lin_vel_w)
    _, slope = sq.pipe_profile(p[:, 0], pipe)
    wall = (p[:, 0].abs() > pipe["L"] / 2) & (slope.abs() > 1e-6)
    vp2 = bv[:, 0] ** 2 + bv[:, 2] ** 2
    neff = G / torch.sqrt(1.0 + slope * slope) + torch.where(wall, vp2 / pipe["R"], torch.zeros_like(vp2))
    f = ((neff - G) / e0).clamp(-1.0, 1.0)
    chain = (st["stage"] == sq.ST_RAMP) & sq.in_pipe(env, 0.0) & torch.isfinite(h)
    gate = chain & tk.soles_on_deck(env)
    # r5rev: ASYMMETRIC clip - a paid move is clipped to dh_max per step, a charged one to 3 dh_max (symmetric clipping
    # let a slow paid extension + a fast (> 0.5 m/s, gravity-driven) unpaid-in-full drop net a profit at a fixed load)
    raw = torch.nan_to_num(h - st["r5_h"])
    dh_pay = raw.clamp(-dh_max, dh_max) / dh_unit
    dh_chg = raw.clamp(-3.0 * dh_max, 3.0 * dh_max) / dh_unit
    ok = chain & torch.isfinite(st["r5_h"])
    on = torch.where(f * raw > 0, f * dh_pay, f * dh_chg)
    pay = torch.where(ok, torch.where(gate, on, -(f * dh_chg).abs()), torch.zeros_like(h))
    st["r5_h"] = torch.where(chain, h, torch.full_like(h, float("nan")))
    st["r5_pay"] = torch.nan_to_num(pay)
    return st


def ramp_pump_dir(env, e0: float = 1.5, dh_unit: float = 0.01) -> torch.Tensor:
    """Dense pump direction (units per second; episode sum = weight x cm of correctly phased CoM travel)."""
    return pump_dir_update(env, e0, dh_unit)["r5_pay"] / env.step_dt


def m_pump_dir(env) -> torch.Tensor:
    """r5: correctly phased CoM travel (cm, signed) summed: seq_pump_dir / episodes."""
    return pump_dir_update(env)["r5_pay"]


def m_head_dev(env) -> torch.Tensor:
    """r5: head-joint L1 deviation from the hand-off head pose (rad) in RAMP: m_head_dev / seq_ramp = mean."""
    ids, hq = sq._head_joint_ids(env)
    d = (sq._robot(env).data.joint_pos[:, ids] - hq).abs().sum(dim=1)
    return torch.nan_to_num(d) * sq._phase(env, sq.RAMP)


def m_trunk_z(env) -> torch.Tensor:
    """r5: trunk deck-frame height (m) in RAMP: m_trunk_z / seq_ramp = mean."""
    return torch.nan_to_num(sq._trunk_deck_z(env)) * sq._phase(env, sq.RAMP)


# ── r5pump (2026-11-21, default off: MICRODUCK_RAMP_R5PUMP=1 in microduck_skate_seq_ramp_env_cfg.py) ──────────────────
# Pay the OUTCOME of pumping, not only the motion: the energy kept from
# one wall visit to the next. At every outward crossing of the flat's edge (|x| = L/2, flat -> transition, RAMP, inside
# the pipe) e_k = |v_sys,xy|^2 / 2 of the ROBOT + BOARD centre of mass (mdp._skate_system_vel_xy: internal motions -
# a fore-aft lunge that shoves the light board - cannot change it; carving only turns it; only the legs' work against
# the transition load (pumping) adds to it, rolling / bad lines remove it). Pay at edge k (review r5pump: the ABSOLUTE
# change, not a ratio - an along-pipe speed reservoir inflated e_{k-1} and so the ratio without any pumping):
#     d_k = e_k - e_{k-1}  (J/kg; r4 / a stiff rider ~ -0.15 .. -0.18 per visit, an ideal 2 cm pump ~ +0.09 of it back)
#     q_k = 1 + d_k / (g rise)   energy kept relative to the visit's CROSS-PIPE energy (its pass height): the r4 audit
#                                showed the loss scales with the pass energy (kept 0.51-0.58 at every height), so a raw
#                                d paid low passes (v2: 7.8 / episode for r4); the along-pipe reservoir is not in g rise
#     pay = clip((q_k - q_lo) / (q_hi - q_lo), 0, 1) * min(rise / z_ref, 1)
# with rise = the highest pass (apex_update) of the visit between edges k-1 and k; nothing unless rise >= z_min (tiny
# passes never pay, and the visibility weight comes from the pass height, not from a speed that may be along the pipe).
# Energy only leaves without pumping, so a deliberately low first pass / dumped energy never raises d; a longer
# diagonal path between edges costs more rolling loss (lower d).


def _rtst(env) -> dict:
    st = sq._st(env)
    if "rt_eprev" not in st:
        n, dev = env.num_envs, env.device
        b = lambda: torch.zeros(n, dtype=torch.bool, device=dev)
        z = lambda: torch.zeros(n, device=dev)
        st.update(rt_flat=b(), rt_evt=b(), rt_eprev=torch.full((n,), float("nan"), device=dev), rt_rise=z(),
                  rt_vrise=z(), rt_d=z(), rt_q=z(), rt_step=-1)
    return st


def reset_retain_state(env, env_ids: torch.Tensor) -> None:
    """Reset event (r5pump on, after reset_ramp): no edge reference, no pass since the last edge."""
    st = _rtst(env)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    i = env_ids.to(env.device).long()
    st["rt_flat"][i] = False
    st["rt_evt"][i] = False
    st["rt_eprev"][i] = float("nan")
    st["rt_rise"][i] = 0.0


def retain_update(env, z_min: float = 0.015) -> dict:
    """Once per env step: st["rt_evt"] (a paid edge crossing), st["rt_d"] (energy change, J/kg), st["rt_vrise"] (the
    pass height of the visit that edge closes)."""
    from micro_skater.tasks import mdp as _mdp

    st = _rtst(env)
    if st["rt_step"] == env.common_step_counter:
        return st
    st["rt_step"] = env.common_step_counter
    ap = apex_update(env)
    pipe = st["pipe"]
    p = torch.nan_to_num(sq.pipe_xyz(env))
    ax = p[:, 0].abs()
    live = (st["stage"] == sq.ST_RAMP) & sq.in_pipe(env, 0.0)
    st["rt_rise"] = torch.where(ap["apex_evt"], torch.maximum(st["rt_rise"], ap["ap_rise"]), st["rt_rise"])
    edge = st["rt_flat"] & (ax > pipe["L"] / 2) & live
    v = torch.nan_to_num(_mdp._skate_system_vel_xy(env))
    e = 0.5 * (v * v).sum(dim=1)
    evt = edge & torch.isfinite(st["rt_eprev"]) & (st["rt_rise"] >= z_min)
    st["rt_evt"] = evt
    st["rt_d"] = torch.where(evt, e - torch.nan_to_num(st["rt_eprev"]), torch.zeros_like(e))
    st["rt_vrise"] = torch.where(evt, st["rt_rise"], torch.zeros_like(e))
    st["rt_q"] = torch.where(evt, 1.0 + st["rt_d"] / (G * st["rt_rise"].clamp(min=z_min)), torch.zeros_like(e))
    st["rt_eprev"] = torch.where(edge, e, st["rt_eprev"])
    st["rt_rise"] = torch.where(edge, torch.zeros_like(e), st["rt_rise"])
    st["rt_flat"] = (ax <= pipe["L"] / 2) & live
    return st


def ramp_retain(env, q_lo: float = 0.5, q_hi: float = 1.0, z_ref: float = 0.025,
                z_min: float = 0.015) -> torch.Tensor:
    """r5pump event: energy kept per wall visit (retain_update), paid at the flat's edge on the way into the next wall."""
    st = retain_update(env, z_min)
    r = ((st["rt_q"] - q_lo) / (q_hi - q_lo)).clamp(0.0, 1.0)
    vis = (st["rt_vrise"] / z_ref).clamp(max=1.0)
    return torch.nan_to_num(st["rt_evt"].float() * r * vis) / env.step_dt


def m_retain(env) -> torch.Tensor:
    """r5pump: energy kept per visit q summed at the paid edges: seq_retain / seq_retain_n = mean q."""
    st = retain_update(env)
    return st["rt_evt"].float() * st["rt_q"]


def m_retain_n(env) -> torch.Tensor:
    """r5pump: paid edge crossings."""
    return retain_update(env)["rt_evt"].float()


# ── r6 (2026-11-21, default off: MICRODUCK_RAMP_R6=1 in microduck_skate_seq_ramp_env_cfg.py) ────────────────────────────
# (audit logs). r5 model_2797 (rolling 0.3): the
# head whip is a NOD - neck_pitch / head_pitch p2p 1.9 / 1.5 rad per RAMP episode (yaw 0.8, roll 0.6), head rate vs
# the board 3.2 rad/s rms about the board's pitch axis (8.1 p95; roll 0.8, yaw 1.3); r5_head_hold's L1 + linear rate
# beyond 2 rad/s charged ~5 raw (-2.5 at -0.5) against ~66 per episode. The pump is capped by the r5 band (8.0-10.5 cm:
# <= 2.5 cm of trunk travel); the point-mass audit pays a 3-4 cm pump far above r5's 2 cm one
# with the r5 outcome terms already, so the band and the head are the limiters, not the pay.


def _neff_exc(env) -> torch.Tensor:
    """(B,) specific normal load excess N_eff - g (m/s^2): the pump_update geometry (surface-normal g term, centripetal
    v_perp^2 / R only on the transition arcs). 0 on the flat, < 0 high on the wall / slow, > 0 in the transition at speed."""
    st = sq.seq_update(env)
    pipe = st["pipe"]
    p = torch.nan_to_num(sq.pipe_xyz(env))
    bv = torch.nan_to_num(tk._board(env).data.root_link_lin_vel_w)
    _, slope = sq.pipe_profile(p[:, 0], pipe)
    wall = (p[:, 0].abs() > pipe["L"] / 2) & (slope.abs() > 1e-6)
    vp2 = bv[:, 0] ** 2 + bv[:, 2] ** 2
    neff = G / torch.sqrt(1.0 + slope * slope) + torch.where(wall, vp2 / pipe["R"], torch.zeros_like(vp2))
    return torch.nan_to_num(neff - G)


def r6_head(env, k: tuple = (1.0, 1.0, 2.0, 1.5), free: float = 0.10, rate_free: float = 1.5,
            rate_k: tuple = (1.0, 1.0, 2.0), rate_w: float = 0.04, cap: float = 3.0,
            stages: tuple = (sq.ST_EXIT, sq.ST_RAMP)) -> torch.Tensor:
    """Cost (EXIT / RAMP, >= 0, NEGATIVE weight): SQUARED head-joint deviation from the hand-off head pose
    (HANDOFF_STANCE: neck_pitch, head_pitch, head_yaw, head_roll weighted k, beyond `free` rad each) + rate_w x the
    squared head angular speed RELATIVE TO THE BOARD beyond rate_free rad/s (board-frame roll / pitch / yaw weighted
    rate_k: yaw counts double). Squared, so a parked 0.3 rad nod costs 0.04 /s and a 1 rad whip 0.81 /s plus its
    rate (8 rad/s: ~1.7 /s), where r5_head_hold (L1 + linear rate) charged 0.9 / 1.5. The board-relative rate leaves
    the trunk / head following the transition (the board pitches at v / R) free. Capped per step at 3 (review r6: at
    cap 8 one big whip could out-cost the alive pay and approach a fall within ~1.5 s)."""
    ids, hq = sq._head_joint_ids(env)
    robot = sq._robot(env)
    kk = torch.tensor(k, device=env.device)
    dev = ((((robot.data.joint_pos[:, ids] - hq).abs() - free).clamp(min=0.0)) ** 2 * kk).sum(dim=1)
    bq = tk._board(env).data.root_link_quat_w
    wr = quat_apply_inverse(bq, robot.data.body_link_ang_vel_w[:, sq._head_ref(env)["id"]]
                            - tk._board(env).data.root_link_ang_vel_w)
    rk = torch.tensor(rate_k, device=env.device)
    w = torch.sqrt((wr * wr * rk).sum(dim=1))
    c = (dev + rate_w * (w - rate_free).clamp(min=0.0) ** 2).clamp(max=cap)
    return torch.nan_to_num(c * _stage_in(env, stages).float())


def r6_trunk_band(env, z_lo: float = 0.080, z_lo_unl: float = 0.065, z_hi: float = 0.110, e0: float = 0.75,
                  unit: float = 0.01, cap: float = 3.0, stages: tuple = (sq.ST_RAMP,)) -> torch.Tensor:
    """Cost (RAMP): r5_trunk_band with a LOAD-DEPENDENT lower edge: z_lo where the deck load is >= g (flat, loaded
    transition), down to z_lo_unl where N_eff - g <= -e0 (high on the wall / slow: the pump's crouch), linear between
    (review r6: e0 1.5 -> 0.75; N_eff - g ~ (v^2 - g rise) / R, so -1.5 needed a >= 6 cm apex and the crouch room
    never opened at the 1.5-5 cm passes the policy makes; -0.75 = a 3 cm apex at rest);
    upper edge z_hi everywhere (the extension). Room for ~4.5 cm of trunk travel in phase, where r5's 8.0-10.5 band
    capped it at 2.5. The deep crouch is only allowed where trunk_low (root world z < 8.3 cm, i.e. ~6.1 cm over a
    deck on the flat) cannot fire: unloaded = on the wall (rise > 0)."""
    zl = z_lo - (z_lo - z_lo_unl) * (-_neff_exc(env) / e0).clamp(0.0, 1.0)
    tz = torch.nan_to_num(sq._trunk_deck_z(env), nan=z_lo)
    c = ((zl - tz) / unit).clamp(min=0.0) + ((tz - z_hi) / unit).clamp(min=0.0)
    return torch.nan_to_num(c.clamp(max=cap) * _stage_in(env, stages).float())


# r6 pass-height maintenance: pay pass k >= 2 for keeping its height relative to an ANCHOR fixed per episode
#     anchor = c_anchor x e_1 / g        (review r6: max(r_1, .) dropped - pumping pass 1 raised the anchor and so
#                                         taxed every later pass; the entry energy alone cannot be lowered by a low pass)
# r_1 = the first pass after the first outward flat-edge crossing in RAMP (inside the pipe), e_1 = the robot + board
# system |v_xy|^2 / 2 at that crossing (mdp._skate_system_vel_xy, all of it: the along-pipe share too). A deliberately
# low first pass (a diagonal line that parks energy along the pipe, a braked visit) cannot lower the anchor below
# c_anchor x e_1 / g (a straight unpumped crossing climbs ~0.75-0.85 e_1 / g at 0.9-1.1 m/s), and energy only leaves
# without pumping. pay_k = clip((r_k / anchor - lo) / (hi - lo), 0, 1) x min(r_k / z_ref, 1), r_k >= z_min.


def _r6st(env) -> dict:
    st = sq._st(env)
    if "r6_e1" not in st:
        n, dev = env.num_envs, env.device
        st.update(r6_e1=torch.full((n,), float("nan"), device=dev), r6_r1=torch.zeros(n, device=dev),
                  r6_flat=torch.zeros(n, dtype=torch.bool, device=dev), r6_pay=torch.zeros(n, device=dev),
                  r6_ratio=torch.zeros(n, device=dev), r6_evt=torch.zeros(n, dtype=torch.bool, device=dev), r6_step=-1)
    return st


def reset_r6_state(env, env_ids: torch.Tensor) -> None:
    """Reset event (r6 on, after reset_ramp): no anchor, no first pass."""
    st = _r6st(env)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    i = env_ids.to(env.device).long()
    st["r6_e1"][i] = float("nan")
    st["r6_r1"][i] = 0.0
    st["r6_flat"][i] = False


def r6_keep_update(env, c_anchor: float = 0.8, lo: float = 0.5, hi: float = 0.8, z_ref: float = 0.025,
                   z_min: float = 0.015) -> dict:
    """Once per env step: st["r6_pay"] (0..1 at a paid pass k >= 2), st["r6_ratio"] (r_k / anchor at it). Cached per
    step: the FIRST caller's params win (r6_keep from the reward manager; the metrics use the defaults - keep the cfg
    on the defaults or pass the same params to both)."""
    from micro_skater.tasks import mdp as _mdp

    st = _r6st(env)
    if st["r6_step"] == env.common_step_counter:
        return st
    st["r6_step"] = env.common_step_counter
    ap = apex_update(env)
    pipe = st["pipe"]
    ax = torch.nan_to_num(sq.pipe_xyz(env))[:, 0].abs()
    live = (st["stage"] == sq.ST_RAMP) & sq.in_pipe(env, 0.0)
    edge = st["r6_flat"] & (ax > pipe["L"] / 2) & live
    v = torch.nan_to_num(_mdp._skate_system_vel_xy(env))
    e = 0.5 * (v * v).sum(dim=1)
    st["r6_e1"] = torch.where(edge & ~torch.isfinite(st["r6_e1"]), e, st["r6_e1"])
    evt = ap["apex_evt"] & live & torch.isfinite(st["r6_e1"])
    rise = ap["ap_rise"]
    first = evt & (st["r6_r1"] <= 0.0)
    later = evt & (st["r6_r1"] > 0.0)
    anchor = (c_anchor * torch.nan_to_num(st["r6_e1"]) / G).clamp(min=z_min)
    ratio = rise / anchor
    pay = ((ratio - lo) / (hi - lo)).clamp(0.0, 1.0) * (rise / z_ref).clamp(max=1.0) * (rise >= z_min).float()
    st["r6_evt"] = later
    st["r6_pay"] = torch.where(later, pay, torch.zeros_like(pay))
    st["r6_ratio"] = torch.where(later, ratio, torch.zeros_like(ratio))
    st["r6_r1"] = torch.where(first, rise.clamp(min=1e-4), st["r6_r1"])
    st["r6_flat"] = (ax <= pipe["L"] / 2) & live
    return st


def r6_keep(env, c_anchor: float = 0.8, lo: float = 0.5, hi: float = 0.8, z_ref: float = 0.025,
            z_min: float = 0.015) -> torch.Tensor:
    """r6 event: pass-height maintenance vs the episode's anchor (r6_keep_update), paid at the pass."""
    st = r6_keep_update(env, c_anchor, lo, hi, z_ref, z_min)
    return torch.nan_to_num(st["r6_pay"]) / env.step_dt


def m_r6_keep(env) -> torch.Tensor:
    """r6: r_k / anchor summed at passes k >= 2: seq_r6_ratio / seq_r6_n = mean height kept vs the anchor."""
    return r6_keep_update(env)["r6_ratio"]


def m_r6_n(env) -> torch.Tensor:
    """r6: passes k >= 2 with an anchor."""
    return r6_keep_update(env)["r6_evt"].float()


def m_head_rate(env) -> torch.Tensor:
    """r6: head angular speed relative to the board (rad/s) in RAMP: m_head_rate / seq_ramp = mean."""
    w = (sq._robot(env).data.body_link_ang_vel_w[:, sq._head_ref(env)["id"]]
         - tk._board(env).data.root_link_ang_vel_w).norm(dim=1)
    return torch.nan_to_num(w) * sq._phase(env, sq.RAMP)


# ── r7 (2026-11-21, default off: MICRODUCK_RAMP_R7=1 in microduck_skate_seq_ramp_env_cfg.py) ────────────────────────────
# (audit logs). r6 model_3596: the "pump" is an apex FOLD of the
# neck / head (~38 % of the mass) - face to the deck, whip back up. ramp_pump / ramp_pump_dir measured the WHOLE-robot
# CoM, so a neck fold moved the paid CoM without any leg work. r7:
#   r7_pump / r7_pump_dir : the pump_update / pump_dir_update pay with h = the CoM of the robot EXCLUDING the neck_pitch
#                           subtree (neck + head bodies) along the deck normal. With the soles on the deck this height is
#                           a function of the leg (and hip) joints only: a neck / head motion cannot move it.
#   r7_head               : head-steadiness cost: head off level vs WORLD up beyond a free band + neck_pitch / head_pitch
#                           away from the hand-off head pose beyond a free band, squared, capped.
# ramp_retain / r6_keep / ramp_apex stay outcome terms (system momentum / pass height): a head fold only moves them by
# REAL pumping work, which r7_head prices.


def _r7_bodies(env) -> dict:
    """Cached: robot body masses (compile time, as mdp._skate_body_masses) and the neck_pitch-subtree mask."""
    c = tk._tc(env)
    if "r7_bodies" not in c:
        import mujoco
        robot = sq._robot(env)
        m = env.sim.mj_model
        gids = robot.indexing.body_ids.cpu().numpy().tolist()
        jid = next(i for i in (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm) for nm in ("robot/neck_pitch", "neck_pitch"))
                   if i >= 0)
        top = int(m.jnt_bodyid[jid])

        def under(b: int) -> bool:
            while b > 0:
                if b == top:
                    return True
                b = int(m.body_parentid[b])
            return False

        head = torch.tensor([under(int(g)) for g in gids], device=env.device)
        mass = torch.tensor(m.body_mass[gids], device=env.device, dtype=torch.float)
        assert head.any() and (~head).any(), "r7: neck subtree not resolved"
        c["r7_bodies"] = dict(mass=mass, head=head, m_head=float(mass[head].sum()), m_all=float(mass.sum()))
    return c["r7_bodies"]


def r7_com_split(env) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(h_all, h_body, h_head): CoM heights above the board origin along the deck normal (m) of the whole robot, the
    robot without the neck_pitch subtree, and the neck + head subtree. h_all = (m_b h_body + m_h h_head) / M."""
    rb = _r7_bodies(env)
    robot = sq._robot(env)
    b = tk._board(env).data
    nz = quat_apply(b.root_link_quat_w, torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(env.num_envs, 3))
    h = ((robot.data.body_com_pos_w - b.root_link_pos_w.unsqueeze(1)) * nz.unsqueeze(1)).sum(dim=2)   # (B, nb)
    wb = rb["mass"] * (~rb["head"]).float()
    wh = rb["mass"] * rb["head"].float()
    hb = (h * wb).sum(dim=1) / wb.sum()
    hh = (h * wh).sum(dim=1) / wh.sum()
    ha = (h * rb["mass"]).sum(dim=1) / rb["mass"].sum()
    return ha, hb, hh


def _r7_load(env) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(h_body, N_eff - g, chain, gate): the pump_update geometry with the below-neck CoM."""
    st = sq.seq_update(env)
    _, hb, _ = r7_com_split(env)
    exc = _neff_exc(env)
    chain = (st["stage"] == sq.ST_RAMP) & sq.in_pipe(env, 0.0) & torch.isfinite(hb)
    gate = chain & tk.soles_on_deck(env)
    return hb, exc, chain, gate


def _r7st(env) -> dict:
    st = sq._st(env)
    if "r7_hp" not in st:
        n, dev = env.num_envs, env.device
        st.update(r7_hp=torch.full((n,), float("nan"), device=dev), r7_hd=torch.full((n,), float("nan"), device=dev),
                  r7_ppay=torch.zeros(n, device=dev), r7_dpay=torch.zeros(n, device=dev), r7_step=-1)
    return st


def reset_r7_state(env, env_ids: torch.Tensor) -> None:
    """Reset event (r7 on): both below-neck CoM chains broken."""
    st = _r7st(env)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    i = env_ids.to(env.device).long()
    st["r7_hp"][i] = float("nan")
    st["r7_hd"][i] = float("nan")


def r7_update(env, e0: float = 1.5, dh_unit: float = 0.01, dh_max: float = PUMP_DH_MAX) -> dict:
    """Once per env step: st["r7_ppay"] = pump_update's work pay (J/kg) and st["r7_dpay"] = pump_dir_update's
    direction pay (units), both with the below-neck CoM height. Same gates, lift-off charge and (dir) asymmetric clip.
    Cached per step: the first caller's params win (keep the cfg on the defaults)."""
    st = _r7st(env)
    if st["r7_step"] == env.common_step_counter:
        return st
    st["r7_step"] = env.common_step_counter
    h, exc, chain, gate = _r7_load(env)
    z = torch.zeros_like(h)
    # work (pump_update): symmetric clip, load excess clipped to [-g, 2g]
    raw = torch.nan_to_num(h - st["r7_hp"])
    ok = chain & torch.isfinite(st["r7_hp"])
    ex = exc.clamp(-G, 2.0 * G)
    dh = raw.clamp(-dh_max, dh_max)
    st["r7_ppay"] = torch.nan_to_num(torch.where(ok, torch.where(gate, ex * dh, -(ex * dh).abs()), z))
    st["r7_hp"] = torch.where(chain, h, torch.full_like(h, float("nan")))
    # direction (pump_dir_update): sign-squashed load, paid moves clipped to dh_max, charged ones to 3 dh_max
    raw = torch.nan_to_num(h - st["r7_hd"])
    ok = chain & torch.isfinite(st["r7_hd"])
    f = (exc / e0).clamp(-1.0, 1.0)
    dh_pay = raw.clamp(-dh_max, dh_max) / dh_unit
    dh_chg = raw.clamp(-3.0 * dh_max, 3.0 * dh_max) / dh_unit
    on = torch.where(f * raw > 0, f * dh_pay, f * dh_chg)
    st["r7_dpay"] = torch.nan_to_num(torch.where(ok, torch.where(gate, on, -(f * dh_chg).abs()), z))
    st["r7_hd"] = torch.where(chain, h, torch.full_like(h, float("nan")))
    return st


def r7_pump(env) -> torch.Tensor:
    """ramp_pump with the below-neck CoM (J/kg per second; POSITIVE weight)."""
    return r7_update(env)["r7_ppay"] / env.step_dt


def r7_pump_dir(env) -> torch.Tensor:
    """ramp_pump_dir with the below-neck CoM (units per second; POSITIVE weight)."""
    return r7_update(env)["r7_dpay"] / env.step_dt


def r7_head_parts(env) -> tuple[torch.Tensor, torch.Tensor]:
    """(lev_deg, dq (B, 2)): head off level vs WORLD up (deg; the angle between the head-frame gravity and its value
    with a level trunk and HOME head joints = the diag / measure2 hlev), and neck_pitch / head_pitch minus the
    hand-off head pose (rad)."""
    ids, hq = sq._head_joint_ids(env)
    robot = sq._robot(env)
    href = sq._head_ref(env)
    qh = robot.data.body_link_quat_w[:, href["id"]]
    down = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(env.num_envs, 3)
    cosv = (quat_apply_inverse(qh, down) * href["g"]).sum(dim=1).clamp(-1.0, 1.0)
    lev = torch.rad2deg(torch.acos(cosv))
    dq = robot.data.joint_pos[:, ids[:2]] - hq[:2]
    return torch.nan_to_num(lev), torch.nan_to_num(dq)


def r7_head(env, free_deg: float = 20.0, unit_deg: float = 20.0, free_q: float = 0.30, unit_q: float = 0.30,
            cap: float = 4.0, stages: tuple = (sq.ST_EXIT, sq.ST_RAMP)) -> torch.Tensor:
    """Cost (EXIT / RAMP, >= 0, NEGATIVE weight): ((lev - free_deg) / unit_deg)^2 beyond the free band + per joint
    ((|dq| - free_q) / unit_q)^2 for neck_pitch / head_pitch beyond theirs; capped per step. Free: a head within 20 deg
    of level and each pitch joint within 0.3 rad (17 deg) of the riding pose (the r5 frozen-head test: the head's small
    balance motions are needed; a 29 deg wall apex needs ~0.5 rad of neck + head pitch to keep the head level, split
    over the two joints = inside the band)."""
    lev, dq = r7_head_parts(env)
    c = ((lev - free_deg) / unit_deg).clamp(min=0.0) ** 2
    c = c + (((dq.abs() - free_q) / unit_q).clamp(min=0.0) ** 2).sum(dim=1)
    return torch.nan_to_num(c.clamp(max=cap) * _stage_in(env, stages).float())


# r7rev (review: the outcome terms still paid the fold's REAL work - the head is 38 % of the mass, a fold adds ~0.1 J/kg
# per episode, worth ~7-10 of ramp_retain / r6_keep per visit vs ~2.5 of capped head cost): the outcome pay is scaled by
# a VISIT head factor hf = 1 - max(r7_head raw cost since the last outward flat-edge crossing) / cap. An in-band head
# keeps 1; a 40 deg nod 0.75; a face-to-deck fold (cap) 0. Pumping with the legs keeps the full outcome pay.


def r7_visit_update(env, cap: float = 4.0) -> dict:
    """Once per env step (after retain_update / r6_keep_update): st["r7_hf"] = the head factor of the visit so far
    (including this step); the latch restarts at each outward flat-edge crossing (RAMP, inside the pipe)."""
    st = _r7st(env)
    if "r7_vmax" not in st:
        n, dev = env.num_envs, env.device
        st.update(r7_vmax=torch.zeros(n, device=dev), r7_flat=torch.zeros(n, dtype=torch.bool, device=dev),
                  r7_hf=torch.ones(n, device=dev), r7_vstep=-1)
    if st["r7_vstep"] == env.common_step_counter:
        return st
    st["r7_vstep"] = env.common_step_counter
    retain_update(env)
    r6_keep_update(env)
    pipe = st["pipe"]
    ax = torch.nan_to_num(sq.pipe_xyz(env))[:, 0].abs()
    live = (st["stage"] == sq.ST_RAMP) & sq.in_pipe(env, 0.0)
    vmax = torch.maximum(st["r7_vmax"], r7_head(env, cap=cap))
    st["r7_hf"] = (1.0 - vmax / cap).clamp(0.0, 1.0)
    edge = st["r7_flat"] & (ax > pipe["L"] / 2) & live
    st["r7_vmax"] = torch.where(edge, torch.zeros_like(vmax), vmax)
    st["r7_flat"] = (ax <= pipe["L"] / 2) & live
    return st


def reset_r7_visit(env, env_ids: torch.Tensor) -> None:
    """Reset event (r7 on): no head cost latched, not on the flat."""
    st = _r7st(env)
    if "r7_vmax" not in st:
        return
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    i = env_ids.to(env.device).long()
    st["r7_vmax"][i] = 0.0
    st["r7_flat"][i] = False


def r7_retain(env) -> torch.Tensor:
    """ramp_retain (default params) x the visit head factor."""
    return ramp_retain(env) * r7_visit_update(env)["r7_hf"]


def r7_keep(env) -> torch.Tensor:
    """r6_keep (default params) x the visit head factor (latched up to the pass)."""
    return r6_keep(env) * r7_visit_update(env)["r7_hf"]


def m_r7_hf(env) -> torch.Tensor:
    """r7: visit head factor at the paid retain edges: seq_r7_hf / seq_retain_n = mean factor."""
    st = r7_visit_update(env)
    return retain_update(env)["rt_evt"].float() * st["r7_hf"]


def m_r7_head(env) -> torch.Tensor:
    """r7: raw r7_head cost in RAMP: seq_r7_head / seq_ramp = mean per step."""
    return r7_head(env) * sq._phase(env, sq.RAMP)


def m_r7_lev(env) -> torch.Tensor:
    """r7: head off level (deg) in RAMP: seq_r7_lev / seq_ramp = mean."""
    return r7_head_parts(env)[0] * sq._phase(env, sq.RAMP)


def m_r7_pump_dir(env) -> torch.Tensor:
    """r7: below-neck correctly phased CoM travel (units, signed): seq_r7_pump_dir / episodes."""
    return r7_update(env)["r7_dpay"]


def m_r7_pump(env) -> torch.Tensor:
    """r7: below-neck pumping work (J/kg): seq_r7_pump / episodes."""
    return r7_update(env)["r7_ppay"]


# ── r8b (2026-11-21, default off: MICRODUCK_RAMP_R8B=1 in microduck_skate_seq_ramp_env_cfg.py) ───────────────────────────
# The 8 cm pipe chain review: at the wall apexes the board pitches to the coping, SLEWS (yaws) under the rider and the deck scrapes the
# wall, the rear sole unweights, the rider folds (head over the feet 16.9 -> 13.1 cm, head nods 25 deg, yaw swings
# 66 deg); other runs: tail down, board nearly on edge. Pumping is accepted as infeasible. r8b
# prices a CLEAN, composed wall visit: every quantity is measured relative to the pipe SURFACE under the board (so a
# straight climb of the wall's own slope costs nothing), all costs >= 0 (NEGATIVE weights), RAMP only.
#   deck    : any deck geom (not a wheel) touching the terrain                              (the "deck_terrain" sensor)
#   slew    : board yaw rate about its own deck normal beyond slew_free (rad/s) - at the apex v ~ 0, so truck steering
#             (v kappa) cannot produce it: it is the board pivoting / skidding under the rider
#   roll    : deck roll relative to the surface beyond roll_free deg (wheels on the surface = ~0)
#   soles   : a sole off the deck, or the lighter sole carrying less than share_free of the two soles' load
#   head    : head (neck subtree CoM) height over the ankles along the deck normal below z_free + head rate relative
#             to the board (pitch + yaw, board frame) beyond rate_free
#   lip     : the leading wheel within lip_margin of the coping (a soft brake well before over_coping's failure)


def _r8b_ids(env) -> dict:
    """Cached body ids: board wheels, robot ankles."""
    c = tk._tc(env)
    if "r8b_ids" not in c:
        b = tk._board(env)
        wid = b.find_bodies(r"^(front|rear)_wheel_[LR]$")[0]
        aid = sq._robot(env).find_bodies(r"^ankle_(left|right)$")[0]
        assert len(wid) == 4 and len(aid) == 2, f"r8b: wheels {wid} ankles {aid}"
        c["r8b_ids"] = dict(wheels=list(wid), ankles=list(aid))
    return c["r8b_ids"]


WHEEL_R = 0.008                # wheel radius (m): wheel centre height - WHEEL_R = wheel bottom above the pipe bottom


def _r8b_st(env) -> dict:
    st = sq._st(env)
    if "r8b_step" not in st:
        st["r8b_step"] = -1
    return st


def r8b_update(env) -> dict:
    """Once per env step: the raw r8b quantities in st (all (B,)):
      r8b_ramp   RAMP and inside the pipe (the cost gate)
      r8b_wall   on a transition (|x| > L/2)
      r8b_roll   deck roll vs the surface normal at the board's x (deg, signed)
      r8b_pitch  deck pitch vs the surface (deg, signed: + = deck x end above the surface tangent)
      r8b_slew   board angular velocity about its deck normal (rad/s, signed)
      r8b_head_dev  heading off the fall line (deg, 0..90, fakie-symmetric)
      r8b_deck   a deck geom touches the terrain (bool)
      r8b_share  the lighter sole's share of the two soles' contact load (0 when a sole is off)
      r8b_hz     head subtree CoM over the ankle midpoint along the deck normal (m)
      r8b_hrate  head angular speed relative to the board, pitch + yaw components in the board frame (rad/s)
      r8b_lead   highest wheel bottom above the pipe bottom (m)"""
    st = _r8b_st(env)
    if st["r8b_step"] == env.common_step_counter:
        return st
    st["r8b_step"] = env.common_step_counter
    sq.seq_update(env)
    pipe = st["pipe"]
    n, dev = env.num_envs, env.device
    ids = _r8b_ids(env)
    b = tk._board(env).data
    q = b.root_link_quat_w
    ex = quat_apply(q, torch.tensor([1.0, 0.0, 0.0], device=dev).expand(n, 3))
    ey = quat_apply(q, torch.tensor([0.0, 1.0, 0.0], device=dev).expand(n, 3))
    ez = quat_apply(q, torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3))
    p = torch.nan_to_num(sq.pipe_xyz(env))
    _, slope = sq.pipe_profile(p[:, 0], pipe)
    ns = torch.stack([-slope, torch.zeros_like(slope), torch.ones_like(slope)], dim=1) / torch.sqrt(1.0 + slope * slope).unsqueeze(1)
    tl = torch.stack([torch.ones_like(slope), torch.zeros_like(slope), slope], dim=1) / torch.sqrt(1.0 + slope * slope).unsqueeze(1)
    st["r8b_roll"] = torch.nan_to_num(torch.rad2deg(torch.asin((ey * ns).sum(1).clamp(-1.0, 1.0))))
    st["r8b_pitch"] = torch.nan_to_num(torch.rad2deg(torch.asin((ex * ns).sum(1).clamp(-1.0, 1.0))))
    w = torch.nan_to_num(b.root_link_ang_vel_w)
    st["r8b_slew"] = (w * ez).sum(1)
    # heading vs the fall line, measured in the surface plane (fakie-symmetric)
    exs = ex - (ex * ns).sum(1, keepdim=True) * ns
    exs = exs / exs.norm(dim=1, keepdim=True).clamp(min=1e-6)
    st["r8b_head_dev"] = torch.nan_to_num(torch.rad2deg(torch.acos((exs * tl).sum(1).abs().clamp(0.0, 1.0))))
    st["r8b_deck"] = tk._contact_any(env, "deck_terrain")
    fl = env.scene.sensors["left_deck"].data.force
    fr = env.scene.sensors["right_deck"].data.force
    fl = torch.nan_to_num(fl.reshape(n, -1, 3).sum(1).norm(dim=1))
    fr = torch.nan_to_num(fr.reshape(n, -1, 3).sum(1).norm(dim=1))
    both = tk.soles_on_deck(env)
    st["r8b_share"] = torch.where(both, torch.minimum(fl, fr) / (fl + fr).clamp(min=1e-3), torch.zeros_like(fl))
    robot = sq._robot(env)
    _, _, hh = r7_com_split(env)                       # head subtree CoM above the board origin along the deck normal
    ank = robot.data.body_link_pos_w[:, ids["ankles"]].mean(dim=1)
    st["r8b_hz"] = torch.nan_to_num(hh - ((ank - b.root_link_pos_w) * ez).sum(1))
    wr = quat_apply_inverse(q, robot.data.body_link_ang_vel_w[:, sq._head_ref(env)["id"]] - w)
    st["r8b_hrate"] = torch.nan_to_num(wr[:, 1:].norm(dim=1))
    wz = tk._board(env).data.body_link_pos_w[:, ids["wheels"], 2] - sq._origin(env)[:, 2:3]
    st["r8b_lead"] = torch.nan_to_num(wz.max(dim=1).values - WHEEL_R)
    st["r8b_ramp"] = (st["stage"] == sq.ST_RAMP) & sq.in_pipe(env, 0.0)
    st["r8b_wall"] = p[:, 0].abs() > pipe["L"] / 2
    return st


def r8b_deck(env) -> torch.Tensor:
    """Cost 1 per step (RAMP): a deck geom (deck / core / tail pad / tapers / kicks / ends) on the terrain."""
    st = r8b_update(env)
    return (st["r8b_ramp"] & st["r8b_deck"]).float()


def r8b_slew(env, free: float = 1.0, unit: float = 1.0, cap: float = 2.0) -> torch.Tensor:
    """Cost (RAMP): ((|board yaw rate about the deck normal| - free) / unit) clamped to [0, cap]. r7 4295 on h8: carving
    on the flat <= 0.6 rad/s; at the apex p50 2.3, p90 4.7 rad/s."""
    st = r8b_update(env)
    c = ((st["r8b_slew"].abs() - free) / unit).clamp(0.0, cap)
    return c * st["r8b_ramp"].float()


def r8b_tilt(env, roll_free: float = 15.0, pitch_free: float = 12.0, unit_deg: float = 10.0, cap: float = 2.0) -> torch.Tensor:
    """Cost (RAMP): deck roll vs the surface beyond roll_free + deck pitch vs the surface beyond pitch_free (a truck off
    the wall: wheelie / tail down; four wheels on the R 0.40 arc stay within ~1 deg), per unit_deg, capped."""
    st = r8b_update(env)
    c = ((st["r8b_roll"].abs() - roll_free) / unit_deg).clamp(min=0.0) + ((st["r8b_pitch"].abs() - pitch_free) / unit_deg).clamp(min=0.0)
    return c.clamp(max=cap) * st["r8b_ramp"].float()


def r8b_soles(env, share_free: float = 0.2) -> torch.Tensor:
    """Cost (RAMP) in [0, 1]: 1 with a sole off the deck, else (share_free - lighter share) / share_free beyond it."""
    st = r8b_update(env)
    c = ((share_free - st["r8b_share"]) / share_free).clamp(0.0, 1.0)
    return c * st["r8b_ramp"].float()


def r8b_head(env, z_free: float = 0.155, unit: float = 0.01, rate_free: float = 3.0, rate_unit: float = 3.0,
             cap: float = 3.0) -> torch.Tensor:
    """Cost (RAMP): head over the ankles below z_free, linear per unit (r7 4295 h8: flat p10 16.2 cm, apex p50 14.4),
    + (head pitch / yaw rate vs the board - rate_free) / rate_unit (apex p50 8.8 rad/s); capped."""
    st = r8b_update(env)
    c = ((z_free - st["r8b_hz"]) / unit).clamp(min=0.0) + ((st["r8b_hrate"] - rate_free) / rate_unit).clamp(min=0.0)
    return c.clamp(max=cap) * st["r8b_ramp"].float()


def r8b_lip(env, margin: float = 0.01, unit: float = 0.01, cap: float = 2.0) -> torch.Tensor:
    """Cost (RAMP): the leading wheel above (coping h - margin), linear per unit, capped (deck-centre rise ~4.8 cm on
    h 8: above the ramp_apex cap of 4 cm, so full apex pay stays below the lip)."""
    st = r8b_update(env)
    c = ((st["r8b_lead"] - (st["pipe"]["h"] - margin)) / unit).clamp(0.0, cap)
    return c * st["r8b_ramp"].float()


R8B_PARTS = dict(deck=r8b_deck, slew=r8b_slew, tilt=r8b_tilt, soles=r8b_soles, head=r8b_head, lip=r8b_lip)
R8B_K = dict(deck=8.0, slew=3.0, tilt=2.0, soles=3.0, head=1.0, lip=2.0)


def r8b_clean(env, k: dict | None = None, cap: float = 8.0, r9_exempt: bool = False) -> torch.Tensor:
    """Cost (RAMP, >= 0, NEGATIVE weight -1): sum_i k_i x part_i (R8B_PARTS at their defaults), capped at `cap` per
    second. One term with a TOTAL cap (review r8b: the uncapped parts could reach ~28 /s, so a rider stuck in a bad state
    could prefer a fall). With the cap, a bad apex state lasting t s costs <= 8 t: the measured bad apex windows last
    ~0.1-0.4 s, and a deliberate bail costs 10 plus every later pay (RAMP alive pay ~3.7 /s + the
    pass outcome terms ~10 per pass). Per-part raw values are logged as metrics seq_r8b_<part>."""
    k = k or R8B_K
    c = torch.zeros(env.num_envs, device=env.device)
    for nm, w in k.items():
        part = R8B_PARTS[nm](env)
        if r9_exempt and nm == "soles":            # r9 (default False): the push foot may leave the deck on the push flat
            part = part * (~r9_update(env)["r9_soles_free"]).float()
        c = c + float(w) * part
    return torch.nan_to_num(c).clamp(max=cap)

# ── r9 (2026-11-21, default off: MICRODUCK_RAMP_R9=1 in microduck_skate_seq_ramp_env_cfg.py) ────────────────────────────
# Keep the speed by
# FOOT-PUSHING on the flat bottom (pumping is infeasible:). Measured (r8b 4794, h 8, ramp_bottom
# n 32): every wall visit is a fakie shuttle (fwd->fakie / fakie->fwd 52 / 42, 0 turns > 90 deg; wall-1 yaw change p50
# 21 deg = a slew, not a kick-turn), so only FORWARD crossings (every other one) are pushed. A forward through-crossing
# lasts 1.8-4.0 s (p10-p90; both axles on the flat 1.6-3.6 s) at |vx| 0.25-0.65 m/s; the ride loses ~0.09 m/s per
# crossing + wall visit (|vx| at the flat 0.66 -> 0.56 -> 0.48). p15 strokes at speed: contact 0.08 s, +0.15 m/s per
# stroke (p10-p90 0.08-0.25) at 0.68 m/s, the step-on 0.30 s after the last lift-off: 1-2 strokes per forward crossing
# replace the loss of two visits. p15 is NOT the reference: its push knee bends backwards (right knee +0.7 rad median
# in PUSH; the normal side is < 0), so r9 prices the knee side and the leg joint speed (servo ~4.7 rad/s no-load).
#   zone       RAMP, inside the pipe, board centre on the flat (|x| < L/2), rolling nose-first (board v_fwd > 0)
#   r9_gain    event at the flat's edge (outward), forward crossing with a push-foot floor contact on it: the energy
#              gained on the flat, both ends capped at the target height (system-CoM |v_xy|^2/2 <= g h_tgt), in cm of
#              wall height (internal motion cannot change the system CoM velocity: only the floor push / the wheels can;
#              review r9: the FULL horizontal speed, so turning along-pipe speed into vx is not paid, and the entry
#              energy is taken only on a flat entry FROM A WALL: a flat spawn's first crossing pays nothing)
#   r9_reach   potential-based (gamma phi' - phi): the push sole below the deck sole with the board sole on the deck
#   r9_drag    push-foot floor contact longer than drag_free (s): a dragging / braking / standing foot
#   r9_late    event at the flat's edge (outward): not both soles on the deck / the push foot still on the floor
#   r9_knee    each knee past its normal side (right <= +0.1 rad, left >= -0.1 rad), squared per 0.1 rad, capped
#   r9_stroke  event at the flat's edge (outward), forward crossing: floor touch on it + both soles back on the deck
#              (an exploration bonus, annealed to 0 by the cfg)
#   (r9_jvel   leg joint speed beyond 5 rad/s: kept, UNUSED - the XL330 no-load speed is 14-22 rad/s)
#   terminations: foot_on_floor lets the RIGHT foot touch the floor in the zone with the sole itself on the flat
#   (|x_sole| < L/2 - sole_margin); board_lost lets the right sole be off the deck up to ramp_off_s when it left the deck
#   in the zone (0.6 s otherwise, as before)
R9_H_TGT = 0.045           # m: gain pay stops at this cross-pipe energy (v ~0.94 m/s; the lip cost starts ~5.6 cm)
R9_KNEE_FREE = 0.10        # rad past zero on the wrong side before the knee cost starts


def _r9_ids(env) -> dict:
    c = tk._tc(env)
    if "r9_ids" not in c:
        robot = sq._robot(env)
        kl = robot.find_joints(r"^left_knee$")[0]
        kr = robot.find_joints(r"^right_knee$")[0]
        legs = robot.find_joints(r"^(left|right)_(hip_yaw|hip_roll|hip_pitch|knee|ankle)$")[0]
        assert len(kl) == 1 and len(kr) == 1 and len(legs) == 10, f"r9 joints {kl} {kr} {legs}"
        c["r9_ids"] = dict(kl=kl[0], kr=kr[0], legs=list(legs),
                           rs=robot.find_geoms(r"^right_sole_deck_collision$")[0][0],
                           ls=robot.find_geoms(r"^left_sole_deck_collision$")[0][0])
    return c["r9_ids"]


def _r9st(env) -> dict:
    st = sq._st(env)
    if "r9_step" not in st:
        n, dev = env.num_envs, env.device
        b = lambda: torch.zeros(n, dtype=torch.bool, device=dev)
        z = lambda: torch.zeros(n, device=dev)
        st.update(r9_step=-1, r9_flat_prev=b(), r9_ein=torch.full((n,), float("nan"), device=dev), r9_touch=b(),
                  r9_rf_prev=b(), r9_phi=torch.full((n,), float("nan"), device=dev), r9_zone=b(), r9_soles_free=b(),
                  r9_allow=b(), r9_gain_evt=b(), r9_gain_cm=z(), r9_late_evt=b(), r9_reach_pay=z(), r9_td=b(),
                  r9_edge=b(), r9_wall_prev=b(), r9_stroke_evt=b(), r9_rd_prev=b(), r9_lift_zone=b())
    return st


def reset_r9_state(env, env_ids: torch.Tensor) -> None:
    """Reset event (r9 on, after reset_ramp): no flat visit open, no reach potential."""
    st = _r9st(env)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    i = env_ids.to(env.device).long()
    st["r9_flat_prev"][i] = False
    st["r9_ein"][i] = float("nan")
    st["r9_touch"][i] = False
    st["r9_rf_prev"][i] = False
    st["r9_phi"][i] = float("nan")
    st["r9_wall_prev"][i] = False
    st["r9_rd_prev"][i] = False
    st["r9_lift_zone"][i] = False


def r9_update(env, h_tgt: float = R9_H_TGT, gain_cap_cm: float = 3.0, sole_margin: float = 0.02,
              reach_m: float = 0.025, gamma: float = 0.99) -> dict:
    """Once per env step: zone / allowance masks, the flat-visit energy bookkeeping, the edge events, the reach pay.
    The first caller's parameters win for that step (all callers use the defaults)."""
    from micro_skater.tasks import mdp as _mdp

    st = _r9st(env)
    if st["r9_step"] == env.common_step_counter:
        return st
    st["r9_step"] = env.common_step_counter
    sq.seq_update(env)
    if "r9b_fresh" in st:                       # r9b (default off: the keys exist only with the r9b reset event)
        _r9b_pre(env, st)
    pipe = st["pipe"]
    L2 = pipe["L"] / 2
    ids = _r9_ids(env)
    robot = sq._robot(env)
    org = sq._origin(env)
    p = torch.nan_to_num(sq.pipe_xyz(env))
    ax = p[:, 0].abs()
    live = (st["stage"] == sq.ST_RAMP) & sq.in_pipe(env, 0.0)
    fwd = sq.board_v_fwd(env) > 0.0
    on_flat = live & (ax < L2)
    zone = on_flat & fwd
    rs = robot.data.geom_pos_w[:, ids["rs"]]
    ls = robot.data.geom_pos_w[:, ids["ls"]]
    rsx = torch.nan_to_num((rs[:, 0] - org[:, 0]).abs(), nan=1.0)
    _, rf = sq._feet_floor(env)
    st["r9_zone"] = zone
    st["r9_soles_free"] = zone
    st["r9_allow"] = zone & (rsx < L2 - sole_margin)
    # flat-visit bookkeeping (cross-pipe energy of the robot + board centre of mass)
    vxy = torch.nan_to_num(_mdp._skate_system_vel_xy(env))
    e = 0.5 * (vxy * vxy).sum(dim=1)
    enter = on_flat & ~st["r9_flat_prev"]
    from_wall = enter & st["r9_wall_prev"]
    st["r9_ein"] = torch.where(from_wall, e, torch.where(enter, torch.full_like(e, float("nan")), st["r9_ein"]))
    st["r9_touch"] = torch.where(enter, torch.zeros_like(st["r9_touch"]), st["r9_touch"]) | (rf & zone)
    edge = st["r9_flat_prev"] & ~on_flat & live & (ax >= L2)
    et = G * h_tgt
    gain = (e.clamp(max=et) - torch.nan_to_num(st["r9_ein"], nan=et).clamp(max=et)) / (G * 0.01)
    st["r9_edge"] = edge
    st["r9_gain_evt"] = edge & fwd & st["r9_touch"] & torch.isfinite(st["r9_ein"])
    st["r9_gain_cm"] = torch.where(st["r9_gain_evt"], gain.clamp(0.0, gain_cap_cm), torch.zeros_like(e))
    on = tk.soles_on_deck(env)
    st["r9_late_evt"] = edge & (~on | rf)
    st["r9_stroke_evt"] = edge & fwd & st["r9_touch"] & on & ~rf
    st["r9_flat_prev"] = on_flat
    st["r9_wall_prev"] = live & (ax >= L2)
    # where the push sole left the deck (board_lost's longer off-deck allowance only for a lift-off in the zone)
    rd = tk._contact_any(env, "right_deck")
    st["r9_lift_zone"] = torch.where(rd, torch.zeros_like(rd), torch.where(st["r9_rd_prev"] & ~rd, zone, st["r9_lift_zone"]))
    st["r9_rd_prev"] = rd
    st["r9_ein"] = torch.where(on_flat, st["r9_ein"], torch.full_like(e, float("nan")))
    # reach potential (telescoping per zone visit): push sole below the deck sole
    drop = torch.nan_to_num(ls[:, 2] - rs[:, 2])
    phi = (zone & tk._contact_any(env, "left_deck")).float() * (drop / reach_m).clamp(0.0, 1.0)
    st["r9_reach_pay"] = torch.nan_to_num(gamma * phi - st["r9_phi"])
    st["r9_phi"] = phi
    st["r9_td"] = rf & ~st["r9_rf_prev"] & zone
    st["r9_rf_prev"] = rf
    if "r9b_fresh" in st:                       # r9b (default off)
        _r9b_post(env, st, on_flat)
    return st


def r9_gain(env) -> torch.Tensor:
    """Event (POSITIVE weight): cm of cross-pipe wall height gained on a pushed forward flat crossing (capped)."""
    st = r9_update(env)
    return st["r9_gain_cm"] / env.step_dt


def r9_stroke(env) -> torch.Tensor:
    """Event (POSITIVE weight, annealed to 0): a forward crossing with a floor touch that reaches the wall with both
    soles back on the deck and the push foot off the floor (review r9: exploration past the coasting optimum)."""
    return r9_update(env)["r9_stroke_evt"].float() / env.step_dt


def r9_reach(env) -> torch.Tensor:
    """Potential-based shaping (POSITIVE weight): d/dt of the push-sole drop below the deck sole in the zone (0..1)."""
    return r9_update(env)["r9_reach_pay"] / env.step_dt


def r9_drag(env, drag_free: float = 0.20, unit: float = 0.10, cap: float = 3.0) -> torch.Tensor:
    """Cost (>= 0): push-foot floor contact held beyond drag_free s (p15 strokes: 0.06-0.08 s), per unit s, capped."""
    _, rf = sq._feet_floor(env)
    ct = env.scene.sensors["right_foot_floor"].data.current_contact_time
    ct = ct.max(dim=-1).values if ct.dim() > 1 else ct
    return torch.nan_to_num(rf.float() * ((ct - drag_free) / unit).clamp(0.0, cap))


def r9_late(env) -> torch.Tensor:
    """Event cost (>= 0): leaving the flat into a wall without both soles on the deck (or the push foot on the floor)."""
    return r9_update(env)["r9_late_evt"].float() / env.step_dt


def r9_knee(env, free: float = R9_KNEE_FREE, unit: float = 0.10, cap: float = 10.0) -> torch.Tensor:
    """Cost (>= 0, every stage): right knee above +free / left knee below -free (rad; the normal sides are right <= 0,
    left >= 0: p15 PUSH right knee median +0.7 = a backwards "bird knee"), squared per unit, capped."""
    ids = _r9_ids(env)
    q = sq._robot(env).data.joint_pos
    c = ((q[:, ids["kr"]] - free) / unit).clamp(min=0.0) ** 2 + ((-free - q[:, ids["kl"]]) / unit).clamp(min=0.0) ** 2
    return torch.nan_to_num(c).clamp(max=cap)


def r9_jvel(env, free: float = 5.0, cap: float = 20.0) -> torch.Tensor:
    """Cost (>= 0, every stage): sum over the 10 leg joints of (|qd| - free)^2 (rad/s) beyond free, capped."""
    qd = sq._robot(env).data.joint_vel[:, _r9_ids(env)["legs"]]
    return torch.nan_to_num(((qd.abs() - free).clamp(min=0.0) ** 2).sum(dim=1)).clamp(max=cap)


def r9_foot_on_floor(env) -> torch.Tensor:
    """sq.seq_foot_on_floor + the r9 allowance: the RIGHT foot may touch the floor in the push zone with its sole on the
    flat bottom (r9_update r9_allow). The left foot on the floor, or the right foot anywhere else, still ends it."""
    st = r9_update(env)
    lf, rf = sq._feet_floor(env)
    push_ok = (st["trick"] == sq.PUSH) | st.get("cmd", torch.zeros_like(lf))
    return lf | (rf & ~push_ok & ~st["r9_allow"])


def r9_board_lost(env, max_dist: float = 0.18, max_off_s: float = 0.6, ramp_off_s: float = 1.0) -> torch.Tensor:
    """sq.seq_board_lost with the push (right) sole allowed off the deck up to ramp_off_s in RAMP when it left the deck
    in the push zone (a stroke from the deck: off, down, kick, back on ~0.6-0.9 s); max_off_s otherwise and for the
    board (left) sole."""
    from micro_skater.tasks.mdp import _skate_air_time

    st = sq.seq_update(env)
    robot = sq._robot(env)
    d = torch.nan_to_num((robot.data.root_link_pos_w[:, :2] - tk._board(env).data.root_link_pos_w[:, :2]).norm(dim=1),
                         nan=1.0)
    l_off = _skate_air_time(env, "left_deck") > max_off_s
    push_ok = (st["trick"] == sq.PUSH) | st.get("cmd", torch.zeros_like(d, dtype=torch.bool))
    lz = r9_update(env)["r9_lift_zone"]
    r_lim = torch.where((st["stage"] == sq.ST_RAMP) & lz, torch.full_like(d, ramp_off_s), torch.full_like(d, max_off_s))
    r_off = (_skate_air_time(env, "right_deck") > r_lim) & ~push_ok
    return (d > max_dist) | l_off | r_off


def m_r9_gain(env) -> torch.Tensor:
    """r9: cm gained, summed at the paid edges (seq_r9_gain / seq_r9_n = mean per pushed crossing)."""
    return r9_update(env)["r9_gain_cm"]


def m_r9_n(env) -> torch.Tensor:
    """r9: pushed forward crossings (edge events with a floor touch)."""
    return r9_update(env)["r9_gain_evt"].float()


def m_r9_stroke(env) -> torch.Tensor:
    """r9: completed strokes (forward edge, floor touch, both soles back on the deck)."""
    return r9_update(env)["r9_stroke_evt"].float()


def m_r9_td(env) -> torch.Tensor:
    """r9: push-foot floor touchdowns in the zone."""
    return r9_update(env)["r9_td"].float()


def m_r9_knee(env) -> torch.Tensor:
    """r9: steps with a knee past its normal side by more than R9_KNEE_FREE."""
    return (r9_knee(env) > 0).float()


def m_r9_jv(env) -> torch.Tensor:
    """r9: steps with any leg joint above 5 rad/s."""
    return (r9_jvel(env) > 0).float()


# ── r9b (2026-11-21, default off: MICRODUCK_RAMP_R9B=1 with R9=1, microduck_skate_seq_ramp_env_cfg._add_r9b) ─────────
# r9 (200 iterations from r7 4295 at 1.0 A) never made one push stroke:
# seq_r9_td (push-foot touchdowns in the zone) stayed exactly 0, r9_stroke was phased out by iteration before any stroke,
# and the r7 prior (a floor touch ended every RAMP episode for ~4300 iterations) never let a foot reach the floor. r9b =
# a REVERSE CURRICULUM, as the push specialist's step_mid spawns:
#   reset_r9b_push  a share of the ramp_bottom spawns start MID-PUSH on the flat bottom: board rolling nose-first at
#                   0.3-0.7 m/s toward a wall with 0.4-0.9 m of flat ahead, the rider in a push-cycle stance with the
#                   push knee on its normal side (push / mount / push_sweep with the right
#                   sole ON the floor, push_low = 12 mm above it, push_lift = 15 mm up behind it)
#   _r9b_pre        first r9_update of an r9b episode: the spawn's flat visit is open with the spawn energy as its
#                   reference (r9_gain pays the spawn crossing too), no spurious touchdown, the longer off-deck
#                   allowance (r9_lift_zone) for a push sole that starts off the deck
#   _r9b_post       "own" strokes (completed strokes off the spawn's first visit, with >= own_cm gained) -> decayed
#                   counters for the stroke-bonus gate
#   r9b_stroke_gate r9_stroke's weight stays w_hi until own strokes reach rate_on per 1000 riding RAMP env-steps, then
#                   phases out over prog_its iterations ABOVE the rate (not on iterations); a hard end at it_end
R9B_STANCE_JSON = str(_data_path("stance_poses_r9b.json"))
R9B_DECAY = 1.0 - 1.0 / (24 * 50)          # per env step: ~50-iteration memory of the own-stroke rate


def _r9b_masks(env, st) -> dict:
    from micro_skater.tasks import mdp as _mdp

    L2 = st["pipe"]["L"] / 2
    p = torch.nan_to_num(sq.pipe_xyz(env))
    live = (st["stage"] == sq.ST_RAMP) & sq.in_pipe(env, 0.0)
    on_flat = live & (p[:, 0].abs() < L2)
    vxy = torch.nan_to_num(_mdp._skate_system_vel_xy(env))
    _, rf = sq._feet_floor(env)
    return dict(on_flat=on_flat, zone=on_flat & (sq.board_v_fwd(env) > 0.0), e=0.5 * (vxy * vxy).sum(dim=1), rf=rf,
                rd=tk._contact_any(env, "right_deck"))


def _r9b_pre(env, st) -> None:
    """Called by r9_update before its bookkeeping (r9b on). Fresh episodes (first update after a reset): rf / rd history
    = now (no spurious touchdown / lift-off events); r9b spawns on the flat: the visit is open (flat_prev) with the spawn
    energy as the gain reference, no touch yet, and a push sole off the deck gets the zone's off-deck allowance."""
    fr = st["r9b_fresh"]
    sp = st["r9b_spawn"] & fr
    m = _r9b_masks(env, st)
    seed = sp & m["on_flat"]
    st["r9_flat_prev"] = st["r9_flat_prev"] | seed
    st["r9_ein"] = torch.where(seed, m["e"], st["r9_ein"])
    st["r9_touch"] = st["r9_touch"] & ~sp
    if "r9c_touch0" in st:                      # r9c (default off): a bank state past its floor touch keeps the touch
        st["r9_touch"] = st["r9_touch"] | (seed & st["r9c_touch0"])
        st["r9c_spawn_lift"] = torch.where(fr, sp & ~m["rd"], st["r9c_spawn_lift"])   # the spawn's own lifted foot
    st["r9_rf_prev"] = torch.where(fr, m["rf"], st["r9_rf_prev"])
    st["r9_lift_zone"] = torch.where(sp & ~m["rd"], m["zone"], st["r9_lift_zone"])
    st["r9_rd_prev"] = torch.where(fr, m["rd"], st["r9_rd_prev"])
    st["r9b_first"] = torch.where(fr, seed, st["r9b_first"])
    st["r9b_fresh"] = torch.zeros_like(fr)


def _r9b_post(env, st, on_flat: torch.Tensor, own_cm: float = 0.3, v_stall: float = 0.10,
              sole_margin: float = 0.02, tap_cm: float = 0.1) -> None:
    """Called at the end of r9_update (r9b on): the push-foot floor allowance with a stall margin, own strokes (not on
    an r9b spawn's first flat visit) and the decayed counters (own strokes, riding RAMP env-steps) for r9b_stroke_gate.
    GPU-only (no host sync). Stall margin (20 / 20 foot_on_floor ends of the mid-push spawns
    were the RIGHT foot on the flat with the board braked to v_fwd -0.005..-0.4 m/s in 0.2-0.7 s, i.e. r9's zone
    (v_fwd > 0) ends every foot touch that stalls the board): the right sole may stay on the flat bottom while the
    board's forward speed is > -v_stall. Gain / touch / stroke / reach still need v_fwd > 0 (forward crossings only);
    standing on the foot is r9_drag's (> 0.2 s contact) and the stop's (ramp_park) cost."""
    L2 = st["pipe"]["L"] / 2
    rs = sq._robot(env).data.geom_pos_w[:, _r9_ids(env)["rs"]]
    rsx = torch.nan_to_num((rs[:, 0] - sq._origin(env)[:, 0]).abs(), nan=1.0)
    st["r9_allow"] = on_flat & (sq.board_v_fwd(env) > -v_stall) & (rsx < L2 - sole_margin)
    first = st["r9b_first"]
    # review r9b (taps): on riding crossings the stroke bonus needs a real gain (>= tap_cm); the spawn's first visit
    # keeps paying the step-back-on (the reverse curriculum's carrot)
    st["r9_stroke_evt"] = st["r9_stroke_evt"] & (first | (st["r9_gain_evt"] & (st["r9_gain_cm"] >= tap_cm)))
    st["r9b_own_evt"] = st["r9_stroke_evt"] & ~first & st["r9_gain_evt"] & (st["r9_gain_cm"] >= own_cm)
    st["r9b_spawn_evt"] = st["r9_stroke_evt"] & first
    st["r9b_first"] = first & on_flat
    if "r9c_res" in st:                         # r9c (default off): spawn outcome + the attempt window
        _r9c_post(env, st, first, on_flat)
    c = tk._tc(env)
    riding = (st["stage"] == sq.ST_RAMP) & ~first
    c["r9b_own_ema"] = c.get("r9b_own_ema", 0.0) * R9B_DECAY + st["r9b_own_evt"].float().sum()
    c["r9b_ride_ema"] = c.get("r9b_ride_ema", 0.0) * R9B_DECAY + riding.float().sum()


def reset_r9b_push(env, env_ids: torch.Tensor, share_stages: list, stances: dict, stance_path: str = R9B_STANCE_JSON,
                   speed: tuple = (0.3, 0.7), x_along: tuple = (-0.40, 0.30), yaw_deg: float = 15.0,
                   z_lift: float = 0.001, joint_noise: float = 0.02, rpy_noise: float = 0.02, pos_noise: float = 0.003,
                   near_x: float = 0.10, near_vmax: float = 0.45, qd_max: float = 10.0) -> None:
    """Reset event (r9b; MUST run right after reset_sequence, before reset_ramp and the other ramp resets). A share of
    the ramp_bottom spawns (share_stages: [(env step, share), ...], the last elapsed stage applies) is re-placed
    MID-PUSH: board level on the flat bottom, heading +-x (+- yaw_deg) toward the wall ahead, its centre at x_along m
    from the flat's centre along the travel (negative = before it), rolling nose-first at U(speed); the rider in a
    stance drawn from `stances` (key -> weight) relative to the deck (stance-file conventions), with the board's
    velocity. Stage / scheduler: the ramp_bottom spawn's (RAMP). Flags st["r9b_spawn"] / st["r9b_fresh"]."""
    import json
    from pathlib import Path

    from mjlab.utils.lab_api.math import quat_mul
    from micro_skater.tasks.mdp import _quat_from_rpy_t

    st = sq._st(env)
    c = tk._tc(env)
    n_all, dev = env.num_envs, env.device
    if "r9b_fresh" not in st:
        z = lambda: torch.zeros(n_all, dtype=torch.bool, device=dev)
        st.update(r9b_fresh=z(), r9b_spawn=z(), r9b_first=z(), r9b_own_evt=z(), r9b_spawn_evt=z(),
                  r9b_key=torch.full((n_all,), -1, dtype=torch.long, device=dev))
    if env_ids is None:
        env_ids = torch.arange(n_all, device=dev)
    env_ids = env_ids.to(dev).long()
    if len(env_ids) == 0:
        return
    st["r9b_fresh"][env_ids] = True
    st["r9b_spawn"][env_ids] = False
    st["r9b_first"][env_ids] = False
    st["r9b_key"][env_ids] = -1
    share = 0.0
    for step, s in share_stages:
        if env.common_step_counter >= step:
            share = float(s)
    n = len(env_ids)
    sel = (st["spawn"][env_ids] == sq.SPAWNS.index("ramp_bottom")) & (torch.rand(n, device=dev) < share)
    if not bool(sel.any()):
        return
    if c.get("r9b_kf_path") != stance_path:
        robot = sq._robot(env)
        keys = list(stances)
        jn = list(tk.load_stance(keys[0], Path(stance_path))["joints"].keys())
        kt = [sq._stance_tensors(env, k, stance_path, jn) for k in keys]
        sd = json.loads(Path(stance_path).read_text())
        qd = torch.tensor([[float(sd[k].get("qd_per_mps", {}).get(j, 0.0)) for j in jn] for k in keys], device=dev)
        c["r9b_kf"] = dict(leg_ids=[list(robot.joint_names).index(j) for j in jn], qd=qd,
                           pos=torch.stack([t[0] for t in kt]), rpy=torch.stack([t[1] for t in kt]),
                           joints=torch.stack([t[2] for t in kt]),
                           w=torch.tensor([float(stances[k]) for k in keys], device=dev),
                           wheel_ids=tk._board(env).find_joints(r"^passive_.*_wheel_[LR]$")[0], keys=keys)
        c["r9b_kf_path"] = stance_path
    kf = c["r9b_kf"]
    ids = env_ids[sel]
    m = len(ids)
    U = lambda lo, hi: lo + torch.rand(m, device=dev) * (hi - lo)
    W = st["pipe"]["W"]
    side = torch.where(torch.rand(m, device=dev) < 0.5, -1.0, 1.0)            # travel direction along +-x
    xa = U(*x_along)
    bx = side * xa
    by = U(-W / 2 + 1.0, W / 2 - 1.5)
    yaw = torch.where(side > 0, 0.0, math.pi) + torch.deg2rad(U(-yaw_deg, yaw_deg))
    v0 = U(*speed)
    # review r9b: the near-end ("nearly done") spawns slower, so the step-on (~0.3 s, p15) fits before the wall
    v0 = torch.where(xa > near_x, speed[0] + (v0 - speed[0]) * (near_vmax - speed[0]) / max(speed[1] - speed[0], 1e-6), v0)
    zero = torch.zeros(m, device=dev)
    qb = _quat_from_rpy_t(zero, zero, yaw)
    bpos = sq._origin(env)[ids] + torch.stack([bx, by, torch.full_like(bx, sq.DECK_Z)], dim=1)
    vdir = quat_apply(qb, torch.tensor([1.0, 0.0, 0.0], device=dev).expand(m, 3))
    vel = torch.zeros(m, 6, device=dev)
    vel[:, :3] = vdir * v0.unsqueeze(1)
    board, robot = tk._board(env), sq._robot(env)
    board.write_root_link_pose_to_sim(torch.cat([bpos, qb], dim=1), ids)
    board.write_root_link_velocity_to_sim(vel, ids)
    bjv = torch.zeros(m, board.num_joints, device=dev)
    bjv[:, kf["wheel_ids"]] = (v0 / sq.BOARD["wheel_r"]).unsqueeze(1)
    board.write_joint_state_to_sim(torch.zeros_like(bjv), bjv, env_ids=ids)
    k = torch.multinomial(kf["w"], m, replacement=True)
    rel = kf["pos"][k].clone()
    rel[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * pos_noise
    rel[:, 2] = rel[:, 2] - sq.DECK_Z + z_lift
    r = kf["rpy"][k] + (torch.rand(m, 3, device=dev) * 2 - 1) * rpy_noise
    rq = quat_mul(qb, _quat_from_rpy_t(r[:, 0], r[:, 1], r[:, 2]))
    robot.write_root_link_pose_to_sim(torch.cat([bpos + quat_apply(qb, rel), rq], dim=1), ids)
    robot.write_root_link_velocity_to_sim(vel, ids)
    jpos = robot.data.default_joint_pos[ids].clone()
    jpos[:, kf["leg_ids"]] = kf["joints"][k] + torch.randn(m, len(kf["leg_ids"]), device=dev) * joint_noise
    # push-sole-still leg velocities ("qd_per_mps"; 0 for keys without): a planted / landing push foot
    # does not slide at the board speed on arrival (zero joint velocities braked 0.47 -> 0.27 m/s in
    # 0.1 s), scaled down uniformly to <= qd_max rad/s
    jv = kf["qd"][k] * v0.unsqueeze(1)
    jv = jv * (qd_max / jv.abs().max(dim=1, keepdim=True).values.clamp(min=1e-6)).clamp(max=1.0)
    jvel = torch.zeros_like(jpos)
    jvel[:, kf["leg_ids"]] = jv
    robot.write_joint_state_to_sim(jpos, jvel, env_ids=ids)
    st["v0"][ids] = v0
    st["r9b_spawn"][ids] = True
    st["r9b_key"][ids] = k


def r9b_stroke_gate(env, env_ids, reward_name: str = "r9_stroke", w_hi: float = 2.0, rate_on: float = 1.0,
                    prog_its: int = 100, it0: int = 0, it_end: int = 10 ** 9) -> torch.Tensor:
    """Curriculum (r9b): r9_stroke weight = w_hi * (1 - prog) from it0; prog += 1 / prog_its per iteration in which own
    strokes (_r9b_post) run at >= rate_on per 1000 riding RAMP env-steps (~50-iteration decayed rate); 0 from it_end.
    prog / the rate live in the env cache (a resume restarts them: the bonus is back at w_hi until strokes show)."""
    del env_ids
    term = env.reward_manager.get_term_cfg(reward_name)
    c = tk._tc(env)
    it = env.common_step_counter // 24
    if it < it0:
        term.weight = 0.0
        return torch.tensor([term.weight])
    if c.get("r9b_gate_it") != it:
        c["r9b_gate_it"] = it
        own, ride = c.get("r9b_own_ema", 0.0), c.get("r9b_ride_ema", 0.0)
        c["r9b_rate"] = float(own) / max(float(ride), 1.0) * 1000.0
        if c["r9b_rate"] >= rate_on:
            c["r9b_prog"] = min(1.0, c.get("r9b_prog", 0.0) + 1.0 / prog_its)
    term.weight = 0.0 if it >= it_end else w_hi * (1.0 - c.get("r9b_prog", 0.0))
    return torch.tensor([term.weight])


def r9b_rate_log(env, env_ids) -> torch.Tensor:
    """Curriculum logger only (changes nothing): the own-stroke rate per 1000 riding RAMP env-steps (r9b_stroke_gate)."""
    del env_ids
    return torch.tensor([float(tk._tc(env).get("r9b_rate", 0.0))])


def r9b_share_log(env, env_ids, share_stages: list) -> torch.Tensor:
    """Curriculum logger only (changes nothing): the current mid-push share of the ramp_bottom spawns."""
    del env_ids
    share = 0.0
    for step, s in share_stages:
        if env.common_step_counter >= step:
            share = float(s)
    return torch.tensor([share])


def m_r9b_own(env) -> torch.Tensor:
    """r9b: own completed strokes (off an r9b spawn's first visit, >= 0.3 cm gained)."""
    st = r9_update(env)
    return st["r9b_own_evt"].float() if "r9b_own_evt" in st else torch.zeros(env.num_envs, device=env.device)


def m_r9b_spawn_stroke(env) -> torch.Tensor:
    """r9b: completed strokes on an r9b spawn's first flat visit (stepped back on, reached the wall)."""
    st = r9_update(env)
    return st["r9b_spawn_evt"].float() if "r9b_spawn_evt" in st else torch.zeros(env.num_envs, device=env.device)


def m_r9b_spawn(env) -> torch.Tensor:
    """r9b: 1 on every step of an episode that started from an r9b mid-push spawn (/ episode length = share)."""
    st = sq._st(env)
    return st["r9b_spawn"].float() if "r9b_spawn" in st else torch.zeros(env.num_envs, device=env.device)


# ── r9c (2026-11-21, default off: MICRODUCK_RAMP_R9C=1 with R9 + R9B, microduck_skate_seq_ramp_env_cfg._add_r9c) ───────
# r9b's mid-push spawns were IK stances that are not viable starting states
# (every key held at its own qpos fell in < 0.5 s, tilt 30-50 deg by 0.25 s; the r9b mini-runs made 0
# strokes). r9c = the same reverse curriculum from REAL mid-stroke states: the push specialist (p16 model_5500, 1.0 A,
# datasheet speed) rolling on the flat floor, recorded relative to the deck (-> bank_strokes.npz,
# phases POST (the step-on swing after the last stroke) / CONTACT / PRE (foot coming down) / AIR (between strokes),
# only states the push policy itself recovered from, right knee <= knee_max), transplanted onto the half-pipe flat
# heading at a wall. Plus "a failed attempt must not cost more than refusing":
#   reset_r9c_bank     replaces reset_r9b_push (same flags / bookkeeping): per-phase ADAPTIVE weights (the push
#                      specialist's bridge ladder: w = base * (floor + 4 p (1 - p)), p = decayed spawn-stroke success of
#                      that phase), a POST / AIR state keeps its floor touch (r9c_touch0: the step-back-on completes it)
#   _r9c_post          spawn outcome (folded into the per-phase counts when the spawn's first flat visit ends), the
#                      push-impulse bookkeeping (r9c_push) and two windows, both only while the board is on / near the
#                      flat (|x| < L/2 + near_m) and tail_s after:
#                        r9c_att  the spawn's first visit or an own attempt -> the head style costs are off
#                        r9c_own  an OWN attempt: the push sole off the deck after a lift-off in the push zone (not a
#                                 spawn's own lifted foot: r9c_spawn_lift), or a LEGAL push-foot floor contact (r9_allow)
#   r9c_fall           seq_fall with a fall inside the PREVIOUS step's r9c_own window (the terminating contact cannot open
#                      its own discount) priced att_lo -> 1 with r9b_stroke_gate's progress, ramped to 1 over the
#                      ramp_its iterations before it_end (the stroke bonus' hard end)
#   r9c_push           the stroke impulse (forward system speed energy over legal push-foot floor contacts in the zone):
#                      losses (braking) are charged at once, gains are HELD and paid only when the push sole is back on
#                      the deck without a termination (review r9c B1: a planted-foot dive / board kick raised the system
#                      energy and was paid as insurance for a fall)
#   r9c_unless_attempt wraps a head style cost (r7_head / r6_head / r5_head_hold): 0 inside r9c_att
R9C_BANK = str(_data_path("bank_strokes.npz"))
# reverse-curriculum RUNGS by the recorded time to the step-on (tau_on, s): t20 = the step-on swing, t35 = the last
# stroke's late contact / lift-off, ... t300 = early strokes (+ bank stats: the knee filter left no
# contact states between tau 0.2 and 0.7 s when the rungs were the phases - a gap in the ladder)
R9C_RUNG_EDGES = (0.0, 0.2, 0.35, 0.5, 0.7, 1.0, 3.0)
R9C_PHASES = ("t20", "t35", "t50", "t70", "t100", "t300")       # rung names (kept as R9C_PHASES: loggers / cfg)
R9C_BANK_PHASES = ("post", "contact", "pre", "air")             # phase codes
R9C_FIELDS = ("rel_pos", "rel_quat", "rel_lin", "rel_ang", "jpos", "jvel", "b_z", "b_tilt", "b_lin", "b_ang", "bj", "bjv",
              "vb", "phase")
R9C_SN_DECAY = 1.0 - 1.0 / (24 * 50)       # per env step: ~50-iteration memory of the per-phase spawn success


def _r9c_bank(env, path: str, knee_max: float) -> tuple:
    """Cached (fields, rows per rung) of a bank: ok states (the push policy recovered) with the right
    knee <= knee_max (rad; its normal side is <= 0) and a step-on ahead, binned by tau_on (R9C_RUNG_EDGES)."""
    c = tk._tc(env)
    key = (path, float(knee_max))
    if c.get("r9c_bank_key") != key:
        import numpy as np
        z = np.load(path)
        assert [str(x) for x in z["joint_names"]] == list(sq._robot(env).joint_names), "r9c bank: robot joint order"
        assert [str(x) for x in z["board_joint_names"]] == list(tk._board(env).joint_names), "r9c bank: board joint order"
        m = (z["ok"] > 0.5) & (z["knee_r"] <= knee_max) & (z["tau_on"] >= 0)
        dev = env.device
        c["r9c_bank"] = {f: torch.tensor(z[f][m], device=dev, dtype=torch.float) for f in R9C_FIELDS}
        tau = torch.tensor(z["tau_on"][m], device=dev, dtype=torch.float)
        e = R9C_RUNG_EDGES
        c["r9c_rows"] = [torch.nonzero((tau >= e[p]) & (tau < e[p + 1])).squeeze(1) for p in range(len(R9C_PHASES))]
        c["r9c_bank_key"] = key
        print(f"[r9c] bank {path}: {int(m.sum())} / {len(m)} states, per phase "
              f"{dict(zip(R9C_PHASES, [len(r) for r in c['r9c_rows']]))}", flush=True)
    return c["r9c_bank"], c["r9c_rows"]


def _r9c_sn(env) -> torch.Tensor:
    """(phases, 2) decayed [successes, spawns] on the device (GPU-only folding, no host sync per step)."""
    c = tk._tc(env)
    if "r9c_sn" not in c:
        c["r9c_sn"] = torch.zeros(len(R9C_PHASES), 2, device=env.device)
    return c["r9c_sn"]


def _r9c_fold(env, key: torch.Tensor, done: torch.Tensor, ok: torch.Tensor) -> None:
    sn = _r9c_sn(env)
    k = key.clamp(min=0)
    m = (done & (key >= 0)).float()
    sn[:, 0].index_add_(0, k, m * ok.float())
    sn[:, 1].index_add_(0, k, m)


def r9c_phase_p(env, k: int) -> float:
    s, n = _r9c_sn(env)[k].tolist()
    return (s + 1.0) / (n + 2.0)


def _r9c_post(env, st, first: torch.Tensor, on_flat: torch.Tensor, tail_s: float = 0.5, near_m: float = 0.05,
              lift_s: float = 0.1, drop_m: float = 0.01) -> None:
    """r9b's _r9b_post hook (r9c on): the spawn's outcome when its first flat visit ends (1 = the stroke completed: both
    soles back on the deck at the wall edge, 2 = not; folded into the per-phase counts at once), the push impulse
    bookkeeping and the r9c_att / r9c_own windows."""
    from micro_skater.tasks import mdp as _mdp

    ended = first & ~on_flat & (st["r9c_res"] == 0)
    st["r9c_res"] = torch.where(ended, torch.where(st["r9b_spawn_evt"], 1, 2), st["r9c_res"])
    sn = _r9c_sn(env)
    sn.mul_(R9C_SN_DECAY)
    _r9c_fold(env, st["r9b_key"], ended, st["r9b_spawn_evt"])
    _, rf = sq._feet_floor(env)
    rd = tk._contact_any(env, "right_deck")
    legal = rf & st["r9_allow"]                   # a push-foot floor contact the terminations allow (r9b stall margin)
    # push impulse: energy of the FORWARD system speed (only floor forces change it on the flat: the push foot, the
    # wheels), capped at R9_H_TGT, in cm of wall height, over the steps with a legal floor contact in the zone (incl. the
    # touchdown / lift-off steps). Losses charged now (r9c_push_neg), gains held (r9c_pend) until the sole is back on
    # the deck (r9c_release; the reward drops them on a termination).
    vf = torch.nan_to_num(_mdp.skate_system_fwd_speed(env)).clamp(min=0.0)
    phi = (0.5 * vf * vf).clamp(max=G * R9_H_TGT) / (G * 0.01)
    cz = legal & st["r9_zone"]
    win = (cz | st["r9c_cz_prev"]) & torch.isfinite(st["r9c_phi"])
    d = torch.where(win, phi - torch.nan_to_num(st["r9c_phi"]), torch.zeros_like(phi))
    st["r9c_push_neg"] = d.clamp(max=0.0)
    st["r9c_gain_now"] = d.clamp(min=0.0)
    st["r9c_pend"] = st["r9c_pend"] + d.clamp(min=0.0)
    on2 = tk.soles_on_deck(env)                   # re-review S3: BOTH soles back on (not a clip of the deck in a tumble)
    st["r9c_release"] = on2 & ~st["r9c_rd_prev"]
    st["r9c_rd_prev"] = on2
    st["r9c_phi"] = phi
    st["r9c_cz_prev"] = cz
    # windows (review r9c B2 / should-fix 1-2): a bank spawn's own lifted foot is not an own attempt; only legal contacts;
    # only on / near the flat
    st["r9c_spawn_lift"] = st["r9c_spawn_lift"] & ~rd
    L2 = st["pipe"]["L"] / 2
    p = torch.nan_to_num(sq.pipe_xyz(env), nan=10.0)
    near = (st["stage"] == sq.ST_RAMP) & sq.in_pipe(env, 0.0) & (p[:, 0].abs() < L2 + near_m)
    # re-review S1: a lift-off counts as an attempt only once the sole has been off the deck >= lift_s AND is >= drop_m
    # below the board sole (sole chatter / a one-step unweighting does not open the window)
    ids = _r9_ids(env)
    gp = sq._robot(env).data.geom_pos_w
    drop = torch.nan_to_num(gp[:, ids["ls"], 2] - gp[:, ids["rs"], 2])
    lifted = st["r9_lift_zone"] & ~st["r9c_spawn_lift"] & (_mdp._skate_air_time(env, "right_deck") > lift_s) & (drop >= drop_m)
    own = (lifted | legal) & ~first & near
    now = (st["r9b_first"] & near) | own
    tail = int(round(tail_s / env.step_dt))
    st["r9c_own_fall"] = st["r9c_own"].clone()    # the previous step's own window prices this step's fall
    st["r9c_att_k"] = torch.where(now, torch.zeros_like(st["r9c_att_k"]), st["r9c_att_k"] + 1)
    st["r9c_att"] = (st["r9c_att_k"] <= tail) & near
    st["r9c_att_any"] = (st["r9c_att_k"] <= tail) & (st["stage"] == sq.ST_RAMP)     # r9c_trunk_low (no `near` cliff)
    st["r9c_own_k"] = torch.where(own, torch.zeros_like(st["r9c_own_k"]), st["r9c_own_k"] + 1)
    st["r9c_own"] = (st["r9c_own_k"] <= tail) & near


def _r9c_att(env, key: str = "r9c_att") -> torch.Tensor:
    st = r9_update(env)
    return st[key] if key in st else torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def reset_r9c_bank(env, env_ids: torch.Tensor, share_stages: list, phase_w: dict, bank_path: str = R9C_BANK,
                   knee_max: float = 0.3, x_along: tuple = (-0.45, 0.0), yaw_deg: float = 12.0, pos_noise: float = 0.003,
                   joint_noise: float = 0.015, adaptive: dict | None = None, v_scale: tuple | None = None) -> None:
    """Reset event (r9c; replaces reset_r9b_push right after reset_sequence). A share of the ramp_bottom spawns
    (share_stages as r9b) restores a recorded p16 mid-stroke state (_r9c_bank) on the half-pipe flat: board centre at
    x_along m from the flat's centre along the travel, heading +-x (+- yaw_deg) at the wall ahead, at its recorded speed;
    the rider / board twists / joints as recorded relative to the deck. Phase drawn by phase_w (name -> base weight),
    adaptive {"floor"} (None = fixed): w = base * (floor + 4 p (1 - p)) with p the phase's decayed spawn-stroke success
    (R9C_SN_DECAY per env step). v_scale (lo, hi) (None = as recorded): every velocity of the state (board / robot
    twists, all joint velocities) times U(lo, hi) - a slower copy of the same stroke (the planted foot stays still; a
    slow push needs a slower hip sweep). Flags: r9b_* as reset_r9b_push, r9c_touch0 (POST / AIR states: the floor touch already
    happened)."""
    from mjlab.utils.lab_api.math import quat_mul
    from micro_skater.tasks.mdp import _quat_from_rpy_t

    st = sq._st(env)
    n_all, dev = env.num_envs, env.device
    if "r9b_fresh" not in st:
        z = lambda: torch.zeros(n_all, dtype=torch.bool, device=dev)
        st.update(r9b_fresh=z(), r9b_spawn=z(), r9b_first=z(), r9b_own_evt=z(), r9b_spawn_evt=z(),
                  r9b_key=torch.full((n_all,), -1, dtype=torch.long, device=dev))
    if "r9c_res" not in st:
        b = lambda: torch.zeros(n_all, dtype=torch.bool, device=dev)
        big = lambda: torch.full((n_all,), 10 ** 6, dtype=torch.long, device=dev)
        st.update(r9c_touch0=b(), r9c_res=torch.full((n_all,), -1, dtype=torch.long, device=dev),
                  r9c_att_k=big(), r9c_att=b(), r9c_own_k=big(), r9c_own=b(), r9c_own_fall=b(), r9c_spawn_lift=b(),
                  r9c_phi=torch.full((n_all,), float("nan"), device=dev), r9c_cz_prev=b(), r9c_rd_prev=b(),
                  r9c_release=b(), r9c_pend=torch.zeros(n_all, device=dev), r9c_push_neg=torch.zeros(n_all, device=dev),
                  r9c_paid=torch.zeros(n_all, device=dev), r9c_buf=torch.zeros(n_all, device=dev),
                  r9c_rel_k=torch.full((n_all,), -1, dtype=torch.long, device=dev), r9c_att_any=b(), r9c_low_ok=b())
    if env_ids is None:
        env_ids = torch.arange(n_all, device=dev)
    env_ids = env_ids.to(dev).long()
    if len(env_ids) == 0:
        return
    # episodes that ended during their spawn's first visit (outcome still pending): failures
    _r9c_fold(env, st["r9b_key"][env_ids], st["r9c_res"][env_ids] == 0, torch.zeros_like(env_ids, dtype=torch.bool))
    st["r9b_fresh"][env_ids] = True
    st["r9b_spawn"][env_ids] = False
    st["r9b_first"][env_ids] = False
    st["r9b_key"][env_ids] = -1
    for k in ("r9c_touch0", "r9c_att", "r9c_own", "r9c_own_fall", "r9c_spawn_lift", "r9c_cz_prev", "r9c_rd_prev",
              "r9c_release", "r9c_att_any", "r9c_low_ok"):
        st[k][env_ids] = False
    st["r9c_res"][env_ids] = -1
    st["r9c_att_k"][env_ids] = 10 ** 6
    st["r9c_own_k"][env_ids] = 10 ** 6
    st["r9c_phi"][env_ids] = float("nan")
    st["r9c_pend"][env_ids] = 0.0
    st["r9c_buf"][env_ids] = 0.0
    st["r9c_rel_k"][env_ids] = -1
    st["r9c_push_neg"][env_ids] = 0.0
    share = 0.0
    for step, s in share_stages:
        if env.common_step_counter >= step:
            share = float(s)
    n = len(env_ids)
    sel = (st["spawn"][env_ids] == sq.SPAWNS.index("ramp_bottom")) & (torch.rand(n, device=dev) < share)
    if not bool(sel.any()):
        return
    bank, rows = _r9c_bank(env, bank_path, knee_max)
    w = [float(phase_w.get(nm, 0.0)) if len(rows[p]) else 0.0 for p, nm in enumerate(R9C_PHASES)]
    if adaptive:
        fl = float(adaptive.get("floor", 0.1))
        sn = _r9c_sn(env).tolist()
        pp = [(s + 1.0) / (nn + 2.0) for s, nn in sn]
        w = [wi * (fl + 4.0 * pp[p] * (1.0 - pp[p])) for p, wi in enumerate(w)]
    assert sum(w) > 0, "r9c: no bank phase with weight"
    ids = env_ids[sel]
    m = len(ids)
    ph = torch.multinomial(torch.tensor(w, device=dev), m, replacement=True)
    pick = torch.zeros(m, dtype=torch.long, device=dev)
    for p in range(len(R9C_PHASES)):
        mp = ph == p
        if bool(mp.any()):
            r = rows[p]
            pick[mp] = r[torch.randint(len(r), (int(mp.sum()),), device=dev)]
    g = {f: v[pick] for f, v in bank.items()}
    U = lambda lo, hi: lo + torch.rand(m, device=dev) * (hi - lo)
    if v_scale is not None:
        vs = U(*v_scale).unsqueeze(1)
        for f in ("b_lin", "b_ang", "bjv", "rel_lin", "rel_ang", "jvel"):
            g[f] = g[f] * vs
        g["vb"] = g["vb"] * vs.squeeze(1)
    W = st["pipe"]["W"]
    side = torch.where(torch.rand(m, device=dev) < 0.5, -1.0, 1.0)
    bx = side * U(*x_along)
    by = U(-W / 2 + 1.0, W / 2 - 1.5)
    zero = torch.zeros(m, device=dev)
    qy = _quat_from_rpy_t(zero, zero, torch.where(side > 0, 0.0, math.pi) + torch.deg2rad(U(-yaw_deg, yaw_deg)))
    nq = lambda q: q / q.norm(dim=1, keepdim=True)
    qb = nq(quat_mul(qy, g["b_tilt"]))
    bpos = sq._origin(env)[ids] + torch.stack([bx, by, g["b_z"]], dim=1)
    board, robot = tk._board(env), sq._robot(env)
    board.write_root_link_pose_to_sim(torch.cat([bpos, qb], dim=1), ids)
    board.write_root_link_velocity_to_sim(torch.cat([quat_apply(qb, g["b_lin"]), quat_apply(qb, g["b_ang"])], dim=1), ids)
    board.write_joint_state_to_sim(g["bj"], g["bjv"], env_ids=ids)
    rpos = bpos + quat_apply(qb, g["rel_pos"])
    rpos[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * pos_noise
    robot.write_root_link_pose_to_sim(torch.cat([rpos, nq(quat_mul(qb, g["rel_quat"]))], dim=1), ids)
    robot.write_root_link_velocity_to_sim(torch.cat([quat_apply(qb, g["rel_lin"]), quat_apply(qb, g["rel_ang"])], dim=1),
                                          ids)
    robot.write_joint_state_to_sim(g["jpos"] + torch.randn_like(g["jpos"]) * joint_noise, g["jvel"], env_ids=ids)
    st["v0"][ids] = g["vb"]
    st["r9b_spawn"][ids] = True
    st["r9b_key"][ids] = ph
    bp = g["phase"].round().long()                 # the recorded phase: POST / AIR states are past a floor touch
    st["r9c_touch0"][ids] = (bp == R9C_BANK_PHASES.index("post")) | (bp == R9C_BANK_PHASES.index("air"))
    st["r9c_res"][ids] = 0
    st["r9c_att_k"][ids] = 0


def r9c_fall(env, att_lo: float = 0.3, manual_scale: float = 0.6, it_end: int = 10 ** 9,
             ramp_its: int = 100) -> torch.Tensor:
    """sq.seq_fall (NEGATIVE weight, the cost of one fall) with a fall inside an OWN attempt window (the previous step's
    r9c_own: not the spawn's first visit or its own lifted foot - a cheap spawn failure taught quitting:
 trunk_low 0.3-1.1 /it) priced att_lo + (1 - att_lo) * prog, prog = max(r9b_stroke_gate's
    progress, the linear ramp over the ramp_its iterations before it_end): a stroke attempt that fails must not cost more
    than refusing it while the skill is discovered; full price once it exists, and from it_end regardless."""
    base = sq.seq_fall(env, manual_scale)
    prog = float(tk._tc(env).get("r9b_prog", 0.0))
    it = env.common_step_counter // 24
    prog = max(prog, min(max((it - (it_end - ramp_its)) / max(ramp_its, 1), 0.0), 1.0))
    s = att_lo + (1.0 - att_lo) * min(max(prog, 0.0), 1.0)
    return torch.where(_r9c_att(env, "r9c_own_fall"), base * s, base)


def r9c_unless_attempt(env, func: str, params: dict | None = None) -> torch.Tensor:
    """Wrapper (r9c): the head style cost `func` (a function of this module) with 0 inside the r9c_att window (a stroke
    needs the head as counterweight; no attempt tax while the stroke is discovered). Near the flat only."""
    v = globals()[func](env, **(params or {}))
    return torch.where(_r9c_att(env), torch.zeros_like(v), v)


def r9c_phase_log(env, env_ids, phase: str) -> torch.Tensor:
    """Curriculum logger only: the decayed spawn-stroke success estimate of a bank phase."""
    del env_ids
    return torch.tensor([r9c_phase_p(env, R9C_PHASES.index(phase))])


def r9c_prog_log(env, env_ids) -> torch.Tensor:
    """Curriculum logger only: r9b_stroke_gate's progress (drives r9c_fall's attempt price)."""
    del env_ids
    return torch.tensor([float(tk._tc(env).get("r9b_prog", 0.0))])


def m_r9c_att(env) -> torch.Tensor:
    """r9c: steps inside the attempt window (head style costs off)."""
    return _r9c_att(env).float()


def m_r9c_own(env) -> torch.Tensor:
    """r9c: steps inside an OWN attempt window (discounted fall)."""
    return _r9c_att(env, "r9c_own").float()


def r9c_push(env, release_steps: int = 25) -> torch.Tensor:
    """Dense (POSITIVE weight, per cm of wall height): the stroke impulse (_r9c_post). Losses (a dragging / braking push
    foot) are charged at once; gains are held and paid release_steps after both soles are back on the deck if no
    termination came in between, dropped on a termination (no pay for a dive over the planted foot). Bounded by the energy actually
    gained (capped at R9_H_TGT): a brake-then-push round trip nets <= 0. Called once per step (reward manager)."""
    st = r9_update(env)
    if "r9c_pend" not in st:
        return torch.zeros(env.num_envs, device=env.device)
    term = env.termination_manager.terminated
    rel = st["r9c_release"]
    zero = torch.zeros_like(st["r9c_pend"])
    # re-review S3: the held gain moves to a buffer at the step-back-on and is paid release_steps later if no termination
    # came in between (a tumble that clips the deck pays nothing)
    st["r9c_buf"] = st["r9c_buf"] + torch.where(rel, st["r9c_pend"], zero)
    st["r9c_pend"] = torch.where(rel | term, zero, st["r9c_pend"])
    st["r9c_rel_k"] = torch.where(rel, torch.full_like(st["r9c_rel_k"], release_steps), (st["r9c_rel_k"] - 1).clamp(min=-1))
    pay = (st["r9c_rel_k"] == 0) & ~term
    paid = st["r9c_push_neg"] + torch.where(pay, st["r9c_buf"], zero)
    st["r9c_buf"] = torch.where(pay | term, zero, st["r9c_buf"])
    st["r9c_paid"] = paid
    return paid / env.step_dt


def m_r9c_push(env) -> torch.Tensor:
    """r9c: cm of wall height paid by r9c_push on the last step (sum = net per episode)."""
    st = sq._st(env)
    return st["r9c_paid"] if "r9c_paid" in st else torch.zeros(env.num_envs, device=env.device)


def m_r9c_gain_cm(env) -> torch.Tensor:
    """r9c: cm of wall height GAINED by legal push-foot floor contacts on this step (raw, before the hold / release)."""
    st = r9_update(env)
    return st["r9c_gain_now"] if "r9c_gain_now" in st else torch.zeros(env.num_envs, device=env.device)


def r9c_trunk_low(env, minimum_height: float = 0.083, att_height: float = 0.072) -> torch.Tensor:
    """trunk_low (root world z above the env origin < minimum_height) with att_height inside the r9c_att window (near
    the flat: a spawn's first visit / an own attempt).: the t35..t100 bank spawns ended in 0.26-0.35
    s, 12-21 / 32 of them on trunk_low with the root minimum at 0.083-0.085 (p16 strokes run at 0.096-0.12; the riding
    root sits ~0.10, 1.7 cm over the limit): the reach to the floor 2 cm below the deck was terminated before the
    policy could learn to stand tall on the board leg. r6_trunk_band still prices the crouch."""
    c = tk._tc(env)
    if not c.get("r9c_origin_ok"):                 # re-review nit: the original term reads world z
        assert float(sq._origin(env)[:, 2].abs().max()) < 1e-6, "r9c_trunk_low: env origins not at z = 0"
        c["r9c_origin_ok"] = True
    st = r9_update(env)
    z = torch.nan_to_num(sq._robot(env).data.root_link_pos_w[:, 2] - sq._origin(env)[:, 2], nan=1.0)
    if "r9c_low_ok" not in st:
        return z < minimum_height
    # re-review S2: hysteresis - the relaxed limit holds while the window is open (no `near` cliff at the wall entry)
    # and after it until the trunk has been back above minimum_height once
    st["r9c_low_ok"] = st["r9c_att_any"] | (st["r9c_low_ok"] & (z < minimum_height))
    lim = torch.where(st["r9c_low_ok"], torch.full_like(z, att_height), torch.full_like(z, minimum_height))
    return z < lim


def m_r9c_own_ride(env) -> torch.Tensor:
    """r9c: steps inside an OWN attempt window in episodes that did NOT start from a bank spawn (riding episodes)."""
    st = r9_update(env)
    if "r9c_own" not in st:
        return torch.zeros(env.num_envs, device=env.device)
    return (st["r9c_own"] & ~st["r9b_spawn"]).float()



# ── r10 (2026-11-21, default off: MICRODUCK_RAMP_R10=1 in microduck_skate_seq_ramp_env_cfg._add_r10) ──────────────────────
# After a chain STROKE hand-back (p17d foot-push) the board
# climbs the next wall NOSE-FIRST at 0.4-0.6 m/s - a state the r9 chain almost never produced (baseline nose-first visits
# at >= 0.4 m/s are rare) - and r9 folds into a joint-limit squat at the turn-around (knees / ankles +-90 deg, head
# pitch +90, neck -70; head camera 3.5-8 cm over the deck vs ~21 cm riding) in 5 / 40 such visits.
# r10 = r9 fine-tune on recorded hand-back states (reset_r10_bank) + realistic-posture costs:
#   r10_head_low   head-camera height above the deck top (deck normal) below `free` m, squared per unit, capped
#   r10_jlim       qpos-side joint-limit proximity / violation of the 10 leg joints + neck / head pitch (beyond
#                  limit - margin, squared per unit_deg, capped); covers the squat (+-90 deg) and the hardware audit's
#                  hip roll / yaw pressed past the MJCF stop
#   r10_hy_heat    the two hip yaws' 2 s low-passed copper loss above p_free (W; skate_heat_cost's own low-pass): the
#                  audit's sustained +-1.0 A hip-yaw squeeze (~99 % of ramp-owner time) on top of the all-servo skate_heat
R10_HEAD_FREE = 0.15          # m: head camera above the deck top (riding median 0.212, p1 0.165, p0.1 0.137)
R10_JLIM_JOINTS = ("left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle", "neck_pitch",
                   "head_pitch", "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle")
R10_JLIM_MARGIN_DEG = dict(hip_yaw=4.0, hip_roll=4.0, hip_pitch=10.0, knee=10.0, ankle=10.0, neck_pitch=10.0,
                           head_pitch=10.0)
# unit multiplier per joint: the hip yaw / roll stops are priced 4x steeper (unit 2.5 deg) so parking a hip yaw on its
# MJCF stop (the stop carries the board squeeze, saving hip-yaw current) costs more than the r10_hy_heat it saves
R10_JLIM_UNIT_SCALE = dict(hip_yaw=0.5, hip_roll=0.5)


def _r10_ids(env) -> dict:
    """Cached: head-camera / deck-top site ids, the r10_jlim joint ids with their MJCF ranges (rad) and margins, the
    hip-yaw indices in the BAM servo order (skate_heat_cost's columns)."""
    c = tk._tc(env)
    if "r10_ids" not in c:
        import mujoco
        from micro_skater.tasks import mdp as _mdp
        robot = sq._robot(env)
        board = tk._board(env)
        hs = robot.find_sites(r"^head_camera$")[0]
        ds = board.find_sites(r"^deck_top$")[0]
        assert len(hs) == 1 and len(ds) == 1, f"r10: sites {hs} {ds}"
        m = env.sim.mj_model
        ids, lo, hi, mg, us = [], [], [], [], []
        for j in R10_JLIM_JOINTS:
            jj = robot.find_joints(rf"^{j}$")[0]
            assert len(jj) == 1, f"r10: joint {j}"
            jid = next(i for i in (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm) for nm in (f"robot/{j}", j)) if i >= 0)
            assert bool(m.jnt_limited[jid]), f"r10: {j} has no range"
            ids.append(jj[0])
            lo.append(float(m.jnt_range[jid][0]))
            hi.append(float(m.jnt_range[jid][1]))
            key = j.replace("left_", "").replace("right_", "")
            mg.append(math.radians(R10_JLIM_MARGIN_DEG[key]))
            us.append(float(R10_JLIM_UNIT_SCALE.get(key, 1.0)))
        names = []
        for a in _mdp._bam_heat_actuators(env):
            names += [str(x).split("/")[-1] for x in a._target_names]
        assert len(names) == 14, f"r10: BAM servo names {names}"
        hy = [names.index("left_hip_yaw"), names.index("right_hip_yaw")]
        dev = env.device
        t = lambda x: torch.tensor(x, device=dev, dtype=torch.float)
        c["r10_ids"] = dict(hs=hs[0], ds=ds[0], j=torch.tensor(ids, device=dev, dtype=torch.long), lo=t(lo), hi=t(hi),
                            mg=t(mg), us=t(us), hy=torch.tensor(hy, device=dev, dtype=torch.long), servo_names=names)
    return c["r10_ids"]


def r10_head_h(env) -> torch.Tensor:
    """(B,) head-camera site height above the board's deck-top site along the deck normal (m) = the squat detector's
    measure."""
    ids = _r10_ids(env)
    robot, board = sq._robot(env), tk._board(env)
    hp = robot.data.site_pos_w[:, ids["hs"]]
    dp = board.data.site_pos_w[:, ids["ds"]]
    nz = quat_apply(board.data.site_quat_w[:, ids["ds"]], torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(env.num_envs, 3))
    return torch.nan_to_num(((hp - dp) * nz).sum(dim=1), nan=R10_HEAD_FREE)


def r10_head_low(env, free: float = R10_HEAD_FREE, unit: float = 0.02, cap: float = 9.0,
                 stages: tuple = (sq.ST_EXIT, sq.ST_RAMP)) -> torch.Tensor:
    """Cost (EXIT / RAMP, >= 0, NEGATIVE weight): ((free - head height) / unit)^2 below free, capped."""
    c = ((free - r10_head_h(env)) / unit).clamp(min=0.0) ** 2
    return torch.nan_to_num(c.clamp(max=cap) * _stage_in(env, stages).float())


def r10_jlim_raw(env, unit_deg: float = 5.0) -> torch.Tensor:
    """(B, 12) per-joint ((q - (hi - margin)) / unit)^2 above / ((lo + margin - q) / unit)^2 below (qpos side)."""
    ids = _r10_ids(env)
    q = sq._robot(env).data.joint_pos[:, ids["j"]]
    u = math.radians(unit_deg) * ids["us"]
    up = ((q - (ids["hi"] - ids["mg"])) / u).clamp(min=0.0)
    dn = (((ids["lo"] + ids["mg"]) - q) / u).clamp(min=0.0)
    return torch.nan_to_num(up * up + dn * dn)


def r10_jlim(env, unit_deg: float = 5.0, cap: float = 10.0, stages: tuple = (sq.ST_EXIT, sq.ST_RAMP)) -> torch.Tensor:
    """Cost (EXIT / RAMP, >= 0, NEGATIVE weight): sum of r10_jlim_raw, capped. A joint at its MJCF limit costs
    (margin / unit)^2 (knee / ankle / hip pitch / neck / head pitch: 4; hip yaw / roll: 0.64), past it more."""
    c = r10_jlim_raw(env, unit_deg).sum(dim=1).clamp(max=cap)
    return c * _stage_in(env, stages).float()


def r10_hy_heat(env, p_free: float = 0.65) -> torch.Tensor:
    """Cost (>= 0, NEGATIVE weight, every stage like skate_heat): sum over the two hip yaws of relu(2 s low-passed
    I^2 R0 - p_free) (W), from skate_heat_cost's own low-pass (updated here first: order-independent)."""
    from micro_skater.tasks import mdp as _mdp
    _mdp.skate_heat_cost(env, cap_w=None)
    ema = env.__dict__["_skate_heat"]["ema"]
    hy = _r10_ids(env)["hy"]
    return torch.nan_to_num((ema[:, hy] - p_free).clamp(min=0.0).sum(dim=1))


def _r10_hy_err(env) -> torch.Tensor:
    """(B, 2) |position target - joint position| (rad) of the two hip yaws (robot joint order, not the servo order)."""
    c = tk._tc(env)
    if "r10_hy_joints" not in c:
        jj = sq._robot(env).find_joints(r"^(left|right)_hip_yaw$")[0]
        assert len(jj) == 2, f"r10: hip yaw joints {jj}"
        c["r10_hy_joints"] = torch.tensor(list(jj), device=env.device, dtype=torch.long)
    d = sq._robot(env).data
    j = c["r10_hy_joints"]
    return torch.nan_to_num((d.joint_pos_target[:, j] - d.joint_pos[:, j]).abs())


def r10_hy_err(env, free: float = 0.30, unit: float = 0.50, cap: float = 30.0) -> torch.Tensor:
    """Cost (>= 0, NEGATIVE weight, every stage; MICRODUCK_RAMP_R10_HYERR_W, default 0 = not registered): the hip-yaw
    squeeze. r9 holds both hip yaws at the 1.0 A clamp ~99 % of the time; the current saturates, so the heat cost
    (I^2 R) is FLAT there and gives no gradient (r10 mini: hip yaws at the limit 0.97 unchanged after 100 its). The
    XL330 duty = kp_fw * error_gain * err = 200 * 4096 / (2 pi * 256 * 885) * err = 0.575 err, static current at 7.4 V
    ~1.5 A per rad: 1.0 A <=> err ~0.66 rad, 0.45 A (the heat cost's 0.65 W) <=> ~0.30 rad. Charge the target error
    above `free` linearly per `unit` (summed over both hip yaws, capped): a gradient all the way through the clamp,
    small overshoot (< free) free (low-kp servos need overshoot). Measured (r9 model_4500):
    mean |err| 4.4 rad - the hip-yaw TARGETS sit far outside the joint range (wide ctrlrange), so unit 0.5 / cap 30
    (~16 now, linear down to 0 at `free`): never flat at the current behaviour."""
    return (((_r10_hy_err(env) - free) / unit).clamp(min=0.0).sum(dim=1)).clamp(max=cap)


def m_r10_hy_err(env) -> torch.Tensor:
    """r10: mean |target - q| of the two hip yaws (rad)."""
    return _r10_hy_err(env).mean(dim=1)


def m_r10_head_h(env) -> torch.Tensor:
    """r10: head height (m) in RAMP (0 elsewhere): / seq_ramp = mean."""
    return r10_head_h(env) * (sq.seq_update(env)["stage"] == sq.ST_RAMP).float()


def m_r10_squat(env) -> torch.Tensor:
    """r10: RAMP steps with the head camera < 10 cm over the deck (the squat detector's threshold)."""
    return ((r10_head_h(env) < 0.10) & (sq.seq_update(env)["stage"] == sq.ST_RAMP)).float()


def m_r10_jlim(env) -> torch.Tensor:
    """r10: raw (uncapped, ungated) joint-limit proximity cost."""
    return r10_jlim_raw(env).sum(dim=1)


def m_r10_over(env) -> torch.Tensor:
    """r10: steps with any r10 joint more than 0.5 deg past its MJCF limit."""
    ids = _r10_ids(env)
    q = sq._robot(env).data.joint_pos[:, ids["j"]]
    e = math.radians(0.5)
    return ((q > ids["hi"] + e) | (q < ids["lo"] - e)).any(dim=1).float()


def m_r10_hy_near(env) -> torch.Tensor:
    """r10: steps with a hip yaw within 1 deg of (or past) its MJCF limit (the stop carrying load)."""
    ids = _r10_ids(env)
    k = [R10_JLIM_JOINTS.index("left_hip_yaw"), R10_JLIM_JOINTS.index("right_hip_yaw")]
    q = sq._robot(env).data.joint_pos[:, ids["j"][k]]
    e = math.radians(1.0)
    return ((q > ids["hi"][k] - e) | (q < ids["lo"][k] + e)).any(dim=1).float()


def m_r10_hy_amp(env) -> torch.Tensor:
    """r10: steps with a hip yaw at the current limit (step RMS current >= 0.95 A)."""
    from micro_skater.tasks import mdp as _mdp
    i2 = _mdp.servo_heat_state(env)["i2"]
    return (i2[:, _r10_ids(env)["hy"]].sqrt() >= 0.95).any(dim=1).float()


def _r10_bank(env, path: str) -> dict:
    """Cached {field: tensor} of a bank (kind "ramp", sq.RAMP_HANDOFF_FIELDS; own cache key: the
    rampbank's _handoff_bank cache holds the EXIT hand-off bank)."""
    c = tk._tc(env)
    if c.get("r10_bank_path") != path:
        import numpy as np
        z = np.load(path)
        assert str(z["kind"]) == "ramp", f"r10 bank {path}: kind {z['kind']}"
        assert [str(x) for x in z["joint_names"]] == list(sq._robot(env).joint_names), "r10 bank: robot joint order"
        assert [str(x) for x in z["board_joint_names"]] == list(tk._board(env).joint_names), "r10 bank: board joint order"
        c["r10_bank"] = {f: torch.tensor(z[f], device=env.device, dtype=torch.float) for f in sq.RAMP_HANDOFF_FIELDS}
        assert bool((c["r10_bank"]["stage"].round() == sq.ST_RAMP).all()), "r10 bank: every row must be in RAMP"
        c["r10_bank_path"] = path
    return c["r10_bank"]


def reset_r10_bank(env, env_ids: torch.Tensor, path: str, share: float, noise: dict | None = None) -> None:
    """Reset event (r10, right after reset_sequence): a share of the spawns (any type) restore a recorded chain STROKE
    hand-back state (_r10_bank) with the rampbank noise (xy shift, yaw turn about the board origin, joint noise, a speed
    offset dv as one velocity scale) - the same writes as sq._ramp_handoff_reset - and the recorded scheduler state
    (stage RAMP, stage_t, timers, clock). Flagged st["handoff"] (the tilt guard is live from the first step, as in the
    chain) and st["r10_bank"]."""
    from mjlab.utils.lab_api.math import quat_mul
    st = sq._st(env)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = env_ids.to(env.device).long()
    n, dev = len(env_ids), env.device
    for k in ("handoff", "r10_bank"):
        if k not in st:
            st[k] = torch.zeros(env.num_envs, dtype=torch.bool, device=dev)
    sel = torch.rand(n, device=dev) < float(share)
    st["r10_bank"][env_ids] = sel
    if not bool(sel.any()):
        return
    nz = {**sq.RAMP_HANDOFF_NOISE, **(noise or {})}
    hb = _r10_bank(env, path)
    m = int(sel.sum())
    pick = torch.randint(len(hb["vb"]), (m,), device=dev)
    g = {f: v[pick] for f, v in hb.items()}
    ids = env_ids[sel]
    U = lambda s: (torch.rand(m, device=dev) * 2 - 1) * float(s)
    z0 = torch.zeros(m, device=dev)
    dq = sq._quat_from_rpy_t(z0, z0, torch.deg2rad(U(nz["yaw_deg"])))
    vs = (1.0 + U(nz["v_scale"]) + U(nz.get("dv", 0.0)) / g["vb"].abs().clamp(min=0.3)).clamp(0.5, 1.5).unsqueeze(1)
    org = sq._origin(env)[ids]
    # x (cross-pipe) noise only where both axles stay on the flat even after the shift: on the transition an x shift
    # without the matching z would sink the wheels into the wall / float them (review item 1)
    flat = (g["b_pos"][:, 0].abs() + sq.BOARD["axle_x"] + float(nz["xy"])) < float(st["pipe"]["L"]) / 2
    bp = g["b_pos"] + torch.stack([U(nz["xy"]) * flat.float(), U(nz["xy"]), z0], dim=1)
    nq = lambda q: q / q.norm(dim=1, keepdim=True)
    board = tk._board(env)
    robot = sq._robot(env)
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
    st["trick"][ids] = torch.tensor(sq.STAGE_PHASE, device=dev)[stg]
    if "obs_ph" in st:
        st["obs_ph"][ids] = st["trick"][ids]
    st["stage_t"][ids] = g["stage_t"]
    st["t_stage"][ids] = -g["stage_t"]
    st["both_on"][ids] = g["both_on"]
    st["wheels_on"][ids] = g["wheels_on"]
    st["entry_s"][ids] = g["entry_s"]
    st["T_exit"][ids] = g["T_exit"]
    st["v_tgt"][ids] = g["v_tgt"]
    st["t_off"][ids] = g["clock"]
    st["wz_lp"][ids] = g["wz_lp"] * vs.squeeze(1)
    st["v0"][ids] = g["vb"] * vs.squeeze(1)
    st["handoff"][ids] = True


def m_r10_bank(env) -> torch.Tensor:
    """r10: steps of episodes that started from a hand-back bank row."""
    st = sq._st(env)
    return st["r10_bank"].float() if "r10_bank" in st else torch.zeros(env.num_envs, device=env.device)
