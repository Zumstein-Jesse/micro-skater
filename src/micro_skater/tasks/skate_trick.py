# micro-skater: part of the author's skateboard add-on for microduck_rl. Apache-2.0 (see LICENSE).
"""Skateboard TRICKS on the double-kick board (2026-11-19). MDP terms for microduck_skate_trick_env_cfg.py.

The rider starts ALREADY ON the board in a SIDEWAYS stance (trunk facing deck +y, one sole over each truck; pose from
data/stance_poses.json). Nothing here uses the forward-stance skate mode
machine (mdp._skate_state); only its geometry-free primitives.

Per env, a TRICK is drawn at reset (one-hot in the obs), plus a travel direction dir = +-1 along deck x (the board is
symmetric: dir = +1 rolls toward the robot's right foot, -1 toward its left foot = regular vs fakie):
  0 ride      rolling 0.25-0.8 m/s, track a curvature command (left turn +), resampled every 2-4 s; both directions.
  1 stop      rolling; bring the board to rest (tail drag = the back end on the floor).
  2 kickturn  at rest / slow: turn the board's heading by a commanded +-90 / +-180 deg and land on four wheels.
  3 tictac    from rest: reach a target speed along dir WITHOUT touching the floor with a foot (nose pivots / pumping).
  4 manual    rolling: balance on the back wheels with the front end up 3-25 deg (some spawns start tilted).
"Front" / "back" always mean relative to dir (the end the board is travelling toward / away from).

Costs are returned >= 0 (use NEGATIVE weights); mjlab multiplies every reward by dt, so per-event costs return 1/dt.
"""

from __future__ import annotations
from micro_skater.paths import data_path as _data_path

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch

from mjlab.entity import Entity
from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, quat_mul, wrap_to_pi

from micro_skater.tasks.mdp import _quat_from_rpy_t, _skate_air_time, _skate_contact, _yaw_from_quat, _unit

TRICKS = ("ride", "stop", "kickturn", "tictac", "manual")
N_TRICK = len(TRICKS)
RIDE, STOP, KICKTURN, TICTAC, MANUAL = range(N_TRICK)
STANCE_JSON = Path(os.environ.get("MICRODUCK_TRICK_STANCE",
                                  str(_data_path("stance_poses.json"))))

# Double-kick geometry (robot/skateboard_variants.py DK_*)
DK_DECK_CENTER_Z = 0.022
DK_AXLE_X = 0.055
DK_AXLE_Z = -0.012
DK_WHEEL_RADIUS = 0.010
KT_RATE = 0.5       # rad/s, kick-turn reference slew (reward and obs); feasibility: ~13 deg per nose lift
KT_T0 = 0.5

TRICK_DEFAULTS = dict(
    kappa_range=0.5,            # 1/m, ride curvature command: the measured carve limit is ~0.39 toe / 0.50 heel
    kappa_zero_prob=0.25,       # share of ride commands that are "straight"
    kappa_resample_s=(2.0, 4.0),
    a_lat=1.2,                  # m/s^2, ride: |kappa| <= a_lat / v^2
    wz_lp_tau=0.25,             # s, low-pass of the board yaw rate for tracking
    kt_angles=(math.pi / 4, math.pi / 2),
    tictac_v=(0.15, 0.35),
)


def load_stance(key: str = "regular", path: Path = STANCE_JSON) -> dict:
    d = json.loads(Path(path).read_text())
    k = d[key]
    return {"pos": list(k["pos"]), "rpy": list(k["rpy"]), "joints": dict(k["joints"])}


# ── action: joint targets centred on the STANCE (obs joint_pos stays relative to HOME, the family convention) ──


class StanceJointPositionAction(JointPositionAction):
    """target = stance + scale * a (instead of HOME + scale * a) for the joints listed in the stance file."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        st = load_stance(cfg.stance_key, Path(cfg.stance_path or STANCE_JSON))
        names = list(self.target_names)
        off = self._offset.clone()
        for j, q in st["joints"].items():
            off[:, names.index(j)] = q
        self._offset = off


@dataclass(kw_only=True)
class StanceJointPositionActionCfg(JointPositionActionCfg):
    stance_key: str = "regular"
    stance_path: str = ""

    def build(self, env):
        return StanceJointPositionAction(self, env)


# ── state ─────────────────────────────────────────────────────────────────────────────────────────────────────────────


def _tc(env) -> dict:
    return env.__dict__.setdefault("_trick_cache", {})


def _state(env) -> dict:
    c = _tc(env)
    if "st" not in c:
        n, dev = env.num_envs, env.device
        z = lambda: torch.zeros(n, device=dev)
        c["st"] = dict(
            trick=torch.zeros(n, dtype=torch.long, device=dev), dir=torch.ones(n, device=dev), kappa=z(),
            kappa_next=z(), v_t=z(), yaw_t=z(), yaw_0=z(), kt_total=z(), wz_lp=z(), v0=z(),
            fresh=torch.ones(n, dtype=torch.bool, device=dev), step=-1, cfg=dict(TRICK_DEFAULTS),
        )
    return c["st"]


def _t(env) -> torch.Tensor:
    return env.episode_length_buf.float() * env.step_dt


def _board(env) -> Entity:
    return env.scene["board"]


def _board_yaw(env) -> torch.Tensor:
    return _yaw_from_quat(_board(env).data.root_link_quat_w)


def board_v_deck(env) -> torch.Tensor:
    """(B, 3) board linear velocity in the deck frame."""
    b = _board(env)
    return torch.nan_to_num(quat_apply_inverse(b.data.root_link_quat_w, b.data.root_link_lin_vel_w))


def board_v_dir(env) -> torch.Tensor:
    """Signed speed along the travel direction dir (+ = rolling the commanded way)."""
    return board_v_deck(env)[:, 0] * _state(env)["dir"]


def front_up_pitch(env) -> torch.Tensor:
    """Pitch of the deck with the FRONT end (relative to dir) up = + (rad)."""
    fx = _unit(_board(env).data.root_link_quat_w, 0)[:, 2]          # world z of the deck +x axis
    return torch.asin(torch.clamp(fx * _state(env)["dir"], -1.0, 1.0))


def _contact_any(env, name: str) -> torch.Tensor:
    return _skate_contact(env, name) > 0


def _contact_all(env, name: str) -> torch.Tensor:
    f = env.scene.sensors[name].data.found
    return (f > 0).all(dim=-1) if f.dim() > 1 else f > 0


def _front_back_wheels(env):
    """(front wheels on the floor, back wheels on the floor), relative to dir (any wheel of the truck)."""
    st = _state(env)
    fw, rw = _contact_any(env, "front_wheels_floor"), _contact_any(env, "rear_wheels_floor")
    plus = st["dir"] > 0
    return torch.where(plus, fw, rw), torch.where(plus, rw, fw)


def _back_end_floor(env) -> torch.Tensor:
    """The back end's kick (relative to dir) touches the floor: tail drag / tail strike."""
    st = _state(env)
    nose, tail = _contact_any(env, "nose_floor"), _contact_any(env, "tail_floor")
    return torch.where(st["dir"] > 0, tail, nose)


def soles_on_deck(env) -> torch.Tensor:
    return _contact_any(env, "left_deck") & _contact_any(env, "right_deck")


def _mask(env, k: int) -> torch.Tensor:
    return (_state(env)["trick"] == k).float()


def trick_update(env) -> dict:
    """Once per env step: ride curvature resampling and the low-passed board yaw rate."""
    st = _state(env)
    if st["step"] == env.common_step_counter:
        return st
    st["step"] = env.common_step_counter
    cfg = st["cfg"]
    t = _t(env)
    ride = st["trick"] == RIDE
    due = ride & (t >= st["kappa_next"])
    if bool(due.any()):
        ids = due.nonzero().squeeze(1)
        _sample_kappa(env, st, ids)
        lo, hi = cfg["kappa_resample_s"]
        st["kappa_next"][ids] = t[ids] + lo + torch.rand(len(ids), device=env.device) * (hi - lo)
    wz = torch.nan_to_num(_board(env).data.root_link_ang_vel_w[:, 2])
    a = min(1.0, env.step_dt / cfg["wz_lp_tau"])
    st["wz_lp"] = torch.where(st["fresh"], wz, st["wz_lp"] + a * (wz - st["wz_lp"]))
    st["fresh"][:] = False
    return st


def _sample_kappa(env, st, ids):
    cfg = st["cfg"]
    n = len(ids)
    k = (torch.rand(n, device=env.device) * 2 - 1) * cfg["kappa_range"]
    k = torch.where(torch.rand(n, device=env.device) < cfg["kappa_zero_prob"], torch.zeros_like(k), k)
    st["kappa"][ids] = k


def ride_wz_target(env) -> torch.Tensor:
    st = trick_update(env)
    v = board_v_deck(env)[:, 0].abs()
    kmax = st["cfg"]["a_lat"] / v.clamp(min=0.05) ** 2
    k = torch.maximum(torch.minimum(st["kappa"], kmax), -kmax)
    return k * v


# ── reset ─────────────────────────────────────────────────────────────────────────────────────────────────────────────


def _rot_about_axis(axis: torch.Tensor, ang: torch.Tensor) -> torch.Tensor:
    h = ang * 0.5
    return torch.cat([torch.cos(h).unsqueeze(1), axis * torch.sin(h).unsqueeze(1)], dim=1)


def reset_trick_state(
    env,
    env_ids: torch.Tensor,
    trick_probs: dict,
    speeds: dict,
    manual_tilt_prob: float = 0.5,
    manual_tilt_deg: tuple = (5.0, 14.0),
    pos_noise: float = 0.003,
    rpy_noise: float = 0.02,
    joint_noise: float = 0.02,
    z_lift: float = 0.001,
    axle_x: float = DK_AXLE_X,
    dir_mode: str = "random",
    alt_spawns: dict | None = None,
    stance_key: str = "regular",
    stance_path: str = str(STANCE_JSON),
):
    """Board at the env origin (random heading), robot standing on it in the sideways stance, both rolling along dir at
    the trick's spawn speed; draws the trick and its command. MUST run after the base reset events."""
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device).long()
    n, dev = len(env_ids), env.device
    robot: Entity = env.scene["robot"]
    board = _board(env)
    st = _state(env)
    cfg = st["cfg"]
    c = _tc(env)
    if "kf" not in c:
        stance = load_stance(stance_key, Path(stance_path))
        jn = list(stance["joints"].keys())
        c["stance_yaw"] = float(stance["rpy"][2])
        c["kf"] = (torch.tensor(stance["pos"], device=dev, dtype=torch.float),
                   torch.tensor(stance["rpy"], device=dev, dtype=torch.float),
                   [list(robot.joint_names).index(j) for j in jn],
                   torch.tensor([stance["joints"][j] for j in jn], device=dev, dtype=torch.float),
                   board.find_joints(r"^passive_.*_wheel_[LR]$")[0])
        p = torch.tensor([float(trick_probs.get(k, 0.0)) for k in TRICKS], device=dev)
        assert p.sum() > 0
        c["p"] = p / p.sum()
        c["sp"] = torch.tensor([speeds[k] for k in TRICKS], device=dev, dtype=torch.float)  # (N_TRICK, 2)
        c["alt"] = []                      # reverse-curriculum spawns: (trick idx, prob, pos, rpy, joints)
        for tname, (akey, aprob) in (alt_spawns or {}).items():
            a = load_stance(akey, Path(stance_path))
            c["alt"].append((TRICKS.index(tname), float(aprob), torch.tensor(a["pos"], device=dev, dtype=torch.float),
                             torch.tensor(a["rpy"], device=dev, dtype=torch.float),
                             torch.tensor([a["joints"][j] for j in jn], device=dev, dtype=torch.float)))
    kpos, krpy, leg_ids, kjoints, wheel_ids = c["kf"]

    trick = torch.multinomial(c["p"], n, replacement=True)
    kp = kpos.unsqueeze(0).repeat(n, 1)
    kr = krpy.unsqueeze(0).repeat(n, 1)
    kj = kjoints.unsqueeze(0).repeat(n, 1)
    for ti, ap, apos, arpy, aj in c.get("alt", []):
        msk = (trick == ti) & (torch.rand(n, device=dev) < ap)
        kp[msk], kr[msk], kj[msk] = apos, arpy, aj
    dirn = (torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0) if dir_mode == "random"
            else torch.ones(n, device=dev))
    lo, hi = c["sp"][trick, 0], c["sp"][trick, 1]
    v0 = lo + torch.rand(n, device=dev) * (hi - lo)

    origin = env.scene.env_origins[env_ids]
    yaw = (torch.rand(n, device=dev) * 2 - 1) * math.pi
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    bpose = torch.zeros(n, 7, device=dev)
    bpose[:, 0:2] = origin[:, :2]
    bpose[:, 2] = origin[:, 2] + DK_DECK_CENTER_Z
    bpose[:, 3] = torch.cos(yaw * 0.5)
    bpose[:, 6] = torch.sin(yaw * 0.5)

    off = kp.clone()
    off[:, :2] += (torch.rand(n, 2, device=dev) * 2 - 1) * pos_noise
    rpose = torch.zeros(n, 7, device=dev)
    rpose[:, 0] = origin[:, 0] + cy * off[:, 0] - sy * off[:, 1]
    rpose[:, 1] = origin[:, 1] + sy * off[:, 0] + cy * off[:, 1]
    rpose[:, 2] = origin[:, 2] + off[:, 2] + z_lift
    r = kr + (torch.rand(n, 3, device=dev) * 2 - 1) * rpy_noise
    rpose[:, 3:7] = _quat_from_rpy_t(r[:, 0], r[:, 1], r[:, 2] + yaw)

    # manual: some spawns start tilted nose-up about the BACK axle (rigid robot + board)
    tilt = (trick == MANUAL) & (torch.rand(n, device=dev) < manual_tilt_prob)
    if bool(tilt.any()):
        th = math.radians(manual_tilt_deg[0]) + torch.rand(n, device=dev) * math.radians(manual_tilt_deg[1] -
                                                                                          manual_tilt_deg[0])
        phi = torch.where(tilt, -dirn * th, torch.zeros_like(th))      # about deck +y: front end (dir) goes up
        axis = torch.stack([-sy, cy, torch.zeros_like(cy)], dim=1)       # deck +y in the world
        q = _rot_about_axis(axis, phi)
        bx = -dirn * axle_x                                           # back axle, deck frame
        piv = torch.stack([bpose[:, 0] + cy * bx, bpose[:, 1] + sy * bx, bpose[:, 2] + DK_AXLE_Z], dim=1)
        for pose in (bpose, rpose):
            pose[:, 0:3] = piv + quat_apply(q, pose[:, 0:3] - piv)
            pose[:, 3:7] = quat_mul(q, pose[:, 3:7])

    vel = torch.zeros(n, 6, device=dev)
    vel[:, 0] = dirn * v0 * cy
    vel[:, 1] = dirn * v0 * sy
    board.write_root_link_pose_to_sim(bpose, env_ids)
    board.write_root_link_velocity_to_sim(vel, env_ids)
    bj = torch.zeros(n, board.num_joints, device=dev)
    bjv = torch.zeros(n, board.num_joints, device=dev)
    bjv[:, wheel_ids] = (dirn * v0 / DK_WHEEL_RADIUS).unsqueeze(1)
    board.write_joint_state_to_sim(bj, bjv, env_ids=env_ids)
    robot.write_root_link_pose_to_sim(rpose, env_ids)
    robot.write_root_link_velocity_to_sim(vel, env_ids)
    jpos = robot.data.default_joint_pos[env_ids].clone()
    jpos[:, leg_ids] = kj + torch.randn(n, len(leg_ids), device=dev) * joint_noise
    robot.write_joint_state_to_sim(jpos, torch.zeros_like(jpos), env_ids=env_ids)

    # trick state
    st["trick"][env_ids] = trick
    st["dir"][env_ids] = dirn
    st["v0"][env_ids] = v0
    st["kappa"][env_ids] = 0.0
    _sample_kappa(env, st, env_ids)
    st["kappa"][env_ids] = torch.where(trick == RIDE, st["kappa"][env_ids], torch.zeros(n, device=dev))
    klo, khi = cfg["kappa_resample_s"]
    st["kappa_next"][env_ids] = klo + torch.rand(n, device=dev) * (khi - klo)
    ang = torch.tensor(cfg["kt_angles"], device=dev)[torch.randint(len(cfg["kt_angles"]), (n,), device=dev)]
    sgn = torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)
    st["yaw_0"][env_ids] = yaw
    st["kt_total"][env_ids] = sgn * ang
    st["yaw_t"][env_ids] = wrap_to_pi(yaw + sgn * ang)
    tlo, thi = cfg["tictac_v"]
    st["v_t"][env_ids] = tlo + torch.rand(n, device=dev) * (thi - tlo)
    st["wz_lp"][env_ids] = 0.0
    st["fresh"][env_ids] = True


# ── observations ──────────────────────────────────────────────────────────────────────────────────────────────────────


def trick_obs(env) -> torch.Tensor:
    """(B, 12): trick one-hot (5), dir, ride curvature cmd, tictac target speed, sin/cos(kickturn heading error to the
    slewed reference the reward tracks), kickturn total turn / pi (signed), episode time / 10 s."""
    st = trick_update(env)
    oh = torch.nn.functional.one_hot(st["trick"], N_TRICK).float()
    ref, _ = _kt_ref(env, KT_RATE, KT_T0)
    e = wrap_to_pi(ref - _board_yaw(env))
    kt = (st["trick"] == KICKTURN).float()
    return torch.cat([
        oh, st["dir"].unsqueeze(1), st["kappa"].unsqueeze(1), (st["v_t"] * (st["trick"] == TICTAC).float()).unsqueeze(1),
        (torch.sin(e) * kt).unsqueeze(1), (torch.cos(e) * kt + (1 - kt)).unsqueeze(1),
        (st["kt_total"] * kt / math.pi).unsqueeze(1), (_t(env) / 10.0).unsqueeze(1),
    ], dim=1)


def board_obs(env) -> torch.Tensor:
    """(B, 28) board block, deck frame: board lin vel (3), ang vel (3), gravity (3), trunk position rel. deck (3),
    sin/cos(trunk yaw - deck yaw - pi/2) (2), box-sole centres (6), sole-deck contacts L/R (2), front/rear truck wheels on
    the floor (2, deck +x truck first), nose/tail kick on the floor (2), truck angles (2)."""
    robot: Entity = env.scene["robot"]
    b = _board(env)
    q = b.data.root_link_quat_w
    pb = b.data.root_link_pos_w
    v = quat_apply_inverse(q, b.data.root_link_lin_vel_w)
    w = quat_apply_inverse(q, b.data.root_link_ang_vel_w)
    g = quat_apply_inverse(q, torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(env.num_envs, 3))
    pr = quat_apply_inverse(q, robot.data.root_link_pos_w - pb)
    dy = wrap_to_pi(_yaw_from_quat(robot.data.root_link_quat_w) - _yaw_from_quat(q) - _tc(env).get("stance_yaw", math.pi / 2))
    c = _tc(env)
    if "sole_ids" not in c:
        c["sole_ids"] = [robot.find_geoms(rf"^{s}_sole_deck_collision$")[0][0] for s in ("left", "right")]
        c["truck_ids"] = b.find_joints(r"^passive_(front|rear)_truck$", preserve_order=True)[0]
    soles = [quat_apply_inverse(q, robot.data.geom_pos_w[:, i] - pb) for i in c["sole_ids"]]
    flags = torch.stack([_contact_any(env, n_).float() for n_ in
                         ("left_deck", "right_deck", "front_wheels_floor", "rear_wheels_floor", "nose_floor",
                          "tail_floor")], dim=1)
    trucks = b.data.joint_pos[:, c["truck_ids"]]
    out = torch.cat([v, w, g, pr, torch.sin(dy).unsqueeze(1), torch.cos(dy).unsqueeze(1), *soles, flags, trucks], dim=1)
    return torch.nan_to_num(out)


# ── rewards (per second; mjlab multiplies by dt) ──────────────────────────────────────────────────────────────────────


def trunk_upright(env, std: float = 0.3) -> torch.Tensor:
    g = env.scene["robot"].data.projected_gravity_b
    return torch.exp(-(g[:, 0] ** 2 + g[:, 1] ** 2) / std**2)


def on_deck(env) -> torch.Tensor:
    return soles_on_deck(env).float()


def trick_fall(env) -> torch.Tensor:
    """1/dt on a non-timeout termination: the weight is the total cost of a fall (not dt-scaled)."""
    return env.termination_manager.terminated.float() / env.step_dt


def contact_cost(env, sensor_name: str) -> torch.Tensor:
    return _contact_any(env, sensor_name).float()


def joint_dev_l2(env, joint_names: tuple) -> torch.Tensor:
    robot: Entity = env.scene["robot"]
    c = _tc(env)
    key = ("jdev", joint_names)
    if key not in c:
        c[key] = robot.find_joints(joint_names)[0]
    ids = c[key]
    d = robot.data.joint_pos[:, ids] - robot.data.default_joint_pos[:, ids]
    return (d**2).sum(dim=1)


def ride_track(env, std: float = 0.12, v_full: float = 0.15) -> torch.Tensor:
    """RIDE: yaw-rate tracking (low-passed board yaw rate vs kappa * |v|), scaled by speed (braking does not help)."""
    st = trick_update(env)
    err = st["wz_lp"] - ride_wz_target(env)
    v = board_v_deck(env)[:, 0].abs()
    gate = ((v - 0.03) / (v_full - 0.03)).clamp(0.0, 1.0)
    return _mask(env, RIDE) * gate * torch.exp(-(err**2) / std**2)


def deck_roll_rate(env) -> torch.Tensor:
    """Cost: deck roll/pitch rate squared (board wobble)."""
    w = quat_apply_inverse(_board(env).data.root_link_quat_w, _board(env).data.root_link_ang_vel_w)
    return torch.nan_to_num(w[:, 0] ** 2 + w[:, 1] ** 2)


def stop_track(env, decel: float = 0.6, std: float = 0.15) -> torch.Tensor:
    """STOP: track a slewed speed v_ref(t) = max(v0 - decel * t, 0) (no jackpot; braking harder than the
    ramp pays nothing extra)."""
    st = _state(env)
    v = board_v_deck(env)[:, :2].norm(dim=1)
    v_ref = (st["v0"] - decel * _t(env)).clamp(min=0.0)
    return _mask(env, STOP) * torch.exp(-((v - v_ref) ** 2) / std**2)


def stop_tail_drag(env, v_min: float = 0.05) -> torch.Tensor:
    """STOP: the back end drags on the floor while still rolling (technique hint, small weight)."""
    v = board_v_deck(env)[:, :2].norm(dim=1)
    return _mask(env, STOP) * (_back_end_floor(env) & (v > v_min)).float()


def _kt_ref(env, rate: float, t0: float) -> torch.Tensor:
    """Slewed kick-turn heading target: from the spawn heading toward yaw_t at `rate` rad/s, starting at t0."""
    st = _state(env)
    total = st["kt_total"]                                  # signed, rad
    prog = ((_t(env) - t0).clamp(min=0.0) * rate).clamp(max=1.0e3)
    step = torch.sign(total) * torch.minimum(prog, total.abs())
    return wrap_to_pi(st["yaw_0"] + step), prog >= total.abs()


def kickturn_track(env, rate: float = KT_RATE, t0: float = 0.5, std: float = 0.35) -> torch.Tensor:
    """KICKTURN: board heading tracks the slewed target (ahead of the ramp pays nothing)."""
    ref, _ = _kt_ref(env, rate, t0)
    e = wrap_to_pi(ref - _board_yaw(env))
    _, back_down = _front_back_wheels(env)
    return _mask(env, KICKTURN) * back_down.float() * torch.exp(-(e**2) / std**2)


def kickturn_done(env, rate: float = KT_RATE, t0: float = 0.5, tol: float = 0.26) -> torch.Tensor:
    """KICKTURN: after the ramp has arrived: heading within tol of the goal, all four wheels down, soles on the deck."""
    st = _state(env)
    _, arrived = _kt_ref(env, rate, t0)
    ok = wrap_to_pi(st["yaw_t"] - _board_yaw(env)).abs() < tol
    down = _contact_all(env, "front_wheels_floor") & _contact_all(env, "rear_wheels_floor")
    return _mask(env, KICKTURN) * (arrived & ok & down & soles_on_deck(env)).float()


def tictac_track(env, accel: float = 0.15, t0: float = 0.5, std: float = 0.15) -> torch.Tensor:
    """TICTAC: speed along dir tracks a slewed v_ref(t) = min(v_t, accel * (t - t0)) (feet never touch the floor:
    that terminates)."""
    st = _state(env)
    v_ref = torch.minimum(st["v_t"], accel * (_t(env) - t0).clamp(min=0.0))
    e = board_v_dir(env) - v_ref
    return _mask(env, TICTAC) * torch.exp(-(e**2) / std**2)


def manual_in(env, lo_deg: float = 3.0, hi_deg: float = 25.0) -> torch.Tensor:
    """MANUAL: front wheels up, back end off the floor, front-up pitch in [lo, hi], both soles on the deck."""
    th = front_up_pitch(env)
    front_down, _ = _front_back_wheels(env)
    _, back_down = _front_back_wheels(env)
    ok = (~front_down) & back_down & (~_back_end_floor(env)) & (th > math.radians(lo_deg)) & (th < math.radians(hi_deg))
    return _mask(env, MANUAL) * (ok & soles_on_deck(env)).float()


def nose_lift(env, lo_deg: float = 2.0, hi_deg: float = 20.0) -> torch.Tensor:
    """KICKTURN / TICTAC shaping: the pivot posture (front wheels up, back wheels down, front-up pitch in band)."""
    th = front_up_pitch(env)
    front_down, back_down = _front_back_wheels(env)
    ok = (~front_down) & back_down & (~_back_end_floor(env)) & (th > math.radians(lo_deg)) & (th < math.radians(hi_deg))
    m = _mask(env, KICKTURN) + _mask(env, TICTAC)
    return m * (ok & soles_on_deck(env)).float()


def manual_shape(env, pts_deg: tuple = (0.0, 6.0, 15.0, 28.0)) -> torch.Tensor:
    """MANUAL: trapezoid on the front-up pitch (shaping toward the band)."""
    th = torch.rad2deg(front_up_pitch(env))
    a, b, c_, d = pts_deg
    up = ((th - a) / (b - a)).clamp(0, 1)
    down = ((d - th) / (d - c_)).clamp(0, 1)
    return _mask(env, MANUAL) * torch.minimum(up, down)


def manual_strike(env) -> torch.Tensor:
    return _mask(env, MANUAL) * _back_end_floor(env).float()


# ── terminations ──────────────────────────────────────────────────────────────────────────────────────────────────────


def board_lost(env, max_dist: float = 0.15, max_off_s: float = 0.6) -> torch.Tensor:
    robot: Entity = env.scene["robot"]
    d = torch.nan_to_num((robot.data.root_link_pos_w[:, :2] - _board(env).data.root_link_pos_w[:, :2]).norm(dim=1),
                         nan=1.0)
    off = (_skate_air_time(env, "left_deck") > max_off_s) | (_skate_air_time(env, "right_deck") > max_off_s)
    return (d > max_dist) | off


def board_flipped(env, min_up_z: float = 0.5) -> torch.Tensor:
    return torch.nan_to_num(_unit(_board(env).data.root_link_quat_w, 2)[:, 2], nan=-1.0) < min_up_z


def foot_on_floor(env, sensor_name: str = "feet_ground_contact") -> torch.Tensor:
    return _contact_any(env, sensor_name)
