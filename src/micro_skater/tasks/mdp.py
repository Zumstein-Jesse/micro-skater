# micro-skater: part of the author's skateboard add-on for microduck_rl. Apache-2.0 (see LICENSE).
"""micro-skater MDP terms: microduck_rl's ``mjlab_microduck.tasks.mdp`` namespace plus the skate / run terms.

Every name of the upstream module (public and private helpers) is re-exported first, so the task configs can use one
``mdp`` namespace; the author's additions follow. Nothing of the upstream module is copied here: it is imported.
"""
from __future__ import annotations

import mjlab_microduck.tasks.mdp as _upstream_mdp

globals().update({_k: _v for _k, _v in vars(_upstream_mdp).items() if not (_k.startswith("__") and _k.endswith("__"))})
from micro_skater.paths import data_path as _data_path  # noqa: E402,F401

# ── run command (Mjlab-Velocity-Run-Flat-MicroDuck) and skate curvature commands ──
class RunVelocityCommand(VelocityCommandCommandOnly):
    """Run task: as VelocityCommandCommandOnly, plus a run bucket. A fraction rel_run_envs of each resample draws vx
    uniformly in run_vx_range (keeping its sampled yaw, lateral scaled by run_vy_scale), so fast forward commands are a
    real share of the data rather than the thin top slice of a wide uniform range."""

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        p = getattr(self.cfg, "rel_run_envs", 0.0)
        if p <= 0.0 or len(env_ids) == 0:
            return
        r = torch.empty(len(env_ids), device=self.device).uniform_(0.0, 1.0)
        run_ids = env_ids[(r < p) & ~self.is_standing_env[env_ids]]
        if len(run_ids) == 0:
            return
        lo, hi = self.cfg.run_vx_range
        self.vel_command_b[run_ids, 0] = torch.empty(len(run_ids), device=self.device).uniform_(lo, hi)
        self.vel_command_b[run_ids, 1] *= self.cfg.run_vy_scale
        self.vel_command_w[run_ids] = self.vel_command_b[run_ids]


@_dataclass(kw_only=True)
class RunVelocityCommandCfg(VelocityCommandCommandOnlyCfg):
    rel_run_envs: float = 0.0
    run_vx_range: tuple[float, float] = (0.5, 1.0)
    run_vy_scale: float = 0.3

    def build(self, env: ManagerBasedRlEnv) -> "RunVelocityCommand":
        return RunVelocityCommand(self, env)


def run_speed_tracking(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str = "twist",
    vx_gate: float = 0.45,
    asset_name: str = "robot",
) -> torch.Tensor:
    """Run task: Gaussian on the forward-speed error, only for envs commanded faster than vx_gate (0 elsewhere). The
    shared track_linear_velocity (std^2 0.1) is too flat to pull a walker out of its ~half-speed compromise; this
    narrower, run-only term prices the forward speed alone. Gated on the COMMAND, not on state: nothing to farm."""
    asset: Entity = env.scene[asset_name]
    cmd = env.command_manager.get_command(command_name)
    vx = asset.data.root_link_lin_vel_b[:, 0]
    gate = (cmd[:, 0] > vx_gate).float()
    return gate * torch.exp(-torch.square(cmd[:, 0] - vx) / std**2)


def run_yaw_tracking(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str = "twist",
    vx_gate: float = 0.3,
    asset_name: str = "robot",
) -> torch.Tensor:
    """Run task (v2): Gaussian on the yaw-rate error for envs commanded forward faster than vx_gate (0 elsewhere). The
    shared track_angular_velocity (std 0.71 rad/s) let run_v1 turn at only ~0.2 rad/s on a 0.8 command at speed."""
    asset: Entity = env.scene[asset_name]
    cmd = env.command_manager.get_command(command_name)
    wz = asset.data.root_link_ang_vel_b[:, 2]
    gate = (cmd[:, 0] > vx_gate).float()
    return gate * torch.exp(-torch.square(cmd[:, 2] - wz) / std**2)


class SkateCurvatureCommand(VelocityCommandCommandOnly):
    """Skate command (v4): twist[0] = cruise speed, twist[2] = path CURVATURE kappa (1/m, + = left), so the board's
    target yaw rate is kappa * forward speed and a turn means the same path at any speed. kappa has its own hold timer
    (curvature_hold_range), independent of the speed resampling: a fraction straight_prob of holds is exactly
    straight, the rest draw |kappa| in [curvature_min_frac, 1] x the range max with a random sign (uniform sampling
    would make tight turns and true straights both rare). ranges.ang_vel_z is the kappa range (curriculum-staged)."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.kappa = torch.zeros(self.num_envs, device=self.device)
        self.kappa_time_left = torch.zeros(self.num_envs, device=self.device)
        # v5 step-off: twist[1] = SKATE_DISMOUNT_FLAG from dismount_at (s into the episode) on, held to the end.
        self.t_ep = torch.zeros(self.num_envs, device=self.device)
        self.dismount_at = torch.full((self.num_envs,), float("inf"), device=self.device)

    def _sample_dismount(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        if n == 0:
            return
        p = float(getattr(self.cfg, "dismount_prob", 0.0))
        lo, hi = getattr(self.cfg, "dismount_time_range", (1.0, 10.0))
        t = torch.empty(n, device=self.device).uniform_(lo, hi)
        st = _skate_cache(self._env).get("state")
        if st is not None and "mounted" in st:   # a step-on spawn first gets time to get on
            extra = torch.full_like(t, float(getattr(self.cfg, "dismount_after_mount_s", 3.0)))
            if st["cfg"].get("v6b", False):
                # v6b walk-ups: after the whole mount budget (skate_board_lost: 4 s + d0 / 6 cm/s); a flag before the
                # mount paid ~7/s for standing anywhere (verifier 2026-11-03: 8-19 % of ring episodes)
                extra = torch.maximum(extra, 4.0 + st["appr_d0"][env_ids] / 0.06)
            t = torch.where(st["mounted"][env_ids], t, t + extra)
        on = torch.rand(n, device=self.device) < p
        if st is not None and "dismount_now" in st and p > 0:   # v5c reverse-curriculum spawns: flag up from t = 0
            now = st["dismount_now"][env_ids]
            t = torch.where(now, torch.zeros_like(t), t)
            on = on | now
        self.dismount_at[env_ids] = torch.where(on, t, torch.full_like(t, float("inf")))

    def _resample_kappa(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        if n == 0:
            return
        lo, hi = self.cfg.ranges.ang_vel_z
        maxr = max(abs(lo), abs(hi))
        u = torch.rand(n, device=self.device)
        sign = torch.where(torch.rand(n, device=self.device) < 0.5, -1.0, 1.0)
        mag = torch.empty(n, device=self.device).uniform_(self.cfg.curvature_min_frac * maxr, maxr)
        # v6: a share of the turn holds at exactly +-max (the play sim's A / D send the max)
        p_max = float(getattr(self.cfg, "max_curvature_prob", 0.0))
        mag = torch.where(torch.rand(n, device=self.device) < p_max, torch.full_like(mag, maxr), mag)
        self.kappa[env_ids] = torch.where(u < self.cfg.straight_prob, torch.zeros_like(mag), sign * mag)
        self.kappa_time_left[env_ids] = torch.empty(n, device=self.device).uniform_(*self.cfg.curvature_hold_range)

    def reset(self, env_ids):
        if isinstance(env_ids, torch.Tensor):
            self._resample_kappa(env_ids)
            self.t_ep[env_ids] = 0.0
            self._sample_dismount(env_ids)
        return super().reset(env_ids)

    def compute(self, dt: float) -> None:
        self.t_ep += dt
        self.kappa_time_left -= dt
        self._resample_kappa((self.kappa_time_left <= 0.0).nonzero().flatten())
        super().compute(dt)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        self.vel_command_b[env_ids, 2] = self.kappa[env_ids]
        # v6 throttle / coast buckets for twist[0]: COAST (no key) and exactly the range max (W held); the rest stays
        # uniform in the range. Standing (STOP) envs are zeroed afterwards by _update_command.
        p_coast = float(getattr(self.cfg, "coast_prob", 0.0))
        p_max = float(getattr(self.cfg, "vmax_prob", 0.0))
        if (p_coast > 0 or p_max > 0) and len(env_ids) > 0:
            u = torch.rand(len(env_ids), device=self.device)
            v = self.vel_command_b[env_ids, 0]
            v = torch.where(u < p_coast, torch.full_like(v, SKATE_COAST_CMD), v)
            v = torch.where((u >= p_coast) & (u < p_coast + p_max),
                            torch.full_like(v, float(self.cfg.ranges.lin_vel_x[1])), v)
            self.vel_command_b[env_ids, 0] = v

    def _update_command(self) -> None:
        self.vel_command_b[:, 2] = self.kappa
        super()._update_command()  # zeroes the STOP (standing) envs
        flag = self.t_ep >= self.dismount_at
        self.vel_command_b[:, 1] = torch.where(flag, torch.full_like(self.t_ep, SKATE_DISMOUNT_FLAG),
                                               self.vel_command_b[:, 1])


@_dataclass(kw_only=True)
class SkateCurvatureCommandCfg(VelocityCommandCommandOnlyCfg):
    straight_prob: float = 0.3
    curvature_min_frac: float = 0.25
    curvature_hold_range: tuple = (2.0, 6.0)
    dismount_prob: float = 0.0            # v5: fraction of episodes that get the step-off flag
    dismount_time_range: tuple = (1.0, 10.0)
    dismount_after_mount_s: float = 3.0   # extra delay for step-on spawns
    coast_prob: float = 0.0               # v6: share of resamples at SKATE_COAST_CMD
    vmax_prob: float = 0.0                # v6: share of resamples at exactly the cruise range max ("W held")
    max_curvature_prob: float = 0.0       # v6: share of turn holds at exactly +-kappa max

    def build(self, env: ManagerBasedRlEnv) -> "SkateCurvatureCommand":
        return SkateCurvatureCommand(self, env)





# --------------------------------------------------------------------------- #
# SKATE task: kick-push a cruiser skateboard (scooter stance, left foot rides)  #
# --------------------------------------------------------------------------- #
# The board is a separate floating, articulated, unactuated entity ("board").
# Command twist[0] = cruise speed v_on (m/s), twist[2] = board yaw rate.
#
# v3: a per-env PUSH / GLIDE / STOP mode machine (see microduck_skate_env_cfg.py).
#   - mode speed v_mode = system (robot + board) forward speed, low-passed (tau 0.1 s). The system speed only
#     changes through external forces (push foot on the floor, wheel drag, braking), so lunging or shoving the
#     deck cannot fake progress or trip the threshold.
#   - PUSH -> GLIDE when v_mode >= v_on, GLIDE -> PUSH when v_mode <= v_off = v_on - gap (hysteresis), each after a
#     min dwell. twist[0] < stop_cmd forces STOP (come to rest, both feet planted).
#   - The machine updates lazily once per env step (first caller after common_step_counter moves). Reset events
#     write the state of the reset envs; the first update after a reset only re-baselines the potentials.

SKATE_PUSH, SKATE_GLIDE, SKATE_STOP = 0, 1, 2

SKATE_MODE_DEFAULTS = dict(
    gap=0.08,            # v_off = v_on - gap
    grace=1.0,           # s after entering GLIDE before the glide terms pay / the floor cost bites
    stop_cmd=0.05,       # twist[0] below this = STOP
    tau_mode=0.1,        # s, low-pass of the mode speed
    tau_cruise=2.0,      # s, low-pass of the cruise (mean) speed
    min_dwell=0.3,       # s in a mode before PUSH <-> GLIDE may switch
    glide_left=(0.0, 0.04),    # board-foot (left) sole xy on the deck in the glide stance
    glide_right=(0.0, -0.04),  # push-foot (right) sole xy on the deck in the glide stance
    push_right=(0.0, -0.09),   # push-foot (right) sole xy, deck frame, planted on the floor in the push stance
    tau_yaw=0.3,         # s, low-pass of the board yaw rate (curvature tracking; the wiggle cost is the remainder)
    tau_heading=4.0,     # s, leak of the integrated heading error (curvature tasks)
    head_clamp=0.5,      # rad, clamp of that integral (v6f 1.5: at 0.5 it saturated at kappa errors > 0.5 1/m)
    mount_load_share=0.25,  # v5: the board foot carrying this share of the weight on the deck latches "mounted"
    step_off_left_y=-0.10,   # v5d: step-off sole spots, deck-frame y (= the STAND spawn's soles, trunk at y -0.145)
    step_off_right_y=-0.185,
    coast_cmd_hi=0.15,       # v6: stop_cmd <= twist[0] < coast_cmd_hi = COAST (glide, no pushing, natural decel)
    coast_decel=0.15,        # v6: m/s^2 of free deceleration in COAST (natural coasting ~0.07, rolling DR x2)
    staging=(0.0, -0.145),   # v6: trunk spot beside the board (canonical deck frame) the step-on starts from
    v6b=False,               # v6b: canonical deck frame everywhere, walk-up potentials, no GLIDE before the mount
    reach_sat=None,          # v6b: mount_reach saturates at this L1 (m); None = unsaturated (v5)
    a_lat=None,              # v7d: m/s^2, the curvature target is clamped to +-a_lat / v^2 (None = kappa as commanded)
    mode_src="true",         # skate student: "sensed" = PUSH/GLIDE switch on the sensed wheel speed (skate_sensor_setup)
    seed_reward_mode=False,  # step 5: with mode_src "sensed", seed the reward machine at reset from the observer's rule
    coast_glide_uphill=False,  # s5b: COAST on a climb is a glide (roll back), not the uphill pump context
    command_name="twist",
    board_name="board",
    board_foot_sensor="board_foot_deck",
    push_foot_sensor="push_foot_deck",
)

# v6 throttle / coast: the play sim sends twist[0] = the cruise range max while W is held (keep pushing, get faster),
# SKATE_COAST_CMD when no key is held (coast: glide on, no pushing) and 0 for S (STOP). 0.10 is -1.5 std of the v5
# normalizer's twist[0] input (mean 0.24, std 0.09).
SKATE_COAST_CMD = 0.10


def skate_coast_active(env, command_name: str = "twist") -> torch.Tensor:
    """(B,) bool: twist[0] in [stop_cmd, coast_cmd_hi): COAST."""
    c = SKATE_MODE_DEFAULTS
    cmd = env.command_manager.get_command(command_name)[:, 0]
    return (cmd >= c["stop_cmd"]) & (cmd < c["coast_cmd_hi"])


# v5 step-off: twist[1] (padding, sampled in +-0.01 otherwise) is set to this value from the step-off request on.
# 0.04 is ~7 std of the v4 normalizer's twist[1] input (std 0.0058), so the warm-started policy sees a clear signal.
SKATE_DISMOUNT_FLAG = 0.04


def skate_dismount_active(env, command_name: str = "twist") -> torch.Tensor:
    """(B,) bool: the step-off flag is up (twist[1] > half the flag value)."""
    return env.command_manager.get_command(command_name)[:, 1] > 0.5 * SKATE_DISMOUNT_FLAG


# ── v7 slopes: a uniform grade is a tilted gravity ─────────────────────────────────────────────────────────────────
# On a plane, "floor flat + gravity tilted by theta" is the same physics as "floor tilted by theta". MJWarp reads
# opt.gravity per world (worldid % len), so skate_slope_setup tiles it to one entry per env (and re-captures the CUDA
# graphs); every entity's gravity_vec_w (projected_gravity obs / rewards) follows it. The tilt s = sin(theta) * the
# downhill direction (world xy) slews toward a per-env target: flat, downhill roughly along the board heading, or
# uphill. Real ramps are short, so a downhill segment ends (target flat again) after a sampled vertical drop.
# Walking up to the board and stepping off stay flat (the walker and the step-off were trained there).
SKATE_SLOPE_DEFAULTS = dict(
    max_deg=0.0,             # downhill grade range [min_deg, max_deg] (curriculum)
    up_max_deg=0.0,          # uphill grade range [min_deg, up_max_deg] (0 = none)
    min_deg=1.0,
    flat_prob=0.35,          # share of segments that are flat
    up_prob=0.15,            # share that are uphill
    lateral_deg=40.0,        # downhill direction spread around the board heading at sampling
    hold_s=(2.0, 6.0),       # segment length (s), unless the drop budget ends a slope first
    slew_deg_s=(4.0, 25.0),  # grade change rate (a ramp transition), deg/s
    drop_m=(0.02, 0.06),     # vertical drop / climb budget of a slope segment, m (curriculum)
    stop_max_deg=2.0,        # STOP command: grade clamped to this
    speed_cap=2.2,           # m/s: no downhill target above this board speed
    eta=0.90,                # passive rolling acceleration / (g sin theta) (wheel inertia, drag; reviewer probe: 0.93
                             # left a 3 deg coast charged -1.8/s vs -0.5/s on the flat)
    up_lateral_deg=None,     # step 5: uphill direction spread (None = lateral_deg, as before)
    stop_up_max_deg=None,    # step 5: STOP clamp of an UPHILL-ahead grade (None = stop_max_deg for every grade)
    back_speed_cap=None,     # s5b: m/s, an uphill-ahead target goes flat while the board rolls BACK faster than this
                             # (the roll-back after a W release; None = no cap, as before)
    board_name="board",
)
SKATE_SLOPE_DOWNHILL_ACC = 0.15   # m/s^2 (~1 deg): above it, over-speed is not charged by the speed terms


def skate_slope_setup(env, env_ids, **params):
    """Startup event (v7): per-env gravity (tiles opt.gravity, re-captures the CUDA graphs) and the slope state."""
    import warp as wp

    del env_ids
    sim = env.sim
    mod = sim.wp_model
    n = env.num_envs
    if mod.opt.gravity.shape[0] != n:
        g0 = mod.opt.gravity.numpy()[0]
        mod.opt.gravity = wp.array(np.tile(g0, (n, 1)).astype(np.float32), dtype=wp.vec3, device=mod.opt.gravity.device)
        sim._model_bridge.clear_cache()
        sim.create_graph()
    G = wp.to_torch(mod.opt.gravity)
    unknown = set(params) - set(SKATE_SLOPE_DEFAULTS)
    assert not unknown, f"unknown slope cfg keys {unknown}"
    cfg = dict(SKATE_SLOPE_DEFAULTS, **params)
    st = _skate_state(env)
    z = lambda: torch.zeros(n, device=env.device)
    st["slope"] = dict(cfg=cfg, G=G, g0=float(np.linalg.norm(mod.opt.gravity.numpy()[0])),
                       s=torch.zeros(n, 2, device=env.device), s_tgt=torch.zeros(n, 2, device=env.device),
                       t_left=z(), rate=z() + math.radians(10.0), drop_left=z(), g_fwd=z(), g_lp=z(),
                       entities=tuple(env.scene.entities.keys()),
                       # step 5 bookkeeping (written, never read by the pre-step-5 code): the current segment is an
                       # uphill one, its initial climb budget, and a per-env segment counter
                       up_seg=torch.zeros(n, dtype=torch.bool, device=env.device), drop0=z(),
                       seg_id=torch.zeros(n, dtype=torch.long, device=env.device))
    _skate_slope_write(env, st["slope"])


def _skate_slope_write(env, sl: dict) -> None:
    s = sl["s"]
    cz = (1.0 - s.pow(2).sum(dim=1)).clamp_min(0.0).sqrt()
    g = torch.cat([s, -cz.unsqueeze(1)], dim=1)
    sl["G"].copy_(sl["g0"] * g)
    for name in sl["entities"]:
        env.scene[name].data.gravity_vec_w.copy_(g)


def _skate_slope_sample(env, sl: dict, ids: torch.Tensor) -> None:
    """New slope segment for envs ``ids``: flat, downhill (direction = board heading +- lateral_deg) or uphill."""
    c = sl["cfg"]
    m = len(ids)
    dev = env.device
    u = torch.rand(m, device=dev)
    lo = math.radians(c["min_deg"])
    dn_hi = math.radians(max(c["max_deg"], c["min_deg"]))
    up_hi = math.radians(max(c["up_max_deg"], c["min_deg"]))
    down = lo + torch.rand(m, device=dev) * (dn_hi - lo)
    up = lo + torch.rand(m, device=dev) * (up_hi - lo)
    is_up = (u >= c["flat_prob"]) & (u < c["flat_prob"] + c["up_prob"]) & (c["up_max_deg"] > c["min_deg"])
    is_down = (u >= c["flat_prob"] + c["up_prob"]) & (c["max_deg"] > c["min_deg"])
    grade = torch.where(is_down, down, torch.where(is_up, up, torch.zeros_like(down)))
    h = skate_board_heading(env, c["board_name"])[ids]
    if c.get("up_lateral_deg") is None:
        ang = (torch.atan2(h[:, 1], h[:, 0]) + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c["lateral_deg"])
               + math.pi * is_up.float())
    else:   # step 5: the uphill direction spread is staged on its own
        lat = torch.where(is_up, torch.full_like(down, math.radians(c["up_lateral_deg"])),
                          torch.full_like(down, math.radians(c["lateral_deg"])))
        ang = torch.atan2(h[:, 1], h[:, 0]) + (torch.rand(m, device=dev) * 2 - 1) * lat + math.pi * is_up.float()
    mag = torch.sin(grade)
    sl["s_tgt"][ids] = torch.stack([mag * torch.cos(ang), mag * torch.sin(ang)], dim=1)
    hl, hh = c["hold_s"]
    sl["t_left"][ids] = hl + torch.rand(m, device=dev) * (hh - hl)
    rl, rh = (math.radians(x) for x in c["slew_deg_s"])
    sl["rate"][ids] = rl + torch.rand(m, device=dev) * (rh - rl)
    dl, dh = c["drop_m"]
    sl["drop_left"][ids] = dl + torch.rand(m, device=dev) * (dh - dl)
    if "up_seg" in sl:      # step 5 bookkeeping (no effect on the slope itself)
        sl["up_seg"][ids] = is_up & (grade > 0)
        sl["drop0"][ids] = sl["drop_left"][ids]
        sl["seg_id"][ids] += 1


def _skate_slope_reset(env, st: dict, env_ids: torch.Tensor) -> None:
    sl = st.get("slope")
    if sl is None:
        return
    sl["s"][env_ids] = 0.0
    sl["s_tgt"][env_ids] = 0.0
    sl["g_fwd"][env_ids] = 0.0
    sl["g_lp"][env_ids] = 0.0
    sl["t_left"][env_ids] = 0.5 + torch.rand(len(env_ids), device=env.device) * (sl["cfg"]["hold_s"][1] - 0.5)
    if "up_seg" in sl:
        sl["up_seg"][env_ids] = False
    _skate_slope_write(env, sl)


def _skate_slope_g_fwd(env, st: dict) -> torch.Tensor:
    """(B,) m/s^2 of passive rolling acceleration along the canonical board heading from the current gravity
    (+ = downhill ahead); zeros without slopes."""
    sl = st.get("slope")
    if sl is None:
        return torch.zeros(env.num_envs, device=env.device)
    h = skate_board_heading(env, sl["cfg"]["board_name"])
    return torch.nan_to_num(sl["cfg"]["eta"] * (sl["G"][:, :2] * h).sum(dim=1), nan=0.0)


def _skate_slope_update(env, st: dict) -> None:
    """Advance the slope segments one env step and write the gravity the next physics step runs with."""
    sl = st.get("slope")
    if sl is None:
        return
    c = sl["cfg"]
    mc = st["cfg"]
    dt = env.step_dt
    board: Entity = env.scene[c["board_name"]]
    v_xy = torch.nan_to_num(board.data.root_link_lin_vel_w[:, :2], nan=0.0)
    # vertical drop / climb along the actual motion; a slope segment ends (flat runout) when its budget is spent
    sl["drop_left"].sub_((v_xy * sl["s"]).sum(dim=1).abs() * dt)
    if c.get("back_speed_cap") is not None:
        # s5b: rolling BACK down an uphill-ahead segment faster than the cap = the ramp's foot: the segment ends now
        # (flat runout, as a spent budget)
        h_b = skate_board_heading(env, c["board_name"])
        v_back = -torch.nan_to_num((v_xy * h_b).sum(dim=1), nan=0.0)
        foot = ((sl["s_tgt"] * h_b).sum(dim=1) < 0.0) & (v_back > c["back_speed_cap"])
        sl["drop_left"].copy_(torch.where(foot, torch.zeros_like(sl["drop_left"]), sl["drop_left"]))
    sloped = sl["s_tgt"].norm(dim=1) > 1e-6
    spent = sloped & (sl["drop_left"] <= 0.0)
    if bool(spent.any()):
        ids = spent.nonzero().squeeze(1)
        sl["s_tgt"][ids] = 0.0
        hl, hh = c["hold_s"]
        sl["t_left"][ids] = hl + torch.rand(len(ids), device=env.device) * (hh - hl)
    sl["t_left"].sub_(dt)
    due = sl["t_left"] <= 0.0
    if bool(due.any()):
        _skate_slope_sample(env, sl, due.nonzero().squeeze(1))
    tgt = sl["s_tgt"]
    cmd = env.command_manager.get_command(mc["command_name"])
    riding = st["mounted"] & ~(cmd[:, 1] > 0.5 * SKATE_DISMOUNT_FLAG)
    speed = torch.nan_to_num(v_xy.norm(dim=1), nan=0.0)
    downhill_ahead = (tgt * skate_board_heading(env, c["board_name"])).sum(dim=1) > 0.0
    keep = riding & ~(downhill_ahead & (speed > c["speed_cap"]))
    tgt = torch.where(keep.unsqueeze(1), tgt, torch.zeros_like(tgt))
    smax = math.sin(math.radians(c["stop_max_deg"]))
    mag = tgt.norm(dim=1, keepdim=True)
    stop = (cmd[:, 0] < mc["stop_cmd"]).unsqueeze(1)
    if c.get("stop_up_max_deg") is None:
        tgt = torch.where(stop & (mag > smax), tgt * (smax / mag.clamp_min(1e-9)), tgt)
    else:   # step 5: STOP on a climb (release W uphill = brake / hold) keeps its grade up to stop_up_max_deg
        up_ahead = ((tgt * skate_board_heading(env, c["board_name"])).sum(dim=1) < 0.0).unsqueeze(1)
        smax_t = torch.where(up_ahead, torch.full_like(mag, math.sin(math.radians(c["stop_up_max_deg"]))),
                             torch.full_like(mag, smax))
        tgt = torch.where(stop & (mag > smax_t), tgt * (smax_t / mag.clamp_min(1e-9)), tgt)
    d = tgt - sl["s"]
    dn = d.norm(dim=1, keepdim=True)
    # off the board (step-off flag up, walking): back to flat at the fastest transition rate (reviewer 2026-11-03)
    rate = torch.where(riding, sl["rate"], torch.full_like(sl["rate"], math.radians(c["slew_deg_s"][1])))
    step = (rate * dt).unsqueeze(1)
    sl["s"].add_(torch.where(dn > step, d * (step / dn.clamp_min(1e-9)), d))
    _skate_slope_write(env, sl)


def skate_slope_curriculum(env, env_ids, stages: list[dict]) -> torch.Tensor:
    """v7: stage the slope cfg, stages = [{"step", <SKATE_SLOPE_DEFAULTS keys>...}] (live state cfg)."""
    del env_ids
    cur = stages[0]
    for s in stages:
        if env.common_step_counter > s["step"]:
            cur = s
    sl = _skate_state(env).get("slope")
    if sl is not None:
        sl["cfg"].update({k: v for k, v in cur.items() if k != "step"})
    return torch.tensor([float(cur.get("max_deg", 0.0))])


def skate_slope_obs_level(env, v: torch.Tensor) -> torch.Tensor:
    """(B, 3) world vectors rotated into the gravity-levelled frame (the frame a real slope world would have: the
    rotation taking the true up, -gravity, onto +z). Identity without slopes."""
    sl = (_skate_cache(env).get("state") or {}).get("slope")
    if sl is None:
        return v
    n0 = -sl["G"] / sl["G"].norm(dim=1, keepdim=True).clamp_min(1e-6)
    z = torch.zeros_like(n0)
    z[:, 2] = 1.0
    axis = torch.cross(n0, z, dim=1)
    s = axis.norm(dim=1, keepdim=True)
    cth = (n0 * z).sum(dim=1, keepdim=True)
    k = axis / s.clamp_min(1e-9)
    out = v * cth + torch.cross(k, v, dim=1) * s + k * (k * v).sum(dim=1, keepdim=True) * (1.0 - cth)
    return torch.where(s > 1e-7, out, v)

# Actor board block, in order: (name, nominal mean, nominal std). The warm-start converter
# seeds the normalizer of the appended inputs from this table.
SKATE_ACTOR_OBS_STATS = (
    ("board_speed", 0.20, 0.12),
    ("mode_speed", 0.20, 0.12),
    ("v_off", 0.14, 0.06),
    ("is_glide", 0.40, 0.50),
    ("is_stop", 0.10, 0.30),
    ("time_in_mode", 1.00, 0.70),
    ("board_yaw_rel", 0.00, 0.20),
    ("left_sole_x", 0.00, 0.02),
    ("left_sole_y", 0.02, 0.025),
    ("right_sole_x", 0.00, 0.03),
    ("right_sole_y", -0.06, 0.03),
    ("left_on_deck", 0.90, 0.30),
    ("right_on_deck", 0.40, 0.50),
    ("right_on_floor", 0.50, 0.50),
)
_SKATE_T_MODE_OBS_MAX = 3.0
SKATE_V6B_SOLE_CLAMP = 0.2    # m, v6b clamp of the sole xy in the board block (riding soles: |x|, |y| < 0.1)
SKATE_V6B_DYAW_CLAMP = 1.0    # rad, v6b clamp of the board-yaw-relative obs


def _skate_cache(env) -> dict:
    return env.__dict__.setdefault("_skate_cache", {})


def _skate_geom_id(env, asset: Entity, name: str) -> int:
    cache = _skate_cache(env)
    key = ("geom", id(asset), name)
    if key not in cache:
        ids, _ = asset.find_geoms(rf"^{name}$")
        assert len(ids) == 1, f"geom {name}: {ids}"
        cache[key] = ids[0]
    return cache[key]


def _yaw_from_quat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _quat_from_rpy_t(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    return torch.stack([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dim=1)


def _unit(q: torch.Tensor, axis: int) -> torch.Tensor:
    e = torch.zeros(q.shape[0], 3, device=q.device, dtype=q.dtype)
    e[:, axis] = 1.0
    return quat_apply(q, e)


def _skate_board_quat(env, board_name: str = "board") -> torch.Tensor:
    """(B, 4) deck orientation in the CANONICAL frame (v6b, st cfg "v6b"): the board quat turned by pi about its up
    axis while st["flip"] (robot on the board's left at the step-on). The board is symmetric under that turn (front /
    rear trucks, wheels and nose / tail mirror, robot/assets/skateboard.xml), so every deck-frame quantity then
    reads as if the robot had stepped on from the right and rides toward deck +x. Raw quat otherwise."""
    q = env.scene[board_name].data.root_link_quat_w
    st = _skate_cache(env).get("state")
    if st is None or not st["cfg"].get("v6b", False):
        return q
    qf = torch.stack([-q[:, 3], q[:, 2], -q[:, 1], q[:, 0]], dim=1)   # q * (0, 0, 0, 1)
    return torch.where(st["flip"].unsqueeze(1), qf, q)


def skate_board_heading(env, board_name: str = "board") -> torch.Tensor:
    """Unit xy vector of the deck's long axis (B, 2), canonical frame (_skate_board_quat)."""
    h = _unit(_skate_board_quat(env, board_name), 0)[:, :2]
    return h / h.norm(dim=1, keepdim=True).clamp_min(1e-6)


def skate_board_fwd_speed(env, board_name: str = "board") -> torch.Tensor:
    board: Entity = env.scene[board_name]
    v = board.data.root_link_lin_vel_w[:, :2]
    return torch.nan_to_num((v * skate_board_heading(env, board_name)).sum(dim=1), nan=0.0)


def _skate_body_masses(env, entity_name: str) -> torch.Tensor:
    cache = _skate_cache(env)
    key = ("mass", entity_name)
    if key not in cache:
        ids = env.scene[entity_name].indexing.body_ids.cpu().numpy()
        cache[key] = torch.tensor(env.sim.mj_model.body_mass[ids], device=env.device, dtype=torch.float)
    return cache[key]


def skate_system_fwd_speed(env, board_name: str = "board") -> torch.Tensor:
    """Forward speed (along the deck) of the robot + board centre of mass. Internal motions (lunges, shoving the
    deck with the board foot) cannot change it; only floor forces can (push foot, wheel drag, braking)."""
    v = _skate_system_vel_xy(env, board_name)
    return torch.nan_to_num((v * skate_board_heading(env, board_name)).sum(dim=1), nan=0.0)


def _skate_system_vel_xy(env, board_name: str = "board") -> torch.Tensor:
    """(B, 2) world xy velocity of the robot + board centre of mass."""
    robot: Entity = env.scene["robot"]
    board: Entity = env.scene[board_name]
    mr, mb = _skate_body_masses(env, "robot"), _skate_body_masses(env, board_name)
    p = (robot.data.body_com_lin_vel_w[..., :2] * mr[None, :, None]).sum(dim=1)
    p = p + (board.data.body_com_lin_vel_w[..., :2] * mb[None, :, None]).sum(dim=1)
    return p / (mr.sum() + mb.sum())


def skate_system_planar_speed(env, board_name: str = "board") -> torch.Tensor:
    """v6 COAST charge speed: |system COM xy velocity|, signed by its projection on the deck heading (board yaw wiggle
    does not change it)."""
    v = _skate_system_vel_xy(env, board_name)
    sgn = torch.where((v * skate_board_heading(env, board_name)).sum(dim=1) >= 0.0, 1.0, -1.0)
    return torch.nan_to_num(v.norm(dim=1) * sgn, nan=0.0)


def _skate_contact(env, sensor_name: str) -> torch.Tensor:
    found = env.scene.sensors[sensor_name].data.found
    if found.dim() > 1:
        found = found.sum(dim=-1)
    return (found > 0).float()


def _skate_air_time(env, sensor_name: str) -> torch.Tensor:
    """Time since ANY primary of the sensor last touched (min over primaries: deck sensors have one primary per
    deck part)."""
    air = env.scene.sensors[sensor_name].data.current_air_time
    return air.min(dim=-1).values if air.dim() > 1 else air


def _skate_recent_contact(env, sensor_name: str, window_s: float) -> torch.Tensor:
    """1 if the sensor is in contact now or lost contact less than window_s ago."""
    return ((_skate_contact(env, sensor_name) > 0) | (_skate_air_time(env, sensor_name) < window_s)).float()


def _skate_normal_force(env, sensor_name: str) -> torch.Tensor:
    """Vertical contact force magnitude (N) summed over the sensor's primaries (B,)."""
    f = env.scene.sensors[sensor_name].data.force
    if f.dim() == 2:
        f = f.unsqueeze(1)
    return torch.nan_to_num(f[..., 2].sum(dim=1).abs(), nan=0.0, posinf=0.0, neginf=0.0)


def _skate_robot_weight(env) -> float:
    cache = _skate_cache(env)
    if "weight" not in cache:
        cache["weight"] = float(_skate_body_masses(env, "robot").sum()) * 9.81
    return cache["weight"]


def skate_sole_in_deck(env, foot_geom: str = "left_foot_collision", board_name: str = "board") -> torch.Tensor:
    """Sole geom position relative to the deck centre, in the (canonical) deck frame (B, 3)."""
    robot: Entity = env.scene["robot"]
    board: Entity = env.scene[board_name]
    gid = _skate_geom_id(env, robot, foot_geom)
    rel = robot.data.geom_pos_w[:, gid] - board.data.root_link_pos_w
    return quat_apply_inverse(_skate_board_quat(env, board_name), rel)


# ── mode machine ────────────────────────────────────────────────────────────


def _skate_glide_potential(env, c: dict) -> torch.Tensor:
    """-(L1 distance of both soles from their glide spots on the deck, xy) / 5 cm."""
    pl = skate_sole_in_deck(env, "left_foot_collision", c["board_name"])[:, :2]
    pr = skate_sole_in_deck(env, "right_foot_collision", c["board_name"])[:, :2]
    tl = torch.tensor(c["glide_left"], device=env.device, dtype=pl.dtype)
    tr = torch.tensor(c["glide_right"], device=env.device, dtype=pl.dtype)
    d = (pl - tl).abs().sum(dim=1) + (pr - tr).abs().sum(dim=1)
    return torch.nan_to_num(-d / 0.05, nan=0.0)


def _skate_push_potential(env, c: dict) -> torch.Tensor:
    """-(L1 distance of the push (right) sole from its push-stance spot beside the deck, xy deck frame) / 5 cm."""
    pr = skate_sole_in_deck(env, "right_foot_collision", c["board_name"])[:, :2]
    tr = torch.tensor(c["push_right"], device=env.device, dtype=pr.dtype)
    return torch.nan_to_num(-(pr - tr).abs().sum(dim=1) / 0.05, nan=0.0)


def _skate_v_on(env, c: dict) -> tuple[torch.Tensor, torch.Tensor]:
    cmd = env.command_manager.get_command(c["command_name"])[:, 0]
    return cmd, cmd.clamp_min(c["stop_cmd"])


def skate_feasible_kappa(kappa: torch.Tensor, v: torch.Tensor, a_lat: float | None, v_floor: float = 0.3) -> torch.Tensor:
    """v7d: the commanded curvature clamped to what the lateral acceleration a_lat allows at forward speed v,
    |kappa| <= a_lat / max(v, v_floor)^2 (None = unclamped). skate_v7c it3449
    turned at kappa +0.46 / -0.30 at 0.45 m/s, +0.17 / -0.03 at 1.4 and ~0 at 2.2 m/s downhill: kappa 1 at 2 m/s
    asks for 4 m/s^2 (a 22 deg lean), so the target was unreachable at speed and the policy stopped steering."""
    if a_lat is None:
        return kappa
    kmax = a_lat / v.clamp_min(v_floor).pow(2)
    return torch.maximum(torch.minimum(kappa, kmax), -kmax)


def _skate_steer_update(env, st: dict) -> None:
    """Steering state (v4), advanced with the mode machine (``fresh`` = first update after a reset). twist[2] is
    the curvature kappa, target yaw rate = kappa * forward board speed. yaw_lp is the board yaw rate low-passed with
    tau_yaw (what the curvature term tracks; the wiggle cost charges the high-pass remainder) and head_err the leaky
    integral of (yaw rate - target): heading drift that a per-step rate error hides."""
    c = st["cfg"]
    dt = env.step_dt
    fresh = st["fresh"]
    board: Entity = env.scene[c["board_name"]]
    w = torch.nan_to_num(board.data.root_link_ang_vel_w[:, 2], nan=0.0)
    kappa = env.command_manager.get_command(c["command_name"])[:, 2]
    vb = skate_board_fwd_speed(env, c["board_name"]).clamp_min(0.0)
    kappa = skate_feasible_kappa(kappa, vb, c.get("a_lat"))
    st["kappa_t"] = kappa
    w_t = kappa * vb
    ay = dt / (c["tau_yaw"] + dt)
    st["yaw_lp"] = torch.where(fresh, w, st["yaw_lp"] + ay * (w - st["yaw_lp"]))
    st["yaw_target"] = w_t
    hc = float(c.get("head_clamp", 0.5))
    he = (st["head_err"] * (1.0 - dt / c["tau_heading"]) + (w - w_t) * dt).clamp(-hc, hc)
    st["head_err"] = torch.where(fresh, torch.zeros_like(he), he)


def _skate_transition_update(env, st: dict) -> None:
    """Step-on / step-off state (v5), advanced with the mode machine.
    - mounted: per-env latch, set once the board foot carries mount_load_share of the weight on the deck (riding
      spawns start latched). premount_prev = not mounted during the step just simulated (what the reach potential is
      credited to).
    - d_reach: change of -(L1 distance of the left sole from its deck spot)/5 cm, step-on shaping.
    - d_off: change of -(number of soles touching the deck), step-off shaping (+1 per foot that leaves the deck)."""
    c = st["cfg"]
    fresh = st.get("rebase", st["fresh"])   # v6b: also re-baseline when the canonical frame turns
    w = _skate_robot_weight(env)
    st["premount_prev"] = ~st["mounted"]
    st["mounted"] = st["mounted"] | (_skate_normal_force(env, c["board_foot_sensor"]) >= c["mount_load_share"] * w)
    pl = skate_sole_in_deck(env, "left_foot_collision", c["board_name"])[:, :2]
    tl = torch.tensor(c["glide_left"], device=env.device, dtype=pl.dtype)
    l1 = (pl - tl).abs().sum(dim=1)
    if c.get("reach_sat") is not None:
        # v6b: saturated, the near field only (the walk-up is approach_progress's; both paying for it double-counts)
        l1 = l1.clamp(max=c["reach_sat"])
    phi_r = torch.nan_to_num(-l1 / 0.05, nan=0.0)
    d_r = (phi_r - st["phi_reach_prev"]).clamp(-0.5, 0.5)
    st["d_reach"] = torch.where(fresh, torch.zeros_like(d_r), d_r)
    st["phi_reach_prev"] = phi_r
    # debounced (contact now or within 0.1 s): sole chatter on the deck would otherwise flip the potential every step
    phi_o = -(_skate_recent_contact(env, c["board_foot_sensor"], 0.1)
              + _skate_recent_contact(env, c["push_foot_sensor"], 0.1))
    d_o = phi_o - st["phi_off_prev"]
    st["d_off"] = torch.where(fresh, torch.zeros_like(d_o), d_o)
    st["phi_off_prev"] = phi_o
    # v5d step-off reach: -(how far each sole is still left of its spot beside the board, deck-frame y) / 5 cm. The
    # reverse of the mount (skate_v5c iteration 500 sat in the push stance 96 % of the time after the flag: no dense
    # path from there to the floor). One-sided: further right than the spot costs nothing.
    if "phi_leave_prev" in st:
        pr = skate_sole_in_deck(env, "right_foot_collision", c["board_name"])[:, :2]
        phi_l = torch.nan_to_num(-((pl[:, 1] - c["step_off_left_y"]).clamp(min=0.0)
                                   + (pr[:, 1] - c["step_off_right_y"]).clamp(min=0.0)) / 0.05, nan=0.0)
        d_l = (phi_l - st["phi_leave_prev"]).clamp(-0.5, 0.5)
        st["d_leave"] = torch.where(fresh, torch.zeros_like(d_l), d_l)
        st["phi_leave_prev"] = phi_l
    if c.get("v6b", False):
        # v6b walk-up (approach from anywhere within 0.5 m): trunk distance to the staging spot (canonical deck
        # frame) and heading error to the canonical deck heading, as potentials.
        #   phi_appr = -max(d - 2 cm, 0) / 5 cm (0.5 m of walking ~ +10 in all, x weight)
        #   phi_head = -|dyaw| / 0.5 rad * clamp(1 - d / 0.3 m, 0, 1) (lining up only matters near the spot)
        robot: Entity = env.scene["robot"]
        board: Entity = env.scene[c["board_name"]]
        qc = _skate_board_quat(env, c["board_name"])
        rel = quat_apply_inverse(qc, robot.data.root_link_pos_w - board.data.root_link_pos_w)[:, :2]
        stg = torch.tensor(c["staging"], device=env.device, dtype=rel.dtype)
        d = torch.nan_to_num((rel - stg).norm(dim=1), nan=1.0)
        st["d_stage"] = d
        dyaw = wrap_to_pi(_yaw_from_quat(robot.data.root_link_quat_w) - _yaw_from_quat(qc))
        phi_a = -(d - 0.02).clamp(min=0.0) / 0.05
        phi_h = torch.nan_to_num(-dyaw.abs() / 0.5 * (1.0 - d / 0.3).clamp(0.0, 1.0), nan=0.0)
        d_a = (phi_a - st["phi_appr_prev"]).clamp(-0.5, 0.5)
        d_h = (phi_h - st["phi_head_prev"]).clamp(-0.5, 0.5)
        st["d_appr"] = torch.where(fresh, torch.zeros_like(d_a), d_a)
        st["d_head"] = torch.where(fresh, torch.zeros_like(d_h), d_h)
        st["phi_appr_prev"] = phi_a
        st["phi_head_prev"] = phi_h


def _skate_flip_update(env, st: dict) -> None:
    """v6 canonical side latch (skate_approach_obs, v6b _skate_board_quat): the deck frame turned by pi while the
    robot is on the board's left. v6a: 3 cm hysteresis around the long axis on the trunk's raw deck-frame y, frozen
    once mounted, written by reset_skate_state (fresh envs keep it). v6b: frozen all episode. st["rebase"] = fresh
    (the potentials' re-baseline mask)."""
    c = st["cfg"]
    fresh = st["fresh"]
    st["rebase"] = fresh
    if c.get("v6b", False):
        # v6b: frozen for the whole episode (reset_skate_state picks the side; the play sim at the hand-off). A
        # latch that turns during the walk-up made crossing the long axis at a board end a positive-sum loop
        # (verifier 2026-11-03: +0.9..3.3 per loop, the dropped potential jumps all favoured it).
        st["flip_changed"] = torch.zeros_like(fresh)
        return
    robot: Entity = env.scene["robot"]
    board: Entity = env.scene[c["board_name"]]
    y = quat_apply_inverse(board.data.root_link_quat_w,
                           robot.data.root_link_pos_w - board.data.root_link_pos_w)[:, 1]
    f = torch.where(st["flip"], y > -0.03, y > 0.03)
    new = torch.where(st["mounted"] | fresh, st["flip"], f)
    st["flip_changed"] = new != st["flip"]
    st["flip"] = new


def _skate_mode_update(env, st: dict) -> None:
    c = st["cfg"]
    dt = env.step_dt
    if "flip" in st:
        _skate_flip_update(env, st)
    cmd, v_on = _skate_v_on(env, c)
    v_off = v_on - c["gap"]
    fresh = st.get("rebase", st["fresh"])   # v6b: also re-baseline when the canonical frame turns
    if "sensor" in st:      # skate student: the board unit's sensor stream + the runtime-side observer machine
        _skate_sensor_update(env, st)
    v_sys = skate_system_fwd_speed(env, c["board_name"])
    phi_g = _skate_glide_potential(env, c)
    # v7 slopes: the passive rolling acceleration of the step just simulated (zeros on the flat): speed that gravity
    # gives is neither paid (push progress) nor charged (coast / brake)
    g_fwd = _skate_slope_g_fwd(env, st)
    if "slope" in st:
        g_fwd = torch.where(fresh, torch.zeros_like(g_fwd), g_fwd)
        st["slope"]["g_fwd"].copy_(g_fwd)
        # low-passed like the mode speed (tau_mode), for the brake term: accel lags a slope transition
        a_lp = dt / (c["tau_mode"] + dt)
        g_lp = st["slope"]["g_lp"]
        g_lp.copy_(torch.where(fresh, g_fwd, g_lp + a_lp * (g_fwd - g_lp)))
    grav = g_fwd * dt

    # Potential differences over the step just simulated, credited to the mode that was active during it.
    # Push potential speed = min(system, board): a forward fall over the planted push foot raises v_sys while the
    # board rolls back (verifier 2026-11-02: holding PUSH_MID and toppling earned +3.97 of push_progress), and
    # shoving the deck raises v_board only. Not clamped: a clamp breaks the telescoping sum (jerk the board,
    # recover slowly, keep the clipped part); resets are handled by ``fresh``. Floored at 0 so a board kicked
    # backward (falls) costs at most the speed that was had, not the kick.
    v_push = torch.minimum(v_sys, skate_board_fwd_speed(env, c["board_name"])).clamp_min(0.0)
    # v7: against the speed a DOWNHILL alone would have given. Uphill losses are charged like wheel drag on the flat
    # (reviewer 2026-11-03: discounting them paid ~10/s for holding speed up a 3 deg grade, and a push / roll-back
    # pump cycle netted +20 * v_peak with no height gained).
    v_exp = (st["v_sys_prev"] + grav.clamp_min(0.0)).clamp_min(0.0)
    d_push = torch.minimum(v_push, v_on) - torch.minimum(v_exp, v_on)
    d_glide = (phi_g - st["phi_glide_prev"]).clamp(-0.5, 0.5)
    phi_p = _skate_push_potential(env, c)
    d_stance = (phi_p - st["phi_stance_prev"]).clamp(-0.5, 0.5)
    st["d_push"] = torch.where(fresh, torch.zeros_like(d_push), d_push)
    # v6 COAST: speed lost faster than natural coasting (coast_decel) is charged at the push-progress rate, natural
    # coasting is free (so braking in COAST to be paid for regaining the speed under throttle nets zero).
    if "d_coast" in st:
        # Charged on the PLANAR system COM speed (signed by the deck heading), not min(v_sys, v_board): the 61 g board's
        # speed jumps with every foot-deck force and a one-sided charge ratchets on that noise (verifier 2026-11-03:
        # ~1-3 / s at 5-10 mm/s step noise); the projection onto the heading leaks board yaw wiggle in. Internal forces
        # cannot move the system COM, so only floor forces (push foot down, wheel drag) change it.
        v_c = skate_system_planar_speed(env, c["board_name"])
        # v7: uphill deceleration is natural coasting; a downhill's extra speed is not owed (checking speed on a
        # ramp is allowed: reviewer 2026-11-03, it would cost ~25/s at 9 deg)
        d_coast = ((v_c - st["v_coast_prev"]) - grav.clamp(max=0.0) + c["coast_decel"] * dt).clamp(max=0.0)
        st["d_coast"] = torch.where(fresh, torch.zeros_like(d_coast), d_coast)
        st["credit_coast"] = st["coast"].clone()
        st["v_coast_prev"] = v_c
    st["d_glide"] = torch.where(fresh, torch.zeros_like(d_glide), d_glide)
    st["d_stance"] = torch.where(fresh, torch.zeros_like(d_stance), d_stance)
    st["v_sys_prev"] = v_push
    st["phi_glide_prev"] = phi_g
    st["phi_stance_prev"] = phi_p
    st["credit_mode"] = st["mode"].clone()

    a = dt / (c["tau_mode"] + dt)
    v_old = st["v_mode"]
    v_new = torch.where(fresh, v_sys, v_old + a * (v_sys - v_old))
    st["accel"] = torch.where(fresh, torch.zeros_like(v_new), (v_new - v_old) / dt)
    st["v_mode"] = v_new
    b = dt / (c["tau_cruise"] + dt)
    st["v_cruise"] = torch.where(fresh, st["v_cruise"], st["v_cruise"] + b * (v_sys - st["v_cruise"]))

    _skate_steer_update(env, st)
    _skate_transition_update(env, st)

    mode = st["mode"]
    t = st["t_mode"] + dt
    can = t >= c["min_dwell"]
    stop = cmd < c["stop_cmd"]
    # skate student: PUSH <-> GLIDE switch on the SENSED mode speed (low-passed received wheel speed), so the mode the
    # rewards use is the one a robot-side runtime can compute; v_mode (true system speed) still feeds the rewards
    v_sw = st["sensor"]["v_s"] if (c.get("mode_src") == "sensed" and "sensor" in st) else v_new
    up = ~stop & (v_sw >= v_on)
    down = ~stop & (v_sw <= v_off)
    new = torch.where(stop, torch.full_like(mode, SKATE_STOP), mode)
    new = torch.where(~stop & (mode == SKATE_STOP),
                      torch.where(up, torch.full_like(mode, SKATE_GLIDE), torch.full_like(mode, SKATE_PUSH)), new)
    new = torch.where((mode == SKATE_PUSH) & can & up, torch.full_like(mode, SKATE_GLIDE), new)
    new = torch.where((mode == SKATE_GLIDE) & can & down, torch.full_like(mode, SKATE_PUSH), new)
    if "coast" in st:
        # v6 COAST: GLIDE from any mode at once, never back to PUSH while it lasts. Riding only (verifier 2026-11-03:
        # forcing GLIDE before the mount latch put a step-on standing beside the board under glide_floor -4/s, so
        # falling beat stepping on: skate_v6a iteration 84 trunk_low terminations rising).
        coast = (cmd >= c["stop_cmd"]) & (cmd < c["coast_cmd_hi"]) & st["mounted"]
        new = torch.where(coast, torch.full_like(mode, SKATE_GLIDE), new)
        st["coast"] = coast
    if c.get("v6b", False):
        # v6b: no GLIDE before the mount latch (walking along the board at v_on would otherwise switch the walk-up
        # into GLIDE and its floor costs)
        new = torch.where(~st["mounted"] & (new == SKATE_GLIDE), torch.full_like(mode, SKATE_PUSH), new)
    changed = new != mode
    st["switches"] += changed.float()
    st["t_mode"] = torch.where(changed, torch.zeros_like(t), t)
    st["mode"] = new
    if "climb" in st:       # step 5: virtual height over the step just simulated (the grade it ran with)
        _skate_climb_update(env, st, fresh)
    st["fresh"] = torch.zeros_like(fresh)
    _skate_slope_update(env, st)   # v7: the gravity of the next physics step
    if "climb" in st:       # step 5: score finished uphill segments (performance-gated grade level)
        _skate_climb_post(env, st)


def _skate_state(env) -> dict:
    """Per-env mode-machine state, advanced once per env step."""
    cache = _skate_cache(env)
    st = cache.get("state")
    if st is None:
        n, dev = env.num_envs, env.device
        z = lambda: torch.zeros(n, device=dev)
        st = dict(
            cfg=dict(SKATE_MODE_DEFAULTS), step=env.common_step_counter,
            mode=torch.zeros(n, dtype=torch.long, device=dev), credit_mode=torch.zeros(n, dtype=torch.long, device=dev),
            t_mode=z(), v_mode=z(), v_cruise=z(), accel=z(), v_sys_prev=z(), phi_glide_prev=z(), d_push=z(),
            d_glide=z(), phi_stance_prev=z(), d_stance=z(), switches=z(), fresh=torch.ones(n, dtype=torch.bool, device=dev),
            yaw_lp=z(), yaw_target=z(), head_err=z(), kappa_t=z(),
            mounted=torch.ones(n, dtype=torch.bool, device=dev), premount_prev=torch.zeros(n, dtype=torch.bool, device=dev),
            phi_reach_prev=z(), d_reach=z(), phi_off_prev=z(), d_off=z(),
            dismount_now=torch.zeros(n, dtype=torch.bool, device=dev), phi_leave_prev=z(), d_leave=z(),
            coast=torch.zeros(n, dtype=torch.bool, device=dev), credit_coast=torch.zeros(n, dtype=torch.bool, device=dev),
            d_coast=z(), flip=torch.zeros(n, dtype=torch.bool, device=dev), v_coast_prev=z(),
            flip_changed=torch.zeros(n, dtype=torch.bool, device=dev), rebase=torch.ones(n, dtype=torch.bool, device=dev),
            d_stage=z(), appr_d0=z(), phi_appr_prev=z(), d_appr=z(), phi_head_prev=z(), d_head=z(),
        )
        cache["state"] = st
    if st["step"] != env.common_step_counter:
        st["step"] = env.common_step_counter
        _skate_mode_update(env, st)
    return st


def skate_mode_setup(
    env,
    env_ids,
    gap: float = SKATE_MODE_DEFAULTS["gap"],
    grace: float = SKATE_MODE_DEFAULTS["grace"],
    glide_left: tuple = SKATE_MODE_DEFAULTS["glide_left"],
    glide_right: tuple = SKATE_MODE_DEFAULTS["glide_right"],
    push_right: tuple = SKATE_MODE_DEFAULTS["push_right"],
    min_dwell: float = SKATE_MODE_DEFAULTS["min_dwell"],
    tau_mode: float = SKATE_MODE_DEFAULTS["tau_mode"],
    extra: dict | None = None,
):
    """Startup event: write the mode-machine constants (gap, grace, glide sole spots, ...) into the state. ``extra``:
    any other SKATE_MODE_DEFAULTS keys (v6b, reach_sat, ...)."""
    del env_ids
    st = _skate_state(env)
    st["cfg"].update(gap=gap, grace=grace, glide_left=tuple(glide_left),
                     glide_right=tuple(glide_right), push_right=tuple(push_right),
                     min_dwell=min_dwell, tau_mode=tau_mode)
    if extra:
        unknown = set(extra) - set(SKATE_MODE_DEFAULTS)
        assert not unknown, f"unknown skate mode cfg keys {unknown}"
        st["cfg"].update(extra)


def skate_mode_curriculum(env, env_ids, stages: list[dict], command_name: str = "twist") -> torch.Tensor:
    """Stage the mode thresholds and the cruise-speed command range: stages = [{"step", "gap", "grace",
    "cmd_range"}, optional "a_lat" (v7d)]. Writes the live state cfg and the live command term cfg."""
    del env_ids
    cur = stages[0]
    for s in stages:
        if env.common_step_counter > s["step"]:
            cur = s
    st = _skate_state(env)
    st["cfg"]["gap"] = cur["gap"]
    st["cfg"]["grace"] = cur["grace"]
    if "a_lat" in cur:      # v7d: staged lateral-acceleration bound of the curvature target
        st["cfg"]["a_lat"] = cur["a_lat"]
    env.command_manager.get_term(command_name).cfg.ranges.lin_vel_x = tuple(cur["cmd_range"])
    return torch.tensor([cur["gap"]])


def skate_approach_ring_curriculum(env, env_ids, stages: list[dict], event_name: str = "reset_skate") -> torch.Tensor:
    """v6b: stage the walk-up spawns, stages = [{"step", "r_max", "yaw"}]: every reset spawn with a "ring" gets
    ring = (its clear, r_max) and yaw_noise = yaw (live EventManager term cfg)."""
    del env_ids
    cur = stages[0]
    for s in stages:
        if env.common_step_counter > s["step"]:
            cur = s
    for s in env.event_manager.get_term_cfg(event_name).params["spawns"]:
        if "ring" in s:
            s["ring"] = (s["ring"][0], cur["r_max"])
            s["yaw_noise"] = cur["yaw"]
    return torch.tensor([float(cur["r_max"])])


def skate_event_param_curriculum(env, env_ids, event_name: str, param: str, stages: list[dict]) -> torch.Tensor:
    """Generic step-staged event parameter: stages = [{"step", "value"}] (live EventManager term cfg)."""
    del env_ids
    cur = stages[0]["value"]
    for s in stages:
        if env.common_step_counter > s["step"]:
            cur = s["value"]
    env.event_manager.get_term_cfg(event_name).params[param] = cur
    v = cur[0] if isinstance(cur, (tuple, list)) else cur
    return torch.tensor([float(v)])


# ── observations ────────────────────────────────────────────────────────────


def skate_actor_obs(
    env,
    board_name: str = "board",
    board_foot_sensor: str = "board_foot_deck",
    push_foot_sensor: str = "push_foot_deck",
    push_floor_sensor: str = "push_foot_ground",
) -> torch.Tensor:
    """Board / mode block, layout SKATE_ACTOR_OBS_STATS (14D)."""
    robot: Entity = env.scene["robot"]
    board: Entity = env.scene[board_name]
    st = _skate_state(env)
    c = st["cfg"]
    cmd, v_on = _skate_v_on(env, c)
    stop = (st["mode"] == SKATE_STOP).float()
    v_off = (v_on - c["gap"]) * (1.0 - stop)
    if "coast" in st:   # v6 COAST: no push threshold
        v_off = v_off * (~st["coast"]).float()
    dyaw = wrap_to_pi(_yaw_from_quat(_skate_board_quat(env, board_name)) - _yaw_from_quat(robot.data.root_link_quat_w))
    left = skate_sole_in_deck(env, "left_foot_collision", board_name)[:, :2]
    right = skate_sole_in_deck(env, "right_foot_collision", board_name)[:, :2]
    if c.get("v6b", False):
        # v6b walk-ups start up to 0.6 m out: unclamped, the soles / dyaw hit the warm-started normalizer at 10-20
        # std (count ~1e9, it will not adapt; verifier 2026-11-03). Riding values sit well inside; the approach block
        # carries the far field.
        left, right = left.clamp(-SKATE_V6B_SOLE_CLAMP, SKATE_V6B_SOLE_CLAMP), right.clamp(
            -SKATE_V6B_SOLE_CLAMP, SKATE_V6B_SOLE_CLAMP)
        dyaw = dyaw.clamp(-SKATE_V6B_DYAW_CLAMP, SKATE_V6B_DYAW_CLAMP)
    out = torch.cat([
        skate_board_fwd_speed(env, board_name).unsqueeze(1),
        st["v_mode"].unsqueeze(1),
        v_off.unsqueeze(1),
        (st["mode"] == SKATE_GLIDE).float().unsqueeze(1),
        stop.unsqueeze(1),
        st["t_mode"].clamp(max=_SKATE_T_MODE_OBS_MAX).unsqueeze(1),
        dyaw.unsqueeze(1),
        left, right,
        _skate_contact(env, board_foot_sensor).unsqueeze(1),
        _skate_contact(env, push_foot_sensor).unsqueeze(1),
        _skate_contact(env, push_floor_sensor).unsqueeze(1),
    ], dim=1)
    assert out.shape[1] == len(SKATE_ACTOR_OBS_STATS)
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def skate_critic_obs(
    env,
    board_name: str = "board",
    board_foot_sensor: str = "board_foot_deck",
    push_foot_sensor: str = "push_foot_deck",
) -> torch.Tensor:
    """22D privileged state: board lin/ang vel (deck frame), deck up vector, trunk position relative to the deck,
    right sole xyz on the deck, truck angles, system speed, mode-speed accel, cruise speed, both soles' deck load
    shares."""
    robot: Entity = env.scene["robot"]
    board: Entity = env.scene[board_name]
    st = _skate_state(env)
    q = _skate_board_quat(env, board_name)
    lin = quat_apply_inverse(q, board.data.root_link_lin_vel_w)
    ang = quat_apply_inverse(q, board.data.root_link_ang_vel_w)
    up = skate_slope_obs_level(env, _unit(q, 2))   # v7: deck up relative to gravity (the floor is flat)
    trunk = quat_apply_inverse(q, robot.data.root_link_pos_w - board.data.root_link_pos_w)
    push_sole = skate_sole_in_deck(env, "right_foot_collision", board_name)
    if st["cfg"].get("v6b", False):   # as skate_actor_obs (critic normalizer warm-started too)
        trunk = trunk.clamp(-0.25, 0.25)
        push_sole = push_sole.clamp(-SKATE_V6B_SOLE_CLAMP, SKATE_V6B_SOLE_CLAMP)
    cache = _skate_cache(env)
    if "trucks" not in cache:
        cache["trucks"], _ = board.find_joints(r"^passive_(front|rear)_truck$")
    trucks = board.data.joint_pos[:, cache["trucks"]]
    if st["cfg"].get("v6b", False):   # canonical frame: the rear truck is the front one (its axis maps onto it)
        trucks = torch.where(st["flip"].unsqueeze(1), trucks.flip(1), trucks)
    w = _skate_robot_weight(env)
    out = torch.cat([
        lin, ang, up, trunk, push_sole, trucks,
        skate_system_fwd_speed(env, board_name).unsqueeze(1),
        st["accel"].clamp(-5.0, 5.0).unsqueeze(1),
        st["v_cruise"].unsqueeze(1),
        (_skate_normal_force(env, board_foot_sensor) / w).clamp(max=3.0).unsqueeze(1),
        (_skate_normal_force(env, push_foot_sensor) / w).clamp(max=3.0).unsqueeze(1),
    ], dim=1)
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# ── rewards ─────────────────────────────────────────────────────────────────


def skate_push_progress(env, scale: float = 0.1, board_foot_sensor: str = "board_foot_deck",
                        floor_window_s: float | None = None, push_floor_sensor: str = "push_foot_ground",
                        terminal_charge: bool = False) -> torch.Tensor:
    """PUSH and GLIDE, potential-based: d min(v_sys, v_board, v_on) / scale per step (divided by dt, so the episode
    sum is weight * dv / scale). Signed: braking or wheel drag is charged; gains count only while the board foot is
    on the deck. Falls over the push foot (board rolls back) and deck shoves (v_sys unchanged) earn nothing, and
    rocking / pump-and-brake net zero.

    Step 5 ``floor_window_s``: gains also need the push foot on the floor now or within that window (a stroke); board
    wiggle (tic-tac) propulsion with both feet on the deck earns nothing. None = no such gate (as before)."""
    st = _skate_state(env)
    d = st["d_push"]
    riding = _skate_contact(env, board_foot_sensor)
    if floor_window_s is not None:
        riding = riding * _skate_recent_contact(env, push_floor_sensor, floor_window_s)
    d = torch.where(d > 0, d * riding, d)
    if "credit_coast" in st:   # v6 COAST: only decel beyond natural coasting is charged, nothing is paid
        d = torch.where(st["credit_coast"], st["d_coast"], d)
    if terminal_charge:
        # Student (2026-11-06 hold-W collapse): the potential was never charged at a fall (Phi(terminal) = 0 missing,
        # ``fresh`` zeroes the next delta), so a robot falling at 1.2 m/s kept the +20..25 banked accelerating and the
        # dt-scaled termination cost is only -0.06: falling at W returned as much as riding it, and PPO drifted to a
        # lunge that pitches over in ~1 s. Charge the potential held at a non-timeout termination; time-outs bootstrap.
        _, v_on = _skate_v_on(env, st["cfg"])
        held = torch.minimum(st["v_sys_prev"], v_on).clamp_min(0.0)    # this step's potential speed
        d = torch.where(env.termination_manager.terminated, d - held, d)
    # Credited in PUSH and GLIDE (not only PUSH): a PUSH-only potential pays every PUSH->GLIDE->PUSH cycle
    # 20 * (v_on - v_end_of_glide) even when the speed was shed by braking in the GLIDE dwell (brake costs only
    # 4 per m/s), i.e. pump-and-brake / rocking over the push foot beat an honest glide.
    return d * (st["credit_mode"] != SKATE_STOP).float() / (scale * env.step_dt)


def skate_in_mode(env, mode: int = SKATE_PUSH) -> torch.Tensor:
    """1 while the env is in ``mode`` (>= 0; a negative weight makes it a time-in-mode cost)."""
    return (_skate_state(env)["mode"] == mode).float()


def _skate_look_forward(
    env,
    board_name: str,
    yaw_std: float,
    head_yaw_std: float,
    neck_std: float,
    hip_yaw_std: float,
    tilt_std: float,
    roll_std: float | None = None,
) -> torch.Tensor:
    """Product of look-forward Gaussians. roll_std None: one factor on the total trunk tilt (tilt_std). With roll_std
    (v4 steering: leaning into a turn is the steering input), pitch keeps tilt_std and roll gets its own, wider std."""
    robot: Entity = env.scene["robot"]
    dyaw = wrap_to_pi(_yaw_from_quat(_skate_board_quat(env, board_name)) - _yaw_from_quat(robot.data.root_link_quat_w))
    q = _servo_joint_pos(env, robot) - _servo_default_joint_pos(env, robot)
    head_yaw, neck_pitch = q[:, 7], q[:, 5]
    hip_yaw_sq = q[:, 0].pow(2) + q[:, 9].pow(2)
    gb = robot.data.projected_gravity_b
    if roll_std is None:
        tilt = torch.acos(-gb[:, 2].clamp(-1.0, 1.0))
        level = torch.exp(-(tilt / tilt_std).pow(2))
    else:
        pitch = torch.atan2(gb[:, 0], -gb[:, 2])
        roll = torch.atan2(gb[:, 1], -gb[:, 2])
        if "slope" in _skate_state(env):
            # v7: pitch relative to the deck. A board rolling freely down a grade accelerates at g sin(theta), so the
            # rider's effective gravity is the deck normal (standing plumb would be leaning back); flat: unchanged.
            gd = quat_apply_inverse(_skate_board_quat(env, board_name), robot.data.gravity_vec_w)
            tilted = _skate_state(env)["slope"]["s"].norm(dim=1) > 0.005     # flat: exactly as before (v6)
            pitch = torch.where(tilted, pitch - torch.atan2(gd[:, 0], -gd[:, 2]), pitch)
        level = torch.exp(-(pitch / tilt_std).pow(2)) * torch.exp(-(roll / roll_std).pow(2))
    f = (torch.exp(-(dyaw / yaw_std).pow(2)) * torch.exp(-(head_yaw / head_yaw_std).pow(2))
         * torch.exp(-(neck_pitch / neck_std).pow(2)) * torch.exp(-hip_yaw_sq / hip_yaw_std**2) * level)
    return torch.nan_to_num(f, nan=0.0)


def skate_glide(
    env,
    board_name: str = "board",
    board_foot_sensor: str = "board_foot_deck",
    push_foot_sensor: str = "push_foot_deck",
    push_floor_sensor: str = "push_foot_ground",
    min_load_share: float = 0.25,
    yaw_std: float = math.radians(15.0),
    head_yaw_std: float = math.radians(7.5),
    neck_std: float = 0.15,
    hip_yaw_std: float = 0.15,
    tilt_std: float = math.radians(7.5),
    platform_half_len: float = 1.0,
    roll_std: float | None = None,
) -> torch.Tensor:
    """GLIDE mode after the grace window: product of (each sole carries >= min_load_share of the robot weight on the
    deck, ramped), (both soles over the standing platform, |x| <= platform_half_len), (push foot off the floor) and
    the look-forward Gaussians (trunk yaw on the board heading, head yaw and neck pitch at HOME, both hip yaws at 0,
    trunk level). In [0, 1]."""
    st = _skate_state(env)
    gate = ((st["mode"] == SKATE_GLIDE) & (st["t_mode"] >= st["cfg"]["grace"])).float()
    xl = skate_sole_in_deck(env, "left_foot_collision", board_name)[:, 0]
    xr = skate_sole_in_deck(env, "right_foot_collision", board_name)[:, 0]
    gate = gate * ((xl.abs() <= platform_half_len) & (xr.abs() <= platform_half_len)).float()
    w = _skate_robot_weight(env)
    load_l = (_skate_normal_force(env, board_foot_sensor) / (w * min_load_share)).clamp(0.0, 1.0)
    load_r = (_skate_normal_force(env, push_foot_sensor) / (w * min_load_share)).clamp(0.0, 1.0)
    off_floor = 1.0 - _skate_contact(env, push_floor_sensor)
    look = _skate_look_forward(env, board_name, yaw_std, head_yaw_std, neck_std, hip_yaw_std, tilt_std, roll_std)
    return gate * load_l * load_r * off_floor * look


def skate_glide_shaping(env) -> torch.Tensor:
    """GLIDE mode, potential-based: progress of both soles toward their glide spots on the deck (per 5 cm, divided
    by dt). Moving away is charged."""
    st = _skate_state(env)
    # v3g: per-mode potentials. GLIDE: both soles toward their glide spots. PUSH: the push sole toward its push-stance
    # spot on the floor beside the deck. (v3c-v3f credited the glide potential in PUSH too, which charged stepping
    # the push foot off the deck: skate_v3f never dismounted to push again.) A full push/glide cycle pays both
    # transitions once; reaching the next cycle needs a real speed loss (coast to v_off), so weight it low.
    d = torch.where(st["credit_mode"] == SKATE_GLIDE, st["d_glide"],
                    torch.where(st["credit_mode"] == SKATE_PUSH, st["d_stance"], torch.zeros_like(st["d_glide"])))
    return d / env.step_dt


def skate_glide_floor(env, push_floor_sensor: str = "push_foot_ground") -> torch.Tensor:
    """1 while the push foot touches the floor in GLIDE mode after the grace window (>= 0: negative weight)."""
    st = _skate_state(env)
    gate = ((st["mode"] == SKATE_GLIDE) & (st["t_mode"] >= st["cfg"]["grace"])).float()
    return gate * _skate_contact(env, push_floor_sensor)


def skate_brake(env, threshold: float = 0.2) -> torch.Tensor:
    """Deceleration of the mode speed beyond ``threshold`` m/s^2 in PUSH / GLIDE (>= 0: negative weight). Coasting
    is about 0.07 m/s^2; braking with a foot is several m/s^2."""
    st = _skate_state(env)
    acc = st["accel"]
    if "slope" in st:   # v7: rolling uphill is not braking; checking speed on a downhill is allowed (only uphill part)
        acc = acc - st["slope"]["g_lp"].clamp(max=0.0)
    return torch.relu(-acc - threshold).clamp(max=10.0) * (st["mode"] != SKATE_STOP).float()


def skate_cruise_track(
    env,
    std: float = 0.08,
    board_foot_sensor: str = "board_foot_deck",
    gate_window_s: float = 0.5,
) -> torch.Tensor:
    """exp(-(v_cruise - (v_on + v_off)/2)^2 / std^2) on the 2 s mean system speed, PUSH / GLIDE, while riding."""
    st = _skate_state(env)
    c = st["cfg"]
    _, v_on = _skate_v_on(env, c)
    err = st["v_cruise"] - (v_on - 0.5 * c["gap"])
    if "slope" in st:   # v7: too fast on a downhill is the grade's doing, not a tracking error
        err = torch.where(st["slope"]["g_fwd"] > SKATE_SLOPE_DOWNHILL_ACC, err.clamp(max=0.0), err)
    gate = _skate_recent_contact(env, board_foot_sensor, gate_window_s) * (st["mode"] != SKATE_STOP).float()
    if "coast" in st:   # v6: no speed target while coasting
        gate = gate * (~st["coast"]).float()
    return torch.exp(-err.pow(2) / std**2) * gate


def skate_speed_track(
    env,
    std: float = 0.15,
    board_name: str = "board",
    board_foot_sensor: str = "board_foot_deck",
    gate_window_s: float = 0.5,
    std_above_pump: float | None = None,
) -> torch.Tensor:
    """Instantaneous speed drive (v3c), PUSH / GLIDE while riding: v = min(mode speed, board speed) (a forward
    fall or a deck shove does not count), target = band centre (v_on + v_off)/2. Below the target the pay is
    linear, v / target (a constant gradient from a standstill: every stroke that adds speed pays at once, unlike the
    2 s mean of skate_cruise_track); above it, exp(-(v - target)^2 / std^2). In [0, 1].

    Step 5 ``std_above_pump``: under the pump flag (W held, twist[0] = the throttle ramp) the above-target Gaussian
    uses this tighter std: being ahead of the ramp pays (almost) nothing. None = std everywhere (as before)."""
    st = _skate_state(env)
    c = st["cfg"]
    _, v_on = _skate_v_on(env, c)
    target = (v_on - 0.5 * c["gap"]).clamp_min(0.02)
    v = torch.minimum(st["v_mode"], skate_board_fwd_speed(env, board_name))
    below = (v / target).clamp(0.0, 1.0)
    above = torch.exp(-(v - target).pow(2) / std**2)
    if std_above_pump is not None:
        above = torch.where(skate_pump_flag_active(env, c["command_name"]),
                            torch.exp(-(v - target).pow(2) / std_above_pump**2), above)
    if "slope" in st:   # v7: too fast on a downhill is the grade's doing
        above = torch.where(st["slope"]["g_fwd"] > SKATE_SLOPE_DOWNHILL_ACC, torch.ones_like(above), above)
    r = torch.where(v < target, below, above)
    gate = _skate_recent_contact(env, board_foot_sensor, gate_window_s) * (st["mode"] != SKATE_STOP).float()
    if "coast" in st:   # v6: no speed target while coasting
        gate = gate * (~st["coast"]).float()
    return torch.nan_to_num(r * gate, nan=0.0)


def skate_board_wiggle(env, command_name: str = "twist", board_name: str = "board") -> torch.Tensor:
    """(board yaw rate - commanded yaw rate)^2 in PUSH / GLIDE (>= 0: negative weight). skate_v3e learned to propel
    the board with both feet on the deck by twisting it side to side (yaw-rate rms 0.33 rad/s, +0.017 m/s^2 from
    rest, never touching the floor): a tic-tac / snakeboard drive instead of the push the task asks for."""
    board: Entity = env.scene[board_name]
    cmd = env.command_manager.get_command(command_name)[:, 2]
    err = board.data.root_link_ang_vel_w[:, 2] - cmd
    st = _skate_state(env)
    return torch.nan_to_num(err.pow(2).clamp(max=4.0) * (st["mode"] != SKATE_STOP).float(), nan=0.0)


def skate_stop(
    env,
    std: float = 0.05,
    board_name: str = "board",
    board_foot_sensor: str = "board_foot_deck",
    push_foot_sensor: str = "push_foot_deck",
    push_floor_sensor: str = "push_foot_ground",
) -> torch.Tensor:
    """STOP mode: board at rest with both feet planted (board foot on the deck, push foot on the deck or floor)."""
    st = _skate_state(env)
    planted = _skate_contact(env, board_foot_sensor) * (
        (_skate_contact(env, push_foot_sensor) + _skate_contact(env, push_floor_sensor)) > 0).float()
    still = torch.exp(-skate_board_fwd_speed(env, board_name).pow(2) / std**2)
    return (st["mode"] == SKATE_STOP).float() * planted * still


def skate_yaw_rate_track(
    env,
    command_name: str = "twist",
    std: float = 0.3,
    board_name: str = "board",
    board_foot_sensor: str = "board_foot_deck",
    gate_window_s: float = 0.5,
) -> torch.Tensor:
    cmd = env.command_manager.get_command(command_name)[:, 2]
    board: Entity = env.scene[board_name]
    err = torch.nan_to_num(board.data.root_link_ang_vel_w[:, 2], nan=0.0) - cmd
    gate = _skate_recent_contact(env, board_foot_sensor, gate_window_s)
    return torch.exp(-err.pow(2) / std**2) * gate


def skate_curvature_track(
    env,
    std: float = 0.15,
    board_foot_sensor: str = "board_foot_deck",
    gate_window_s: float = 0.5,
) -> torch.Tensor:
    """v4 steering: exp(-(yaw_lp - kappa * v_board)^2 / std^2) while riding in PUSH / GLIDE. yaw_lp is the board
    yaw rate low-passed with tau_yaw (sway is the wiggle cost's business, not this term's). In [0, 1]."""
    st = _skate_state(env)
    err = st["yaw_lp"] - st["yaw_target"]
    gate = _skate_recent_contact(env, board_foot_sensor, gate_window_s) * (st["mode"] != SKATE_STOP).float()
    return torch.nan_to_num(torch.exp(-err.pow(2) / std**2) * gate, nan=0.0)


def skate_curvature_track_k(
    env,
    std: float = 0.4,
    v_min: float = 0.15,
    board_name: str = "board",
    board_foot_sensor: str = "board_foot_deck",
    gate_window_s: float = 0.5,
    std_wide: float | None = None,
    wide_share: float = 0.5,
) -> torch.Tensor:
    """v6 steering in path-curvature units: exp(-((yaw_lp - kappa v) / max(v_board, v_min))^2 / std^2) (std in 1/m)
    while riding outside STOP. skate_v5d sat at ~40 % of the commanded curvature (kappa got 0.36-0.45 at kappa 1):
    the rad/s std of skate_curvature_track paid 0.37 there and 0.89 at 0.8, too little against the lean cost in the
    glide product; per 1/m the same gap is 0.10 vs 0.78. In [0, 1].

    v6c std_wide: (1 - wide_share) * the std Gaussian + wide_share * a std_wide one. skate_v6a2 it500 turned left
    (+0.42 at kappa +1) but not right (-0.03 at kappa -1): at an error of 1 / m the std 0.4 Gaussian pays 0.002 with a
    slope of ~0.03, 30x less than at the left turn's 0.55, so the right turn had no gradient (turn-physics probe
    2026-11-03: the board and rider are symmetric, a scripted lean reaches the same |kappa| both ways)."""
    st = _skate_state(env)
    vb = skate_board_fwd_speed(env, board_name)
    v = vb.clamp_min(v_min)
    err = (st["yaw_lp"] - st["yaw_target"]) / v
    gate = _skate_recent_contact(env, board_foot_sensor, gate_window_s) * (st["mode"] != SKATE_STOP).float()
    # Pay scaled by min(v / v_min, 1) (verifier 2026-11-03: the target kappa * v is met trivially at rest, so slowing
    # or stopping during a turn hold paid the full term, 1.0 at rest vs 0.10 riding at 40 % of the curvature).
    slow = (vb / v_min).clamp(0.0, 1.0)
    pay = torch.exp(-err.pow(2) / std**2)
    if std_wide is not None:
        pay = (1.0 - wide_share) * pay + wide_share * torch.exp(-err.pow(2) / std_wide**2)
    return torch.nan_to_num(pay * gate * slow, nan=0.0)


def skate_heading_drift(env) -> torch.Tensor:
    """|leaky integral of (board yaw rate - kappa * v)| (rad, tau_heading leak, clamped 0.5) outside STOP (>= 0:
    negative weight). Catches a steady bias the rate term barely sees (skate_v3g drifted ~4 deg per 8 s straight)."""
    st = _skate_state(env)
    return torch.nan_to_num(st["head_err"].abs() * (st["mode"] != SKATE_STOP).float(), nan=0.0)


def skate_joint_limit_proximity(env, asset_cfg: SceneEntityCfg, margin: float = 0.25) -> torch.Tensor:
    """v6f: sum over the selected joints of relu(|q| - margin) in rad (>= 0: negative weight), qpos side. skate_v6e
    glided splayed with both hip rolls parked at +-0.30-0.35 of their +-0.38 stop (targets 1.3-1.7 rad, slammed into
    the limit as a brace), which left no hip-roll range for the lean that steers; the stock dof_pos_limits only bites
    in the last ~7.5 % of the range."""
    asset: Entity = env.scene[asset_cfg.name]
    q = torch.nan_to_num(asset.data.joint_pos[:, asset_cfg.joint_ids], nan=0.0)
    return (q.abs() - margin).clamp_min(0.0).sum(dim=-1)


def skate_hip_splay(env, s_max: float = 0.12, asset_name: str = "robot") -> torch.Tensor:
    """v7f: relu(splay - s_max) (rad, >= 0: negative weight), splay = (q_right_hip_roll - q_left_hip_roll) / 2, the
    common legs-apart part of the two hip rolls (GLIDE keyframe 0.087). Leaning moves both hip rolls the same way
    (the differential part) and is not charged. skate_v7e it4150 glided splayed at -0.23 / +0.21 (a brace, just under
    the per-joint hip_roll_limit margin 0.25) with 0.15 rad left to the +-0.38 stops: no lean range, deck roll only
    +-1 deg at 1.4 m/s (kappa +-0.2) and ~0 downhill."""
    asset: Entity = env.scene[asset_name]
    names = list(asset.joint_names)
    q = torch.nan_to_num(asset.data.joint_pos, nan=0.0)
    splay = 0.5 * (q[:, names.index("right_hip_roll")] - q[:, names.index("left_hip_roll")])
    return (splay - s_max).clamp_min(0.0)


def skate_board_wiggle_hp(env, board_name: str = "board") -> torch.Tensor:
    """v4: (board yaw rate - its tau_yaw low-pass)^2, clamped at 4, outside STOP (>= 0: negative weight). The
    high-pass part only: tic-tac / snakeboard propulsion and sway are charged, a held turn is free (the v3f version
    charged (yaw rate - command)^2, which also taxes every turn the curvature command asks for)."""
    board: Entity = env.scene[board_name]
    st = _skate_state(env)
    hp = torch.nan_to_num(board.data.root_link_ang_vel_w[:, 2], nan=0.0) - st["yaw_lp"]
    return torch.nan_to_num(hp.pow(2).clamp(max=4.0) * (st["mode"] != SKATE_STOP).float(), nan=0.0)


def skate_command_range_curriculum(env, env_ids, stages: list[dict], attr: str = "ang_vel_z",
                                   command_name: str = "twist") -> torch.Tensor:
    """Step-staged command range: stages = [{"step", "range"}] written to the live command term cfg."""
    del env_ids
    cur = stages[0]["range"]
    for s in stages:
        if env.common_step_counter > s["step"]:
            cur = s["range"]
    setattr(env.command_manager.get_term(command_name).cfg.ranges, attr, tuple(cur))
    return torch.tensor([float(max(abs(cur[0]), abs(cur[1])))])


def skate_on_board(env, board_foot_sensor: str = "board_foot_deck", gate_window_s: float = 0.5) -> torch.Tensor:
    return _skate_recent_contact(env, board_foot_sensor, gate_window_s)


def skate_board_foot_slip(
    env,
    board_name: str = "board",
    board_foot_sensor: str = "board_foot_deck",
    foot_body: str = "ankle_left",
) -> torch.Tensor:
    """Squared horizontal velocity of the board foot relative to the deck while in contact (>= 0: negative weight).
    Uses the deck's linear velocity (deck rotation is small)."""
    robot: Entity = env.scene["robot"]
    board: Entity = env.scene[board_name]
    cache = _skate_cache(env)
    key = ("body", foot_body)
    if key not in cache:
        ids, _ = robot.find_bodies(rf"^{foot_body}$")
        cache[key] = ids[0]
    vel = robot.data.body_link_lin_vel_w[:, cache[key], :2] - board.data.root_link_lin_vel_w[:, :2]
    return torch.nan_to_num(vel.pow(2).sum(dim=1), nan=0.0) * _skate_contact(env, board_foot_sensor)


def skate_contact_cost(env, sensor_name: str) -> torch.Tensor:
    """1 while the sensor reports any contact (>= 0: negative weight)."""
    return _skate_contact(env, sensor_name)


# ── v5: step on / step off ──────────────────────────────────────────────────


def skate_stop_fall(env, all_modes: bool = False, mount_grace_s: float | None = None) -> torch.Tensor:
    """Student (2026-11-06 standstill diagnosis, s1r3 it1250): 1/dt on the step a mounted rider falls (non-timeout
    termination) while the credit mode is STOP, so the weight is the total cost per fall (mjlab dt-scales rewards).
    A STOP fall was 3.5-6.5x cheaper than one at speed (no push charge in STOP, `brake` off, `terminated` -3 -> -0.06)
    and the student lost 67 % of brakes from W vs the teacher's 18 %. Wrap with skate_riding_only.

    ``all_modes`` (s1r4 lesson): charging STOP falls only moved the failures to low-speed COAST (GLIDE), where a fall
    still cost ~0.5: mount -> COAST 64/64 lost, COAST up 2 deg 0/40. With it every mounted fall costs the weight.

    ``mount_grace_s`` (s1r8, opt-in MICRODUCK_RIDING_FALL_GRACE_S): no charge for a fall within this many seconds of
    a step-on's mount latch (riding spawns, latched at the reset, get no grace). The mount lineage collapse: refusing
    the step-on ends at the 4 s mount timeout (terminated -3), stepping on and falling costs this -8 on top, so the
    policy stopped stepping on where a fall right after the latch was likely (COAST at a standstill first)."""
    st = _skate_state(env)
    hit = env.termination_manager.terminated
    if not all_modes:
        hit = hit & (st["credit_mode"] == SKATE_STOP)
    if mount_grace_s is not None:
        hit = hit & (skate_time_since_mount(env) >= mount_grace_s)
    return hit.float() / env.step_dt


def skate_time_since_mount(env) -> torch.Tensor:
    """(B,) seconds since this episode's step-on latched the mount (1e6 for riding spawns latched at the reset and
    for envs not mounted yet). Advanced once per env step (cached by common_step_counter). Read-only on the state."""
    st = _skate_state(env)
    mounted = st["mounted"]
    c = env.__dict__.setdefault("_skate_mount_timer", {})
    if c.get("step") != env.common_step_counter or "since" not in c:
        since = c.get("since")
        prev = c.get("prev")
        if since is None or since.shape != mounted.shape:
            since = torch.full(mounted.shape, 1e6, device=mounted.device)
            prev = mounted.clone()
        fresh = env.episode_length_buf <= 1
        new = mounted & ~prev & ~fresh
        since = torch.where(new, torch.zeros_like(since), since + env.step_dt)
        since = torch.where(fresh, torch.full_like(since, 1e6), since)
        c.update(step=env.common_step_counter, since=since, prev=mounted.clone())
    return c["since"]


def skate_riding_gated(func):
    """Wrap a riding reward so it pays / costs nothing while the step-off flag is up (the step-off has its own
    terms). Same signature as ``func`` (functools.wraps), returns func(env, ...) * (1 - dismount)."""
    import functools

    @functools.wraps(func)
    def gated(env, *args, **kwargs):
        return func(env, *args, **kwargs) * (~skate_dismount_active(env)).float()

    gated.skate_riding_gated = True
    return gated


def skate_riding_only(func):
    """v6: wrap a reward term so it reads 0 unless the env is riding (mounted latch set, no step-off flag). For the
    riding smoothness tax: the step-on / step-off phases stay untaxed (no attempt-tax while those are discovered)."""
    import functools

    @functools.wraps(func)
    def riding(env, *args, **kwargs):
        st = _skate_state(env)
        on = st["mounted"] & ~skate_dismount_active(env)
        return func(env, *args, **kwargs) * on.float()

    riding.skate_riding_only = True
    return riding


def skate_approach_obs(env, board_name: str = "board", clip: float = 0.6, mounted_override: bool = True) -> torch.Tensor:
    """v6 approach block (4D, appended LAST to actor and critic): [dx, dy] of the staging trunk spot beside the board
    in the robot's yaw frame (clipped +-clip m), sin / cos of (robot yaw - canonical deck yaw). The deck is fore/aft
    symmetric: the canonical frame is the deck frame turned by pi when the robot is on the board's left (st["flip"],
    hysteresis latch), so the staging spot is always on the robot's side. Once mounted it reads [0, 0, 0, 1]."""
    robot: Entity = env.scene["robot"]
    board: Entity = env.scene[board_name]
    st = _skate_state(env)
    sx, sy = st["cfg"]["staging"]
    byaw = _yaw_from_quat(board.data.root_link_quat_w) + math.pi * st["flip"].float()   # canonical deck yaw
    cb, sb = torch.cos(byaw), torch.sin(byaw)
    px = board.data.root_link_pos_w[:, 0] + cb * sx - sb * sy                          # staging spot, world
    py = board.data.root_link_pos_w[:, 1] + sb * sx + cb * sy
    ryaw = _yaw_from_quat(robot.data.root_link_quat_w)
    cr, sr = torch.cos(ryaw), torch.sin(ryaw)
    ex = px - robot.data.root_link_pos_w[:, 0]
    ey = py - robot.data.root_link_pos_w[:, 1]
    dx = (cr * ex + sr * ey).clamp(-clip, clip)
    dy = (-sr * ex + cr * ey).clamp(-clip, clip)
    dyaw = wrap_to_pi(ryaw - byaw)
    out = torch.stack([dx, dy, torch.sin(dyaw), torch.cos(dyaw)], dim=1)
    if mounted_override:    # False: the raw geometry (skate student localization model; the latch is privileged)
        mounted = st["mounted"].unsqueeze(1)
        out = torch.where(mounted, torch.tensor([0.0, 0.0, 0.0, 1.0], device=out.device).expand_as(out), out)
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# Normalizer seed for the appended approach block: (mean, std) per dim.
SKATE_APPROACH_OBS_STATS = ((0.0, 0.2), (0.0, 0.2), (0.0, 0.5), (0.9, 0.3))


def skate_mount_reach(env) -> torch.Tensor:
    """Step-on shaping, potential-based: progress of the left sole toward its deck spot (per 5 cm, divided by dt),
    credited only before the mount latch and outside step-off. Moving away is charged."""
    st = _skate_state(env)
    gate = st["premount_prev"] & ~skate_dismount_active(env)
    return st["d_reach"] * gate.float() / env.step_dt


def _skate_premount_gate(env) -> torch.Tensor:
    st = _skate_state(env)
    return (st["premount_prev"] & ~skate_dismount_active(env)).float()


def skate_approach_progress(env) -> torch.Tensor:
    """v6b walk-up shaping, potential-based: trunk progress toward the staging spot beside the board (canonical deck
    frame, per 5 cm, divided by dt) before the mount latch. Walking away is charged; standing still pays 0."""
    return _skate_state(env)["d_appr"] * _skate_premount_gate(env) / env.step_dt


def skate_approach_heading(env) -> torch.Tensor:
    """v6b walk-up shaping, potential-based: lining the trunk yaw up with the canonical deck heading (per 0.5 rad),
    weighted in over the last 0.3 m to the staging spot, before the mount latch."""
    return _skate_state(env)["d_head"] * _skate_premount_gate(env) / env.step_dt


def skate_premount_time(env) -> torch.Tensor:
    """v6b: 1 per step before the mount latch (>= 0: negative weight), the urgency of the step-on (replaces the
    riding costs board_foot_on_floor / push_time, which v6b gates to riding: walking needs the board foot on the floor)."""
    return _skate_premount_gate(env)


def skate_board_kick(env, board_name: str = "board") -> torch.Tensor:
    """Before the mount latch: board planar speed^2 + 0.05 * yaw rate^2 (>= 0: negative weight). The board starts at
    rest beside the robot; shoving it away is not a step-on."""
    st = _skate_state(env)
    board: Entity = env.scene[board_name]
    v2 = board.data.root_link_lin_vel_w[:, :2].pow(2).sum(dim=1) + 0.05 * board.data.root_link_ang_vel_w[:, 2].pow(2)
    return torch.nan_to_num(v2.clamp(max=4.0), nan=0.0) * (~st["mounted"]).float()


def skate_dismount_progress(env) -> torch.Tensor:
    """Step-off shaping, potential-based on -(soles touching the deck): +1 per foot that leaves the deck (divided by
    dt), charged when a foot goes back on. Only while the step-off flag is up."""
    st = _skate_state(env)
    return st["d_off"] * skate_dismount_active(env).float() / env.step_dt


def skate_dismount_reach(env) -> torch.Tensor:
    """v5d step-off shaping, potential-based: progress of both soles toward their spots beside the board (right of
    it, deck-frame y, per 5 cm, divided by dt) while the flag is up. Moving back toward the deck is charged."""
    st = _skate_state(env)
    return st["d_leave"] * skate_dismount_active(env).float() / env.step_dt


def _skate_feet_on_floor(env, feet_sensor: str) -> torch.Tensor:
    """(B,) 1 when both soles touch the floor (feet_ground_contact has one primary per sole)."""
    found = env.scene.sensors[feet_sensor].data.found
    if found.dim() == 1:
        return (found > 1).float()
    return (found > 0).all(dim=-1).float()


def skate_dismount_stand(
    env,
    feet_sensor: str = "feet_ground_contact",
    speed_std: float = 0.15,
    tilt_std: float = math.radians(15.0),
) -> torch.Tensor:
    """Step-off goal while the flag is up: product of (no sole on the deck), (both soles on the floor), trunk at rest
    (exp(-v_xy^2 / speed_std^2)) and trunk upright (exp(-tilt^2 / tilt_std^2)). In [0, 1]: standing beside the board,
    ready for the walking policy."""
    st = _skate_state(env)
    c = st["cfg"]
    robot: Entity = env.scene["robot"]
    off_deck = (1.0 - _skate_contact(env, c["board_foot_sensor"])) * (1.0 - _skate_contact(env, c["push_foot_sensor"]))
    floor = _skate_feet_on_floor(env, feet_sensor)
    v2 = robot.data.root_link_lin_vel_w[:, :2].pow(2).sum(dim=1)
    tilt = torch.acos(-robot.data.projected_gravity_b[:, 2].clamp(-1.0, 1.0))
    r = off_deck * floor * torch.exp(-v2 / speed_std**2) * torch.exp(-(tilt / tilt_std).pow(2))
    return torch.nan_to_num(r * skate_dismount_active(env).float(), nan=0.0)


def skate_dismount_pose(env, std: float = 0.3) -> torch.Tensor:
    """While the flag is up and no sole is on the deck: exp(-mean squared leg-joint deviation from HOME / std^2), a
    hand-off posture the walking policy knows. In [0, 1]."""
    st = _skate_state(env)
    c = st["cfg"]
    robot: Entity = env.scene["robot"]
    q = _servo_joint_pos(env, robot) - _servo_default_joint_pos(env, robot)
    legs = torch.cat([q[:, 0:5], q[:, 9:14]], dim=1)
    off_deck = (1.0 - _skate_contact(env, c["board_foot_sensor"])) * (1.0 - _skate_contact(env, c["push_foot_sensor"]))
    r = torch.exp(-legs.pow(2).mean(dim=1) / std**2) * off_deck
    return torch.nan_to_num(r * skate_dismount_active(env).float(), nan=0.0)


def skate_dismount_feet(env, feet_sensor: str = "feet_ground_contact") -> torch.Tensor:
    """v5b dense step-off term while the flag is up: per sole, 1 if it is on the floor and not on the deck (0..2).
    Pays as soon as one foot is down beside the board, more for both (skate_v5 iteration 1000 never found the sparse
    stand product)."""
    st = _skate_state(env)
    c = st["cfg"]
    found = env.scene.sensors[feet_sensor].data.found
    if found.dim() == 1:
        found = found.unsqueeze(1)
    floor_l, floor_r = (found[:, 0] > 0).float(), (found[:, -1] > 0).float()
    # v5c: board-foot centric. skate_v5b iteration 750 parked in the push stance (push foot on the floor paid 1/s,
    # board foot left on the deck): the push foot alone now pays nothing; the board foot down beside the board pays
    # 1, both feet down 2.
    board_down = floor_l * (1.0 - _skate_contact(env, c["board_foot_sensor"]))
    push_down = floor_r * (1.0 - _skate_contact(env, c["push_foot_sensor"]))
    r = board_down * (1.0 + push_down)
    return r * skate_dismount_active(env).float()


def skate_dismount_failed(env) -> torch.Tensor:
    """v5b: 1 on a termination while the step-off flag is up (>= 0: negative weight). skate_v5 iteration 1000 ended
    every step-off episode on purpose (trunk_low): the per-foot-on-deck time cost outweighed the -3 termination."""
    return env.termination_manager.terminated.float() * skate_dismount_active(env).float()


def skate_dismount_pending(env) -> torch.Tensor:
    """1 per sole still touching the deck while the step-off flag is up (>= 0: negative weight)."""
    st = _skate_state(env)
    c = st["cfg"]
    on = _skate_contact(env, c["board_foot_sensor"]) + _skate_contact(env, c["push_foot_sensor"])
    return on * skate_dismount_active(env).float()


# ── terminations ────────────────────────────────────────────────────────────


def skate_board_lost(
    env,
    board_name: str = "board",
    max_dist: float = 0.12,
    board_foot_sensor: str = "board_foot_deck",
    max_off_s: float = 1.0,
    premount_max_dist: float = 0.3,
    mount_timeout_s: float = 4.0,
    approach_slack: float | None = None,
    approach_speed: float = 0.06,
) -> torch.Tensor:
    """Terminate when the trunk is more than max_dist (xy) from the deck centre or the board foot has been off the
    deck for longer than max_off_s. v5: before the mount latch, only a trunk farther than premount_max_dist or no
    mount within mount_timeout_s of the reset ends the episode; never while the step-off flag is up.
    v6b (approach_slack set): per env, from the spawn's distance d0 to the staging spot: the trunk wandering more than
    approach_slack beyond d0 (at least premount_max_dist) from the spot, or no mount within
    mount_timeout_s + d0 / approach_speed."""
    robot: Entity = env.scene["robot"]
    board: Entity = env.scene[board_name]
    dist = torch.nan_to_num((robot.data.root_link_pos_w[:, :2] - board.data.root_link_pos_w[:, :2]).norm(dim=1),
                            nan=1.0)
    ride_lost = (dist > max_dist) | (_skate_air_time(env, board_foot_sensor) > max_off_s)
    cache = _skate_cache(env)
    st = cache.get("state")
    if st is None or "mounted" not in st:
        return ride_lost
    st = _skate_state(env)
    t = env.episode_length_buf.float() * env.step_dt
    if approach_slack is not None and st["cfg"].get("v6b", False):
        d0 = st["appr_d0"]
        far = st["d_stage"] > (d0 + approach_slack).clamp(min=premount_max_dist)
        pre_lost = far | (t > mount_timeout_s + d0 / approach_speed)
    else:
        pre_lost = (dist > premount_max_dist) | (t > mount_timeout_s)
    lost = torch.where(st["mounted"], ride_lost, pre_lost)
    return lost & ~skate_dismount_active(env)


def skate_board_flipped(env, board_name: str = "board", min_up_z: float = 0.5) -> torch.Tensor:
    """Board upside down / on its side; ignored once the step-off flag is up (the rider is leaving it)."""
    board: Entity = env.scene[board_name]
    flipped = torch.nan_to_num(_unit(board.data.root_link_quat_w, 2)[:, 2], nan=-1.0) < min_up_z
    return flipped & ~skate_dismount_active(env)


def skate_state_is_nan(env, board_name: str = "board", sensor_names: tuple = ()) -> torch.Tensor:
    bad = robot_state_is_nan(env, sensor_names=sensor_names)
    d = env.scene[board_name].data
    for t in (d.joint_pos, d.joint_vel, d.root_link_pos_w, d.root_link_quat_w, d.root_link_lin_vel_w,
              d.root_link_ang_vel_w):
        bad |= ~torch.isfinite(t).flatten(start_dim=1).all(dim=1)
    return bad


# ── reset ───────────────────────────────────────────────────────────────────

_SKATE_MODE_NAMES = {"push": SKATE_PUSH, "glide": SKATE_GLIDE}


def skate_spawn_speed_range(kind, cmd_range: tuple, gap: float, glide_margin: float = 0.05) -> tuple:
    """Board speed range of a spawn: an explicit (lo, hi) or a kind relative to the cruise command range:
    zero, push (below the range), glide (v_off .. v_on + margin), mount (inside the range), dismount (at v_off)."""
    if isinstance(kind, (tuple, list)):
        lo, hi = kind
    else:
        c_lo, c_hi = cmd_range
        lo, hi = {
            "zero": (0.0, 0.0),
            "push": (0.0, c_lo),
            "glide": (c_lo - gap, c_hi + glide_margin),
            "mount": (c_lo, c_hi),
            "dismount": (c_lo - gap, c_lo - gap + 0.03),
            "fast": (c_hi, c_hi + 0.6),   # v7: above the push range (a downhill's runout)
        }[kind]
    lo = max(0.0, float(lo))
    return lo, max(lo, float(hi))


def reset_skate_state(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    keyframes: dict,
    spawns: list,
    board_name: str = "board",
    command_name: str = "twist",
    pos_noise: float = 0.004,
    rpy_noise: float = 0.03,
    joint_noise: float = 0.03,
    z_lift: float = 0.002,
    deck_center_z: float = 0.01325,
    wheel_radius: float = 0.005,
    board_extent: tuple = (0.112, 0.055),
):
    """Place the board at the env origin with a random heading and the robot on it in a keyframe pose, both moving
    forward together (wheels spinning to match), and set the mode machine.

    ``spawns``: list of {"key": keyframe name, "prob": float, "speed": kind or (lo, hi), "mode": "push"|"glide",
    "settled": bool}. Spawns whose keyframe is missing from ``keyframes`` are dropped and the rest renormalized.
    "settled" glide spawns start past the grace window. Must run AFTER the base reset events."""
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device).long()
    n = len(env_ids)
    dev = env.device
    robot: Entity = env.scene["robot"]
    board: Entity = env.scene[board_name]
    st = _skate_state(env)

    valid = [s for s in spawns if s["key"] in keyframes and s["prob"] > 0]
    assert valid, f"no spawn keyframe present: {[s['key'] for s in spawns]} vs {list(keyframes)}"
    names = tuple(keyframes)
    cache = _skate_cache(env)
    ck = ("kf", names)
    if ck not in cache:
        pos = torch.tensor([keyframes[k]["pos"] for k in names], device=dev, dtype=torch.float)
        rpy = torch.tensor([keyframes[k]["rpy"] for k in names], device=dev, dtype=torch.float)
        joint_names = list(keyframes[names[0]]["joints"].keys())
        ids = [list(robot.joint_names).index(j) for j in joint_names]
        joints = torch.tensor([[keyframes[k]["joints"][j] for j in joint_names] for k in names], device=dev,
                              dtype=torch.float)
        wheel_ids, _ = board.find_joints(r"^passive_.*_wheel_[LR]$")
        cache[ck] = (pos, rpy, ids, joints, wheel_ids)
    kpos, krpy, leg_ids, kjoints, wheel_ids = cache[ck]

    cmd_range = tuple(env.command_manager.get_term(command_name).cfg.ranges.lin_vel_x)
    gap = st["cfg"]["gap"]
    sp_kf = torch.tensor([names.index(s["key"]) for s in valid], device=dev)
    sp_lo, sp_hi = zip(*[skate_spawn_speed_range(s["speed"], cmd_range, gap) for s in valid])
    sp_lo = torch.tensor(sp_lo, device=dev)
    sp_hi = torch.tensor(sp_hi, device=dev)
    sp_mode = torch.tensor([_SKATE_MODE_NAMES[s["mode"]] for s in valid], device=dev)
    sp_t = torch.tensor([3.0 if s.get("settled", False) else 0.0 for s in valid], device=dev)
    p = torch.tensor([float(s["prob"]) for s in valid], device=dev)
    # v5 step-on spawns: "xy" = ((x_lo, x_hi), (y_lo, y_hi)) trunk position in the deck frame (replaces the
    # keyframe's xy), "yaw_noise" = extra trunk yaw range (rad), "mounted" False = standing on the floor.
    sp_xy = torch.tensor([s.get("xy", ((0.0, 0.0), (0.0, 0.0))) for s in valid], device=dev, dtype=torch.float)
    sp_has_xy = torch.tensor(["xy" in s for s in valid], device=dev)
    sp_yaw = torch.tensor([float(s.get("yaw_noise", 0.0)) for s in valid], device=dev)
    sp_mounted = torch.tensor([bool(s.get("mounted", True)) for s in valid], device=dev)
    sp_dis = torch.tensor([bool(s.get("dismount", False)) for s in valid], device=dev)   # v5c: flag up from t = 0
    # v6b: "ring" = (clear, r_max): trunk anywhere (uniform in area) whose distance to the board's footprint rectangle
    # (board_extent half sizes, nose included) is in [clear, r_max], on the canonical right half (y <= 0, the ends
    # included); "flip_prob": share of these spawns mirrored to the board's left (pi about the deck centre: robot
    # pose and motion turned, canonical latch set; the board is symmetric under it).
    sp_ring = torch.tensor([s.get("ring", (0.0, 0.0)) for s in valid], device=dev, dtype=torch.float)
    sp_has_ring = torch.tensor(["ring" in s for s in valid], device=dev)
    sp_flip = torch.tensor([float(s.get("flip_prob", 0.0)) for s in valid], device=dev)

    j = torch.multinomial(p / p.sum(), n, replacement=True)
    k = sp_kf[j]
    origin = env.scene.env_origins[env_ids]
    yaw = (torch.rand(n, device=dev) * 2 - 1) * math.pi
    cy, sy = torch.cos(yaw), torch.sin(yaw)

    # Board at the env origin, random heading
    bpose = torch.zeros(n, 7, device=dev)
    bpose[:, 0:2] = origin[:, :2]
    bpose[:, 2] = origin[:, 2] + deck_center_z
    bpose[:, 3] = torch.cos(yaw * 0.5)
    bpose[:, 6] = torch.sin(yaw * 0.5)

    # Robot trunk: keyframe pose in the deck frame, rotated by the board yaw
    off = kpos[k].clone()
    xy = sp_xy[j]
    u = torch.rand(n, 2, device=dev)
    beside = xy[:, :, 0] + u * (xy[:, :, 1] - xy[:, :, 0])
    off[:, :2] = torch.where(sp_has_xy[j].unsqueeze(1), beside, off[:, :2])
    ring = sp_has_ring[j]
    if bool(ring.any()):
        ids = ring.nonzero().squeeze(1)
        clear, rmax = sp_ring[j[ids], 0:1], sp_ring[j[ids], 1:2]
        hl, hw = board_extent
        cand = torch.rand(len(ids), 16, 2, device=dev)
        cx = (cand[..., 0] * 2 - 1) * (hl + rmax)
        cyy = -cand[..., 1] * (hw + rmax)
        dist = torch.hypot((cx.abs() - hl).clamp(min=0.0), (cyy.abs() - hw).clamp(min=0.0))
        ok = (dist >= clear) & (dist <= rmax)
        first = ok.float().argmax(dim=1, keepdim=True)
        stg = torch.tensor(st["cfg"]["staging"], device=dev, dtype=torch.float)
        px = torch.where(ok.any(dim=1), cx.gather(1, first).squeeze(1), stg[0])
        py = torch.where(ok.any(dim=1), cyy.gather(1, first).squeeze(1), stg[1])
        off[ids, 0], off[ids, 1] = px, py
    off[:, :2] += (torch.rand(n, 2, device=dev) * 2 - 1) * pos_noise
    stg = torch.tensor(st["cfg"]["staging"], device=dev, dtype=torch.float)
    d0 = (off[:, :2] - stg).norm(dim=1)                     # canonical distance to the staging spot (v6b timeout)
    flip = torch.rand(n, device=dev) < sp_flip[j]
    fs = torch.where(flip, -1.0, 1.0)                         # pi about the deck centre: (x, y) -> (-x, -y)
    rpose = torch.zeros(n, 7, device=dev)
    ox, oy = off[:, 0] * fs, off[:, 1] * fs
    rpose[:, 0] = origin[:, 0] + cy * ox - sy * oy
    rpose[:, 1] = origin[:, 1] + sy * ox + cy * oy
    rpose[:, 2] = origin[:, 2] + off[:, 2] + z_lift
    r = krpy[k] + (torch.rand(n, 3, device=dev) * 2 - 1) * rpy_noise
    r[:, 2] += (torch.rand(n, device=dev) * 2 - 1) * sp_yaw[j]
    rpose[:, 3:7] = _quat_from_rpy_t(r[:, 0], r[:, 1], r[:, 2] + yaw + math.pi * flip.float())

    # Shared forward speed (canonical forward: deck -x when flipped)
    v0 = sp_lo[j] + torch.rand(n, device=dev) * (sp_hi[j] - sp_lo[j])
    vel = torch.zeros(n, 6, device=dev)
    vel[:, 0] = v0 * cy * fs
    vel[:, 1] = v0 * sy * fs

    board.write_root_link_pose_to_sim(bpose, env_ids)
    board.write_root_link_velocity_to_sim(vel, env_ids)
    bj = torch.zeros(n, board.num_joints, device=dev)
    bjv = torch.zeros(n, board.num_joints, device=dev)
    bjv[:, wheel_ids] = (fs * v0 / wheel_radius).unsqueeze(1)
    board.write_joint_state_to_sim(bj, bjv, env_ids=env_ids)

    robot.write_root_link_pose_to_sim(rpose, env_ids)
    robot.write_root_link_velocity_to_sim(vel, env_ids)
    jpos = robot.data.default_joint_pos[env_ids].clone()
    jpos[:, leg_ids] = kjoints[k] + torch.randn(n, len(leg_ids), device=dev) * joint_noise
    robot.write_joint_state_to_sim(jpos, torch.zeros_like(jpos), env_ids=env_ids)

    # Mode machine
    st["mode"][env_ids] = sp_mode[j]
    st["credit_mode"][env_ids] = sp_mode[j]
    st["t_mode"][env_ids] = sp_t[j]
    st["v_mode"][env_ids] = v0
    st["v_cruise"][env_ids] = v0
    st["v_sys_prev"][env_ids] = v0
    st["accel"][env_ids] = 0.0
    st["d_push"][env_ids] = 0.0
    st["d_glide"][env_ids] = 0.0
    st["switches"][env_ids] = 0.0
    st["mounted"][env_ids] = sp_mounted[j]
    st["premount_prev"][env_ids] = ~sp_mounted[j]
    if "dismount_now" in st:
        st["dismount_now"][env_ids] = sp_dis[j]
    st["d_reach"][env_ids] = 0.0
    st["d_off"][env_ids] = 0.0
    if "d_leave" in st:
        st["d_leave"][env_ids] = 0.0
    if "d_coast" in st:
        st["d_coast"][env_ids] = 0.0
        st["credit_coast"][env_ids] = False
    if "flip" in st:
        st["flip"][env_ids] = flip    # canonical side: the spawn's mirror (no flip_prob: always the board's right)
        st["flip_changed"][env_ids] = False
        st["appr_d0"][env_ids] = d0
        st["d_appr"][env_ids] = 0.0
        st["d_head"][env_ids] = 0.0
    st["fresh"][env_ids] = True
    _skate_slope_reset(env, st, env_ids)   # v7: every episode starts on the flat
    g_deck = None
    if "slope" in st and any(s.get("slope_up", False) for s in valid):
        # step 5 reverse-curriculum spawns (UP_PUSH / UP_SLOW / UP_GLIDE): the episode starts ON an uphill
        up = torch.tensor([bool(s.get("slope_up", False)) for s in valid], device=dev)[j]
        up_deg = None
        if any("up_deg" in s for s in valid):   # s5b: a spawn's own grade range (else the live level's)
            nan = float("nan")
            up_deg = torch.tensor([tuple(s.get("up_deg", (nan, nan))) for s in valid], device=dev)[j]
        g_deck = _skate_slope_spawn_up(env, st, env_ids, up, yaw + math.pi * flip.float(), up_deg=up_deg)
    if "pump_now" in st:    # step 5: spawns that start with W held (the command reads it at its reset resample)
        pp = torch.tensor([float(s.get("pump_prob", 1.0 if s.get("pump", False) else 0.0)) for s in valid],
                          device=dev)[j]
        st["pump_now"][env_ids] = torch.rand(n, device=dev) < pp
    if "coast_now" in st:   # s5b: spawns that start with no key (COAST: the roll-back of a release on a climb)
        st["coast_now"][env_ids] = torch.tensor([bool(s.get("coast", False)) for s in valid], device=dev)[j]
    cache["last_spawn"] = (env_ids, j, v0)
    if "sensor" in st:      # skate student: re-arm the board unit / radio / localization models for the new episode
        _skate_sensor_reset(env, st, env_ids, v0, vel[:, :3], g_deck=g_deck)


# ══ SKATE STUDENT (2026-11-06): real-sensor observation model + servo heat ═══════════════════════════════════════════
# The teacher skater (skate_v7f) reads 14 board numbers no robot has (true board / CoM speed, board-vs-robot yaw, both
# soles in the deck frame, three contact flags) and a privileged approach block. The student reads only what a
# "smart board" (ESP32 + IMU + a hall-effect magnetic wheel encoder, AS5600-class 12 bit) radios to the robot, plus an
# optional, explicitly labelled localization input (head-camera fiducial on the deck), all through the models below.
# has the numbers and their sources.
#
# Timing: _skate_sensor_update runs inside the lazy mode-machine update (once per env step, after the physics step,
# before the observations); _skate_sensor_reset runs at the end of reset_skate_state.

SKATE_SENSOR_DEFAULTS = dict(
    # radio link (ESP-NOW board -> robot), per policy step of 20 ms
    latency_steps=(1, 3),        # per-episode base latency
    latency_jitter_prob=0.2,     # per step: latency +-1 (kept >= 1)
    drop_prob=(0.05, 0.10),      # per-episode packet-loss rate (lost packet = hold the last received sample)
    burst_prob=0.005,            # per step: a burst of lost packets starts
    burst_len=(3, 8),
    age_norm=10.0,               # obs: steps since the held sample was MEASURED, / this, clamped to 1
    # board IMU (deck-mounted, canonical deck frame)
    gyro_noise=0.015,            # rad/s white
    gyro_vib=0.03,               # rad/s extra white noise per m/s of board speed (rolling vibration)
    gyro_bias=0.02,              # rad/s per-axis per-episode bias, uniform +-
    imu_mis_deg=3.0,             # mounting misalignment, per episode, random axis
    grav_noise=0.015,            # white, per component of the fused gravity unit vector
    grav_acc_leak=(0.05, 0.25),  # share of the deck's linear acceleration / g leaking into the fused gravity estimate
    # wheel encoder (one rear wheel, angle-difference speed over the 20 ms send period)
    wheel_joint="passive_rear_wheel_L",
    wheel_counts=4096,           # counts / rev (12-bit magnetic angle sensor)
    wheel_radius=0.010,          # m, nominal (SKATEBOARD_WHEEL_RADIUS)
    wheel_radius_err=0.03,       # per-episode scale error, uniform +-
    wheel_noise=0.003,           # m/s white (angle jitter)
    # robot-side mode machine on the received speed
    tau_mode=0.1,                # s, low-pass of the received wheel speed (as the teacher's mode speed)
    # localization (head camera + fiducial on the deck), optional block
    loc_period=5,                # steps (10 Hz)
    loc_latency=(2, 5),          # steps (40-100 ms), per episode
    loc_miss_prob=0.10,          # per sample
    loc_pos_noise=0.01,          # m white
    loc_yaw_noise=0.04,          # rad white
    loc_pos_bias=0.01,           # m per-episode, uniform +-
    loc_yaw_bias=0.03,           # rad per-episode, uniform +-
    loc_hold_s=0.5,              # s: older samples are dropped (valid = 0)
    loc_off_prob=0.25,           # share of episodes with no localization at all (operator-placed start)
    loc_occlude=(0.16, 0.09),    # trunk over the deck (canonical |x| <, |y| <): the fiducial is under the robot
    loc_clip=0.8,                # m, as the teacher's approach block
    board_name="board",
)
_SKATE_RAW_DIM = 7          # [wheel speed, gyro xyz, gravity xyz]
_SKATE_RING = 6             # radio ring buffer (>= max latency + 2)
_SKATE_LOC_RING = 8


def skate_sensor_setup(env, env_ids, **params):
    """Startup event: allocate the board-unit / radio / localization model state (st["sensor"])."""
    del env_ids
    unknown = set(params) - set(SKATE_SENSOR_DEFAULTS)
    assert not unknown, f"unknown skate sensor cfg keys {unknown}"
    c = dict(SKATE_SENSOR_DEFAULTS, **params)
    st = _skate_state(env)
    n, dev = env.num_envs, env.device
    board: Entity = env.scene[c["board_name"]]
    wid, _ = board.find_joints("^" + c["wheel_joint"] + "$")
    assert len(wid) == 1, (c["wheel_joint"], wid)
    z = lambda *s: torch.zeros(n, *s, device=dev)
    zl = lambda *s: torch.zeros(n, *s, dtype=torch.long, device=dev)
    st["sensor"] = dict(
        cfg=c, wheel_id=int(wid[0]), g=0,
        ring=z(_SKATE_RING, _SKATE_RAW_DIM), ring_g=zl(_SKATE_RING) - 10**6,
        held=z(_SKATE_RAW_DIM), held_g=zl(), lat=zl() + 1, drop_p=z() + 0.05, burst=zl(),
        gyro_bias=z(3), mis=torch.eye(3, device=dev).repeat(n, 1, 1), leak=z() + 0.1, wscale=z() + 1.0,
        wheel_q_prev=z(), v_prev_w=z(3),
        v_s=z(), mode_o=zl(), t_o=z(),
        loc_on=torch.ones(n, dtype=torch.bool, device=dev), loc_lat=zl() + 3, loc_phase=zl(),
        loc_bias=z(3), loc_ring=z(_SKATE_LOC_RING, 4), loc_held=z(4), loc_age=z() + 1e3,
        loc_ring_g=zl(_SKATE_LOC_RING) - 10**6,                                  # capture step of each slot
        loc_ring_occ=torch.ones(n, _SKATE_LOC_RING, dtype=torch.bool, device=dev),   # occluded AT CAPTURE
        loc_occ=torch.zeros(n, dtype=torch.bool, device=dev),   # diagnostics only (current truth, never observed)
        seed_o=torch.ones(n, dtype=torch.bool, device=dev),     # observer mode not yet seeded for this episode
    )


def _skate_raw_board_sample(env, st: dict) -> torch.Tensor:
    """(N, 7) what the board unit measures this step (before the radio): wheel-encoder speed (canonical forward),
    gyro (canonical deck frame), fused gravity unit vector (canonical deck frame). Updates the encoder / velocity
    memories."""
    s = st["sensor"]
    c = s["cfg"]
    dt = env.step_dt
    board: Entity = env.scene[c["board_name"]]
    n, dev = env.num_envs, env.device
    sign = torch.where(st["flip"], -1.0, 1.0) if "flip" in st else torch.ones(n, device=dev)
    # wheel: quantized angle difference over the send period (measures wheel SPIN: slip / lift-off show up honestly)
    q = torch.nan_to_num(board.data.joint_pos[:, s["wheel_id"]], nan=0.0)
    cnt = 2.0 * math.pi / c["wheel_counts"]
    qq = torch.round(q / cnt) * cnt
    w = (qq - s["wheel_q_prev"]) / dt
    s["wheel_q_prev"] = qq
    v_w = sign * w * c["wheel_radius"] * s["wscale"] + torch.randn(n, device=dev) * c["wheel_noise"]
    # IMU in the canonical deck frame, misaligned
    qc = _skate_board_quat(env, c["board_name"])
    ang = quat_apply_inverse(qc, torch.nan_to_num(board.data.root_link_ang_vel_w, nan=0.0))
    vw = torch.nan_to_num(board.data.root_link_lin_vel_w, nan=0.0)
    acc_w = (vw - s["v_prev_w"]) / dt
    s["v_prev_w"] = vw
    acc = quat_apply_inverse(qc, acc_w)
    if "slope" in st:      # v7: a grade is a tilted gravity; the deck IMU reads it
        G = st["slope"]["G"]
        g_w = G / G.norm(dim=1, keepdim=True).clamp_min(1e-6)
    else:
        g_w = board.data.gravity_vec_w
    g = quat_apply_inverse(qc, g_w)
    vib = c["gyro_noise"] + c["gyro_vib"] * v_w.abs().clamp(max=3.0)
    gyro = (torch.bmm(s["mis"], ang.unsqueeze(-1)).squeeze(-1) + s["gyro_bias"]
            + torch.randn(n, 3, device=dev) * vib.unsqueeze(1))
    gf = g - s["leak"].unsqueeze(1) * acc.clamp(-30.0, 30.0) / 9.81    # accel-fused gravity estimate (unit-ish)
    gf = gf / gf.norm(dim=1, keepdim=True).clamp_min(1e-6)
    gf = torch.bmm(s["mis"], gf.unsqueeze(-1)).squeeze(-1) + torch.randn(n, 3, device=dev) * c["grav_noise"]
    return torch.nan_to_num(torch.cat([v_w.unsqueeze(1), gyro, gf], dim=1), nan=0.0, posinf=0.0, neginf=0.0)


def _skate_observer_step(c: dict, mode, t, v, cmd, gap, dt):
    """The robot-side mode machine (the runtime has only the command and the received wheel speed): the reward machine's
    rules without the privileged mounted latch. Returns (mode, t_mode)."""
    v_on = cmd.clamp_min(c["stop_cmd"])
    v_off = v_on - gap
    t = t + dt
    can = t >= c["min_dwell"]
    stop = cmd < c["stop_cmd"]
    up = ~stop & (v >= v_on)
    down = ~stop & (v <= v_off)
    new = torch.where(stop, torch.full_like(mode, SKATE_STOP), mode)
    new = torch.where(~stop & (mode == SKATE_STOP),
                      torch.where(up, torch.full_like(mode, SKATE_GLIDE), torch.full_like(mode, SKATE_PUSH)), new)
    new = torch.where((mode == SKATE_PUSH) & can & up, torch.full_like(mode, SKATE_GLIDE), new)
    new = torch.where((mode == SKATE_GLIDE) & can & down, torch.full_like(mode, SKATE_PUSH), new)
    coast = (cmd >= c["stop_cmd"]) & (cmd < c["coast_cmd_hi"])
    new = torch.where(coast, torch.full_like(mode, SKATE_GLIDE), new)
    changed = new != mode
    return new, torch.where(changed, torch.zeros_like(t), t)


def _skate_loc_capture_ok(ring_g: torch.Tensor, ring_occ: torch.Tensor, cap_g: torch.Tensor) -> torch.Tensor:
    """(N,) bool: the localization ring holds the frame captured at step cap_g (written this episode, not an older or
    reset slot) and the fiducial was visible when it was captured."""
    ar = torch.arange(ring_g.shape[0], device=ring_g.device)
    slot = cap_g % ring_g.shape[1]
    return (ring_g[ar, slot] == cap_g) & ~ring_occ[ar, slot]


def _skate_observer_seed(st: dict, cmd: torch.Tensor) -> None:
    """Seed the robot-side mode machine of freshly reset envs from its own rule on the sensed speed and the command
    (STOP below stop_cmd, GLIDE under COAST or at >= v_on, else PUSH), time in mode 0."""
    s = st["sensor"]
    if not bool(s["seed_o"].any()):
        return
    mc = st["cfg"]
    v_on = cmd.clamp_min(mc["stop_cmd"])
    coast = (cmd >= mc["stop_cmd"]) & (cmd < mc["coast_cmd_hi"])
    rule = torch.where(cmd < mc["stop_cmd"], torch.full_like(s["mode_o"], SKATE_STOP),
                       torch.where(coast | (s["v_s"] >= v_on), torch.full_like(s["mode_o"], SKATE_GLIDE),
                                   torch.full_like(s["mode_o"], SKATE_PUSH)))
    seed = s["seed_o"]
    s["mode_o"] = torch.where(seed, rule, s["mode_o"])
    s["t_o"] = torch.where(seed, torch.zeros_like(s["t_o"]), s["t_o"])
    s["seed_o"] = torch.zeros_like(seed)
    if mc.get("seed_reward_mode", False) and mc.get("mode_src") == "sensed":
        # step 5 (reviewer nit): the REWARD machine of the same envs starts from the same rule, so the rewarded and
        # the observed mode agree from step 0 (not the privileged spawn mode). Its own latches still apply: no GLIDE
        # before the mount latch (v6b), COAST only once mounted.
        rr = rule
        if mc.get("v6b", False):
            rr = torch.where(~st["mounted"] & (rr == SKATE_GLIDE), torch.full_like(rr, SKATE_PUSH), rr)
        st["mode"] = torch.where(seed, rr, st["mode"])
        st["credit_mode"] = torch.where(seed, rr, st["credit_mode"])
        st["t_mode"] = torch.where(seed, torch.zeros_like(st["t_mode"]), st["t_mode"])


def _skate_loc_true(env, st: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """(raw approach geometry (N, 4) without the mounted override, trunk-over-deck occlusion (N,) bool)."""
    s = st["sensor"]
    c = s["cfg"]
    raw = skate_approach_obs(env, c["board_name"], clip=c["loc_clip"], mounted_override=False)
    robot: Entity = env.scene["robot"]
    board: Entity = env.scene[c["board_name"]]
    rel = quat_apply_inverse(_skate_board_quat(env, c["board_name"]),
                             robot.data.root_link_pos_w - board.data.root_link_pos_w)
    rel = torch.nan_to_num(rel, nan=0.0)
    ox, oy = c["loc_occlude"]
    occ = (rel[:, 0].abs() < ox) & (rel[:, 1].abs() < oy)
    return raw, occ


def _skate_sensor_update(env, st: dict) -> None:
    s = st["sensor"]
    c = s["cfg"]
    mc = st["cfg"]
    n, dev = env.num_envs, env.device
    dt = env.step_dt
    s["g"] += 1
    g = s["g"]
    # board unit -> radio ring
    raw = _skate_raw_board_sample(env, st)
    slot = g % _SKATE_RING
    ring = s["ring"].clone()
    ring[:, slot] = raw
    ring_g = s["ring_g"].clone()
    ring_g[:, slot] = g
    s["ring"], s["ring_g"] = ring, ring_g
    jit = torch.where(torch.rand(n, device=dev) < c["latency_jitter_prob"],
                      torch.randint(-1, 2, (n,), device=dev), torch.zeros(n, dtype=torch.long, device=dev))
    lat = (s["lat"] + jit).clamp(1, _SKATE_RING - 2)
    src_g = g - lat
    src = src_g % _SKATE_RING
    ar = torch.arange(n, device=dev)
    have = ring_g[ar, src] == src_g
    start = (s["burst"] <= 0) & (torch.rand(n, device=dev) < c["burst_prob"])
    lo, hi = c["burst_len"]
    s["burst"] = torch.where(start, torch.randint(lo, hi + 1, (n,), device=dev), (s["burst"] - 1).clamp_min(0))
    lost = (torch.rand(n, device=dev) < s["drop_p"]) | (s["burst"] > 0)
    take = have & ~lost & (src_g > s["held_g"])
    s["held"] = torch.where(take.unsqueeze(1), ring[ar, src], s["held"])
    s["held_g"] = torch.where(take, src_g, s["held_g"])
    # robot side: low-passed received speed and the observer mode machine
    a = dt / (c["tau_mode"] + dt)
    s["v_s"] = s["v_s"] + a * (s["held"][:, 0] - s["v_s"])
    cmd = env.command_manager.get_command(mc["command_name"])[:, 0]
    _skate_observer_seed(st, cmd)
    s["mode_o"], s["t_o"] = _skate_observer_step(mc, s["mode_o"], s["t_o"], s["v_s"], cmd, mc["gap"], dt)
    # localization: 10 Hz sample-and-hold of the delayed, noisy approach geometry
    true4, occ = _skate_loc_true(env, st)
    lslot = g % _SKATE_LOC_RING
    loc_ring = s["loc_ring"].clone()
    loc_ring[:, lslot] = true4
    loc_ring_g = s["loc_ring_g"].clone()
    loc_ring_g[:, lslot] = g
    loc_ring_occ = s["loc_ring_occ"].clone()
    loc_ring_occ[:, lslot] = occ
    s["loc_ring"], s["loc_ring_g"], s["loc_ring_occ"] = loc_ring, loc_ring_g, loc_ring_occ
    # a fix = the frame CAPTURED loc_lat steps ago; it exists only if that slot was written this episode at exactly
    # that step (the ring is invalidated at reset), and the fiducial was visible THEN. The current occlusion is never
    # used: the robot learns "occluded" only through fixes that stop arriving (valid decays after loc_hold_s).
    cap_g = g - s["loc_lat"]
    cslot = cap_g % _SKATE_LOC_RING
    tick = ((g + s["loc_phase"]) % c["loc_period"]) == 0
    have = _skate_loc_capture_ok(loc_ring_g, loc_ring_occ, cap_g)
    got = tick & s["loc_on"] & have & (torch.rand(n, device=dev) >= c["loc_miss_prob"])
    old = loc_ring[ar, cslot]
    yaw = torch.atan2(old[:, 2], old[:, 3]) + s["loc_bias"][:, 2] + torch.randn(n, device=dev) * c["loc_yaw_noise"]
    xy = old[:, :2] + s["loc_bias"][:, :2] + torch.randn(n, 2, device=dev) * c["loc_pos_noise"]
    meas = torch.cat([xy.clamp(-c["loc_clip"], c["loc_clip"]), torch.sin(yaw).unsqueeze(1),
                      torch.cos(yaw).unsqueeze(1)], dim=1)
    s["loc_held"] = torch.where(got.unsqueeze(1), meas, s["loc_held"])
    s["loc_age"] = torch.where(got, torch.zeros_like(s["loc_age"]), s["loc_age"] + dt)
    s["loc_occ"] = occ


def _skate_sensor_reset(env, st: dict, env_ids: torch.Tensor, v0: torch.Tensor,
                        vel_w: torch.Tensor | None = None, g_deck: torch.Tensor | None = None) -> None:
    """New episode: per-episode sensor parameters, buffers primed with the spawn state (the stream was running). The
    state tensors may be inference tensors (rebound during rollouts), so they are rebuilt out of place.

    ``vel_w`` = the (m, 3) world linear velocity reset_skate_state just wrote (the board's data.root_link_* still holds
    the previous episode until the next forward pass, so nothing here reads the board pose).
    ``g_deck`` (step 5, uphill spawns): (m, 3) unit gravity in the canonical deck frame at the spawn; None = flat."""
    s = st["sensor"]
    c = s["cfg"]
    dev = env.device
    ids = env_ids.to(dev).long()
    m = len(ids)
    u = lambda lo, hi, *sh: lo + (hi - lo) * torch.rand(m, *sh, device=dev)

    def put(key, val):
        t = s[key].clone()
        t[ids] = val
        s[key] = t

    put("lat", torch.randint(c["latency_steps"][0], c["latency_steps"][1] + 1, (m,), device=dev))
    put("drop_p", u(*c["drop_prob"]))
    put("burst", 0)
    put("gyro_bias", u(-c["gyro_bias"], c["gyro_bias"], 3))
    axis = torch.randn(m, 3, device=dev)
    axis = axis / axis.norm(dim=1, keepdim=True).clamp_min(1e-6)
    ang = u(0.0, math.radians(c["imu_mis_deg"]))
    put("mis", matrix_from_quat(quat_from_angle_axis(ang, axis)))
    put("leak", u(*c["grav_acc_leak"]))
    put("wscale", u(1.0 - c["wheel_radius_err"], 1.0 + c["wheel_radius_err"]))
    board: Entity = env.scene[c["board_name"]]
    cnt = 2.0 * math.pi / c["wheel_counts"]
    put("wheel_q_prev", torch.round(torch.nan_to_num(board.data.joint_pos[ids, s["wheel_id"]], nan=0.0) / cnt) * cnt)
    v0 = v0.to(dev).float().reshape(-1)
    if vel_w is None:     # fallback (no caller does this): spawn at rest
        vel_w = torch.zeros(m, 3, device=dev)
    put("v_prev_w", vel_w.to(dev).float().reshape(m, 3))
    prime = torch.zeros(m, _SKATE_RAW_DIM, device=dev)
    prime[:, 0] = v0
    # every episode starts on the flat (_skate_slope_reset), so the deck reads gravity -z through its misalignment
    g_down = torch.tensor([0.0, 0.0, -1.0], device=dev).expand(m, 3)
    if g_deck is not None:  # step 5 uphill spawns start on a grade
        g_down = g_deck.to(dev).float().reshape(m, 3)
    prime[:, 4:7] = torch.bmm(s["mis"][ids], g_down.unsqueeze(-1)).squeeze(-1)
    put("ring", prime.unsqueeze(1).expand(m, _SKATE_RING, _SKATE_RAW_DIM))
    put("ring_g", -10**6)
    put("held", prime)
    put("held_g", s["g"])
    if "grade_lp" in s:     # step 5 (skate_sensed_grade_deg allocates it): new calibration residual, filter re-primed
        put("grade_cal_err", _skate_grade_cal_err(m, dev))
        g_cal = torch.bmm(s["mis"][ids].transpose(1, 2), prime[:, 4:7].unsqueeze(-1)).squeeze(-1)
        put("grade_lp", skate_grade_from_gravity_deg(g_cal) + s["grade_cal_err"][ids])
    put("v_s", v0)
    # observer machine: seeded from its own rule (sensed v0 + the episode's command, t = 0) at the first use after the
    # reset, not from the privileged spawn mode; the command manager resamples AFTER the reset events
    put("mode_o", SKATE_STOP)
    put("t_o", 0.0)
    put("seed_o", True)
    put("loc_on", torch.rand(m, device=dev) >= c["loc_off_prob"])
    put("loc_lat", torch.randint(c["loc_latency"][0], c["loc_latency"][1] + 1, (m,), device=dev))
    put("loc_phase", torch.randint(0, c["loc_period"], (m,), device=dev))
    put("loc_bias", torch.cat([u(-c["loc_pos_bias"], c["loc_pos_bias"], 2),
                               u(-c["loc_yaw_bias"], c["loc_yaw_bias"], 1)], dim=1))
    put("loc_ring", 0.0)
    put("loc_ring_g", -10**6)    # no frame captured this episode yet: no fix until one is (no zero-slot sentinel)
    put("loc_ring_occ", True)
    put("loc_held", 0.0)
    put("loc_age", 1e3)       # no fix yet: the camera acquires at its next tick after the latency
    put("loc_occ", False)


# Student board block, in order: (name, nominal mean, nominal std) - documentation / normalizer sanity.
SKATE_STUDENT_BOARD_OBS = (
    ("wheel_speed", 0.4, 0.4), ("board_gyro_x", 0.0, 0.2), ("board_gyro_y", 0.0, 0.2), ("board_gyro_z", 0.0, 0.3),
    ("board_grav_x", 0.0, 0.05), ("board_grav_y", 0.0, 0.05), ("board_grav_z", -1.0, 0.01), ("packet_age", 0.2, 0.1),
    ("mode_speed_sensed", 0.4, 0.4), ("v_off", 0.3, 0.3), ("is_glide", 0.4, 0.5), ("is_stop", 0.1, 0.3),
    ("time_in_mode", 1.0, 0.7),
)
SKATE_LOCALIZATION_OBS = ("loc_dx", "loc_dy", "loc_sin_dyaw", "loc_cos_dyaw", "loc_valid")


def _skate_sensor_state(env) -> dict:
    """The mode-machine state with the sensor model allocated (the observation manager calls the obs terms once at
    construction, before the startup events: defaults until skate_sensor_setup writes its params)."""
    st = _skate_state(env)
    if "sensor" not in st:
        skate_sensor_setup(env, None)
    return st


def skate_student_board_raw(env) -> torch.Tensor:
    """(N, 7) the last RECEIVED board-unit sample [wheel speed, gyro xyz, gravity xyz] (radio latency + drops)."""
    st = _skate_sensor_state(env)
    return st["sensor"]["held"].clone()


def skate_student_board_obs(env) -> torch.Tensor:
    """(N, 13) student board block (SKATE_STUDENT_BOARD_OBS): received sample, its age, and the robot-side mode
    machine on the sensed speed (mode speed, v_off, GLIDE, STOP, time in mode)."""
    st = _skate_sensor_state(env)
    s = st["sensor"]
    c = s["cfg"]
    mc = st["cfg"]
    cmd = env.command_manager.get_command(mc["command_name"])[:, 0]
    _skate_observer_seed(st, cmd)
    v_on = cmd.clamp_min(mc["stop_cmd"])
    stop = (s["mode_o"] == SKATE_STOP).float()
    coast = ((cmd >= mc["stop_cmd"]) & (cmd < mc["coast_cmd_hi"])).float()
    v_off = (v_on - mc["gap"]) * (1.0 - stop) * (1.0 - coast)
    age = ((s["g"] - s["held_g"]).float() / c["age_norm"]).clamp(0.0, 1.0)
    out = torch.cat([
        s["held"], age.unsqueeze(1), s["v_s"].unsqueeze(1), v_off.unsqueeze(1),
        (s["mode_o"] == SKATE_GLIDE).float().unsqueeze(1), stop.unsqueeze(1),
        s["t_o"].clamp(max=_SKATE_T_MODE_OBS_MAX).unsqueeze(1),
    ], dim=1)
    assert out.shape[1] == len(SKATE_STUDENT_BOARD_OBS)
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def skate_localization_obs(env) -> torch.Tensor:
    """(N, 5) OPTIONAL localization block (head-camera fiducial on the deck): the teacher's approach geometry [dx, dy,
    sin, cos] sampled at 10 Hz, 40-100 ms late, noisy and biased, missing 10 % of samples, no new fix from a frame
    captured while the trunk was over the deck (fiducial occluded) and none for whole episodes (loc_off_prob); last =
    valid flag (a fix younger than loc_hold_s). Zeros when invalid."""
    st = _skate_sensor_state(env)
    s = st["sensor"]
    c = s["cfg"]
    valid = s["loc_on"] & (s["loc_age"] <= c["loc_hold_s"])   # never the current occlusion (that is truth now)
    v = valid.float().unsqueeze(1)
    return torch.nan_to_num(torch.cat([s["loc_held"] * v, v], dim=1), nan=0.0)


# ── servo heat (servo_heat.py) ──────────────────────────────────────────────────────────────────────────────────────


def _bam_heat_actuators(env, asset_name: str = "robot") -> list:
    from micro_skater.actuator.friction_dr_bam import FrictionDRBamActuator

    return [a for a in env.scene[asset_name].actuators if isinstance(a, FrictionDRBamActuator)]


def servo_heat_state(env, ema_tau: float = 2.0, asset_name: str = "robot") -> dict:
    """Per env step (lazy, keyed on common_step_counter): i2 = (N, 14) mean motor current^2 over the step's physics
    substeps (A^2, servo order), i = the mean signed current (A; i^2 vs i2 splits a held load from dither), p = copper loss at R0 (W), p_ema = its ema_tau low-pass (restarted with the episode)."""
    cache = env.__dict__.setdefault("_servo_heat", {})
    if cache.get("step") == env.common_step_counter and "p" in cache:
        return cache
    from micro_skater import servo_heat as _sh

    acts = list(_bam_heat_actuators(env, asset_name))
    parts = [a.pop_i2_mean() for a in acts]
    if parts and all(p is not None for p in parts):
        i2 = torch.nan_to_num(torch.cat(parts, dim=1), nan=0.0, posinf=0.0)
        i = torch.nan_to_num(torch.cat([a.i_last for a in acts], dim=1), nan=0.0, posinf=0.0, neginf=0.0)
    else:
        i2 = torch.zeros(env.num_envs, 14, device=env.device)
        i = torch.zeros_like(i2)
    p = i2 * _sh.R0_OHM
    ema = cache.get("p_ema")
    if ema is None or ema.shape != p.shape:
        ema = p.clone()
    a = env.step_dt / (ema_tau + env.step_dt)
    ema = ema + a * (p - ema)
    fresh = (env.episode_length_buf <= 1).unsqueeze(1)
    ema = torch.where(fresh, p, ema)
    cache.update(step=env.common_step_counter, i2=i2, i=i, p=p, p_ema=ema)
    return cache


def servo_heat_power(env, asset_name: str = "robot") -> torch.Tensor:
    """Cost (>= 0, NEGATIVE weight): total copper loss of the 14 servos, sum I^2 R0 (W)."""
    return servo_heat_state(env, asset_name=asset_name)["p"].sum(dim=1)


def servo_heat_excess(env, p_cont: float = 0.65, asset_name: str = "robot") -> torch.Tensor:
    """Cost (>= 0, NEGATIVE weight): sum over servos of relu(2 s low-passed copper loss - p_cont) (W). p_cont = the
    hot-corner continuous limit (servo_heat.continuous_power_limit(XL330_HOT) ~ 0.65 W): a joint held loaded (a brace
    parked against a stop) heats past it; brief peaks average out."""
    return (servo_heat_state(env, asset_name=asset_name)["p_ema"] - p_cont).clamp_min(0.0).sum(dim=1)


def skate_heat_cost(env, p_cont: float = 0.65, cap_w: float = 8.0, it0: int = 0, asset_name: str = "robot") -> torch.Tensor:
    """Cost (>= 0, NEGATIVE weight; real2 2026-11-21, MICRODUCK_SKATE_HEAT=1 in microduck_skate_sequence_env_cfg.py):
    servo_heat_excess = sum over servos of relu(2 s low-passed copper loss I^2 R0 - p_cont) (W), capped at cap_w so the
    cost rate stays below the alive pay (a fall stays the worst outcome), 0 before iteration it0 (env steps / 24). p_cont
    0.65 W = 0.48 A RMS (servo_heat.continuous_power_limit(XL330_HOT)): sustained current above it heats the servo;
    a static squeeze at the 1.0 A limit is 2.81 W per servo, brief strokes average out in the 2 s low-pass.
    Own low-pass seeded at 0 at the episode start (as skate_manual.skate_manual_heat_excess; the shared
    servo_heat_state p_ema seeds with the first step's loss, which charges a 20 ms reset transient as ~2 s of heat and
    rewards falling early). The shared p_ema is left untouched for the older tasks."""
    hs = servo_heat_state(env, asset_name=asset_name)
    st = env.__dict__.setdefault("_skate_heat", {})
    p = hs["p"]
    if st.get("step") != env.common_step_counter or "ema" not in st or st["ema"].shape != p.shape:
        ema = st.get("ema")
        if ema is None or ema.shape != p.shape:
            ema = torch.zeros_like(p)
        a = env.step_dt / (2.0 + env.step_dt)
        fresh = (env.episode_length_buf <= 1).unsqueeze(1)
        ema = torch.where(fresh, a * p, ema + a * (p - ema))
        st.update(step=env.common_step_counter, ema=ema)
    c = (st["ema"] - p_cont).clamp_min(0.0).sum(dim=1)
    if cap_w is not None:
        c = c.clamp_max(cap_w)
    if it0 and env.common_step_counter < it0 * 24:
        return torch.zeros_like(c)
    return c


# ══ SKATE STEP 5 (2026-11-06): continuous pumping under hold-W, pushing uphill ════════════════════════════════════════
# task cfg.
# Everything below is used only by that task (Mjlab-Skate-Student-Pump-Flat-Backlash-MicroDuck); the hooks it needs in
# the code above are all gated by cfg keys / state keys the older tasks do not have, so they behave bit-for-bit as
# before (proof:).
#
# Throttle: W held = twist[1] = SKATE_PUMP_FLAG (-0.04; the step-off flag is twist[1] > +0.02, never tripped) and
# twist[0] = the ramp r(t) = min(v_top, r0 + a_W t), r0 = max(0.2, sensed wheel speed at the press). The same ramp
# function drives the command term, and (later) the play Director.

SKATE_PUMP_FLAG = -0.04          # twist[1] while W is held (step-off flag is +0.04, its test is > +0.02)
SKATE_PUMP_R_MIN = 0.2           # m/s, the ramp's floor at the press
SKATE_PUMP_UPHILL_ACC = 0.15     # m/s^2: g_fwd (low-passed) below -this = uphill pump context (~1 deg)
SKATE_PUMP_PUSH_LEG = ("right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle")


def skate_throttle_r0(v_sensed, r_min: float = SKATE_PUMP_R_MIN):
    """Ramp start at the W press: max(r_min, sensed wheel speed). Floats or tensors."""
    if isinstance(v_sensed, torch.Tensor):
        return v_sensed.clamp_min(r_min)
    return max(float(r_min), float(v_sensed))


def skate_throttle_ramp(t, r0, v_top, a_w):
    """Hold-W throttle ramp r(t) = min(v_top, r0 + a_W * t) (t = seconds since the press, >= 0). Monotone in t, capped
    at v_top (r0 above v_top gives v_top). Floats or tensors (broadcast)."""
    if any(isinstance(x, torch.Tensor) for x in (t, r0, v_top, a_w)):
        ref = next(x for x in (t, r0, v_top, a_w) if isinstance(x, torch.Tensor))

        def as_t(x):
            return x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=ref.dtype, device=ref.device)

        t, r0, v_top, a_w = (as_t(x) for x in (t, r0, v_top, a_w))
        return torch.minimum(v_top, r0 + a_w * t.clamp_min(0.0))
    return min(float(v_top), float(r0) + float(a_w) * max(0.0, float(t)))


# Grade cap of the hold-W command (step-5 probes P3 + the v7f calibration: feasible climb speeds
# 3 deg 0.3-1.0, 4 deg 0.25-0.8, 6 deg 0.1-0.4 m/s): on a climb the ramp never asks for more than the grade allows.
# (grade deg along the board heading, + = uphill ahead; cap m/s), linear in between, no cap below the first point.
SKATE_PUMP_GRADE_CAP = ((2.0, 1.4), (3.0, 1.0), (4.0, 0.8), (6.0, 0.4), (9.0, 0.2))


def skate_throttle_grade_cap(grade_deg, table=SKATE_PUMP_GRADE_CAP):
    """Speed cap (m/s) of the hold-W command at an uphill grade (deg, + = uphill ahead): +inf up to the table's first
    grade, piecewise linear through the table, its last speed beyond. The command is min(ramp, cap). Floats or
    tensors. The Director applies the same cap with the deck IMU's grade."""
    g = [float(a) for a, _ in table]
    v = [float(b) for _, b in table]
    if isinstance(grade_deg, torch.Tensor):
        x = grade_deg.float()
        out = torch.full_like(x, float("inf"))
        for (g0, v0), (g1, v1) in zip(zip(g, v), zip(g[1:], v[1:])):
            seg = (x > g0) & (x <= g1)
            out = torch.where(seg, v0 + (v1 - v0) * (x - g0) / (g1 - g0), out)
        return torch.where(x > g[-1], torch.full_like(x, v[-1]), out)
    x = float(grade_deg)
    if x <= g[0]:
        return float("inf")
    for (g0, v0), (g1, v1) in zip(zip(g, v), zip(g[1:], v[1:])):
        if x <= g1:
            return v0 + (v1 - v0) * (x - g0) / (g1 - g0)
    return v[-1]


def skate_pump_flag_active(env, command_name: str = "twist") -> torch.Tensor:
    """(B,) bool: W held (twist[1] below half the pump flag, < -0.02)."""
    return env.command_manager.get_command(command_name)[:, 1] < 0.5 * SKATE_PUMP_FLAG


def _skate_riding(env, st: dict) -> torch.Tensor:
    return st["mounted"] & ~skate_dismount_active(env)


def skate_uphill_grade(env, st: dict | None = None) -> torch.Tensor:
    """(B,) rad, the current grade along the canonical board heading, + = uphill ahead (0 without slopes)."""
    st = st if st is not None else _skate_state(env)
    sl = st.get("slope")
    if sl is None:
        return torch.zeros(env.num_envs, device=env.device)
    h = skate_board_heading(env, sl["cfg"]["board_name"])
    return torch.nan_to_num(torch.asin((-(sl["s"] * h).sum(dim=1)).clamp(-1.0, 1.0)), nan=0.0)


SKATE_SENSED_GRADE_TAU = 0.5     # s, low-pass of the sensed grade (the command's grade cap and release latch; Director)
SKATE_GRADE_CAL_STD = 0.5        # deg, per-episode residual of the one-time flat-ground board IMU calibration ...
SKATE_GRADE_CAL_MAX = 1.0        # deg, ... clipped here


def skate_grade_from_gravity_deg(g_deck: torch.Tensor) -> torch.Tensor:
    """deg, + = uphill ahead, from a gravity vector in the canonical deck frame (x forward, z up; gravity points
    down, so on an uphill ahead its x part is negative): asin(-g_x / |g|). The Director computes the same from the
    received board sample (gravity = sample[4:7])."""
    gx = g_deck[..., 0] / g_deck.norm(dim=-1).clamp_min(1e-6)
    return torch.nan_to_num(torch.rad2deg(torch.asin((-gx).clamp(-1.0, 1.0))), nan=0.0)


def _skate_grade_calibrated(s: dict, g_deck: torch.Tensor, err: torch.Tensor) -> torch.Tensor:
    """deg: the grade of a received deck-gravity sample after the board's one-time flat-ground IMU calibration (the
    episode's sampled mounting misalignment s["mis"] removed) plus the calibration's residual error ``err``."""
    g = torch.bmm(s["mis"].transpose(1, 2), g_deck.unsqueeze(-1)).squeeze(-1)
    return skate_grade_from_gravity_deg(g) + err


def _skate_grade_cal_err(m: int, device) -> torch.Tensor:
    return (torch.randn(m, device=device) * SKATE_GRADE_CAL_STD).clamp(-SKATE_GRADE_CAL_MAX, SKATE_GRADE_CAL_MAX)


def skate_sensed_grade_deg(env, st: dict | None = None, tau: float = SKATE_SENSED_GRADE_TAU) -> torch.Tensor:
    """(B,) deg, the grade along the board as the ROBOT knows it: skate_grade_from_gravity_deg of the last received
    board sample (radio latency / drops, acceleration leak, noise), CALIBRATED: the board IMU's mounting offset is
    measured once, static on known-flat ground, and stored on the board (a systematic mounting bias is a runtime
    calibration), so the episode's sampled misalignment is removed and a per-episode residual
    N(0, SKATE_GRADE_CAL_STD) clipped at +-SKATE_GRADE_CAL_MAX deg is added. The policy's board obs keep the full
    misalignment DR. Low-passed with time constant ``tau`` (first-order, a = dt / (tau + dt) per sensor tick), lazily
    advanced once per sensor tick (st["sensor"]["g"]) by whoever calls it first; re-primed at episode reset
    (_skate_sensor_reset). Without the sensor model: the true grade (skate_uphill_grade)."""
    st = st if st is not None else _skate_state(env)
    s = st.get("sensor")
    if s is None:
        return torch.rad2deg(skate_uphill_grade(env, st))
    if "grade_cal_err" not in s:
        s["grade_cal_err"] = _skate_grade_cal_err(env.num_envs, env.device)
    raw = _skate_grade_calibrated(s, s["held"][:, 4:7], s["grade_cal_err"])
    g = int(s["g"])
    if "grade_lp" not in s:
        s["grade_lp"], s["grade_g"] = raw.clone(), g
    k = g - s["grade_g"]
    if k > 0:
        keep = (tau / (tau + env.step_dt)) ** k if tau > 0 else 0.0
        s["grade_lp"] = s["grade_lp"] + (1.0 - keep) * (raw - s["grade_lp"])
        s["grade_g"] = g
    return s["grade_lp"].clone()


def skate_pump_context(env, uphill_acc: float = SKATE_PUMP_UPHILL_ACC) -> torch.Tensor:
    """(B,) bool, the pump context: riding (mounted, no step-off flag) and (W held, or uphill: low-passed passive
    rolling acceleration g_fwd < -uphill_acc). There glide / push_time / glide_floor are off and a mode-neutral riding
    pay is on: pushing as often as the speed needs is not taxed against a glide that is not wanted (or not possible)."""
    st = _skate_state(env)
    ctx = skate_pump_flag_active(env, st["cfg"]["command_name"])
    if "slope" in st:
        up = st["slope"]["g_lp"] < -uphill_acc
        if st["cfg"].get("coast_glide_uphill", False) and "coast" in st:
            # s5b: COAST on a climb (W released: the hill rolls the robot back) is a glide: glide / glide_floor /
            # push_time apply as on the flat, no pump-context pay
            up = up & ~st["coast"]
        ctx = ctx | up
    return ctx & _skate_riding(env, st)


def skate_outside_pump(func):
    """Wrap a riding term so it reads 0 in the pump context (glide, push_time, glide_floor)."""
    import functools

    @functools.wraps(func)
    def outside(env, *args, **kwargs):
        return func(env, *args, **kwargs) * (~skate_pump_context(env)).float()

    outside.skate_outside_pump = True
    return outside


def skate_flag_off(func):
    """Wrap a term so it reads 0 while W is held (the 2 s mean cruise term lags a moving ramp: being ahead of the ramp
    would raise it)."""
    import functools

    @functools.wraps(func)
    def off(env, *args, **kwargs):
        return func(env, *args, **kwargs) * (~skate_pump_flag_active(env)).float()

    off.skate_flag_off = True
    return off


def skate_pump_ride(env, board_foot_sensor: str = "board_foot_deck", gate_window_s: float = 0.5,
                    stroke_window_s: float | None = None, push_floor_sensor: str = "push_foot_ground",
                    below_margin: float = 0.1) -> torch.Tensor:
    """Mode-neutral riding pay in the pump context: 1 while the board foot is on the deck (now or within the window).
    It replaces the glide pay there (glide is gated off), so continuous pumping is not poorer than the s1 push / glide
    cycle. ``stroke_window_s`` (flop audit, s5 reward-mass probe: a parked rider under the flag kept the whole pay):
    while the speed is more than ``below_margin`` under the speed term's target, the pay also needs a push-foot stroke
    within that window (_skate_recent_stroke: a touchdown or lift-off, not a foot planted on the floor: the first probe
    showed a parked rider braking with the push foot down kept the pay); at the target no stroke is needed. Below the
    target the pay is also scaled by clamp(v / target, 0, 1) (review: a dab every 1.5 s defeated the stroke gate alone,
    a parked rider kept 1.74/s on a 3 deg climb): a park earns ~0 whatever the feet do.
    In [0, 1]; wrap riding-only in the cfg."""
    pay = skate_pump_context(env).float() * _skate_recent_contact(env, board_foot_sensor, gate_window_s)
    if stroke_window_s is not None:
        st = _skate_state(env)
        c = st["cfg"]
        _, v_on = _skate_v_on(env, c)
        target = (v_on - 0.5 * c["gap"]).clamp_min(0.02)
        v = torch.minimum(st["v_mode"], skate_board_fwd_speed(env, c["board_name"]))
        below = v < target - below_margin
        stroke = _skate_recent_stroke(env, push_floor_sensor, stroke_window_s)
        frac = (v / target).clamp(0.0, 1.0)
        pay = pay * torch.where(below, stroke, torch.ones_like(stroke)) * frac
    return torch.nan_to_num(pay, nan=0.0)


def _skate_recent_stroke(env, sensor_name: str, window_s: float) -> torch.Tensor:
    """1 if the sensor's foot is in a stroke: in contact for less than window_s (a touchdown within the window), or
    lifted less than window_s ago. A foot PLANTED longer than the window (a parked brake / scooter stance) is not a
    stroke."""
    data = env.scene.sensors[sensor_name].data
    ct = data.current_contact_time
    ct = ct.max(dim=-1).values if ct.dim() > 1 else ct
    on = _skate_contact(env, sensor_name) > 0
    return torch.where(on, ct < window_s, _skate_air_time(env, sensor_name) < window_s).float()


# ── climb progress: a potential on virtual height ──────────────────────────────────────────────────────────────────
# A grade is a tilted gravity, so "height" is virtual: dh = -(v_COM,xy . s) dt with s = sin(theta) x the downhill
# direction (the system COM: internal motions cannot move it). Counted on UPHILL segments only (a downhill descent is
# not charged). Gains pay only under the pump flag; losses (rolling back) are charged whenever riding: over any loop
# on the same uphill the sum is <= 0, and exactly 0 under the flag (it telescopes).


def skate_climb_setup(env, env_ids, ok_m: float = 0.03, ok_frac: float = 0.6, min_work_m: float = 0.02):
    """Startup event (step 5): allocate the climb state (st["climb"]) and the uphill scoring for the grade level."""
    del env_ids
    st = _skate_state(env)
    n, dev = env.num_envs, env.device

    def z():
        return torch.zeros(n, device=dev)

    st["climb"] = dict(dh=z(), h_seg=z(), v_open=z(), cred=z(), open=torch.zeros(n, dtype=torch.bool, device=dev),
                       seg_prev=torch.zeros(n, dtype=torch.long, device=dev),
                       sloped_prev=torch.zeros(n, dtype=torch.bool, device=dev),
                       n_ok=torch.zeros((), device=dev), n_all=torch.zeros((), device=dev),
                       level=0, level_step=None, ok_m=float(ok_m), ok_frac=float(ok_frac),
                       min_work_m=float(min_work_m))


def skate_climb_need(drop0: torch.Tensor, v_open: torch.Tensor, ok_frac: float, ok_m: float, eta: float,
                     g0: float = 9.81, min_work_m: float = 0.02) -> tuple[torch.Tensor, torch.Tensor]:
    """(need, scorable) of an uphill segment (F3). Momentum at the segment start would coast it up
    h_coast = v_open^2 / (2 eta g) of virtual height (passive deceleration eta g sin(theta) over a distance of
    h / sin(theta)), so only the climb BEYOND that is pumping: need = h_coast + min(ok_frac x (drop0 - h_coast), ok_m).
    The need never exceeds the segment's budget (the segment ends when it is spent). A segment whose budget leaves
    less than ``min_work_m`` above h_coast measures coasting, not pumping: not scorable."""
    h_coast = v_open.clamp_min(0.0).pow(2) / (2.0 * eta * g0)
    work = drop0 - h_coast
    need = h_coast + torch.minimum(ok_frac * work.clamp_min(0.0), torch.full_like(work, ok_m))
    return need, work >= min_work_m


def skate_climb_delta(dh: torch.Tensor, up_seg: torch.Tensor, pump: torch.Tensor) -> torch.Tensor:
    """Credited height change (m): on uphill segments only; a gain only under the pump flag, a loss always."""
    d = dh * up_seg.float()
    return torch.where(d > 0, d * pump.float(), d)


def _skate_climb_update(env, st: dict, fresh: torch.Tensor) -> None:
    cl = st["climb"]
    sl = st.get("slope")
    if sl is None:
        cl["dh"] = torch.zeros(env.num_envs, device=env.device)
        return
    v = _skate_system_vel_xy(env, st["cfg"]["board_name"])
    dh = torch.nan_to_num(-(v * sl["s"]).sum(dim=1) * env.step_dt, nan=0.0).clamp(-0.05, 0.05)
    dh = torch.where(fresh, torch.zeros_like(dh), dh) * sl["up_seg"].float()
    cl["dh"] = dh
    cl["h_seg"] = cl["h_seg"] + dh
    if "cred" in cl:     # a new episode banks nothing yet
        cl["cred"] = torch.where(fresh, torch.zeros_like(cl["cred"]), cl["cred"])


def _skate_climb_score(env, st: dict, ids: torch.Tensor, failed: torch.Tensor | None = None) -> None:
    """Score the open uphill segments of ``ids`` that ended while W was held: ok = climbed at least
    skate_climb_need (the coastable height from the speed at the segment start, plus min(ok_frac x the rest of the
    budget, ok_m)) and did not fail. Segments that momentum alone could nearly finish are not scored."""
    cl = st["climb"]
    if len(ids) == 0:
        return
    sl = st["slope"]
    pump = skate_pump_flag_active(env, st["cfg"]["command_name"])[ids] & st["mounted"][ids]
    need, scorable = skate_climb_need(sl["drop0"][ids], cl["v_open"][ids], cl["ok_frac"], cl["ok_m"],
                                      float(sl["cfg"]["eta"]), float(sl.get("g0", 9.81)), cl["min_work_m"])
    sel = cl["open"][ids] & pump & scorable
    ok = sel & (cl["h_seg"][ids] >= need)
    if failed is not None:
        ok = ok & ~failed
    cl["n_ok"] = cl["n_ok"] + ok.float().sum()
    cl["n_all"] = cl["n_all"] + sel.float().sum()
    o = cl["open"].clone()
    o[ids] = False
    cl["open"] = o


def _skate_climb_post(env, st: dict) -> None:
    """After the slope update: score uphill segments that ended (budget spent = flat runout, or resampled)."""
    cl = st["climb"]
    sl = st.get("slope")
    if sl is None:
        return
    sloped = sl["s_tgt"].norm(dim=1) > 1e-6
    new_seg = sl["seg_id"] != cl["seg_prev"]
    spent = cl["sloped_prev"] & ~sloped & ~new_seg
    ended = (new_seg | spent) & cl["open"]
    if bool(ended.any()):
        _skate_climb_score(env, st, ended.nonzero().squeeze(1))
    # a new segment restarts the height count; an uphill one opens for scoring
    cl["open"] = torch.where(new_seg, sl["up_seg"], cl["open"])
    cl["h_seg"] = torch.where(new_seg, torch.zeros_like(cl["h_seg"]), cl["h_seg"])
    # the climb potential banked in this segment (terminal_charge): kept once the segment is done
    cl["cred"] = torch.where(new_seg | spent, torch.zeros_like(cl["cred"]), cl["cred"])
    if bool(new_seg.any()):    # F3: the forward system speed at the segment start (its coastable height)
        h = skate_board_heading(env, sl["cfg"]["board_name"])
        v_f = (_skate_system_vel_xy(env, sl["cfg"]["board_name"]) * h).sum(dim=1).clamp_min(0.0)
        cl["v_open"] = torch.where(new_seg, torch.nan_to_num(v_f, nan=0.0), cl["v_open"])
    cl["seg_prev"] = sl["seg_id"].clone()
    cl["sloped_prev"] = sloped


def skate_climb_progress(env, scale: float = 0.01, terminal_charge: bool = False,
                         coast_loss_cap: bool = False) -> torch.Tensor:
    """Step 5 climb potential: credited virtual height change (skate_climb_delta) per `scale` m, divided by dt (the
    episode sum is weight * dh / scale: weight 0.7 at scale 0.01 = 0.7 per cm). The uphill-segment gate is already in
    the state's dh. Wrap riding-only in the cfg.

    ``terminal_charge`` (as skate_push_progress, the s1r hold-W collapse): the height credited in the current uphill
    segment (cl["cred"], while riding; reset when the segment ends: budget spent or a new segment) is charged back at a
    non-timeout termination, so a fall halfway up does not keep the climb pay. The mounted latch only sets, so the
    riding-only wrapper passes the charge.

    ``coast_loss_cap`` (s5b, needs terminal_charge: W released on a climb = COAST, the hill rolls the robot back): a
    loss under COAST is charged only down to the height banked in this segment (cl["cred"] >= 0), so rolling back
    gives back what the climb paid and no more. Gains still pay under W only: a climb / roll-back loop nets 0, and a
    commanded roll-back past the banked height costs nothing (the policy is not taught to brake against COAST)."""
    st = _skate_state(env)
    cl = st.get("climb")
    if cl is None or "slope" not in st:
        return torch.zeros(env.num_envs, device=env.device)
    pump = skate_pump_flag_active(env, st["cfg"]["command_name"])
    d = skate_climb_delta(cl["dh"], torch.ones_like(cl["dh"], dtype=torch.bool), pump)
    if coast_loss_cap and terminal_charge and "coast" in st:
        coast = st["coast"] & ~pump
        d = torch.where(coast & (d < 0), torch.maximum(d, -cl["cred"].clamp_min(0.0)), d)
    if terminal_charge:
        on = (st["mounted"] & ~skate_dismount_active(env)).float()     # what the riding-only wrapper pays
        cred = cl["cred"] + d * on
        term = env.termination_manager.terminated.bool()
        d = torch.where(term, d - cred.clamp_min(0.0), d)
        cl["cred"] = torch.where(term, torch.zeros_like(cred), cred)
    return d / (scale * env.step_dt)


SKATE_LEVEL_STEP_JUMP = 1000      # env steps: a larger jump of common_step_counter between calls = a resume


def skate_uphill_level_curriculum(env, env_ids, levels: list[dict], start_step: int = 0, promote: float = 0.6,
                                  min_segments: int = 200, min_steps: int = 100 * 24,
                                  relative: bool = False) -> torch.Tensor:
    """Performance-gated uphill grade level (step 5). levels = [{"up_max_deg", "up_lateral_deg"}, ...] (live slope
    cfg; an optional per-level "promote" overrides ``promote`` for leaving that level). Uphill segments that end while
    W is held are scored (skate_climb_setup / _skate_climb_score; an episode that TERMINATES on an open one is a
    failure). The level goes up once at least ``min_segments`` were scored at the current level, ``promote`` of them
    climbed, and it has held ``min_steps``. Never down.
      - Sliding window: both counts are halved whenever 2 x min_segments are scored, so early failures fade (a
        cumulative count pinned the rate under the gate long after the skill existed).
      - The ``min_steps`` floor is re-based whenever the step counter jumps by more than SKATE_LEVEL_STEP_JUMP (a
        resume: the wrapper's first reset runs at counter 0, then runner.load jumps it to the checkpoint's).
      - The level is env memory, not checkpointed: a resume restarts at level 0, or at MICRODUCK_S5_START_LEVEL
        (env var, read once at the first call) for a crash-resume."""
    st = _skate_state(env)
    cl = st.get("climb")
    sl = st.get("slope")
    if cl is None or sl is None:
        return torch.tensor([0.0])
    if env_ids is not None and len(env_ids) > 0:
        ids = torch.as_tensor(env_ids, device=env.device).long()
        tm = env.termination_manager
        failed = (tm.terminated[ids] & ~tm.time_outs[ids]).bool()
        _skate_climb_score(env, st, ids, failed=failed)
    step = env.common_step_counter
    if not cl.get("level_init"):
        cl["level_init"] = True
        import os
        lv0 = os.environ.get("MICRODUCK_S5_START_LEVEL")
        if lv0:
            cl["level"] = max(0, min(int(lv0), len(levels) - 1))
            print(f"[skate s5] uphill level {cl['level']} {levels[cl['level']]} from MICRODUCK_S5_START_LEVEL")
    last = cl.get("last_step")
    if cl["level_step"] is None or last is None or step < last or step - last > SKATE_LEVEL_STEP_JUMP:
        cl["level_step"] = step       # first call, or a counter jump (resume): the floor counts from here
    cl["last_step"] = step
    lvl = cl["level"]
    n_all, n_ok = float(cl["n_all"]), float(cl["n_ok"])
    need = float(levels[lvl].get("promote", promote))     # a level may ask for more before the next one
    t0 = start_step + (skate_s5_base_step(env) if relative else 0)
    if (lvl + 1 < len(levels) and step >= t0 and n_all >= min_segments and n_ok >= need * n_all
            and step - cl["level_step"] >= min_steps):
        lvl += 1
        cl["level"] = lvl
        cl["level_step"] = step
        cl["n_ok"] = torch.zeros_like(cl["n_ok"])
        cl["n_all"] = torch.zeros_like(cl["n_all"])
        print(f"[skate s5] uphill level -> {lvl} {levels[lvl]} at step {step} (it {step // 24}); scored {n_all:.0f}, "
              f"climbed {n_ok / max(n_all, 1):.0%}")
    elif n_all >= 2 * min_segments:   # sliding window: early failures fade
        cl["n_ok"] = cl["n_ok"] * 0.5
        cl["n_all"] = cl["n_all"] * 0.5
    sl["cfg"].update({k: v for k, v in levels[lvl].items() if k in SKATE_SLOPE_DEFAULTS})
    return torch.tensor([float(levels[lvl]["up_max_deg"])])


def skate_uphill_success_rate(env) -> torch.Tensor:
    """Metrics helper: the current window's uphill climb success rate (same for every env; 0 before any score)."""
    cl = _skate_state(env).get("climb")
    if cl is None:
        return torch.zeros(env.num_envs, device=env.device)
    r = cl["n_ok"] / cl["n_all"].clamp_min(1.0)
    return r.expand(env.num_envs).clone()


def _skate_slope_spawn_up(env, st: dict, env_ids: torch.Tensor, up: torch.Tensor, yaw_c: torch.Tensor,
                          start_deg: float = 2.0, up_deg: torch.Tensor | None = None) -> torch.Tensor:
    """Step 5 reverse-curriculum spawns: the envs flagged ``up`` start an uphill segment (grade uniform in [min_deg,
    up_max_deg], direction = the canonical board heading +- the uphill spread, uphill ahead), already tilted by
    min(grade, start_deg) and slewing to the full grade at the segment's rate (the keyframes are flat-gravity poses).
    ``yaw_c`` = (n,) canonical board yaw of the spawn (the data still holds the old pose). Returns the (n, 3) unit
    gravity in the canonical deck frame for the sensor priming (flat for the other envs). ``up_deg`` (s5b): (n, 2)
    per-env grade range in degrees overriding [min_deg, up_max_deg] (NaN rows = the live range)."""
    sl = st["slope"]
    c = sl["cfg"]
    dev = env.device
    n = len(env_ids)
    g_deck = torch.tensor([0.0, 0.0, -1.0], device=dev).repeat(n, 1)
    sel = up.nonzero().squeeze(1)
    m = len(sel)
    if m == 0:
        return g_deck
    ids = env_ids[sel]
    lo = math.radians(c["min_deg"])
    hi = math.radians(max(c["up_max_deg"], c["min_deg"]))
    u = torch.rand(m, device=dev)
    if up_deg is None:
        grade = lo + u * (hi - lo)
    else:
        r = torch.deg2rad(up_deg[sel])
        own = ~torch.isnan(r[:, 0])
        lo_t = torch.where(own, r[:, 0], torch.full_like(u, lo))
        hi_t = torch.where(own, r[:, 1], torch.full_like(u, hi))
        grade = lo_t + u * (hi_t - lo_t)
    lat = math.radians(c["up_lateral_deg"] if c.get("up_lateral_deg") is not None else c["lateral_deg"])
    ang = yaw_c[sel] + math.pi + (torch.rand(m, device=dev) * 2 - 1) * lat
    d = torch.stack([torch.cos(ang), torch.sin(ang)], dim=1)
    sl["s_tgt"][ids] = torch.sin(grade).unsqueeze(1) * d
    sl["s"][ids] = torch.sin(torch.clamp(grade, max=math.radians(start_deg))).unsqueeze(1) * d
    hl, hh = c["hold_s"]
    sl["t_left"][ids] = hl + torch.rand(m, device=dev) * (hh - hl)
    rl, rh = (math.radians(x) for x in c["slew_deg_s"])
    sl["rate"][ids] = rl + torch.rand(m, device=dev) * (rh - rl)
    dl, dh = c["drop_m"]
    sl["drop_left"][ids] = dl + torch.rand(m, device=dev) * (dh - dl)
    sl["up_seg"][ids] = True
    sl["drop0"][ids] = sl["drop_left"][ids]
    sl["seg_id"][ids] += 1
    _skate_slope_write(env, sl)
    s = sl["s"][ids]
    gw = torch.cat([s, -(1.0 - s.pow(2).sum(dim=1, keepdim=True)).clamp_min(0.0).sqrt()], dim=1)
    cy, sy = torch.cos(yaw_c[sel]), torch.sin(yaw_c[sel])
    g_deck[sel] = torch.stack([cy * gw[:, 0] + sy * gw[:, 1], -sy * gw[:, 0] + cy * gw[:, 1], gw[:, 2]], dim=1)
    return g_deck


# ── pump command ────────────────────────────────────────────────────────────────────────────────────────────────────


class SkatePumpCommand(SkateCurvatureCommand):
    """Step 5 skate command: SkateCurvatureCommand plus
      - a PUMP bucket (W held): a share pump_prob of the speed resamples of mounted, unflagged envs (and the
        reverse-curriculum spawns that ask for it) press W for its own hold time (pump_hold_range): twist[1] =
        SKATE_PUMP_FLAG, twist[0] = skate_throttle_ramp(t since the press, r0 = max(0.2, sensed wheel speed at the
        press), v_top ~ pump_v_top_range (exactly pump_v_top_play with pump_play_prob: the play default), a_W ~
        pump_a_range). The base resample timer keeps running underneath; a timer resample keeps an active pump.
      - RELEASE: when a pump ends, the env goes to "no key" for release_hold_range s. No key = COAST, except on a climb
        (the SENSED grade, skate_sensed_grade_deg, above release_up_deg, latched off below release_off_deg) where it is
        STOP (twist 0, 0, 0): brake / hold rather than coast and roll back (step 5 decision, item 8). The grade cap
        under W reads the same sensed grade. A release zeroes twist[1]. Play's Director additionally sends STOP at a
        standstill (sensed wheel speed < 0.1 m/s) instead of COAST: see the pump env cfg docstring. COAST resamples follow the same
        rule. Play's Director must apply the same mapping (see the step-5 report).
      - release_mode "coast" (s5b, the author's rule 2026-11-06): no key = COAST everywhere, a climb included (the hill
        rolls the robot back through the turnaround); STOP (0, 0, 0) only once the board is on the flat and slow
        (skate_nokey_stop_latch: |sensed grade| <= flat_stop_deg and |sensed wheel speed| < stop_speed, held while
        < stop_speed + stop_speed_hyst, the grade not re-checked: flat-ground grade noise). The uphill STOP hold is
        not used.
        Spawns with ``coast`` start in no key (st["coast_now"], skate_pump_setup(coast_spawns=True)).
    Set ``scripted`` True (skate_eval.set_command does) to freeze all of it: the eval writes twist itself."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        n, dev = self.num_envs, self.device
        self.pump_on = torch.zeros(n, dtype=torch.bool, device=dev)
        self.pump_t = torch.zeros(n, device=dev)
        self.pump_left = torch.zeros(n, device=dev)
        self.pump_r0 = torch.zeros(n, device=dev)
        self.pump_v_top = torch.ones(n, device=dev)
        self.pump_a = torch.full((n,), 0.3, device=dev)
        self.nokey = torch.zeros(n, dtype=torch.bool, device=dev)
        self.up_latch = torch.zeros(n, dtype=torch.bool, device=dev)
        self.stop_latch = torch.zeros(n, dtype=torch.bool, device=dev)     # s5b: no-key STOP on the flat
        self.scripted = False

    def _sensed_speed(self, ids: torch.Tensor) -> torch.Tensor:
        st = _skate_cache(self._env).get("state")
        if st is not None and "sensor" in st:
            return st["sensor"]["v_s"][ids]
        if st is not None:
            return st["v_mode"][ids]
        return torch.zeros(len(ids), device=self.device)

    def _start_pump(self, ids: torch.Tensor) -> None:
        m = len(ids)
        if m == 0:
            return
        cfg = self.cfg
        dev = self.device
        vt = torch.empty(m, device=dev).uniform_(*cfg.pump_v_top_range)
        play = torch.rand(m, device=dev) < cfg.pump_play_prob
        vt = torch.where(play, torch.full_like(vt, min(cfg.pump_v_top_play, cfg.pump_v_top_range[1])), vt)
        self.pump_v_top[ids] = vt
        self.pump_a[ids] = torch.empty(m, device=dev).uniform_(*cfg.pump_a_range)
        self.pump_r0[ids] = skate_throttle_r0(torch.nan_to_num(self._sensed_speed(ids), nan=0.0), cfg.pump_r_min)
        self.pump_t[ids] = 0.0
        self.pump_left[ids] = torch.empty(m, device=dev).uniform_(*cfg.pump_hold_range)
        self.pump_on[ids] = True

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        if self.scripted or len(env_ids) == 0:
            return
        v = self.vel_command_b[env_ids, 0]
        self.nokey[env_ids] = (v == SKATE_COAST_CMD) & ~self.is_standing_env[env_ids]
        st = _skate_cache(self._env).get("state")
        mounted = st["mounted"][env_ids] if st is not None else torch.ones_like(self.nokey[env_ids])
        flagged = self.dismount_at[env_ids] <= self.t_ep[env_ids]
        forced = torch.zeros_like(mounted)
        if st is not None and "pump_now" in st:
            forced = st["pump_now"][env_ids].clone()
            st["pump_now"][env_ids] = False
        if st is not None and "coast_now" in st:    # s5b: a spawn that starts with no key (never W at this resample)
            cn = st["coast_now"][env_ids].clone()
            st["coast_now"][env_ids] = False
            if bool(cn.any()):
                ids_c = env_ids[cn]
                self.vel_command_b[ids_c, 0] = SKATE_COAST_CMD
                self.is_standing_env[ids_c] = False
                self.nokey[ids_c] = True
                mounted = mounted & ~cn
        new = (~self.pump_on[env_ids] & mounted & ~flagged
               & ((torch.rand(len(env_ids), device=self.device) < self.cfg.pump_prob) | forced))
        self._start_pump(env_ids[new])
        on = env_ids[self.pump_on[env_ids]]
        self.is_standing_env[on] = False
        self.nokey[on] = False

    def reset(self, env_ids):
        if isinstance(env_ids, torch.Tensor):
            self.pump_on[env_ids] = False
            self.nokey[env_ids] = False
            self.up_latch[env_ids] = False
            self.stop_latch[env_ids] = False
        return super().reset(env_ids)

    def compute(self, dt: float) -> None:
        if not self.scripted:
            self.pump_t += dt
            self.pump_left -= dt
            flagged = self.dismount_at <= self.t_ep + dt      # the step-off flag ends W
            ended = self.pump_on & ((self.pump_left <= 0.0) | flagged)
            if bool(ended.any()):
                ids = ended.nonzero().flatten()
                self.pump_on[ids] = False
                self.nokey[ids] = True
                self.vel_command_b[ids, 1] = 0.0     # W released: the flag goes (the parent re-writes a step-off flag)
                self.time_left[ids] = torch.empty(len(ids), device=self.device).uniform_(*self.cfg.release_hold_range)
        super().compute(dt)

    def _update_command(self) -> None:
        super()._update_command()
        if self.scripted:
            return
        st = _skate_cache(self._env).get("state")
        flag = self.t_ep >= self.dismount_at
        pump = self.pump_on & ~flag
        # the grade the robot can know (F2): received deck gravity, low-passed; advanced every step
        g_s = skate_sensed_grade_deg(self._env, st) if st is not None and "slope" in st else None
        if bool(pump.any()):
            r = skate_throttle_ramp(self.pump_t, self.pump_r0, self.pump_v_top, self.pump_a)
            if self.cfg.grade_cap and g_s is not None:
                r = torch.minimum(r, skate_throttle_grade_cap(g_s))
            self.vel_command_b[:, 0] = torch.where(pump, r, self.vel_command_b[:, 0])
            self.vel_command_b[:, 1] = torch.where(pump, torch.full_like(r, SKATE_PUMP_FLAG), self.vel_command_b[:, 1])
        # belt and braces (B1): no W and no step-off flag -> twist[1] never reads as the pump flag
        tw1 = self.vel_command_b[:, 1]
        self.vel_command_b[:, 1] = torch.where(~pump & ~flag & (tw1 < 0.5 * SKATE_PUMP_FLAG), torch.zeros_like(tw1), tw1)
        if g_s is not None and self.cfg.release_mode == "coast":
            # s5b: no key = COAST (a climb included: roll back); STOP only on the flat and slow (the Director rule)
            c = self.cfg
            nk = self.nokey & ~pump & ~self.is_standing_env
            v_s = st["sensor"]["v_s"] if "sensor" in st else st["v_mode"]
            latch = skate_nokey_stop_latch(self.stop_latch, g_s, torch.nan_to_num(v_s, nan=0.0), c.flat_stop_deg,
                                           c.stop_speed, c.stop_speed_hyst, c.flat_stop_hyst_deg)
            self.stop_latch = latch & nk
            hold = self.stop_latch & ~flag
            v = self.vel_command_b[:, 0]
            v = torch.where(hold, torch.zeros_like(v), torch.where(nk, torch.full_like(v, SKATE_COAST_CMD), v))
            self.vel_command_b[:, 0] = v
            for k in (1, 2):
                self.vel_command_b[:, k] = torch.where(hold, torch.zeros_like(v), self.vel_command_b[:, k])
        elif g_s is not None and self.cfg.release_up_deg is not None:
            latch = torch.where(self.up_latch, g_s > self.cfg.release_off_deg, g_s > self.cfg.release_up_deg)
            self.up_latch = latch & st["mounted"]
            hold = self.nokey & ~pump & self.up_latch & ~self.is_standing_env & ~flag
            free = self.nokey & ~pump & ~self.up_latch & ~self.is_standing_env
            v = self.vel_command_b[:, 0]
            v = torch.where(hold, torch.zeros_like(v), torch.where(free, torch.full_like(v, SKATE_COAST_CMD), v))
            self.vel_command_b[:, 0] = v
            for k in (1, 2):    # the hold is the s1 STOP command exactly (0, 0, 0): no lateral jitter, kappa 0
                self.vel_command_b[:, k] = torch.where(hold, torch.zeros_like(v), self.vel_command_b[:, k])


@_dataclass(kw_only=True)
class SkatePumpCommandCfg(SkateCurvatureCommandCfg):
    pump_prob: float = 0.30                      # share of speed resamples (mounted, unflagged) that press W
    pump_hold_range: tuple = (4.0, 12.0)         # s, how long W stays held
    pump_v_top_range: tuple = (0.6, 1.0)         # m/s (curriculum: up to 1.4)
    pump_v_top_play: float = 1.0                 # the play default v_top (1.4 crashes)
    pump_play_prob: float = 0.3                  # share of presses at exactly pump_v_top_play
    pump_a_range: tuple = (0.15, 0.4)            # m/s^2, ramp slope a_W
    pump_r_min: float = SKATE_PUMP_R_MIN
    release_hold_range: tuple = (2.0, 5.0)       # s of "no key" after a release before the next resample
    release_up_deg: float | None = 1.5           # no key on a climb above this grade = STOP (hold); None = always COAST
    release_off_deg: float = 1.0                 # hysteresis
    grade_cap: bool = True                       # twist[0] = min(ramp, skate_throttle_grade_cap(grade)) under W
    release_mode: str = "stop"                   # "stop": the s5 rule above; "coast" (s5b): COAST on a climb too,
                                                 # STOP only on the flat and slow (release_up_deg then unused)
    flat_stop_deg: float = 1.0                   # s5b: |sensed grade| at or below this = the flat
    flat_stop_hyst_deg: float | None = None      # s5b: None = a STOP latch is released on speed only
    stop_speed: float = 0.1                      # s5b: |sensed wheel speed| below this on the flat = STOP
    stop_speed_hyst: float = 0.02

    def build(self, env: ManagerBasedRlEnv) -> "SkatePumpCommand":
        return SkatePumpCommand(self, env)


def skate_pump_setup(env, env_ids, coast_spawns: bool = False):
    """Startup event (step 5): the spawn -> command hand-over of "start with W held" (reset_skate_state writes
    st["pump_now"] from the spawn's pump / pump_prob, the command's reset resample consumes it). ``coast_spawns``
    (s5b): also st["coast_now"] from the spawn's ``coast`` (start with no key)."""
    del env_ids
    st = _skate_state(env)
    st["pump_now"] = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if coast_spawns:
        st["coast_now"] = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def skate_nokey_stop_latch(latch: torch.Tensor, grade_deg: torch.Tensor, v_s: torch.Tensor, flat_deg: float = 1.0,
                           v_stop: float = 0.1, v_hyst: float = 0.02,
                           grade_hyst: float | None = None) -> torch.Tensor:
    """s5b Director no-key rule (the author's choice 2026-11-06): COAST, a climb included, except STOP once the board
    is on the flat (|sensed grade| <= flat_deg) and slow (|sensed wheel speed| < v_stop). The STOP latch then holds
    while |v| < v_stop + v_hyst, whatever the grade reads: on flat riding the sensed grade is above 1 deg on ~1 % of
    the steps (coordinator, play sim), and a grade release would drop a standing robot into COAST, the s1r5 fall.
    ``grade_hyst`` (optional) also releases it above flat_deg + grade_hyst. A noise spike only delays the engagement.
    Returns the new latch (True = STOP). The pump command (release_mode "coast"), skate_eval's ROLLBACK battery and
    play's Director share this rule."""
    va, ga = v_s.abs(), grade_deg.abs()
    hold = va < v_stop + v_hyst
    if grade_hyst is not None:
        hold = hold & (ga <= flat_deg + grade_hyst)
    return torch.where(latch, hold, (va < v_stop) & (ga <= flat_deg))


def skate_s5_base_step(env) -> int:
    """The env step the step-5 run started at (its schedules count from here): MICRODUCK_S5_START_IT x 24 if set
    (a crash-resume of s5), else the step counter at the first call, re-based whenever the counter jumps by more than
    SKATE_LEVEL_STEP_JUMP or goes back (the wrapper's first reset runs at counter 0, then runner.load jumps it to the
    restored iteration's). Shared by the step-5 curricula."""
    import os
    b = env.__dict__.setdefault("_skate_s5_base", {})
    step = int(env.common_step_counter)
    pinned = os.environ.get("MICRODUCK_S5_START_IT")
    if pinned:
        b["base"] = int(pinned) * 24
    else:
        last = b.get("last")
        if b.get("base") is None or last is None or step < last or step - last > SKATE_LEVEL_STEP_JUMP:
            if b.get("base") != step:
                print(f"[skate s5] schedules start at step {step} (it {step // 24})")
            b["base"] = step
    b["last"] = step
    return b["base"]


def skate_pump_curriculum(env, env_ids, stages: list[dict], command_name: str = "twist",
                          relative: bool = False) -> torch.Tensor:
    """Step-staged pump bucket: stages = [{"step", "v_top": (lo, hi), "a_w": (lo, hi)}] (live command term cfg).
    ``relative``: stage steps count from skate_s5_base_step (the restored iteration)."""
    del env_ids
    t = env.common_step_counter - (skate_s5_base_step(env) if relative else 0)
    cur = stages[0]
    for s in stages:
        if t > s["step"]:
            cur = s
    tcfg = env.command_manager.get_term(command_name).cfg
    tcfg.pump_v_top_range = tuple(cur["v_top"])
    tcfg.pump_a_range = tuple(cur["a_w"])
    return torch.tensor([float(cur["v_top"][1])])


def skate_s5_slope_stages(env, env_ids, stages: list[dict]) -> torch.Tensor:
    """s5b: stage live slope cfg keys by the env steps since the s5 start (skate_s5_base_step), stages = [{"step",
    <SKATE_SLOPE_DEFAULTS keys>}]: the roll-back cap grows with training time, so it is trained even when the uphill
    level stays low. Returns the first staged value."""
    del env_ids
    st = _skate_state(env)
    sl = st.get("slope")
    t = env.common_step_counter - skate_s5_base_step(env)
    cur = stages[0]
    for s in stages:
        if t > s["step"]:
            cur = s
    keys = [k for k in cur if k != "step"]
    assert all(k in SKATE_SLOPE_DEFAULTS for k in keys), keys
    if sl is not None:
        sl["cfg"].update({k: cur[k] for k in keys})
    v = cur[keys[0]] if keys else 0.0
    return torch.tensor([float(v if v is not None else 0.0)])


def skate_s5_info_obs(env) -> torch.Tensor:
    """(B, 2) step-5 bookkeeping obs group (no policy reads it; the student algorithm's BC mask does): the true grade
    along the canonical board heading in degrees (+ = uphill ahead) and the pump flag (1 = W held)."""
    st = _skate_state(env)
    g = torch.rad2deg(skate_uphill_grade(env, st))
    f = skate_pump_flag_active(env, st["cfg"]["command_name"]).float()
    return torch.nan_to_num(torch.stack([g, f], dim=1), nan=0.0)


def skate_mounted_obs(env) -> torch.Tensor:
    """(B, 1) the mounted latch (1 = mounted), a bookkeeping obs group for skate_student's bc_mount / bc_turn gating
    (s1r7 mount_info). Read-only."""
    return _skate_state(env)["mounted"].float().unsqueeze(1)


def skate_s5b_info_obs(env) -> torch.Tensor:
    """(B, 3) s5b bookkeeping obs group: skate_s5_info_obs plus the mounted latch (1 = mounted; skate_student's
    bc_mount floor reads it: BC on the step-on frames before the latch)."""
    st = _skate_state(env)
    return torch.cat([skate_s5_info_obs(env), st["mounted"].float().unsqueeze(1)], dim=1)


def skate_pump_context_metric(env) -> torch.Tensor:
    """Metrics helper: 1 in the pump context."""
    return skate_pump_context(env).float()


# ── heat split ──────────────────────────────────────────────────────────────────────────────────────────────────────


def _servo_heat_names(env, asset_name: str = "robot") -> list:
    c = env.__dict__.setdefault("_servo_heat_names", {})
    if asset_name not in c:
        c[asset_name] = [nm for a in _bam_heat_actuators(env, asset_name) for nm in a.target_names]
    return c[asset_name]


def servo_heat_joint_power(env, joint: str = "left_knee", asset_name: str = "robot") -> torch.Tensor:
    """Metrics helper: copper loss I^2 R0 of one servo this step (W)."""
    names = _servo_heat_names(env, asset_name)
    p = servo_heat_state(env, asset_name=asset_name)["p"]
    if joint not in names:
        return torch.zeros(env.num_envs, device=env.device)
    return p[:, names.index(joint)]


def servo_heat_excess_split(env, p_cont: float = 0.65, p_cont_pump: float = 1.24, tau_pump: float = 12.0,
                            pump_joints: tuple = SKATE_PUMP_PUSH_LEG, asset_name: str = "robot") -> torch.Tensor:
    """Step 5 cost (>= 0, NEGATIVE weight), the split servo_heat_excess. Every servo outside the pump context, and
    the board leg / neck always: relu(2 s low-passed copper loss - p_cont) (the s1 term, 0.65 W = the hot-corner
    continuous limit). The push leg INSIDE the pump context: relu(tau_pump (12 s) low-passed loss - p_cont_pump), 1.24
    W = the nominal continuous limit: continuous pumping loads the push leg by design, and its housing node (tau ~6
    min) integrates a stroke train, so only its sustained mean above what it can shed is charged. Sum over servos."""
    hs = servo_heat_state(env, asset_name=asset_name)
    c = env.__dict__.setdefault("_servo_heat_s5", {})
    if c.get("step") != env.common_step_counter or "ema" not in c:
        p = hs["p"]
        ema = c.get("ema")
        if ema is None or ema.shape != p.shape:
            ema = p.clone()
        a = env.step_dt / (tau_pump + env.step_dt)
        ema = ema + a * (p - ema)
        ema = torch.where((env.episode_length_buf <= 1).unsqueeze(1), p, ema)
        c.update(step=env.common_step_counter, ema=ema)
    names = _servo_heat_names(env, asset_name)
    key = ("mask", tuple(pump_joints), len(names))
    if key not in c:
        flags = [nm in pump_joints for nm in names] if names else [False] * hs["p"].shape[1]
        c[key] = torch.tensor(flags, device=env.device)
    mask = c[key]
    e2 = (hs["p_ema"] - p_cont).clamp_min(0.0)
    e12 = (c["ema"] - p_cont_pump).clamp_min(0.0)
    use12 = skate_pump_context(env).unsqueeze(1) & mask.unsqueeze(0)
    return torch.where(use12, e12, e2).sum(dim=1)


# ── Skate whole-robot mass DR (2026-11-21, audit_real item 7; default off) ───────────────────────────────────────────
# The training MJCF weighs 0.737 kg, the runtime's 0.744 kg, the README "~800 g"; the inherited trunk-only +-5 %
# pseudo_inertia DR never reaches 0.80 kg. Wired by MICRODUCK_SKATE_MASS_RANGE in microduck_skate_sequence_env_cfg.py.
from mjlab.managers.event_manager import RecomputeLevel as _RecomputeLevel  # noqa: E402


@requires_model_fields("body_mass", "body_inertia", recompute=_RecomputeLevel.set_const)
def skate_robot_mass_scale(env, env_ids, scale_range: tuple[float, float],
                           asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """ONE uniform scale s per env applied to every selected body's mass AND principal inertia (uniform density change:
    COM and principal axes unchanged, so it composes with the CoM-offset DR, which only writes body_ipos).
    Restore-then-apply: always written from the compile-time defaults (sim.get_default_field), never accumulates."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    elif isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)[env_ids]
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)
    asset = env.scene[asset_cfg.name]
    idx = asset.indexing.body_ids[asset_cfg.body_ids]
    def_mass = env.sim.get_default_field("body_mass")[idx]          # (B,)
    def_inertia = env.sim.get_default_field("body_inertia")[idx]    # (B, 3)
    lo, hi = scale_range
    s = torch.rand(len(env_ids), 1, device=env.device) * (hi - lo) + lo
    rows = env_ids[:, None]
    env.sim.model.body_mass[rows, idx] = def_mass.unsqueeze(0) * s
    env.sim.model.body_inertia[rows, idx] = def_inertia.unsqueeze(0) * s.unsqueeze(-1)
