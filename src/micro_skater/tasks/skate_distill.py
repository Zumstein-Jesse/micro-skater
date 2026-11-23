# micro-skater: part of the author's skateboard add-on for microduck_rl. Apache-2.0 (see LICENSE).
"""PPO + walk-up behavior cloning from the frozen walker (skate v6d).

Why (skate_v6c it250, 2026-11-03): the v6b walk-up (ring spawns up to 0.5 m from the board, any yaw) never got off
the ground. The skater was only ever trained to stand beside the board and take ONE step onto it; from the ring
spawns it crouched into its step-on motion within 0.2 s and fell (125 of 128 ring spawns fell or were lost within
~0.4 s, any side, any distance). The approach potentials never pay because no partial walk ever happens: RL would
have to discover a gait from scratch. A gait exists: the 61D velocity walker the play runtime already drives to the
board (skate_play.Approach). So we distill it, as velstand distills the stand expert (distill.py).

Mechanism: after every PPO update, a BC pass over the observations the STUDENT just collected. Walk-up frames
(before the mount latch, no step-off flag, and before the env's trunk first entered the trained step-on box + margin
this episode) regress the student's deterministic action onto the walker's at weight ``coef``; every other frame is
anchored to the frozen warm-start skater at ``anchor_coef`` (distill.py's run-3 lesson: without it one pass took the
warm start's step-on from 124 to 0 of 128). The walker reads
the student's first 61 obs dims (the family layout; skate_play feeds it the same slice) with its twist slot replaced
by ``walkup_command``: a stateless, vectorized port of skate_play.Approach.command (P controller toward the staging
spot, the corner waypoint round the deck end when the straight line crosses the keep-out box), computed from the
approach obs block (staging dx, dy in the robot's yaw frame and sin / cos of robot - canonical deck yaw). Inside the
near zone, and once mounted, PPO alone: the step-on, riding and step-off are not touched by the teacher.

The walker has its own architecture (61D input) and obs normalizer, so it is loaded from its state dict here
(``WalkTeacher``), not deep-copied from the student like distill.py's experts.

Obs-contract dependencies (actor obs, 79D): family block [0:61] with twist at [48:51] (step-off flag = twist[1]),
approach block last [75:79] (it reads [0, 0, 0, 1] exactly once mounted).
"""

from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass, field

import torch
from rsl_rl.algorithms import PPO
from tensordict import TensorDict

from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg

from micro_skater.paths import checkpoint_path

# The walking policy (microduck_rl's stock velocity task, 5000 iterations) that drives the walk-up to the board.
# Default: checkpoints/walker/model.pt of this repo; override with MICRO_SKATER_WALKER.
WALKER_CHECKPOINT = os.environ.get("MICRO_SKATER_WALKER") or str(checkpoint_path("walker/model.pt"))


# v6d only (not shipped): the v6c checkpoint the walk-up BC anchored to. Set MICRO_SKATER_ANCHOR to use it.
ANCHOR_CHECKPOINT = os.environ.get("MICRO_SKATER_ANCHOR", "")


def default_walkup_bc_cfg() -> dict:
    # v6d-1 (lr 3e-4, coef 1, no anchor, 5 cm zone) wrecked the warm start in ONE pass: core / wide step-on 124 ->
    # 0 / 128 (verifier 2026-11-03, distill.py's run-3 lesson again). Offline sweep (one
    # iteration-0 batch, 5 passes without PPO repair): lr 1e-4, coef 0.3, anchor 1 on every other frame, the trained
    # step-on box + 4 cm latched per episode kept step-on 122 / 128 and riding 124 / 125 while the walk-up error fell
    # 0.76 -> 0.46.
    return {
        "walker_checkpoint": WALKER_CHECKPOINT,
        "anchor_checkpoint": ANCHOR_CHECKPOINT,
        "coef": 0.3,
        # v6d-2 it160: walk-up frames are ~1 % of a training batch once riding episodes run long (9 % in the sweep's
        # iteration-0 batch), and the walk-up error stalled at ~0.45. The walker weight is scaled up to the sweep's
        # loss mass, coef * max(1, ref_frac / frac) capped at max_boost. (Per-group means were tested instead: at
        # coef 0.3 they took step-on 122 -> 76 / 128, walk-up states next to the box look like the step-on.)
        "ref_frac": 0.09,
        "max_boost": 10.0,
        "anchor_coef": 1.0,
        "learning_rate": 1e-4,
        "epochs": 2,
        "mini_batches": 4,
        "family_dim": 61,
        "twist_slice": (48, 51),
        "approach_slice": (75, 79),
        "dismount_flag": 0.04,         # mdp.SKATE_DISMOUNT_FLAG in twist[1]
        # the trained step-on box (STEP_ON_XY_WIDE, STEP_ON_YAW_WIDE) + margin, canonical deck frame: once an env's
        # trunk enters it, the rest of its episode is PPO's (+ the anchor), never the walker's
        "box_x": 0.06,
        "box_y": (-0.19, -0.13),
        "box_yaw": 0.5,
        "box_margin": 0.04,
        # planner (skate_play.Approach constants; the canonical deck frame keeps the robot on the board's right)
        "staging": (0.0, -0.145),
        "keepout_in": (0.16, 0.10),
        "corner": (0.19, -0.16),
        "corner_far": 0.24,
        "face_travel_wp": 0.2,
        "gain": (2.0, 2.0, 2.0),
        "max": (0.3, 0.25, 0.8),
        # v6f hand-off gate (walk-up diagnosis 2026-11-03, skate_v6e it500): first entry into the wide box handed over
        # at its outer edge (45 % at y < -0.17), turned (|dyaw| > 0.2) or with the board foot in swing (39 %), and
        # 46 % of walk-ups were lost; handing over only inside y [-0.17, -0.13], |dyaw| <= 0.25 with the board (left)
        # foot on the floor took riding-after-walk-up 54 % -> 76 % untrained. Off (False) = the v6e first-entry rule.
        "handoff_gate": False,
        "handoff_y": (-0.17, -0.13),
        "handoff_yaw": 0.25,
        "handoff_sensor": "board_foot_ground",   # left sole on the floor; None = no foot condition
        # v6f reviewer: ~3-6 % of walkers park 0.5-4 cm outside the y band (the planner's small vy is inside the
        # walker's lateral deadband) and time out; after this long in the wide box, the foot condition alone hands off
        "handoff_stall_s": 1.0,
    }


@dataclass
class PpoWithWalkupBcCfg(PpoWithSymmetryCfg):
    """PpoWithSymmetryCfg + walk-up behavior cloning from the walker."""

    walkup_bc_cfg: dict | None = field(default_factory=default_walkup_bc_cfg)
    class_name: str = "micro_skater.tasks.skate_distill.PpoWithWalkupBc"


class WalkTeacher(torch.nn.Module):
    """rsl_rl actor (EmpiricalNormalization + ELU MLP), deterministic mean, frozen; loaded from its checkpoint."""

    def __init__(self, path: str, device):
        super().__init__()
        sd = torch.load(path, map_location=device, weights_only=False)["actor_state_dict"]
        self.register_buffer("mean", sd["obs_normalizer._mean"].clone())
        self.register_buffer("std", sd["obs_normalizer._std"].clone())
        self.eps = 1e-2
        ws, bs = [], []
        i = 0
        while f"mlp.{i}.weight" in sd:
            ws.append(sd[f"mlp.{i}.weight"].clone())
            bs.append(sd[f"mlp.{i}.bias"].clone())
            i += 2
        self.ws = torch.nn.ParameterList([torch.nn.Parameter(w, requires_grad=False) for w in ws])
        self.bs = torch.nn.ParameterList([torch.nn.Parameter(b, requires_grad=False) for b in bs])
        self.in_dim = int(self.mean.shape[-1])

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = (x - self.mean) / (self.std + self.eps)
        n = len(self.ws)
        for j in range(n):
            h = torch.nn.functional.linear(h, self.ws[j], self.bs[j])
            if j < n - 1:
                h = torch.nn.functional.elu(h)
        return h


def _segment_hits_box(px, py, qx, qy, hx: float, hy: float) -> torch.Tensor:
    """Vectorized skate_play.segment_hits_box: does segment p -> q cross the centred box |x| < hx, |y| < hy
    (Liang-Barsky clip)."""
    dx, dy = qx - px, qy - py
    t0 = torch.zeros_like(px)
    t1 = torch.ones_like(px)
    for p, q in ((-dx, px + hx), (dx, hx - px), (-dy, py + hy), (dy, hy - py)):
        par = p.abs() < 1e-12
        out = par & (q < 0)
        r = q / torch.where(par, torch.ones_like(p), p)
        t0 = torch.where(~par & (p < 0), torch.maximum(t0, r), t0)
        t1 = torch.where(~par & (p > 0), torch.minimum(t1, r), t1)
        t1 = torch.where(out, torch.full_like(t1, -1.0), t1)
    return t0 <= t1


def walkup_state(appr: torch.Tensor, staging: tuple[float, float]):
    """From the approach obs (dx, dy of the staging spot in the robot's yaw frame, sin / cos of robot - deck yaw):
    robot trunk xy in the canonical deck frame and the yaw offset."""
    dx, dy, s, c = appr.unbind(dim=1)
    dyaw = torch.atan2(s, c)
    cy, sy = torch.cos(dyaw), torch.sin(dyaw)
    # staging - robot, rotated from the robot's yaw frame into the deck frame
    ex, ey = cy * dx - sy * dy, sy * dx + cy * dy
    return staging[0] - ex, staging[1] - ey, dyaw


def walkup_command(appr: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Walker twist (vx, vy, wz) toward the staging pose, as skate_play.Approach.command, stateless."""
    sx, sy = cfg["staging"]
    x, y, dyaw = walkup_state(appr, (sx, sy))
    hx, hy = cfg["keepout_in"]
    cx, cy = cfg["corner"]
    tx = torch.full_like(x, sx)
    ty = torch.full_like(y, sy)
    way = _segment_hits_box(x, y, tx, ty, hx, hy)
    end = torch.where(x >= 0.0, torch.ones_like(x), -torch.ones_like(x))
    # from the board's left (y > 0) whose line to the near corner crosses the box: round the end at corner_far first
    around = way & (y > 0.0) & _segment_hits_box(x, y, end * cx, torch.full_like(y, cy), hx, hy)
    wpx = torch.where(around, end * cfg["corner_far"], end * cx)
    wpy = torch.where(around, torch.full_like(y, -cy), torch.full_like(y, cy))
    tx = torch.where(way, wpx, tx)
    ty = torch.where(way, wpy, ty)
    # target in the robot's yaw frame
    c, s = torch.cos(dyaw), torch.sin(dyaw)
    gx, gy = tx - x, ty - y
    ex, ey = c * gx + s * gy, -s * gx + c * gy
    far = way & (torch.hypot(ex, ey) > cfg["face_travel_wp"])
    eyaw = torch.where(far, torch.atan2(ey, ex), -dyaw)
    (kx, ky, kw), (mx, my, mw) = cfg["gain"], cfg["max"]
    vx, vy = kx * ex, ky * ey
    k = torch.minimum(mx / vx.abs().clamp_min(1e-9), my / vy.abs().clamp_min(1e-9))
    k = torch.where(way, k, k.clamp(max=1.0))
    return torch.stack([k * vx, k * vy, (kw * eyaw).clamp(-mw, mw)], dim=1)


def in_step_on_box(flat: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Trunk inside the trained step-on box + margin (canonical deck frame, from the approach obs)."""
    a0, a1 = cfg["approach_slice"]
    x, y, dyaw = walkup_state(flat[:, a0:a1], cfg["staging"])
    mg = cfg["box_margin"]
    ylo, yhi = cfg["box_y"]
    return ((x.abs() <= cfg["box_x"] + mg) & (y >= ylo - mg) & (y <= yhi + mg)
            & (dyaw.abs() <= cfg["box_yaw"] + 4 * mg))


def in_handoff_box(flat: torch.Tensor, cfg: dict) -> torch.Tensor:
    """v6f hand-off pose: |x| <= box_x, y in handoff_y, |dyaw| <= handoff_yaw (canonical deck frame, approach obs).
    The foot condition is the env's (contact sensor), see SkateWalkerApproachAction."""
    a0, a1 = cfg["approach_slice"]
    x, y, dyaw = walkup_state(flat[:, a0:a1], cfg["staging"])
    ylo, yhi = cfg["handoff_y"]
    return (x.abs() <= cfg["box_x"]) & (y >= ylo) & (y <= yhi) & (dyaw.abs() <= cfg["handoff_yaw"])


def unmounted_unflagged(flat: torch.Tensor, cfg: dict) -> torch.Tensor:
    a0, a1 = cfg["approach_slice"]
    mounted = (flat[:, a0:a1] == torch.tensor([0.0, 0.0, 0.0, 1.0], device=flat.device)).all(dim=1)
    t0 = cfg["twist_slice"][0]
    flagged = (flat[:, t0 + 1] - cfg["dismount_flag"]).abs() < 1e-3
    return ~mounted & ~flagged


def walkup_mask(flat: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Walk-up candidates: before the mount latch (approach obs not exactly [0, 0, 0, 1]), no step-off flag, trunk
    outside the step-on box. The per-episode box latch (PpoWithWalkupBc._latched_box) is applied on top."""
    return unmounted_unflagged(flat, cfg) & ~in_step_on_box(flat, cfg)


def teacher_input(flat: torch.Tensor, cfg: dict) -> torch.Tensor:
    a0, a1 = cfg["approach_slice"]
    t0, t1 = cfg["twist_slice"]
    x = flat[:, : cfg["family_dim"]].clone()
    x[:, t0:t1] = walkup_command(flat[:, a0:a1], cfg)
    return x


# ── v6e: the walker EXECUTES the walk-up in the env ─────────────────────────────────────────────────────────────────
# v6d2 it350: the BC'd skater imitated the walker to a 0.1 rad^2 action error but still could not walk (0 / 128 ring
# spawns mounted, falls after ~1.3 s), and a gait-sized BC signal on the shared weights costs the step-on. So the walk
# is not distilled: in training (and in play, skate_play.Director) the walker drives every env that is unmounted,
# flag down and has not yet entered the step-on box this episode; from the first frame in the box the skater acts.
# The skater trains its step-on on the walker's real arrival states from every ring spawn; PpoWithWalkupBc zeroes
# the advantages of the walker-driven frames (the stored actions there were never executed).

import weakref  # noqa: E402

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg  # noqa: E402

# The live env term (training is one env per process): PpoWithWalkupBc reads which frames the walker executed from
# it, so the advantage mask is exactly the env's hand-off rule (v6f's foot condition is not in the obs).
_WALKER_TERM = None


def _live_walker_term():
    return _WALKER_TERM() if _WALKER_TERM is not None else None


@dataclass(kw_only=True)
class SkateWalkerApproachActionCfg(JointPositionActionCfg):
    walker_checkpoint: str = WALKER_CHECKPOINT
    planner: dict = field(default_factory=default_walkup_bc_cfg)

    def build(self, env):
        return SkateWalkerApproachAction(self, env)


class SkateWalkerApproachAction(JointPositionAction):
    """JointPositionAction whose rows in walk-up mode carry the walker's action (also written into the action
    manager's current-action slot, so last_action / action-rate terms see what was executed)."""

    def __init__(self, cfg: SkateWalkerApproachActionCfg, env):
        super().__init__(cfg, env)
        self._skate_env = env
        self.walker = WalkTeacher(cfg.walker_checkpoint, env.device).to(env.device)
        self._entered = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.walker_active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        # the walker drove this step's action (NOT cleared by reset: PPO reads it after env.step, whose resets would
        # otherwise wipe the flags of the envs that just ended)
        self.executed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._in_box_t = torch.zeros(env.num_envs, device=env.device)   # s in the wide box (v6f stall fallback)
        # False: the policy's actions pass through untouched (export reference rollouts, probes that drive an
        # unmounted robot with their own actions)
        self.enabled = True
        global _WALKER_TERM
        _WALKER_TERM = weakref.ref(self)

    def handoff_ready(self, x: torch.Tensor) -> torch.Tensor:
        pl = self.cfg.planner
        if not pl.get("handoff_gate"):
            return in_step_on_box(x, pl)
        ok = in_handoff_box(x, pl)
        wide = in_step_on_box(x, pl)
        # in place (a rebind under inference_mode breaks reset() outside it)
        self._in_box_t.copy_(torch.where(wide, self._in_box_t + self._skate_env.step_dt, torch.zeros_like(self._in_box_t)))
        ok = ok | (wide & (self._in_box_t >= pl.get("handoff_stall_s", math.inf)))
        name = pl.get("handoff_sensor")
        if name:
            found = self._skate_env.scene[name].data.found
            ok = ok & (found.reshape(found.shape[0], -1) > 0).any(dim=1)
        return ok

    def process_actions(self, actions: torch.Tensor):
        env = self._skate_env
        obs = getattr(env, "obs_buf", None)
        # skate student: the planner reads the privileged "teacher" group (the 79D teacher layout), the student's own
        # actor obs carry no approach block in that layout
        x = obs.get(self.cfg.planner.get("obs_group", "actor")) if isinstance(obs, dict) else None
        active = None
        if self.enabled and isinstance(x, torch.Tensor) and x.shape[-1] >= self.cfg.planner["approach_slice"][1]:
            pl = self.cfg.planner
            self._entered |= self.handoff_ready(x)
            if pl.get("handoff_gate"):   # v6f: the walker keeps the robot (inside the wide box too) until the gate
                active = unmounted_unflagged(x, pl) & ~self._entered
            else:
                active = walkup_mask(x, pl) & ~self._entered
            self.walker_active.copy_(active)     # in place (a rebind under inference_mode breaks reset() outside it)
            self.executed.copy_(active)
        else:
            self.executed.fill_(False)
        if active is not None and active.any():
                with torch.no_grad():
                    wa = self.walker(teacher_input(x[active], pl))
                actions = actions.clone()
                actions[active] = wa
                env.action_manager._action[active] = wa
        super().process_actions(actions)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self._entered[env_ids] = False
        self.walker_active[env_ids] = False
        self._in_box_t[env_ids] = 0.0


class PpoWithWalkupBc(PPO):
    def __init__(self, actor, critic, storage, *args, walkup_bc_cfg: dict | None = None, **kwargs) -> None:
        super().__init__(actor, critic, storage, *args, **kwargs)
        self.walkup_bc_cfg = walkup_bc_cfg
        self.teacher = None
        self.anchor = None
        self._box_latch = None          # (N,) bool: entered the step-on box this episode (carried across rollouts)
        self._walker_frames = None      # (T*N,) this iteration's walker-driven frames (v6e "executed")
        self._executed_steps: list[torch.Tensor] = []   # per rollout step, the env term's executed flags (v6f)
        self._walker_mask_source = None
        if walkup_bc_cfg and walkup_bc_cfg.get("executed"):
            print("[walkup] v6e: the walker executes the walk-up in the env; those frames get zero advantage")
        if walkup_bc_cfg and walkup_bc_cfg.get("coef", 0.0) > 0:
            self.teacher = WalkTeacher(walkup_bc_cfg["walker_checkpoint"], self.device).to(self.device)
            assert self.teacher.in_dim == walkup_bc_cfg["family_dim"], (self.teacher.in_dim, walkup_bc_cfg)
            if walkup_bc_cfg.get("anchor_coef", 0.0) > 0:
                ack = torch.load(walkup_bc_cfg["anchor_checkpoint"], map_location=self.device, weights_only=False)
                self.anchor = copy.deepcopy(self.actor)
                self.anchor.load_state_dict(ack["actor_state_dict"], strict=True)
                self.anchor.eval()
                for p in self.anchor.parameters():
                    p.requires_grad_(False)
            self.bc_optimizer = torch.optim.Adam(self.actor.parameters(), lr=walkup_bc_cfg["learning_rate"])
            print(f"[walkup-bc] walker teacher ON ({walkup_bc_cfg['walker_checkpoint']}), coef {walkup_bc_cfg['coef']}, "
                  f"anchor {walkup_bc_cfg.get('anchor_checkpoint') if self.anchor is not None else 'OFF'} coef "
                  f"{walkup_bc_cfg.get('anchor_coef', 0.0)}, lr {walkup_bc_cfg['learning_rate']}, epochs "
                  f"{walkup_bc_cfg['epochs']} x {walkup_bc_cfg['mini_batches']}")

    def _latched_box(self, flat: torch.Tensor, T: int, N: int) -> torch.Tensor:
        """(T*N,) frames at or after the env's first entry into the step-on box this episode. dones[t] ends the
        episode after step t; the latch carries into the next rollout."""
        box = in_step_on_box(flat, self.walkup_bc_cfg).reshape(T, N)
        dones = self.storage.dones.reshape(T, N).bool()
        if self._box_latch is None or self._box_latch.shape[0] != N:
            self._box_latch = torch.zeros(N, dtype=torch.bool, device=flat.device)
        ent = self._box_latch
        out = torch.zeros(T, N, dtype=torch.bool, device=flat.device)
        for t in range(T):
            ent = ent | box[t]
            out[t] = ent
            ent = ent & ~dones[t]
        self._box_latch = ent
        return out.reshape(-1)

    def _walkup_bc_update(self) -> dict[str, float]:
        cfg = self.walkup_bc_cfg
        T, N = self.storage.observations.batch_size[:2]
        obs_td: TensorDict = self.storage.observations.flatten(0, 1)
        groups = list(self.actor.obs_groups)
        flat = torch.cat([obs_td[g] for g in groups], dim=-1)
        use = walkup_mask(flat, cfg) & ~self._latched_box(flat, T, N)
        frac = use.float().mean().item()
        boost = min(cfg.get("max_boost", 1.0), max(1.0, cfg.get("ref_frac", 0.0) / max(frac, 1e-6)))
        stats = {"walkup_bc_frac": frac, "walkup_bc_boost": boost}
        with torch.no_grad():
            if self.anchor is not None:
                target = self.anchor(obs_td)
                w = torch.full((flat.shape[0],), cfg["anchor_coef"], device=self.device)
            else:
                idx = use.nonzero().flatten()
                obs_td, flat, use = obs_td[idx], flat[idx], use[idx]
                target = torch.zeros(flat.shape[0], self.storage.actions.shape[-1], device=self.device)
                w = torch.zeros(flat.shape[0], device=self.device)
            if use.any():
                target[use] = self.teacher(teacher_input(flat[use], cfg))
                w[use] = cfg["coef"] * boost
        n = flat.shape[0]
        if n == 0 or not (w > 0).any():
            stats["walkup_bc"] = 0.0
            return stats
        mb = max(1, n // cfg["mini_batches"])
        total, steps = 0.0, 0
        walk_err = 0.0
        for _ in range(cfg["epochs"]):
            perm = torch.randperm(n, device=self.device)
            for s in range(0, n, mb):
                sel = perm[s:s + mb]
                err = (self.actor(obs_td[sel]) - target[sel]).pow(2).mean(dim=1)
                loss = (w[sel] * err).mean()
                self.bc_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.bc_optimizer.step()
                total += loss.item()
                us = use[sel]
                if us.any():
                    walk_err += err[us].mean().item()
                steps += 1
        stats["walkup_bc"] = total / max(steps, 1)
        stats["walkup_bc_walk_err"] = walk_err / max(steps, 1)
        return stats

    def process_env_step(self, obs, rewards, dones, extras) -> None:
        cfg = self.walkup_bc_cfg
        if cfg and cfg.get("executed"):
            term = _live_walker_term()
            if term is not None:
                self._executed_steps.append(term.executed.clone())
        super().process_env_step(obs, rewards, dones, extras)

    def compute_returns(self, obs) -> None:
        super().compute_returns(obs)
        cfg = self.walkup_bc_cfg
        if not (cfg and cfg.get("executed")):
            return
        T, N = self.storage.observations.batch_size[:2]
        steps, self._executed_steps = self._executed_steps, []
        if len(steps) == T and steps[0].shape[0] == N:
            # what the env actually executed (exact under any hand-off rule)
            walk = torch.stack(steps).to(self.storage.advantages.device).reshape(-1)
            src = "env"
            if not self.normalize_advantage_per_mini_batch and not walk.all():
                # v6f reviewer: normalize over the policy's own frames only (walker frames are 15-30 % of a batch
                # under the gate; their never-executed advantages skewed the mean / std the policy frames get)
                st = self.storage
                raw = st.returns - st.values
                pol = ~walk.reshape(T, N)
                m, s = raw[pol].mean(), raw[pol].std()
                st.advantages = (raw - m) / (s + 1e-8)
        else:   # no live term (or a partial rollout): the v6e obs-side reconstruction (no foot condition)
            assert not cfg.get("handoff_gate"), f"v6f hand-off gate needs the live env term ({len(steps)} / {T} steps)"
            obs_td = self.storage.observations.flatten(0, 1)
            flat = torch.cat([obs_td[g] for g in self.actor.obs_groups], dim=-1)
            walk = walkup_mask(flat, cfg) & ~self._latched_box(flat, T, N)
            src = "obs"
        if src != self._walker_mask_source:
            print(f"[walkup] walker-frame advantage mask from the {src}")
            self._walker_mask_source = src
        self._walker_frames = walk
        # after the normalization in super(): zero surrogate gradient on actions the env never executed
        self.storage.advantages[walk.reshape(T, N)] = 0.0

    def update(self) -> dict[str, float]:
        loss_dict = super().update()
        if self._walker_frames is not None:
            loss_dict["walker_frame_frac"] = self._walker_frames.float().mean().item()
        if self.teacher is not None:
            loss_dict.update(self._walkup_bc_update())
        return loss_dict


def _self_test() -> None:
    """Planner sanity: behind the staging spot -> forward; beside it facing along the deck -> ~0; off the nose ->
    heads for the near corner (vy < 0); lined up but turned -> turns back."""
    cfg = default_walkup_bc_cfg()
    sx, sy = cfg["staging"]

    def appr(x, y, yaw):
        ex, ey = sx - x, sy - y
        c, s = math.cos(yaw), math.sin(yaw)
        return torch.tensor([[c * ex + s * ey, -s * ex + c * ey, math.sin(yaw), math.cos(yaw)]])

    back = walkup_command(appr(-0.3, -0.145, 0.0), cfg)[0]
    assert back[0] > 0.2 and abs(back[1]) < 1e-6 and abs(back[2]) < 1e-6, back
    at = walkup_command(appr(0.0, -0.145, 0.0), cfg)[0]
    assert at.abs().max() < 1e-6, at
    x, y, yaw = walkup_state(appr(0.3, 0.02, 1.0), (sx, sy))
    assert abs(x.item() - 0.3) < 1e-6 and abs(y.item() - 0.02) < 1e-6 and abs(yaw.item() - 1.0) < 1e-6
    nose = walkup_command(appr(0.25, 0.0, math.pi), cfg)[0]       # off the nose, facing back along the deck
    # corner (0.19, -0.16) in the robot frame (yaw pi): ex = -(0.19 - 0.25) > 0, ey = -(-0.16) > 0 -> walks toward it
    assert nose[0] > 0 and nose[1] > 0, nose
    turned = walkup_command(appr(0.0, -0.145, 0.5), cfg)[0]
    assert turned[2] < -0.5, turned


if __name__ == "__main__":
    _self_test()
    print("ok")
