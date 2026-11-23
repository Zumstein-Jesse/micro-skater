# micro-skater: part of the author's skateboard add-on for microduck_rl. Apache-2.0 (see LICENSE).
"""Evaluate a Mjlab-Skate-Sequence-MicroDuck checkpoint: per-spawn batteries (the scheduler runs from each spawn's stage)
with a per-stage funnel, plus continuous videos of the full chain (from START: one foot on the floor, at rest).

    python sequence_eval.py CKPT [--task Mjlab-Skate-Sequence-MicroDuck] [--n 64]
                            [--spawns start,push,glide,manual,exit,ramp_bottom,ramp_wall]
                            [--seconds 18] [--render-dir D] [--render typical,best] [--label L] [--json OUT]

--task: a specialist (e.g. Mjlab-Skate-SeqPush-MicroDuck, scheduler capped at SETTLE) is evaluated on its own spawn mix
(default --spawns = its spawn types with a non-zero probability); "success" = the episode ended on the specialist's
success termination (a time-out term), "stepped_on" = PUSH episodes that reached SETTLE.

Funnel per episode (env): furthest stage reached (0 push .. 4 ramp), and per stage: PUSH time to exit + top speed +
strokes (right-foot floor touch-downs), MANUAL in-band fraction of the MANUAL stage + longest hold, EXIT time to four
wheels down + tail strikes, RAMP: passes (latched apex counter, r2: alternating walls or same-wall returns, rise
> 1 cm) and their rise, time alive in RAMP.
"""
from __future__ import annotations

import argparse
import atexit
import copy
import json
import math
import os
from dataclasses import asdict

os.environ.setdefault("WANDB_MODE", "disabled")
import numpy as np
import torch

DEVICE = os.environ.get("MICRO_SKATER_DEVICE", "cuda:0")
TASK = "Mjlab-Skate-Sequence-MicroDuck"
STAGE_NAMES = ("PUSH", "SETTLE", "MANUAL", "EXIT/approach", "HALF-PIPE")


def build(checkpoint: str, n: int, task: str = TASK):
    import micro_skater.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = n
    # a params curriculum on the spawn event (manual specialist: spawn mix / speeds) would overwrite the forced
    # spawn_probs at every reset (curricula run before the reset events): drop it (its stage-0 params stay in the cfg)
    for k in [k for k, t in cfg.curriculum.items() if (t.params or {}).get("event_name") == "reset_sequence"]:
        cfg.curriculum.pop(k)
    env = ManagerBasedRlEnv(cfg=cfg, device=DEVICE)
    agent_cfg = load_rl_cfg(task)
    vec = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = load_runner_cls(task)(vec, asdict(agent_cfg), device=DEVICE)
    runner.load(checkpoint, load_cfg={"actor": True}, strict=True, map_location=DEVICE)
    return vec, runner.get_inference_policy(device=DEVICE)


def run_spawn(vec, policy, spawn: str, seconds: float):
    from micro_skater.tasks import skate_sequence as sq
    from micro_skater.tasks import skate_trick as tk

    env = vec.unwrapped
    term = env.event_manager.get_term_cfg("reset_sequence")
    p0 = dict(term.params["spawn_probs"])
    term.params["spawn_probs"] = {spawn: 1.0}
    tk._tc(env).pop("seq_kf", None)
    obs, _ = vec.reset()
    term.params["spawn_probs"] = p0
    tk._tc(env).pop("seq_kf", None)
    n, dt = env.num_envs, env.step_dt
    T = int(round(seconds / dt))
    alive = torch.ones(n, dtype=torch.bool, device=env.device)
    end = torch.full((n,), T, dtype=torch.long, device=env.device)
    reason = ["time_out"] * n
    tm = env.termination_manager
    tos = {nm for nm in tm.active_terms if tm.get_term_cfg(nm).time_out}
    L = {k: [] for k in ("qpos", "stage", "v", "pitch", "manual", "band", "rise", "x", "lf", "rf", "wheels", "tail",
                          "cmd", "both_on", "fwd")}   # cmd / both_on / fwd: render labels only
    pipe = sq._st(env)["pipe"]
    st = sq._st(env)
    apex_n = torch.zeros(n, device=env.device)
    apex_rise = torch.zeros(n, device=env.device)
    # r2 (2026-11-20): latched pass counter = skate_seq_ramp.apex_update (the old vz_prev > 0.02 -> vz <= 0 test missed
    # 85-90 % of real apexes: near an apex vz changes only 0.005-0.02 per 20 ms step). Max rise of the wall visit, a
    # pass once the rise is APEX_DROP below it with the max > 1 cm; re-armed on the inner flat |x| < APEX_REARM_X or
    # outside the pipe; alternating walls and same-wall returns both count. Per step: L["apex"], L["apex_z"].
    APEX_DROP, APEX_REARM_X, APEX_MIN = 0.003, 0.30, 0.01
    ap_zmax = torch.zeros(n, device=env.device)
    ap_armed = torch.zeros(n, dtype=torch.bool, device=env.device)
    L["apex"], L["apex_z"] = [], []
    with torch.no_grad():
        for k in range(T):
            st = sq.seq_update(env)
            p = sq.pipe_xyz(env)
            rise = (p[:, 2] - sq.DECK_Z).clamp(min=0)
            L["qpos"].append(env.sim.data.qpos.clone().cpu().numpy())
            L["stage"].append(st["stage"].clone())
            L["v"].append(sq.board_v_fwd(env).clone())
            L["pitch"].append(torch.rad2deg(tk.front_up_pitch(env)).clone())
            fd, bd = tk._front_back_wheels(env)
            L["fwd"].append(fd.clone())                  # render8: front wheels down (MANUAL label from the lift-off)
            th = tk.front_up_pitch(env)
            L["manual"].append((~fd) & bd & ~tk._back_end_floor(env) & (th > math.radians(3)) & (th < math.radians(25))
                               & tk.soles_on_deck(env))
            L["band"].append(sq._in_manual_band(env, (8.0, 15.0)))      # the demo band (m_manual_band / band_t)
            L["rise"].append(rise.clone())
            L["x"].append(p.clone())
            lf, rf = sq._feet_floor(env)
            L["lf"].append(lf.clone())
            L["rf"].append(rf.clone())
            L["wheels"].append(sq._wheels_all_down(env))
            L["tail"].append(tk._contact_any(env, "tail_floor"))
            L["cmd"].append(st["cmd"].clone() if "cmd" in st else torch.zeros_like(st["stage"], dtype=torch.bool))
            L["both_on"].append(st["both_on"] > 0)
            vz = env.scene["board"].data.root_link_lin_vel_w[:, 2]
            ax = p[:, 0].abs()
            inp = sq.in_pipe(env, 0.0)
            on_wall = (ax > pipe["L"] / 2) & inp
            if k == 0:                                   # a wall spawn rolling down is not armed (its own height)
                ap_armed = ~on_wall | (vz > 0)
            ap_armed = ap_armed | (ax < APEX_REARM_X) | ~inp
            ap_zmax = torch.where(on_wall & ap_armed, torch.maximum(ap_zmax, rise), torch.zeros_like(rise))
            fire = ap_armed & on_wall & (ap_zmax - rise > APEX_DROP) & (ap_zmax > APEX_MIN) & (st["stage"] == 4)
            new = fire & alive
            apex_n += new.float()
            apex_rise = torch.maximum(apex_rise, torch.where(new, ap_zmax, torch.zeros_like(rise)))
            L["apex"].append(new.clone())
            L["apex_z"].append(torch.where(new, ap_zmax, torch.zeros_like(rise)))
            ap_armed = ap_armed & ~fire
            ap_zmax = torch.where(fire, torch.zeros_like(ap_zmax), ap_zmax)
            obs, _, dones, _ = vec.step(policy(obs))
            newly = dones.bool() & alive
            if bool(newly.any()):
                tm = env.termination_manager
                for i in newly.nonzero().squeeze(1).tolist():
                    rs = [nm for nm in tm.active_terms if bool(tm.get_term(nm)[i])]
                    reason[i] = ",".join(rs) or "time_out"
                    end[i] = k + 1
                alive &= ~newly
    for key in L:
        if key == "qpos":
            L[key] = np.stack(L[key])
        else:
            L[key] = torch.stack(L[key]).cpu().numpy()
    L["timeout_terms"] = tos
    return L, end.cpu().numpy(), reason, apex_n.cpu().numpy(), apex_rise.cpu().numpy()


def _ended_ok(r: str, tos: set) -> bool:
    """The episode ended on time-out terms only (time_out, a specialist's success, past_pipe_end)."""
    return all(nm in tos for nm in r.split(","))


def funnel(L, end, reason, apex_n, apex_rise, dt):
    tos = L.get("timeout_terms", {"time_out"})
    n = len(end)
    eps = []
    for i in range(n):
        e = int(end[i])
        stg = L["stage"][:e, i]
        rec = dict(end_s=e * dt, reason=reason[i], max_stage=int(stg.max()) if e else 0, apexes=int(apex_n[i]),
                   apex_rise_cm=float(apex_rise[i] * 100))
        push = stg == 0
        if push.any():
            rf = L["rf"][:e, i]
            rec["push_s"] = float(push.sum() * dt) if (stg > 0).any() else None
            rec["push_top_v"] = float(L["v"][:e, i][push].max())
            rec["strokes"] = int(((rf[1:] & ~rf[:-1]) & push[1:]).sum())
        man = stg == 2
        if man.any():
            m = L["manual"][:e, i][man]
            run = best = 0
            for b in m:
                run = run + 1 if b else 0
                best = max(best, run)
            rec["manual_frac"] = float(m.mean())
            bm = L["band"][:e, i][man]
            run = bb = 0
            for b in bm:
                run = run + 1 if b else 0
                bb = max(bb, run)
            rec["band_frac"] = float(bm.mean())
            rec["band_longest_s"] = bb * dt
            rec["manual_longest_s"] = best * dt
            rec["manual_v"] = float(L["v"][:e, i][man].mean())
        ex = stg == 3
        if ex.any():
            rec["exit_tail_strike_s"] = float(L["tail"][:e, i][ex].sum() * dt)
            rec["approach_s"] = float(ex.sum() * dt)
        rp = stg == 4
        if rp.any():
            rec["ramp_s"] = float(rp.sum() * dt)
            rec["ramp_entry_v"] = float(L["v"][:e, i][rp][0])
            rec["ramp_max_rise_cm"] = float(L["rise"][:e, i][rp].max() * 100)
        settle = stg == 1
        if settle.any():
            rec["settle_s"] = float(settle.sum() * dt)
        rec["success"] = "success" in reason[i].split(",")
        rec["score"] = rec["max_stage"] + 0.25 * rec["apexes"] + 0.01 * rec["end_s"] + 0.5 * rec["success"]
        eps.append(rec)
    agg = dict(n=n, survival=float(np.mean([r["reason"] == "time_out" for r in eps])),
               end_s_mean=float(np.mean([r["end_s"] for r in eps])))
    # specialists: ended on a time-out term only (time_out / success / past_pipe_end), success rate, step-on rate
    agg["ended_ok"] = float(np.mean([_ended_ok(r["reason"], tos) for r in eps]))
    if "success" in tos:
        agg["success"] = float(np.mean([r["success"] for r in eps]))
        pe = [r for r in eps if r.get("push_top_v") is not None]
        if pe:
            agg["stepped_on"] = float(np.mean([r["max_stage"] >= 1 for r in pe]))
        sv = [r["settle_s"] for r in eps if r.get("settle_s") is not None]
        if sv:
            agg["settle_s_median"] = float(np.median(sv))
    for s in range(5):
        agg[f"reached_{STAGE_NAMES[s]}"] = float(np.mean([r["max_stage"] >= s for r in eps]))
    for key in ("push_s", "push_top_v", "strokes", "manual_frac", "band_frac", "band_longest_s", "manual_longest_s", "manual_v", "exit_tail_strike_s",
                "approach_s", "ramp_s", "ramp_entry_v", "ramp_max_rise_cm", "apexes", "apex_rise_cm"):
        vals = [r[key] for r in eps if r.get(key) is not None]
        if vals:
            agg[key + "_median"] = float(np.median(vals))
            agg[key + "_mean"] = float(np.mean(vals))
    agg["ramp_2plus_passes"] = float(np.mean([r["apexes"] >= 2 for r in eps]))
    rs = {}
    for r in eps:
        rs[r["reason"]] = rs.get(r["reason"], 0) + 1
    agg["end_reasons"] = rs
    return agg, eps


# ── render (visual only) ──────────────────────────────────────────────────────────────────────────────────────────────
# Everything below draws from a RENDER-TIME copy of the scene spec (sky / floor textures, lights, copings and painted
# wall marks are added to the copy, which is compiled separately). env.sim.mj_model and the Warp model used for
# training are never modified. chain_play.py reuses RenderScene / CamRig / the HUD helpers.

RENDER_W, RENDER_H = 1280, 720
SKY_TOP, SKY_HORIZON = (0.36, 0.56, 0.84), (0.86, 0.90, 0.95)
FLOOR_A, FLOOR_B, FLOOR_LINE = (0.60, 0.60, 0.58), (0.53, 0.53, 0.52), (0.42, 0.42, 0.41)
FLOOR_TILE = 0.25                     # m: one checker square (a sense of scale and of speed)
PIPE_RGB, DECK_RGB, COPING_RGB = (0.34, 0.51, 0.72), (0.20, 0.30, 0.44), (0.86, 0.87, 0.90)
WALL_MARKS = (0.04, 0.08)             # m: painted height lines on the transitions (the "height" telemetry)
KEY_DIR = (-0.42, 0.0, -0.86)         # sun direction (world): contact shadows; its x-component shades the two walls apart
#                                       render7: y-component 0.30 -> 0 (chain_r6 review: jagged teeth along the wall top /
#                                       the painted lines at f660 = shadow-map staircase of the coping / mark shadows, whose
#                                       edges run along world y, diagonal to the shadow-map texel grid while the light had a
#                                       y-component; with the light in the xz plane they are texel-aligned and clean,
#                                       verified on a standalone pipe render; a smaller shadowclip only shrank the teeth)
FOVY = 40.0

# stage cameras: key -> (azimuth: offset from the board yaw, or absolute for "ramp"; elevation; distance;
# lookat height above the board origin). Free-camera azimuth a looks along (cos a, sin a), so a = yaw + 90 sits on the
# board's -y side (the side the rider faces), a = yaw + 35 behind the board on that side; the sequence rotates around
# the rider: rear three-quarter (PUSH) -> profile (RIDE, MANUAL, the set-down and >= RAMP_HOLD_S after it) -> a slow
# blend down the pipe axis once the rider carves in (270 = yaw + ~175 there). "exit" is no longer used by the videos.
# render3: PUSH / STEP ON use a close near-side view (yaw + 72: on the side the rider faces, a little behind the board)
# so the push foot's sweep runs across the frame (the old yaw + 35 looked along the travel and foreshortened it) and the
# rider fills ~50 % of the frame height; the later blend to the profile (yaw + 90) is a small, slow move.
# render5 (first-time-viewer review: from yaw + 72 the torso faced the lens and the push foot swung toward it, a
# sideways shuffle): PUSH / STEP ON from a rear three-quarter on the rider's BACK side (yaw - 45, the board's +y side).
# The push foot (right, ~10-15 cm off the -y edge) sweeps back toward the tail along the board, in the clear (A/B of
# yaw +35 / +50 / +72 / -35 / -50 on p12 model_3398: from the -y side the foot comes at the lens and overlaps the
# body; from the +y side the forward lean + backward kick read as a stroke and the board-foot leg never hides it).
# The profile (yaw + 90, MANUAL) is reached by a slow orbit behind the rider starting PUSH_CAM_SETTLE_S into SETTLE.
# render6 (chain v1 review: the -45 -> +90 orbit crossed yaw + 0, ~1.5 s of the rider's back during RIDE): the MANUAL
# profile is taken from the push camera's side (yaw - 90), so the RIDE move is a 45 deg orbit from the rear
# three-quarter to the side profile that never passes behind the rider.
CAMS = dict(push=(-45.0, -11.0, 0.76, 0.10),       # rear three-quarter, back side (render5; render3: 72, -9, 0.78, 0.11)
            manual=(-90.0, -3.0, 0.72, 0.075),      # low side-on profile: nose lift and tail clearance (render6: was +90)
            exit=(160.0, -14.0, 1.15, 0.08),       # pulls back, front three-quarter
            ramp=(90.0, -10.0, 1.55, 0.06),        # render9: down the pipe axis FROM THE OPEN END (az 90 looks along
#   +y, into the pipe; the camera sits just outside the mouth): both transitions in profile, the pipe receding behind the
#   rider, the board side-on. chain_final_h8 review: az 270 / 0.92 m sat inside the pipe looking back at the mouth (the
#   rider front-on, one wall in frame, the board slew read as a bail) and its blend from the carve camera (az ~58)
#   took the short way through az 0, i.e. outside the -x wall (f520-556: the dark back face, the board hidden).
#   render4: (270, -9, 0.92, 0.085)
#                                                   wall height reads against the coping / painted 4 / 8 cm lines
            carve=(-38.0, -16.0, 1.3, 0.07))      # render8: tighter follow-cam (chain_final review: at -62/-22/1.6 the
#   rider was ~1/10-1/5 of the frame height): rear three-quarter on the profile side, rider ~1/3 of the frame, the pipe
#   opening at the frame edge ahead. render7: CARVE IN (chain only, cam_keys(carve_cam=True)): a high
#   rear-side three-quarter; azimuth relative to the carve direction (board -> pipe centre at the carve start, CamRig
#   carve_dir), on the profile camera's side (no crossing behind the rider), lookat CARVE_LEAD_M ahead of the board along
#   that direction: the path, the floor sliding by and the pipe ahead read (chain_r6 review: end-on down the pipe axis
#   the 4.6 s carve-in was a static figure mid-frame)
CARVE_LEAD_M = 0.30                   # m: carve camera lookat lead toward the pipe (render8: 0.45 -> 0.30, tighter cam)
RAMP_CAM_LEAD_S = 1.0                 # s: render8: the ramp camera blend starts this long before RAMP entry
RAMP_CAM_MIN_CARVE_S = 1.0            # s: render8: ... but after at least this long on the carve camera
CARVE_HOLD_S = 0.3                    # s: the profile stays this long after the four wheels are down before the carve move
RAMP_FOLLOW = 0.3                     # render9: 0.9 -> 0.3 (the wider axis shot keeps both copings in frame; it pans a
#                                       little with the rider instead of centring it). render6: ramp lookat x = origin +
#                                       RAMP_FOLLOW * (board x - origin) (render4: 0.6,
#                                       the rider sat at the frame edge on the walls)
RAMP_FOLLOW_SIG_S = 0.3               # s: zero-phase smoothing of the board x the ramp camera follows, twiced (render6:
#                                       _zero_phase_twice keeps the wall-to-wall swing, removes the jitter; was 0.35 plain)
FAR_FOG_M = (9.0, 30.0)               # m: render9 establishing shot fog start / end (the clip's: 2 / 7.5 m)
EST_S = 1.3                           # s: render9 establishing shot under the title (chain clips), then a cut to PUSH
EST_FOVY = 24.0                       # deg: render9 establishing shot lens (longer than FOVY: the pipe ~10 m away reads)
EST_LEAD_S = 0.3                      # s: render9: real frames between the cut to the push camera and the first push
HEIGHT_BAR_CM = 10.0                  # render4: HUD height bar full scale (ticks at the painted WALL_MARKS)


def _norm(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def _smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def _styled_model(env):
    """Compile a render copy of the scene spec with sky, a checker floor of FLOOR_TILE squares, shaded half-pipe
    materials, copings, wall height marks and a shadow-casting key light. Returns (model, pipe dict | None)."""
    import mujoco
    from micro_skater.tasks import skate_sequence as sq

    sp = env.scene.spec.copy()
    Tx, Bt = mujoco.mjtTexture, mujoco.mjtBuiltin
    sp.add_texture(name="rv_sky", type=Tx.mjTEXTURE_SKYBOX, builtin=Bt.mjBUILTIN_GRADIENT, rgb1=SKY_TOP,
                   rgb2=SKY_HORIZON, width=256, height=1536)
    sp.add_texture(name="rv_floor", type=Tx.mjTEXTURE_2D, builtin=Bt.mjBUILTIN_CHECKER, rgb1=FLOOR_A, rgb2=FLOOR_B,
                   mark=mujoco.mjtMark.mjMARK_EDGE, markrgb=FLOOR_LINE, width=512, height=512)

    def mat(name, rgb, tex=None, **kw):
        mt = sp.add_material(name=name, rgba=list(rgb) + [1.0], **kw)
        if tex:
            t = [""] * int(mujoco.mjtTextureRole.mjNTEXROLE)
            t[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = tex
            mt.textures = t

    rep = 1.0 / FLOOR_TILE                          # measured: one checker square per 1 / texrepeat m on the plane
    mat("rv_floor", (1.0, 1.0, 1.0), "rv_floor", texrepeat=[rep, rep], texuniform=True, reflectance=0.0,
        specular=0.1, shininess=0.15)
    mat("rv_pipe", PIPE_RGB, specular=0.25, shininess=0.35)
    mat("rv_deck", DECK_RGB, specular=0.15, shininess=0.2)
    mat("rv_coping", COPING_RGB, specular=0.9, shininess=0.9)
    mat("rv_mark", (0.95, 0.96, 0.98), specular=0.1)

    G = mujoco.mjtGeom
    tb = sp.body("terrain")
    bodies = [b for b in (tb, sp.worldbody) if b is not None]
    boxes = []
    for b in bodies:
        for g in b.geoms:
            if g.type == G.mjGEOM_PLANE:
                g.material, g.rgba = "rv_floor", [1.0, 1.0, 1.0, 1.0]
                g.size = [60.0, 60.0, FLOOR_TILE]
            elif g.type == G.mjGEOM_BOX and b is tb:
                boxes.append(g)
    pipe = None
    if boxes:
        decks = [g for g in boxes if g.size[2] > 0.03]
        for g in boxes:
            dk = g in decks                         # explicit rgba (a non-default geom rgba overrides the material's)
            g.material, g.rgba = ("rv_deck" if dk else "rv_pipe"), list(DECK_RGB if dk else PIPE_RGB) + [1.0]
        if len(decks) == 2:
            cx = 0.5 * (decks[0].pos[0] + decks[1].pos[0])
            cy = decks[0].pos[1]
            P = dict(sq.PIPE)
            W = 2.0 * decks[0].size[1]
            for g in decks:                         # copings: steel tubes along the platforms' inner edges
                s = 1.0 if g.pos[0] > cx else -1.0
                xc = g.pos[0] - s * g.size[0]
                zt = g.pos[2] + g.size[2]
                tb.add_geom(type=G.mjGEOM_CYLINDER, size=[0.006, 0, 0], fromto=[xc, cy - W / 2, zt, xc, cy + W / 2, zt],
                            material="rv_coping", rgba=list(COPING_RGB) + [1.0], contype=0, conaffinity=0, group=0)
                for zm in WALL_MARKS:               # painted height lines tangent to the transition
                    if zm >= P["h"]:
                        continue
                    phi = math.acos(1.0 - zm / P["R"])
                    x = P["L"] / 2 + P["R"] * math.sin(phi)
                    nx, nz = -math.sin(phi), math.cos(phi)
                    off = 0.0015
                    th = s * phi
                    tb.add_geom(type=G.mjGEOM_BOX, size=[0.004, W / 2, 0.0006],
                                pos=[cx + s * (x + nx * off), cy, zm + nz * off],
                                quat=[math.cos(-th / 2), 0.0, math.sin(-th / 2), 0.0],
                                material="rv_mark", rgba=[0.80, 0.86, 0.93, 1.0], contype=0, conaffinity=0, group=0)
            pipe = dict(P, cx=cx, cy=cy, W=W)
    key = sp.worldbody.add_light(name="rv_key", castshadow=True, pos=[0.0, 0.0, 4.0], dir=list(_norm(KEY_DIR)),
                                 diffuse=[0.62, 0.60, 0.55], specular=[0.25, 0.25, 0.25], ambient=[0.0, 0.0, 0.0])
    fill = sp.worldbody.add_light(name="rv_fill", castshadow=False, pos=[0.0, 0.0, 4.0], dir=list(_norm((0.5, -0.3, -0.8))),
                                  diffuse=[0.16, 0.18, 0.22], specular=[0.0, 0.0, 0.0], ambient=[0.0, 0.0, 0.0])
    for lt in (key, fill):
        try:
            lt.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        except AttributeError:                      # older MuJoCo: boolean flag
            lt.directional = True
    return sp.compile(), pipe


class RenderScene:
    """Render model + data + mujoco.Renderer for an env (cached per env)."""

    def __init__(self, env):
        import mujoco
        self.mj = mujoco
        m_sim = env.sim.mj_model
        try:
            m, pipe = _styled_model(env)
            assert m.nq == m_sim.nq and m.nbody == m_sim.nbody and m.njnt == m_sim.njnt, "layout differs"
            assert all(m.joint(j).name == m_sim.joint(j).name for j in range(m.njnt)), "joint names differ"
        except Exception as ex:                       # never fail a render on the cosmetics: plain copy
            print(f"[render] styled scene unavailable ({ex!r}); using a plain copy", flush=True)
            m, pipe = copy.deepcopy(m_sim), None
        self.m, self.pipe = m, pipe
        ext = float(m.stat.extent) or 1.0
        m.vis.global_.offwidth, m.vis.global_.offheight = RENDER_W, RENDER_H
        m.vis.global_.fovy = FOVY
        m.vis.quality.shadowsize = 8192
        m.vis.quality.offsamples = 8
        m.vis.headlight.ambient[:] = (0.26, 0.27, 0.30)
        m.vis.headlight.diffuse[:] = (0.10, 0.10, 0.10)
        m.vis.headlight.specular[:] = (0.0, 0.0, 0.0)
        m.vis.rgba.haze[:] = SKY_HORIZON + (1.0,)
        m.vis.rgba.fog[:] = SKY_HORIZON + (1.0,)
        m.vis.map.fogstart, m.vis.map.fogend = 2.0 / ext, 7.5 / ext
        m.vis.map.znear, m.vis.map.zfar = 0.01 / ext, 60.0 / ext
        m.vis.map.shadowclip = 7.0 / ext           # directional shadow box +-7 m around the camera target (fog hides beyond)
        self.key = -1
        for i in range(m.nlight):
            if m.light(i).name == "rv_key":
                self.key = i
            elif m.light(i).name != "rv_fill":
                m.light_castshadow[i] = 0              # the scene's own light: soft, no second shadow
                m.light_diffuse[i] *= 0.25
        if self.key < 0 and m.nlight:
            m.light_castshadow[0] = 1
        self.d = mujoco.MjData(m)
        frees = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        self.bfree = [m.jnt_qposadr[j] for j in frees if m.joint(j).name.startswith("board/")][0]
        # tail-end collision boxes of the deck (for the tail-clearance readout)
        bid = [b for b in range(m.nbody) if m.body(b).name == "board/skateboard"]
        self.tail = [g for g in range(m.ngeom) if bid and m.geom_bodyid[g] == bid[0] and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
                     and m.geom_contype[g] | m.geom_conaffinity[g] and m.geom_pos[g][0] < -0.03]
        # render8 (chain_final review B: the deck core brushed the wall at left-wall apexes, <= 2.1 mm for 2-3 steps,
        # without a TAIL DOWN chip; the env's tail_floor sensor covers deck_tail_* only): every deck collision geom of
        # the board (tail / nose / core) against the terrain (floor plane + half-pipe), from the render copy's contacts
        bnames = {b: m.body(b).name for b in range(m.nbody)}
        self.deck_kind = {}
        for g in range(m.ngeom):
            nm = m.geom(g).name.split("/")[-1]
            if (bnames[m.geom_bodyid[g]].startswith("board/") and nm.startswith("deck") and nm.endswith("_collision")
                    and m.geom_contype[g] | m.geom_conaffinity[g]):
                self.deck_kind[g] = "tail" if nm.startswith("deck_tail_") else "nose" if nm.startswith("deck_nose_") else "core"
        self.terrain_g = {g for g in range(m.ngeom) if bnames[m.geom_bodyid[g]] in ("terrain", "world")
                          and m.geom_contype[g] | m.geom_conaffinity[g]}
        self.r = mujoco.Renderer(m, RENDER_H, RENDER_W)
        atexit.register(self.close)                     # free the GL context before interpreter teardown
        for f, v in ((mujoco.mjtRndFlag.mjRND_SHADOW, 1), (mujoco.mjtRndFlag.mjRND_FOG, 1),
                     (mujoco.mjtRndFlag.mjRND_SKYBOX, 1), (mujoco.mjtRndFlag.mjRND_REFLECTION, 1)):
            self.r.scene.flags[f] = v
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    def close(self):
        if getattr(self, "r", None) is not None:
            self.r.close()
            self.r = None
        if getattr(self, "r_far", None) is not None:
            self.r_far.close()
            self.r_far = None

    @classmethod
    def get(cls, env) -> "RenderScene":
        if id(env) not in _RENDER_CACHE:
            _RENDER_CACHE[id(env)] = cls(env)
        return _RENDER_CACHE[id(env)]

    def set_state(self, qpos):
        """Forward kinematics on the render data; returns board (xyz, yaw deg)."""
        self.d.qpos[:] = qpos
        self.mj.mj_forward(self.m, self.d)
        a = self.bfree
        w_, x_, y_, z_ = self.d.qpos[a + 3:a + 7]
        yaw = math.degrees(math.atan2(2 * (w_ * z_ + x_ * y_), 1 - 2 * (y_ * y_ + z_ * z_)))
        return np.array(self.d.qpos[a:a + 3]), yaw

    def ground_z(self, x: float, y: float, origin) -> float:
        p = self.pipe
        if p is None or abs(y - p["cy"]) > p["W"] / 2:
            return float(origin[2])
        h, R, Lp = p["h"], p["R"], p["L"]
        ax = abs(x - p["cx"])
        x_end = Lp / 2 + R * math.sin(math.acos(1.0 - h / R))
        if ax <= Lp / 2:
            return float(origin[2])
        if ax <= x_end:
            u = ax - Lp / 2
            return float(origin[2] + R - math.sqrt(max(R * R - u * u, 0.0)))
        return float(origin[2] + h)

    def deck_touch(self) -> set:
        """render8: the deck parts ("tail" / "nose" / "core") in contact (dist < 0: touching / penetrating) with the
        terrain at the current set_state (mj_forward ran the collision pass)."""
        d, out = self.d, set()
        if d.ncon == 0 or not self.deck_kind:
            return out
        g1, g2, dist = d.contact.geom1[:d.ncon], d.contact.geom2[:d.ncon], d.contact.dist[:d.ncon]
        for a, b, ds in zip(g1.tolist(), g2.tolist(), dist.tolist()):
            if ds >= 0.0:
                continue
            if a in self.deck_kind and b in self.terrain_g:
                out.add(self.deck_kind[a])
            elif b in self.deck_kind and a in self.terrain_g:
                out.add(self.deck_kind[b])
        return out

    def tail_gap(self, origin) -> float | None:
        """Lowest tail-end deck corner above the surface below it (m); None without tail boxes."""
        best = None
        for g in self.tail:
            s = self.m.geom_size[g]
            R = self.d.geom_xmat[g].reshape(3, 3)
            c = self.d.geom_xpos[g]
            for sx in (-1, 1):
                for sy in (-1, 1):
                    for sz in (-1, 1):
                        p = c + R @ (s * np.array([sx, sy, sz]))
                        gap = p[2] - self.ground_z(p[0], p[1], origin)
                        best = gap if best is None else min(best, gap)
        return best

    def render(self, az: float, el: float, dist: float, look, far: bool = False) -> np.ndarray:
        """far (render9, the establishing shot): a second renderer with the fog pushed out to FAR_FOG_M (the fog lives
        in the GL context, set at its creation), so the half-pipe ~10 m ahead of the start line is visible."""
        self.cam.azimuth, self.cam.elevation, self.cam.distance = az, el, dist
        self.cam.lookat[:] = look
        if self.key >= 0:                              # shadow box follows the camera target
            self.m.light_pos[self.key] = np.asarray(look) - 4.0 * _norm(KEY_DIR)
            self.mj.mj_forward(self.m, self.d)
        r = self.r
        if far:
            if getattr(self, "r_far", None) is None:
                vm = self.m.vis.map
                ext = float(self.m.stat.extent) or 1.0
                f0, f1, sc = vm.fogstart, vm.fogend, vm.shadowclip
                vm.fogstart, vm.fogend, vm.shadowclip = FAR_FOG_M[0] / ext, FAR_FOG_M[1] / ext, 14.0 / ext
                self.r_far = self.mj.Renderer(self.m, RENDER_H, RENDER_W)
                vm.fogstart, vm.fogend, vm.shadowclip = f0, f1, sc
                for f, v in ((self.mj.mjtRndFlag.mjRND_SHADOW, 1), (self.mj.mjtRndFlag.mjRND_FOG, 1),
                             (self.mj.mjtRndFlag.mjRND_SKYBOX, 1), (self.mj.mjtRndFlag.mjRND_REFLECTION, 1)):
                    self.r_far.scene.flags[f] = v
            r = self.r_far
        if far:
            fv = self.m.vis.global_.fovy
            self.m.vis.global_.fovy = EST_FOVY
            r.update_scene(self.d, camera=self.cam, scene_option=self.opt)
            self.m.vis.global_.fovy = fv
        else:
            r.update_scene(self.d, camera=self.cam, scene_option=self.opt)
        return r.render()

    def board_roll_rel(self) -> float:
        """render9 (demo score): |board roll| (deg) relative to the surface under the board (the transition slope runs
        along x, so a board crossing the wall diagonally is not counted as rolled): asin(board y-axis . surface normal)."""
        if getattr(self, "_bbody", None) is None:
            j = [j for j in range(self.m.njnt) if self.m.jnt_qposadr[j] == self.bfree][0]
            self._bbody = int(self.m.jnt_bodyid[j])
        Rm = self.d.xmat[self._bbody].reshape(3, 3)
        p = self.d.xpos[self._bbody]
        o = np.zeros(3)
        e = 0.005
        s = (self.ground_z(p[0] + e, p[1], o) - self.ground_z(p[0] - e, p[1], o)) / (2 * e)
        n = np.array([-s, 0.0, 1.0]) / math.hypot(s, 1.0)
        return abs(math.degrees(math.asin(float(np.clip(Rm[:, 1] @ n, -1.0, 1.0)))))


STAGE_CAM = ("push", "manual", "manual", "manual", "ramp")  # legacy per-stage map; the videos use cam_keys() below
CAM_BLEND = dict(ramp=1.4, manual=1.3, carve=1.4)  # s: smoothstep blend INTO a camera (default CamRig.blend_s); slow moves
#                                       render5: manual 1.5 -> 1.7 (the push -> profile orbit was 135 deg);
#                                       render6: 1.3 (the orbit is 45 deg now, done before MANUAL)
PUSH_CAM_SETTLE_S = 0.2               # s: the push camera stays this long into SETTLE (never moves during the step-on;
#                                       render5: 0.5 -> 0.2 so the longer orbit is ~done when MANUAL starts, SETTLE 1.2-1.5 s)
RAMP_HOLD_S = 1.2                     # s: the profile stays this long after the four wheels are down before the ramp move
SETDOWN_WHEELS_S = 0.15               # s: four wheels continuously down = set-down complete (the chain's wheels gate)
# render3: no fade from black (it hid the standstill and the first stroke): the clip opens on INTRO_STILL_S of the
# first frame, un-faded, with the opening title; the episode then plays.
# render5 (first-time-viewer review): the title is pinned top-left: a title card (TITLE_HOLD_S of clip time, fading out
# over TITLE_OUT_S) that hands over to a small persistent caption at the same anchor (never jumps to the bottom). A
# start-spawn clip skips the standstill before the first push: INTRO_STILL_S of the frame PUSH_LEAD_S before the board
# first passes PUSH_V_START (clip_start), then the episode plays. End card: END_CARD_S freeze (slow push-in), the last
# END_FADE_S fading to dark.
TITLE_FADE_S, TITLE_HOLD_S = 0.0, 1.2
INTRO_STILL_S = 0.5
TITLE_OUT_S = 0.35                    # s: title card fade-out (= persistent caption fade-in), ends at TITLE_HOLD_S
PUSH_V_START, PUSH_LEAD_S = 0.2, 0.2    # m/s, s: clip_start (first push = the board passes PUSH_V_START; at 0.06
#                                         the board's slow creep while the foot plants still left ~1.1 s of stillness)
END_CARD_S, END_FADE_S = 1.5, 0.45    # s: end card hold (freeze + slow push-in); fade to dark over its last END_FADE_S
CREDITS_S = 2.5                       # s: render8: 3.0 -> 2.5 (2 lines + a footnote). render6 closing card (credits / disclosure block) after the end card (render7: 2.5
#                                       -> 3.0, fewer and larger lines)
START_STILL_S = 1.1                   # s: render7 (chain): clip time before the first push under the title card: the real
#                                       episode frames from START_STILL_S before the push (from frame 0 at the latest),
#                                       topped up by a held FIRST frame when the episode has fewer (intro_hold_frames)
STEP_ON_MIN_S = 0.4                   # s: render7: STEP ON shown at least this long from the last floor contact
LABEL_MIN_S = 0.2                     # s: a stage label shorter than this is merged into the previous one (no flicker)
_RENDER_CACHE: dict = {}


# ── per-frame render schedule (labels, camera keys) from the logged episode: offline, render-only ──

def setdown_track(stage, wheels, dt: float, need_s: float = SETDOWN_WHEELS_S):
    """Per frame: set-down complete (latched once the four wheels stayed down need_s in EXIT or later) and the time the
    wheels came down for that run (s; nan before)."""
    T = len(stage)
    done, t_wd = np.zeros(T, bool), np.full(T, np.nan)
    run0 = latched = None
    for k in range(T):
        if stage[k] >= 3 and bool(wheels[k]):
            run0 = k if run0 is None else run0
            if latched is None and (k - run0 + 1) * dt >= need_s:
                latched = run0
        else:
            run0 = None
        if latched is not None:
            done[k], t_wd[k] = True, latched * dt
    return done, t_wd


def cam_keys(stage, setdown, t_wd, dt: float, ramp_ok: bool = True, hold_s: float = RAMP_HOLD_S,
             profile: bool = True, carve_cam: bool = False, carve_hold_s: float = CARVE_HOLD_S):
    """carve_cam (render7, chain videos): the set-down's CARVE IN (EXIT with the set-down complete, >= carve_hold_s after
    the four wheels came down) gets the high "carve" camera; the ramp camera follows at RAMP entry (stage 4). Without a
    carve phase (RAMP entered before the carve move would start) the old rule applies."""
    if carve_cam and ramp_ok:
        keys = _cam_keys(stage, setdown, t_wd, dt, ramp_ok=False, profile=profile)
        T = len(stage)
        k4 = next((k for k in range(T) if int(stage[k]) >= 4), T)          # RAMP entry
        kc = next((k for k in range(k4) if int(stage[k]) == 3 and setdown[k] and k * dt - t_wd[k] >= carve_hold_s), None)
        if kc is not None:
            # render8: the ramp camera move starts RAMP_CAM_LEAD_S before RAMP entry (>= RAMP_CAM_MIN_CARVE_S of carve
            # camera first): started at the entry, the low carve camera orbited behind the pipe platform during the
            # first wall ride (rt8 demo: the 6.3 cm ride was hidden behind the wall)
            kr = min(k4, max(kc + int(round(RAMP_CAM_MIN_CARVE_S / dt)), k4 - int(round(RAMP_CAM_LEAD_S / dt))))
            keys[kc:kr] = ["carve"] * (kr - kc)
            keys[kr:] = ["ramp"] * (T - kr)
            return keys
    return _cam_keys(stage, setdown, t_wd, dt, ramp_ok, hold_s, profile)
    return _cam_keys(stage, setdown, t_wd, dt, ramp_ok, hold_s, profile)


def _cam_keys(stage, setdown, t_wd, dt: float, ramp_ok: bool = True, hold_s: float = RAMP_HOLD_S,
              profile: bool = True):
    """Camera key per frame: the rear three-quarter push camera through PUSH, the step-on and the first
    PUSH_CAM_SETTLE_S of SETTLE (render3: the move never starts during the step-on), then the side profile (slow
    CAM_BLEND["manual"] blend) through MANUAL and the set-down, and the ramp camera only once the rider carves toward
    the pipe (RAMP, or EXIT with the set-down complete), never earlier than hold_s after the four wheels came down.
    Latched (never swings back). profile=False (render5: the push specialist's clip, no MANUAL follows): the push
    camera holds through SETTLE (no half-done orbit frozen in the end card)."""
    keys, ramp, prof, t1 = [], False, False, None
    for k in range(len(stage)):
        s, t = int(stage[k]), k * dt
        if s >= 1 and t1 is None:
            t1 = t
        if not prof and (s >= 2 or (profile and s == 1 and t - t1 >= PUSH_CAM_SETTLE_S)):
            prof = True
        if ramp_ok and not ramp:
            if s >= 4 and (np.isnan(t_wd[k]) or t - t_wd[k] >= hold_s):
                ramp = True
            elif s == 3 and setdown[k] and t - t_wd[k] >= hold_s:
                ramp = True
        keys.append("ramp" if ramp else "manual" if prof else "push")
    return keys


SETDOWN_LABEL_S = 0.5                 # s: render8: SET-DOWN shown this long from the four wheels touching down
DECK_CHIP_HOLD_S = 0.3                # s: render8: a TAIL DOWN / DECK SCRAPE chip stays on screen this long per contact


def stage_labels(stage, cmd, both_on, setdown, dt: float, passes=None, carve: bool = True,
                 min_s: float = LABEL_MIN_S, foot_floor=None, front_down=None, wheels=None):
    """Panel label per frame. PUSH; STEP ON only while the push foot moves onto the deck (PUSH with the step command
    latched, both feet not yet on); RIDE once on the board (SETTLE, or PUSH with both feet on); MANUAL; SET-DOWN, then
    CARVE IN after the set-down (carve=True: a chain heading for the pipe); HALF-PIPE · pass k. Runs shorter than
    min_s are merged into the previous label.
    foot_floor (render7, (T,) bool: either foot on the floor): STEP ON starts when the feet do, i.e. after the last
    floor contact in PUSH / SETTLE (the push foot's last touch-down); a stroke after the scheduler's step command stays
    PUSH (chain_r6 review: the 4th push ran under "STEP ON").
    front_down / wheels (render8, (T,) bool: front wheels down / all four wheels down; chain_final review: MANUAL
    started 0.18 s before the nose lifted, SET-DOWN 0.26 s before it dropped): MANUAL runs from the first frame in
    MANUAL (stage 2) with the front wheels off the ground (stage 2 entry if they never lift) to the four wheels touching
    down in EXIT (stage 3); SET-DOWN then shows SETDOWN_LABEL_S from that touch-down, then CARVE IN."""
    T = len(stage)
    stage = np.asarray(stage).astype(int)
    k_up = k_dn = None
    if front_down is not None and wheels is not None:
        fdn, wh = np.asarray(front_down, bool), np.asarray(wheels, bool)
        k2 = next((k for k in range(T) if stage[k] >= 2), None)
        if k2 is not None and stage[k2] == 2:
            nl = max(1, int(round(0.1 / dt)))           # the lift-off that lasts >= 0.1 s (no single-step flicker)
            k_up = next((k for k in range(k2, T) if stage[k] == 2 and not fdn[k:k + nl].any()), k2)
            k3 = next((k for k in range(k2, T) if stage[k] >= 3), None)
            if k3 is not None and stage[k3] == 3:
                k_dn = next((k for k in range(k3, T) if stage[k] >= 4 or wh[k]), None)
    cmd = np.zeros(T, bool) if cmd is None else np.asarray(cmd, bool)
    both_on = np.zeros(T, bool) if both_on is None else np.asarray(both_on, bool)
    k_last = -1
    if foot_floor is not None:
        ff = np.asarray(foot_floor, bool) & (np.asarray(stage) <= 1)
        hit = np.nonzero(ff)[0]
        k_last = int(hit[-1]) if len(hit) else -1
    out = []
    for k in range(T):
        s = int(stage[k])
        if s == 0:
            lab = ("RIDE" if both_on[k] else "STEP ON" if k > k_last else "PUSH") if cmd[k] else "PUSH"
        elif s == 1:
            lab = "RIDE"
        elif s == 2:
            lab = "MANUAL"
        elif s == 3:
            lab = "CARVE IN" if carve and setdown[k] else "SET-DOWN"
        else:
            p = 0 if passes is None else int(passes[k])
            lab = f"HALF-PIPE · wall ride {p}" if p else "HALF-PIPE"   # render8: "pass" -> "wall ride"
        if k_up is not None:                              # render8: labels follow the wheel contacts
            if s == 2 and k < k_up:
                lab = "RIDE"
            elif s == 3 and k_dn is not None and k < k_dn:
                lab = "MANUAL"
            elif s == 3 and k_dn is not None and k < k_dn + int(round(SETDOWN_LABEL_S / dt)):
                lab = "SET-DOWN"
            elif s == 3 and k_dn is None:
                lab = "MANUAL"
        out.append(lab)
    if foot_floor is not None and 0 <= k_last < T - 1 and cmd[k_last + 1]:
        # render7: the step-on itself (push foot off the floor -> both soles on the deck) often takes < LABEL_MIN_S and
        # was merged away; STEP ON is shown STEP_ON_MIN_S from the last floor contact (PUSH / SETTLE frames only)
        for k in range(k_last + 1, min(T, k_last + 1 + int(round(STEP_ON_MIN_S / dt)))):
            if int(stage[k]) <= 1:
                out[k] = "STEP ON"
    n_min = max(1, int(round(min_s / dt)))
    k = 0
    while k < T:                                          # merge short runs (flicker) into the previous label
        j = k
        while j < T and out[j] == out[k]:
            j += 1
        if j - k < n_min and k > 0 and not out[k].startswith("HALF-PIPE"):
            out[k:j] = [out[k - 1]] * (j - k)
        k = j
    return out


def _zero_phase(x, sigma_frames: float):
    """Gaussian smoothing without lag (centred kernel), odd-reflected ends: a target moving at constant speed is
    reproduced exactly, including the first and last frames (no start-up transient)."""
    x = np.asarray(x, float)
    if sigma_frames < 0.5 or len(x) < 3:
        return x.copy()
    r = int(math.ceil(3 * sigma_frames))
    i = np.arange(-r, r + 1)
    kern = np.exp(-0.5 * (i / sigma_frames) ** 2)
    kern /= kern.sum()
    xp = np.pad(x, r, mode="reflect", reflect_type="odd")
    return np.convolve(xp, kern, mode="valid")


def _zero_phase_twice(x, sigma_frames: float):
    """render6: Tukey twicing of _zero_phase (s + G(x - s)): response 2H - H^2, so the slow wall-to-wall swing passes
    almost unattenuated (no apparent lag / undershoot at the walls) while fast jitter is still removed."""
    s = _zero_phase(x, sigma_frames)
    return s + _zero_phase(np.asarray(x, float) - s, sigma_frames)


class CritDamp:
    """Critically damped second-order tracker with target-velocity feed-forward (kept for CamRig.__call__, the online
    API; the videos use CamRig.plan, which has no lag). Starts AT the target with the target's own velocity once a
    second sample exists, so it never trails a moving target at the start."""

    def __init__(self, omega: float):
        self.w, self.x, self.v, self.tp, self.tv = omega, None, None, None, None

    def __call__(self, tgt, dt: float):
        tgt = np.asarray(tgt, float)
        if self.x is None:
            self.x, self.v, self.tp, self.tv = tgt.copy(), np.zeros_like(tgt), tgt.copy(), None
            return self.x.copy()
        tv = (tgt - self.tp) / dt
        self.tp = tgt.copy()
        if self.tv is None:                               # second frame: adopt the target's velocity, stay on it
            self.tv, self.v, self.x = tv.copy(), tv.copy(), tgt.copy()
            return self.x.copy()
        self.tv = self.tv + min(1.0, 0.5 * self.w * dt) * (tv - self.tv)     # low-passed target velocity
        a = self.w * self.w * (tgt - self.x) + 2.0 * self.w * (self.tv - self.v)
        self.v = self.v + a * dt
        self.x = self.x + self.v * dt
        return self.x.copy()


class CamRig:
    """Stage cameras (CAMS) with a smoothstep blend on a camera change (CAM_BLEND[key] or blend_s).
    plan(): the whole clip offline: blended targets per frame, then zero-phase Gaussian smoothing of azimuth /
    elevation / distance (sig_ang) and of the lookat (sig_look): the camera sits on the true target from frame 0
    with no lag and no over-correction. __call__: the older online version (critically damped)."""

    def __init__(self, dt: float, origin, cams=None, blend_s: float = 0.8, w_look: float = 6.0, w_ang: float = 4.0,
                 carve_dir: float = 90.0):
        self.dt, self.origin, self.cams, self.blend_s = dt, np.asarray(origin, float), cams or CAMS, blend_s
        self.carve_dir = float(carve_dir)              # render7: deg, world heading of the carve-in (the "carve" key)
        self.key = self.frm = None
        self.t_sw = -1e9
        self.yaw_c = self.az_c = None
        self.f_look, self.f_ang = CritDamp(w_look), CritDamp(w_ang)

    def _params(self, key, b, yaw, bx_s=None):
        az, el, dist, dz = self.cams[key]
        if key == "ramp":                              # render4: smooth lateral follow of the rider along the pipe
            o = self.origin
            bx = b[0] if bx_s is None else bx_s
            return az, el, dist, np.array([o[0] + RAMP_FOLLOW * (bx - o[0]), b[1], o[2] + dz])
        if key == "carve":                             # render7: absolute azimuth from the carve direction, lookat ahead
            c = math.radians(self.carve_dir)
            return (self.carve_dir + az, el, dist,
                    np.array([b[0] + CARVE_LEAD_M * math.cos(c), b[1] + CARVE_LEAD_M * math.sin(c), b[2] + dz]))
        return yaw + az, el, dist, np.array([b[0], b[1], b[2] + dz])

    def plan(self, keys, boards, yaws, sig_ang: float = 0.2, sig_look=(0.10, 0.10, 0.18)):
        T = len(keys)
        yaw = np.degrees(np.unwrap(np.radians(np.asarray(yaws, float))))
        P = np.zeros((T, 6))
        key = frm = None
        t_sw, bs = 0.0, self.blend_s
        bxs = _zero_phase_twice([float(b_[0]) for b_ in boards], RAMP_FOLLOW_SIG_S / self.dt)
        for k in range(T):
            t = k * self.dt
            if keys[k] != key:
                frm, key, t_sw = key, keys[k], t
                bs = CAM_BLEND.get(key, self.blend_s)
            az, el, di, lk = self._params(key, boards[k], yaw[k], bxs[k])
            s = _smoothstep((t - t_sw) / bs) if frm is not None else 1.0
            if s < 1.0:
                az0, el0, di0, lk0 = self._params(frm, boards[k], yaw[k], bxs[k])
                az, el, di, lk = az0 + s * _wrap180(az - az0), el0 + s * (el - el0), di0 + s * (di - di0), lk0 + s * (lk - lk0)
            P[k] = (az, el, di, *lk)
        P[:, 0] = np.degrees(np.unwrap(np.radians(P[:, 0])))
        sa = sig_ang / self.dt
        for c, sg in zip(range(6), (sa, sa, sa) + tuple(s_ / self.dt for s_ in sig_look)):
            P[:, c] = _zero_phase(P[:, c], sg)
        return [(float(p[0]), float(p[1]), float(p[2]), p[3:6].copy()) for p in P]

    def __call__(self, key, b, yaw, t):
        self.yaw_c = yaw if self.yaw_c is None else self.yaw_c + _wrap180(yaw - self.yaw_c)   # continuous yaw
        if key != self.key:
            self.frm, self.key, self.t_sw = self.key, key, t
        az, el, di, lk = self._params(key, b, self.yaw_c)
        bs = CAM_BLEND.get(key, self.blend_s)
        s = _smoothstep((t - self.t_sw) / bs) if self.frm is not None else 1.0
        if s < 1.0:
            az0, el0, di0, lk0 = self._params(self.frm, b, self.yaw_c)
            az, el, di, lk = az0 + s * _wrap180(az - az0), el0 + s * (el - el0), di0 + s * (di - di0), lk0 + s * (lk - lk0)
        self.az_c = az if self.az_c is None else self.az_c + _wrap180(az - self.az_c)
        az_f, el_f, di_f = self.f_ang(np.array([self.az_c, el, di]), self.dt)
        return float(az_f), float(el_f), float(di_f), self.f_look(lk, self.dt)


# ── HUD ──
# render5 (first-time-viewer review): sized for a phone (a 1280x720 frame is shown ~360-400 px wide there): stage label
# 46 px, values 32 px, keys 24 px light grey, the honesty note 24 px on its own pill. One fixed place per element: title
# card -> persistent caption top-left, stage panel bottom-left (the end card's result panel takes its place), honesty
# note bottom-right, policy tag top-right.

HUD_X, HUD_Y, HUD_BOTTOM = 28, 22, 26   # px: left / top / bottom margins
KEY_RGB = (214, 222, 234)              # telemetry keys (render4: (190, 200, 215) at 17 px)
PANEL_RGBA = (10, 14, 22, 178)


def hud_fonts() -> dict:
    from PIL import ImageFont

    def f(names, sz):
        for nm in names:
            try:
                return ImageFont.truetype(nm, sz)
            except OSError:
                pass
        return ImageFont.load_default()
    semi, reg = ("seguisb.ttf", "segoeuib.ttf", "arialbd.ttf"), ("segoeui.ttf", "arial.ttf")
    return dict(label=f(semi, 46), chip=f(semi, 22), tele=f(semi, 32), tele_k=f(reg, 24), note=f(semi, 24),
                cap=f(semi, 26), title=f(semi, 50), sub=f(reg, 28), card=f(semi, 50), card_sub=f(reg, 26),
                tick=f(semi, 18), foot=f(semi, 26), foot_s=f(semi, 24), cred=f(semi, 42), cred_s=f(reg, 34),
                cred_xs=f(reg, 25), fast=f(semi, 44), ptag=f(reg, 22), est=f(semi, 30))
    # render9: fast = the 2x badge (centre-bottom), ptag = the small policy tag inside the stage panel, est = the
    # establishing shot's "half-pipe" pointer   # render7: closing card 30/26/22 -> 42/34/25 px (fewer lines)


def _txt(dr, xy, s, font, fill=(255, 255, 255), a=255, shadow=True):
    a = int(max(0, min(255, a)))
    if shadow:
        dr.text((xy[0] + 1, xy[1] + 2), s, font=font, fill=(0, 0, 0, int(a * 0.55)))
    dr.text(xy, s, font=font, fill=tuple(fill[:3]) + (a,))


def _chip(dr, x, y, s, font, bg, a=255):
    tw = dr.textlength(s, font=font)
    h = font.size + 12
    dr.rounded_rectangle([x, y, x + tw + 24, y + h], radius=h // 2, fill=tuple(bg) + (int(a * 0.92),))
    dr.text((x + 12, y + 3), s, font=font, fill=(255, 255, 255, int(a)))
    return x + tw + 24


def _pill_text(dr, x, y, s, font, fill, a=255, pad_x=14, pad_y=7, right=False):
    """Text on a dark rounded pill (contrast on the light floor / sky); x is the right edge when right=True."""
    a = int(max(0, min(255, a)))
    tw = dr.textlength(s, font=font)
    w, h = tw + 2 * pad_x, font.size + 2 * pad_y + 4
    if right:
        x = x - w
    dr.rounded_rectangle([x, y, x + w, y + h], radius=12, fill=PANEL_RGBA[:3] + (int(PANEL_RGBA[3] * a / 255),))
    dr.text((x + pad_x, y + pad_y - 2), s, font=font, fill=tuple(fill[:3]) + (a,))
    return w, h


def draw_note(dr, F, note, clear_x: float | None = None):
    """The small honesty note (e.g. "typical of 16 robots"), bottom-right, on its own pill. clear_x: the stage panel's
    right edge; a note that would touch it sits one row higher (wide MANUAL panel + a long chain note)."""
    if note:
        h = F["note"].size + 2 * 7 + 4
        y = RENDER_H - HUD_BOTTOM - h
        if clear_x is not None and RENDER_W - HUD_X - (dr.textlength(note, font=F["note"]) + 28) < clear_x + 16:
            y -= h + 10
        _pill_text(dr, RENDER_W - HUD_X, y, note, F["note"], (240, 243, 248), right=True)


def draw_footer(dr, F, foot_l="", foot_r="", foot_l_a=255):
    """render5: kept for callers; the title lives top-left now (draw_title), only the honesty note is drawn here."""
    draw_note(dr, F, foot_r)


def draw_height_bar(dr, F, x0, y0, h, rise_cm, best_cm=None, full_cm=HEIGHT_BAR_CM):
    """render4: small vertical gauge of the rider's height on the wall (0..full_cm), ticks + labels at the painted
    wall marks, a thin peak-hold line at the best height so far."""
    w, pad = 58, 12
    dr.rounded_rectangle([x0, y0, x0 + w, y0 + h], radius=14, fill=PANEL_RGBA)
    bx0, bx1, by0, by1 = x0 + 32, x0 + 46, y0 + pad, y0 + h - pad
    dr.rounded_rectangle([bx0, by0, bx1, by1], radius=5, fill=(255, 255, 255, 46))

    def yv(cm):
        return by1 - (by1 - by0) * min(max(cm / full_cm, 0.0), 1.0)
    fy = yv(rise_cm)
    if fy < by1 - 1:
        dr.rounded_rectangle([bx0, fy, bx1, by1], radius=5, fill=(255, 210, 90, 240))
    for m in WALL_MARKS:
        ty = yv(m * 100)
        dr.line([bx0 - 3, ty, bx1 + 3, ty], fill=(236, 238, 242, 220), width=2)
        _txt(dr, (x0 + 8, ty - 12), f"{m * 100:.0f}", F["tick"], KEY_RGB, shadow=False)
    if best_cm is not None and best_cm > 0.2:
        py = yv(best_cm)
        dr.line([bx0 - 5, py, bx1 + 5, py], fill=(255, 255, 255, 245), width=3)


def draw_hud(dr, F, label, tele, chips=(), label_a=255, tag=None, foot_l="", foot_r="", pitch=None, foot_l_a=255,
             height=None, ptag=None):
    """Bottom-left panel: stage label + chips, the stage's telemetry below (key/value slots of fixed width, so values
    never push each other around; a board glyph tilted by the true pitch after "nose-up"); optional height bar right
    of the panel; optional tag chip top-right; the honesty note (foot_r) bottom-right. render5: foot_l / foot_l_a are
    unused (the title is drawn by draw_title, pinned top-left)."""
    W, H = RENDER_W, RENDER_H
    pad = 16
    lw = dr.textlength(label, font=F["label"])
    cw = sum(dr.textlength(c, font=F["chip"]) + 24 + 10 for c, _ in chips)
    slots = []
    for k_, v_, tmpl in tele:
        kw = dr.textlength(k_, font=F["tele_k"])
        vw = max(dr.textlength(tmpl, font=F["tele"]), dr.textlength(v_, font=F["tele"]))
        slots.append((k_, v_, kw, vw))
    glyph = pitch is not None and any(k_ == "nose-up" for k_, *_ in slots)
    gw = 54 if glyph else 0
    tw = sum(kw + 8 + vw + 26 for _, _, kw, vw in slots) - 26 + gw
    ptw = dr.textlength(ptag, font=F["ptag"]) if ptag else 0     # render9: small policy tag after the label / chips
    pw = max(lw + (16 + cw if chips else 0) + (18 + ptw if ptag else 0), tw) + 2 * pad
    ph = pad + F["label"].size + 12 + F["tele"].size + pad + 4
    x0, y0 = HUD_X, H - HUD_BOTTOM - ph
    draw_note(dr, F, foot_r, clear_x=x0 + pw + (68 if height is not None else 0))
    dr.rounded_rectangle([x0, y0, x0 + pw, y0 + ph], radius=16, fill=PANEL_RGBA)
    _txt(dr, (x0 + pad, y0 + pad - 8), label, F["label"], a=label_a)
    cx = x0 + pad + lw + 16
    for c, col in chips:
        cx = _chip(dr, cx, y0 + pad + 6, c, F["chip"], col) + 10
    if ptag:
        _txt(dr, (x0 + pw - pad - ptw, y0 + pad + 10), ptag, F["ptag"], (176, 186, 202), shadow=False)
    ty = y0 + pad + F["label"].size + 12
    x = x0 + pad
    for k_, v_, kw, vw in slots:
        _txt(dr, (x, ty + 6), k_, F["tele_k"], KEY_RGB, shadow=False)
        x += kw + 8
        _txt(dr, (x, ty), v_, F["tele"], (255, 255, 255), shadow=False)
        x += vw + 26
        if k_ == "nose-up" and glyph:                  # board glyph tilted by the true pitch
            th = math.radians(pitch)
            gx, gy = x - 14, ty + F["tele"].size - 2
            ex, ey = gx + 40 * math.cos(th), gy - 40 * math.sin(th)
            dr.line([gx, gy, ex, ey], fill=(255, 210, 90, 255), width=5)
            for px, py in ((gx + 8 * math.cos(th), gy - 8 * math.sin(th)), (gx + 32 * math.cos(th), gy - 32 * math.sin(th))):
                dr.ellipse([px - 4 + 2 * math.sin(th), py + 1, px + 4 + 2 * math.sin(th), py + 9], fill=(255, 255, 255, 235))
            x += gw
    if height is not None:                             # render4: (rise_cm, best_cm) in RAMP: height bar right of the panel
        draw_height_bar(dr, F, x0 + pw + 10, y0, ph, *height)
    if tag:
        tw_ = dr.textlength(tag, font=F["chip"]) + 24
        _chip(dr, W - HUD_X - tw_, HUD_Y, tag, F["chip"], (40, 48, 64))
    return x0 + pw


def draw_fast_badge(dr, F, clear_x: float, s: str = "2× speed"):
    """render9: the 2x badge, large and centred at the bottom (chain_final_h8 review: the small orange chip was easy to
    miss); shifted right just enough to clear the stage panel (clear_x = its right edge)."""
    f = F["fast"]
    tw = dr.textlength(s, font=f)
    w, h = tw + 2 * 26, f.size + 2 * 10 + 4
    x = max((RENDER_W - w) / 2, clear_x + 20)
    y = RENDER_H - HUD_BOTTOM - h
    dr.rounded_rectangle([x, y, x + w, y + h], radius=h // 2, fill=(214, 120, 24, 240))
    dr.text((x + 26, y + 8), s, font=f, fill=(255, 255, 255, 255))


def cam_project(cam, p, fovy: float = FOVY):
    """render9: pixel (x, y) of world point p for a free camera (az, el, dist, lookat) at RENDER_W x RENDER_H / FOVY;
    None behind the camera."""
    az, el, di, lk = cam
    a, e = math.radians(az), math.radians(el)
    fwd = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    pos = np.asarray(lk, float) - di * fwd
    right = _norm(np.cross(fwd, [0.0, 0.0, 1.0]))
    up = np.cross(right, fwd)
    v = np.asarray(p, float) - pos
    z = float(v @ fwd)
    if z <= 1e-3:
        return None
    f = (RENDER_H / 2) / math.tan(math.radians(fovy / 2))
    return RENDER_W / 2 + f * float(v @ right) / z, RENDER_H / 2 - f * float(v @ up) / z


def telemetry(v, pitch=None, rise_cm=None, tail_gap_cm=None, t=None, best_cm=None):
    """(key, value, width template) triples for draw_hud; None hides a slot (chosen per stage by stage_telemetry).
    render8 (plain language): "height" -> "height on wall" (+ "highest" so far, replacing the unlabelled gauge),
    "tail gap" -> "tail clearance"."""
    out = [("speed", f"{v:+.2f} m/s", "+0.00 m/s")]
    if pitch is not None:
        out.append(("nose-up", f"{pitch:+.1f}°", "+00.0°"))
    if rise_cm is not None:
        out.append(("height on wall", f"{rise_cm:.1f} cm", "00.0 cm"))
    if best_cm is not None:
        out.append(("highest", f"{best_cm:.1f} cm", "00.0 cm"))
    if tail_gap_cm is not None:
        out.append(("tail clearance", f"{tail_gap_cm:.1f} cm", "00.0 cm"))
    if t is not None:
        out.append(("t", f"{t:.2f} s", "00.00 s"))
    return out


def ramp_rise_best(stage, rise):
    """render4: running max of the wall height (m) over the RAMP frames (the height bar's peak-hold line)."""
    r = np.where(np.asarray(stage) == 4, np.asarray(rise, float), 0.0)
    return np.maximum.accumulate(r) if len(r) else r


def ramp_finish(reason: str, end_stage: int) -> bool:
    """render4: the half-pipe's normal finish (ramp_done: RAMP time or apex budget used up; or time_out in RAMP) is a
    success on the end card, not "ended early"."""
    rs_ = reason.split(",")
    return "ramp_done" in rs_ or (end_stage == 4 and all(r in ("time_out", "ramp_done") for r in rs_))


def ramp_card(passes: int, best_cm: float) -> str:
    return f"half-pipe · {passes} pass{'' if passes == 1 else 'es'} · best {best_cm:.1f} cm"


def stage_telemetry(rs, origin, stg: int, v: float, pitch: float, rise_m: float, t: float | None = None,
                    setdown: bool = False, best_m: float | None = None):
    """render5: only what the stage is about. Speed always; nose-up (+ board glyph) and the tail gap in MANUAL,
    nose-up in the SET-DOWN until the four wheels are down (setdown); the wall height in HALF-PIPE. No clock (t is
    ignored; kept for callers)."""
    tg = rs.tail_gap(origin) if stg == 2 else None
    show_pitch = stg == 2 or (stg == 3 and not setdown)
    return telemetry(v, pitch if show_pitch else None, rise_m * 100 if stg == 4 else None,
                     None if tg is None else max(tg, 0.0) * 100,
                     best_cm=best_m * 100 if (stg == 4 and best_m is not None and best_m * 100 >= 0.95) else None)


def opening_alpha(tc: float, hold_s: float = TITLE_HOLD_S, out_s: float = TITLE_OUT_S) -> float:
    """Title card opacity at clip time tc (the intro still included): full from frame 0, fades out over out_s ending
    at hold_s. The persistent caption (same anchor) uses 1 - this."""
    return _smoothstep((hold_s - tc) / out_s)


def intro_still_frames(dt: float) -> int:
    return max(0, int(round(INTRO_STILL_S / dt)))


def clip_start(stage, v, dt: float, v_start: float = PUSH_V_START, lead_s: float = PUSH_LEAD_S) -> int:
    """render5: first episode frame of a clip. A clip that starts at rest in PUSH (the start spawn) begins lead_s
    before the board first passes v_start (the first push), so it holds INTRO_STILL_S still and then pushes, instead
    of ~1 s of standstill. 0 for any other clip."""
    stage, v = np.asarray(stage), np.asarray(v, float)
    if len(stage) == 0 or int(stage[0]) != 0 or abs(float(v[0])) > 0.5 * v_start:
        return 0
    hit = np.nonzero(v > v_start)[0]
    if not len(hit):
        return 0
    return max(0, int(hit[0]) - int(round(lead_s / dt)))


def intro_hold_frames(stage, v, dt: float, k0: int, total_s: float = START_STILL_S, v_start: float = PUSH_V_START) -> int:
    """render7: held copies of the clip's first frame k0 so that total_s of clip time passes before the first push
    (the board first passing v_start): the real frames k0 .. push come first in the count, a held frame only tops up
    what the episode does not have (a start spawn pushes ~0.5 s in). 0 when the clip does not start at rest."""
    stage, v = np.asarray(stage), np.asarray(v, float)
    if len(stage) == 0 or int(stage[0]) != 0:
        return 0
    hit = np.nonzero(v > v_start)[0]
    if not len(hit):
        return 0
    return max(0, int(round(total_s / dt)) - max(0, int(hit[0]) - int(k0)))


def draw_title(dr, F, tc: float, title: str, sub: str = "", caption: bool = True):
    """render5: the title pinned top-left. A title card (title + optional sub line on a dark panel) from frame 0 that
    fades out by TITLE_HOLD_S of clip time, handing over to a small persistent caption at the same anchor."""
    if not title:
        return
    a_card = opening_alpha(tc)
    a_cap = _smoothstep((tc - (TITLE_HOLD_S - 0.12)) / 0.3)   # caption fades in as the card is ~gone (no double text)
    if a_cap > 0.0 and caption:                        # persistent caption, same anchor as the card (render9: optional)
        _pill_text(dr, HUD_X, HUD_Y, title, F["cap"], (255, 255, 255), 255 * a_cap)
    if a_card > 0.0:
        pad = 18
        tw = dr.textlength(title, font=F["title"])
        # render10 (2026-11-22): a title wider than the frame (the foot-push title) shrinks to fit; fitting titles unchanged
        W_ = dr.im.size[0]
        if tw + 2 * pad > W_ - 2 * HUD_X and hasattr(F["title"], "font_variant"):
            F = dict(F, title=F["title"].font_variant(size=max(12, int(F["title"].size * (W_ - 2 * HUD_X - 2 * pad) / tw))))
            tw = dr.textlength(title, font=F["title"])
        sw = dr.textlength(sub, font=F["sub"]) if sub else 0
        pw = max(tw, sw) + 2 * pad
        ph = pad + F["title"].size + (10 + F["sub"].size if sub else 0) + pad + 6
        a = 255 * a_card
        dr.rounded_rectangle([HUD_X, HUD_Y, HUD_X + pw, HUD_Y + ph], radius=16,
                             fill=PANEL_RGBA[:3] + (int(200 * a_card),))
        _txt(dr, (HUD_X + pad, HUD_Y + pad - 8), title, F["title"], (255, 255, 255), a)
        if sub:
            _txt(dr, (HUD_X + pad + 2, HUD_Y + pad + F["title"].size + 6), sub, F["sub"], (226, 230, 238), a,
                 shadow=False)


def title_overlay(dr, F, t: float, title: str, sub: str = "", fade_s: float = TITLE_FADE_S,
                  hold_s: float = TITLE_HOLD_S):
    """Title at clip time t (the intro still included): draw_title (render5: card -> caption, pinned top-left).
    fade_s > 0 restores a fade from black. Draw it last."""
    if t < fade_s:
        dr.rectangle([0, 0, RENDER_W, RENDER_H], fill=(6, 8, 12, int(255 * (1.0 - _smoothstep(t / fade_s)))))
    draw_title(dr, F, t, title, sub)


def _result_icon(dr, cx, cy, r, ok, a):
    col = (52, 168, 92) if ok else (210, 72, 58) if ok is False else (90, 100, 120)
    dr.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col + (int(a),))
    w = max(3, r // 4)
    if ok:                                             # check mark (the UI font has no U+2713)
        dr.line([cx - 0.5 * r, cy + 0.02 * r, cx - 0.12 * r, cy + 0.42 * r, cx + 0.55 * r, cy - 0.38 * r],
                fill=(255, 255, 255, int(a)), width=w, joint="curve")
    elif ok is False:
        for sx in (-1, 1):
            dr.line([cx - 0.4 * r, cy - sx * 0.4 * r, cx + 0.4 * r, cy + sx * 0.4 * r], fill=(255, 255, 255, int(a)),
                    width=w)


def _font_sized(font, size: int):
    try:
        return font.font_variant(size=size)
    except Exception:                                  # bitmap default font: no variants
        return font


def card_fonts(dr, F, title: str, sub: str, foot_r: str = ""):
    """render6: end-card fonts shrunk until the result panel fits beside the bottom-right note (or the frame width
    without one): the card text never collides with a footer / label."""
    note_w = (dr.textlength(foot_r, font=F["note"]) + 28 + 16) if foot_r else 0
    avail = RENDER_W - 2 * HUD_X - note_w
    fc, fs = F["card"], F["card_sub"]
    for _ in range(12):
        r = fc.size // 2 + 2
        need = 2 * 18 + 2 * r + 16 + max(dr.textlength(title, font=fc), dr.textlength(sub, font=fs) if sub else 0)
        if need <= avail or fc.size <= 26:
            break
        fc, fs = _font_sized(fc, int(fc.size * 0.92)), _font_sized(fs, max(18, int(fs.size * 0.92)))
    return fc, fs


def credits_frames(lines, dt, F=None, hold_s: float = CREDITS_S, fade_s: float = 0.35):
    """render6: closing card after the end card: a small centred credits block on the dark frame the end card faded
    to. lines: [(text, style)] with style "head" (semi 30, white), "body" (26, light grey) or "fine" (22, grey); fades
    in / out over fade_s. Text wider than the frame is shrunk to fit."""
    from PIL import Image, ImageDraw
    F = F or hud_fonts()
    st = dict(head=(F["cred"], (255, 255, 255)), body=(F["cred_s"], (214, 220, 230)), fine=(F["cred_xs"], (160, 168, 182)))
    W, H = RENDER_W, RENDER_H
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    rows = []
    for txt, sty in lines:
        f, col = st[sty]
        while probe.textlength(txt, font=f) > W - 2 * 60 and f.size > 14:
            f = _font_sized(f, f.size - 1)
        rows.append((txt, f, col, 30 if sty == "head" else 26 if sty == "body" else 14))
    tot = sum(f.size + gap for _, f, _, gap in rows) + 8
    n = max(1, int(round(hold_s / dt)))
    nf = max(1, int(round(fade_s / dt)))
    for i in range(n):
        a = 255 * min(_smoothstep((i + 1) / nf), _smoothstep((n - i) / nf))
        im = Image.new("RGB", (W, H), (6, 8, 12))
        dr = ImageDraw.Draw(im, "RGBA")
        y = (H - tot) / 2
        for txt, f, col, gap in rows:
            tw = dr.textlength(txt, font=f)
            dr.text(((W - tw) / 2, y), txt, font=f, fill=tuple(col) + (int(a),))
            y += f.size + gap
        yield np.asarray(im)


def end_card_frames(last_rgb, title, sub, dt, hold_s=None, F=None, ok: bool | None = None, foot_l: str = "",
                    foot_r: str = "", tag: str | None = None):
    """render5 end card, hold_s (default END_CARD_S = 1.5 s): a freeze of the last frame with a slow push-in; the
    result appears in the stage panel's place (bottom-left: check / cross icon, result word, optional sub line); the
    caption (foot_l, top-left), the honesty note (foot_r, bottom-right) and the tag stay; the last END_FADE_S fade to
    black."""
    from PIL import Image, ImageDraw
    F = F or hud_fonts()
    hold_s = END_CARD_S if hold_s is None else hold_s
    base = Image.fromarray(last_rgb)
    W, H = base.size
    n = max(1, int(round(hold_s / dt)))
    n_fade = min(n, int(round(END_FADE_S / dt)))
    for i in range(n):
        p = i / max(1, n - 1)
        z = 1.0 + 0.025 * _smoothstep(p)
        cw, ch = W / z, H / z
        im = base.crop(((W - cw) / 2, (H - ch) / 2, (W + cw) / 2, (H + ch) / 2)).resize((W, H), Image.BILINEAR)
        dr = ImageDraw.Draw(im, "RGBA")                 # RGBA ink on an RGB image: every primitive blends
        draw_note(dr, F, foot_r)
        draw_title(dr, F, 1e9, foot_l)                  # the persistent caption
        if tag:
            tw_ = dr.textlength(tag, font=F["chip"]) + 24
            _chip(dr, W - HUD_X - tw_, HUD_Y, tag, F["chip"], (40, 48, 64))
        a = 255 * _smoothstep(min(1.0, (i + 1) * dt / 0.25))
        fc, fs = card_fonts(dr, F, title, sub, foot_r)
        pad, r = 18, fc.size // 2 + 2
        tw = dr.textlength(title, font=fc)
        sw = dr.textlength(sub, font=fs) if sub else 0
        pw = 2 * pad + 2 * r + 16 + max(tw, sw)
        ph = 2 * pad + fc.size + (fs.size + 10 if sub else 0) + 6
        x0, y0 = HUD_X, H - HUD_BOTTOM - ph
        dr.rounded_rectangle([x0, y0, x0 + pw, y0 + ph], radius=16, fill=(10, 14, 22, int(190 * a / 255)))
        _result_icon(dr, x0 + pad + r, y0 + pad + fc.size // 2 + 6, r, ok, a)
        tx = x0 + pad + 2 * r + 16
        _txt(dr, (tx, y0 + pad - 4), title, fc, (255, 255, 255), a)
        if sub:
            _txt(dr, (tx + 2, y0 + pad + fc.size + 8), sub, fs, (220, 226, 236), a, shadow=False)
        j = i - (n - n_fade)
        if j >= 0:                                      # slow fade to black over the last END_FADE_S
            dr.rectangle([0, 0, W, H], fill=(0, 0, 0, int(255 * _smoothstep((j + 1) / max(1, n_fade)))))
        yield np.asarray(im)


FAST_MIN_S = 2.5                      # s: render6: a flat coasting stretch (CARVE IN, after the ramp camera move, until
FAST_TAIL_S = 0.5                     #    FAST_TAIL_S before the next stage) at least this long plays at 2x, labelled
FAST_FLAT_M = 0.005                   # m: "flat" = deck rise below this
FAST_LEAD_S = 1.0                     # s: render7: 2x starts this long after the carve camera move starts (the
#                                       1.4 s smoothstep is ~80 % done: at 0.3 s the crane-out played too fast)
FAST_MIN_S_CHAIN = 2.5                # s: render7: chain videos (carve camera) play CARVE IN stretches >= this at 2x
#                                       (render8: 1.2 -> 2.5: with the tighter carve cam the carve-in is no longer dead
#                                       footage; only a long coast still plays at the labelled 2x)
PIPE_FLAT_MIN_S = 0.8                 # s: render9: 1.0 -> 0.8, every flat coast between the walls (2x minimum 0.5 -> 0.2,
#                                       the ramp camera exclusion now ends with the ramp camera blend). render8: a HALF-PIPE stretch on the flat (rise < FAST_FLAT_M) longer than this
PIPE_FLAT_EDGE_S = 0.3                #    plays at the labelled 2x, except PIPE_FLAT_EDGE_S at both ends (1x off / onto
PIPE_FLAT_2X_MIN_S = 0.2              #    the walls) and the ramp camera move; only if >= PIPE_FLAT_2X_MIN_S is left


def fast_mask(stage, labels, rise, t_wd, dt: float, min_s: float = FAST_MIN_S, tail_s: float = FAST_TAIL_S,
              keys=None, lead_s: float = FAST_LEAD_S, pipe_flat: bool = False):
    """render6: per frame True = shown at 2x (every other frame dropped, a "2x speed" chip on screen). Only CARVE IN
    (the coast from the set-down to the pipe, no stage change), on the flat, from the end of the ramp camera move
    (t_wd + RAMP_HOLD_S + CAM_BLEND["ramp"]) to tail_s before the label changes; a stretch shorter than min_s stays 1x.
    Time is never cut, only played faster with the label.
    keys (render7, cam_keys(carve_cam=True)): the stretch starts lead_s after the carve camera move starts (the
    move finishes at 2x) instead of after the ramp camera move: chain_r6's demo carve-in (3.5 s of CARVE IN) never
    reached FAST_MIN_S between the end of the 2.6 s profile hold + ramp move and the RAMP entry, so it played at 1x."""
    T = len(stage)
    ok = np.zeros(T, bool)
    kc = next((k for k in range(T) if keys[k] == "carve"), None) if keys is not None else None
    for k in range(T):
        if kc is not None:                                # render8: never during the ramp camera move
            t_ok = k >= kc + int(round(lead_s / dt)) and keys[k] == "carve"
        else:
            t_ok = not np.isnan(t_wd[k]) and k * dt >= t_wd[k] + RAMP_HOLD_S + CAM_BLEND["ramp"]
        ok[k] = labels[k] == "CARVE IN" and float(rise[k]) < FAST_FLAT_M and t_ok
    out = np.zeros(T, bool)
    k = 0
    while k < T:
        if not ok[k]:
            k += 1
            continue
        j = k
        while j < T and ok[j]:
            j += 1
        j_end = j - int(round(tail_s / dt)) if j < T else j    # 1x into the next stage
        if (j_end - k) * dt >= min_s:
            out[k:j_end] = True
        k = j
    if pipe_flat:                                         # render8: flat stretches between the walls
        stage = np.asarray(stage).astype(int)
        k4 = next((k for k in range(T) if stage[k] >= 4), T)
        k_ok = k4 + int(round((CAM_BLEND["ramp"] + 0.2) / dt))        # not during the ramp camera move
        kr = next((k for k in range(T) if keys[k] == "ramp"), None) if keys is not None else None
        if kr is not None:                                # render9: the move's real end (it starts before RAMP entry)
            k_ok = kr + int(round((CAM_BLEND["ramp"] + 0.2) / dt))
        flat = (stage == 4) & (np.asarray(rise, float) < FAST_FLAT_M)
        ed = int(round(PIPE_FLAT_EDGE_S / dt))
        k = 0
        while k < T:
            if not flat[k]:
                k += 1
                continue
            j = k
            while j < T and flat[j]:
                j += 1
            if j < T and (j - k) * dt > PIPE_FLAT_MIN_S:                # a stretch the clip leaves (not its end)
                a_, b_ = max(k + ed, k_ok), j - ed
                if (b_ - a_) * dt >= PIPE_FLAT_2X_MIN_S:
                    out[a_:b_] = True
            k = j
    return out


def clip_frames(rs, F, L, idx, origin, stage, labels, cams, setdown, dt, k0=0, title="", sub="", note="",
                tag_fn=None, fast=None, n_still=None, est_cams=None, est_mark=None, caption=True, ptag_fn=None,
                fast_big=False):
    """render5: the clip's frames, shared by render() and chain_play.render_chain(): INTRO_STILL_S of frame k0 (the
    title card), then frames k0 .. len(stage)-1 with the stage panel (label, chips, per-stage telemetry, height bar in
    HALF-PIPE), the pinned title card / caption top-left, the honesty note bottom-right and an optional tag
    (tag_fn(k) -> str) top-right. fast (render6, fast_mask): frames played at 2x (every other one dropped, "2x speed"
    chip). Yields (frame, raw render without the HUD).
    render9 (chain clips): est_cams = one camera per held intro frame (the establishing shot, rendered with the far fog,
    title card only + an optional "half-pipe" pointer at world point est_mark; no stage panel), then a cut to the clip
    camera at frame k0; caption=False drops the persistent title caption after the card; ptag_fn(k) -> the small
    policy tag inside the stage panel; fast_big = the large centre-bottom 2x badge instead of the chip."""
    from PIL import Image, ImageDraw
    n_still = intro_still_frames(dt) if n_still is None else int(n_still)   # render7: chain = intro_hold_frames
    rise_best = ramp_rise_best(stage, L["rise"][:len(stage), idx])
    # render8: deck contact chips. TAIL DOWN = the env's tail_floor sensor or a deck_tail_* geom touching the terrain in
    # the render copy; DECK SCRAPE = the deck core / nose geoms touching it (review B: the deck core brushed the wall at
    # the apexes with no chip). Each contact keeps its chip on screen DECK_CHIP_HOLD_S (a 2-step brush is 40 ms).
    T_ = len(stage)
    tail_on, scrape_on = np.zeros(T_, bool), np.zeros(T_, bool)
    for k in range(k0, T_):
        rs.set_state(L["qpos"][k, idx])
        tch = rs.deck_touch()
        tail_on[k] = bool(L["tail"][k, idx]) or "tail" in tch
        scrape_on[k] = bool(tch & {"core", "nose"})
    hold = max(1, int(round(DECK_CHIP_HOLD_S / dt)))

    def _held(x):
        y = x.copy()
        for k in np.nonzero(x)[0]:
            y[k:k + hold] = True
        return y
    tail_chip, scrape_chip = _held(tail_on), _held(scrape_on) & ~_held(tail_on)
    print(f"[render] deck contacts in the clip: tail {int(tail_on.sum())} steps, core/nose {int(scrape_on.sum())} steps "
          f"(at {', '.join(f'{k * dt:.2f}' for k in np.nonzero(scrape_on)[0][:12])} s)", flush=True)
    lab, lab_t0 = None, 0.0
    run0 = None
    for k in range(k0, len(stage)):
        is_fast = fast is not None and bool(fast[k])
        run0 = (k if run0 is None else run0) if is_fast else None
        if is_fast and (k - run0) % 2 == 1:
            continue
        rs.set_state(L["qpos"][k, idx])
        stg, t = int(stage[k]), k * dt
        raw = rs.render(*cams[k])
        if labels[k] != lab:                            # the first label is on screen from frame 0 (no fade-in)
            lab, lab_t0 = labels[k], (t if lab is not None else t - 1.0)
        chips = []
        if L["manual"][k, idx] and labels[k] == "MANUAL":  # render6: MANUAL only (render8: the MANUAL label)
            chips.append(("NOSE UP", (38, 150, 84)))
        if tail_chip[k]:
            chips.append(("TAIL DOWN", (200, 62, 48)))
        elif scrape_chip[k]:
            chips.append(("DECK SCRAPE", (200, 62, 48)))
        if is_fast and not fast_big:
            chips.append(("2× speed", (214, 120, 24)))
        pitch = float(L["pitch"][k, idx])
        tele = stage_telemetry(rs, origin, stg, float(L["v"][k, idx]), pitch, float(L["rise"][k, idx]),
                               setdown=bool(setdown[k]), best_m=float(rise_best[k]))
        tag = tag_fn(k) if tag_fn else None
        ptag = ptag_fn(k) if ptag_fn else None
        for j in range(n_still + 1 if k == k0 else 1):
            tc = (j if k == k0 else n_still + k - k0) * dt          # clip time (intro still included)
            est = k == k0 and est_cams is not None and j < len(est_cams) and j < n_still
            raw_j = rs.render(*est_cams[j], far=True) if est else raw
            im = Image.fromarray(raw_j)
            dr = ImageDraw.Draw(im, "RGBA")              # RGBA ink on the RGB frame: every primitive blends
            if est:                                      # render9: establishing shot: title card (+ pointer) only
                if est_mark is not None:
                    px = cam_project(est_cams[j], est_mark, EST_FOVY)
                    a_ = 255 * _smoothstep((tc - 0.25) / 0.3)
                    if px is not None and 0 < px[0] < RENDER_W and 0 < px[1] < RENDER_H and a_ > 0:
                        s_ = "half-pipe ahead"
                        w_ = dr.textlength(s_, font=F["est"]) + 28
                        bx, by = px[0] - w_ / 2, px[1] - 92
                        dr.line([px[0], by + F["est"].size + 18, px[0], px[1] - 8], fill=(255, 255, 255, int(a_)),
                                width=3)
                        _pill_text(dr, bx, by, s_, F["est"], (255, 255, 255), a_)
                draw_title(dr, F, tc, title, sub, caption=caption)
                yield np.asarray(im), raw_j
                continue
            right = draw_hud(dr, F, lab, tele, chips, label_a=255 * min(1.0, (t - lab_t0) / 0.4), tag=tag,
                             foot_r=note, pitch=pitch, height=None, ptag=ptag)  # render8: no unlabelled gauge
            if is_fast and fast_big:
                draw_fast_badge(dr, F, right)
            draw_title(dr, F, tc, title, sub, caption=caption)
            yield np.asarray(im), raw


def render(env, L, end, idx, out, caption, label, dt, end_msg="episode ended (fall / foot down / lost board)",
           hold_s=None, end_ok: bool | None = None, end_sub: str = "", carve: bool = True, profile: bool = True):
    """caption: the small honesty note (e.g. "typical of 32"); label: the clean title (title card + caption);
    profile: cam_keys(profile=...) (False for the push specialist: the rear three-quarter holds to the end)."""
    import imageio.v2 as imageio

    rs = RenderScene.get(env)
    origin = env.scene.env_origins[idx].detach().cpu().numpy()
    F = hud_fonts()
    e = int(end[idx])
    title = label or "Microduck skate sequence"
    stage = L["stage"][:e, idx].astype(int)
    setdown, t_wd = setdown_track(stage, L["wheels"][:e, idx], dt)
    keys = cam_keys(stage, setdown, t_wd, dt, ramp_ok=rs.pipe is not None, profile=profile)
    labels = stage_labels(stage, L["cmd"][:e, idx] if "cmd" in L else None,
                          L["both_on"][:e, idx] if "both_on" in L else None, setdown, dt, carve=carve)
    poses = [rs.set_state(L["qpos"][k, idx]) for k in range(e)]
    cams = CamRig(dt, origin).plan(keys, [p[0] for p in poses], [p[1] for p in poses])
    k0 = clip_start(stage, L["v"][:e, idx], dt)
    hold_s = END_CARD_S if hold_s is None else hold_s
    r0 = next((k for k, lb in enumerate(labels) if lb == "RIDE"), None)
    ride_s = 0.0 if r0 is None else (e - r0) * dt
    print(f"[render] {os.path.basename(out)}: episode {e * dt:.2f} s, clip from {k0 * dt:.2f} s (+ {INTRO_STILL_S:.1f} s "
          f"still), riding on screen {ride_s:.2f} s before the end (+ {hold_s:.2f} s end card)", flush=True)
    last = None
    with imageio.get_writer(out, fps=int(round(1 / dt)), codec="libx264", quality=8, macro_block_size=8) as w:
        for fr, raw in clip_frames(rs, F, L, idx, origin, stage, labels, cams, setdown, dt, k0=k0, title=title,
                                   note=caption):
            w.append_data(fr)
            last = raw                                   # the end card starts from the frame without the HUD
        if last is not None and e < L["qpos"].shape[0]:
            ok = end_msg.startswith("success") if end_ok is None else end_ok
            for fr in end_card_frames(last, end_msg, end_sub, dt, hold_s, F, ok=ok, foot_l=title, foot_r=caption):
                w.append_data(fr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--task", default=TASK)
    ap.add_argument("--spawns", default=None,
                    help="default: the chain task's start,push,glide,manual,exit,ramp_bottom,ramp_wall; another task's "
                         "spawn types with a non-zero probability")
    ap.add_argument("--seconds", type=float, default=18.0)
    ap.add_argument("--render-dir", default="")
    ap.add_argument("--render", default="typical,best")
    ap.add_argument("--render-spawns", default="start")
    ap.add_argument("--label", default="")
    ap.add_argument("--json", default="")
    ap.add_argument("--manual-tilt-prob", type=float, default=None,
                    help="override reset_sequence manual_tilt_prob (e.g. 1.0: every MANUAL spawn tilted)")
    ap.add_argument("--curriculum-step", type=int, default=None,
                    help="set env.common_step_counter after the checkpoint load (which restores the checkpoint's own "
                         "counter, i.e. the curricula's final stages); 0 = the conditions of a warm start's iteration 0")
    a = ap.parse_args()
    vec, policy = build(a.ckpt, a.n, a.task)
    env = vec.unwrapped
    dt = env.step_dt
    if a.curriculum_step is not None:
        env.common_step_counter = a.curriculum_step
    if a.manual_tilt_prob is not None:
        env.event_manager.get_term_cfg("reset_sequence").params["manual_tilt_prob"] = a.manual_tilt_prob
    if a.spawns is None:
        if a.task == TASK:
            a.spawns = "start,push,glide,manual,exit,ramp_bottom,ramp_wall"
        else:
            from micro_skater.tasks import skate_sequence as sq
            probs = env.event_manager.get_term_cfg("reset_sequence").params["spawn_probs"]
            a.spawns = ",".join(s for s in sq.SPAWNS if probs.get(s, 0.0) > 0)
    print(f"task {a.task}  spawns {a.spawns}", flush=True)
    res = {}
    for sp in a.spawns.split(","):
        L, end, reason, apex_n, apex_rise = run_spawn(vec, policy, sp, a.seconds)
        agg, eps = funnel(L, end, reason, apex_n, apex_rise, dt)
        res[sp] = agg
        print(f"RESULT {sp}: " + json.dumps(agg), flush=True)
        if a.render_dir and sp in a.render_spawns.split(","):
            os.makedirs(a.render_dir, exist_ok=True)
            order = np.argsort([r["score"] for r in eps])
            picks = {"best": int(order[-1]), "typical": int(order[len(order) // 2])}
            for which in a.render.split(","):
                idx = picks[which]
                out = os.path.join(a.render_dir, f"sequence_{sp}_{which}.mp4")
                cap = f"{which} of {a.n} robots"              # small honesty note (footer right)
                r_ = eps[idx]
                ok_word = ("manual" if "SeqManual" in a.task else "on the board" if "SeqPush" in a.task else "landed")
                msg = ok_word if r_["success"] else "ended early"
                sub = "" if r_["success"] else r_["reason"].replace("_", " ").replace(",", ", ")
                if r_["success"] and r_.get("band_longest_s"):
                    sub = f"nose up {r_['band_longest_s']:.1f} s · four wheels down"
                e_i = int(end[idx])
                ok_ = bool(r_["success"])
                # render4; render5: no green check for a half-pipe with 0 passes
                if e_i > 0 and ramp_finish(r_["reason"], int(L["stage"][e_i - 1, idx])) and r_["apexes"] >= 1:
                    msg, sub, ok_ = ramp_card(r_["apexes"], r_["apex_rise_cm"]), "", True
                render(env, L, end, idx, out, cap, a.label, dt, end_msg=msg, end_ok=ok_, end_sub=sub,
                       carve="SeqManual" not in a.task and "SeqPush" not in a.task, profile="SeqPush" not in a.task)
                print("wrote", out, json.dumps(eps[idx]), flush=True)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
