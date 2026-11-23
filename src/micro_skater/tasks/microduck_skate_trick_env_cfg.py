# micro-skater: part of the author's skateboard add-on for microduck_rl. Apache-2.0 (see LICENSE).
"""Skateboard TRICKS on the double-kick board (2026-11-19). Task family Mjlab-Skate-Trick-*-MicroDuck.

The rider starts ALREADY ON the board in a SIDEWAYS stance (skate_trick.py docstring); one multi-trick policy, the trick
drawn per episode and given as a one-hot (ride/carve incl. fakie, tail stop, kick-turn, tic-tac, manual).
Stances: data/stance_poses.json; board: robot/assets/skateboard_doublekick.xml.

Built on the forward-stance skate TEACHER cfg (make_microduck_skate_env_cfg) for its proven obs noise / delays / IMU
misalignment / encoder bias / BAM + DR event stack (deep-copied), with everything tied to the push-stance mode machine
removed. Obs: the 61D family block first (twist / head / body command slots zero-padded with tiny ranges),
then the 28D board block and the 11D trick block. Privileged board state (a teacher; a sensor-only student is a later
distillation, as for the cruiser).
"""

from __future__ import annotations
from micro_skater.paths import data_path as _data_path

import math
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers import CurriculumTermCfg, EventTermCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from micro_skater.robot.microduck_skate_trick_robot import MICRODUCK_SKATE_TRICK_ROBOT_CFG
from micro_skater.robot.skateboard_variants import MICRODUCK_SKATEBOARD_DOUBLEKICK_CFG
from micro_skater.tasks import mdp as microduck_mdp
from micro_skater.tasks import skate_trick as tk
from micro_skater.tasks.microduck_skate_env_cfg import make_microduck_skate_env_cfg
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg

EPISODE_LENGTH_S = 8.0
ACTION_SCALE = 0.5
INIT_STD = 0.5
TRUCK_K_RANGE = (0.8, 1.0)          # N m/rad; the board's armature is sized for <= 1.0
ENABLE_PUSHES = False
BOARD_ROLLING_RANGE = (0.3, 1.0)   # x wheel rolling friction (teacher 0.5-2.0): the rider cannot push here

# Feasibility: only ride/carve/fakie pass in the sideways stance; the others are kept as RL attempts.
TRICK_PROBS_ALL = dict(ride=0.5, stop=0.125, kickturn=0.125, tictac=0.125, manual=0.125)
SPAWN_SPEEDS = dict(ride=(0.25, 0.8), stop=(0.3, 0.8), kickturn=(0.0, 0.08), tictac=(0.0, 0.0), manual=(0.3, 0.6))
ACTION_RATE_STAGES = [{"step": 0, "weight": -0.01}, {"step": 1000 * 24, "weight": -0.05},
                      {"step": 2000 * 24, "weight": -0.1}]

_DECK_GEOMS = r"^deck.*_collision$"
_ROBOT_BODY_NO_FEET = r"^(trunk_base|hip_l|leg|hip_l_2|leg_2|jaw_soft)$"
HEAD_JOINTS = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")
ALL_SENSORS = ("feet_ground_contact", "left_deck", "right_deck", "body_board", "front_wheels_floor",
               "rear_wheels_floor", "nose_floor", "tail_floor")


def _sensors():
    terrain = ContactMatch(mode="body", pattern="terrain")

    def floor(name, pattern, entity, fields=("found",)):
        return ContactSensorCfg(name=name, primary=ContactMatch(mode="geom", pattern=pattern, entity=entity),
                                secondary=terrain, fields=fields, reduce="netforce", num_slots=1)

    def deck(name, sole):
        # primaries = the 4 sole spheres (a regex SECONDARY keeps only its first match in mjlab 1.3); the spheres can
        # only touch deck geoms (contype/conaffinity bit 4), so the board subtree secondary is exact.
        return ContactSensorCfg(name=name, primary=ContactMatch(mode="geom", pattern=sole, entity="robot"),
                                secondary=ContactMatch(mode="subtree", pattern="skateboard", entity="board"),
                                fields=("found", "force"), reduce="netforce", num_slots=1, track_air_time=True)

    feet = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="geom", pattern=r"^(left_foot_collision|right_foot_collision)$", entity="robot"),
        secondary=terrain, fields=("found", "force"), reduce="netforce", num_slots=1, track_air_time=True)
    body_board = ContactSensorCfg(
        name="body_board", primary=ContactMatch(mode="body", pattern=_ROBOT_BODY_NO_FEET, entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="skateboard", entity="board"), fields=("found",),
        reduce="netforce", num_slots=1)
    return (feet, deck("left_deck", r"^left_sole_deck_s[0-3]_collision$"),
            deck("right_deck", r"^right_sole_deck_s[0-3]_collision$"),
            body_board, floor("front_wheels_floor", r"^front_wheel_[LR]_collision$", "board"),
            floor("rear_wheels_floor", r"^rear_wheel_[LR]_collision$", "board"),
            floor("nose_floor", r"^deck_nose_.*_collision$", "board"),
            floor("tail_floor", r"^deck_tail_.*_collision$", "board"))


def make_microduck_skate_trick_env_cfg(play: bool = False, trick_probs: dict | None = None, board_cfg=None,
                                       stance_key: str = "regular", stance_path: str = "", axle_x: float = 0.055,
                                       dir_mode: str = "random", speeds: dict | None = None,
                                       alt_spawns: dict | None = None, nose_lift_w: float = 0.0) -> ManagerBasedRlEnvCfg:
    """Defaults = the sideways double-kick task; the forward-stance variant passes its board / stance / axle."""
    teacher = make_microduck_skate_env_cfg(play=play)
    cfg = make_velocity_env_cfg()

    # ── scene / sim ──
    cfg.scene.entities = {"robot": MICRODUCK_SKATE_TRICK_ROBOT_CFG, "board": board_cfg or MICRODUCK_SKATEBOARD_DOUBLEKICK_CFG}
    cfg.scene.sensors = _sensors()
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.sim = deepcopy(teacher.sim)
    cfg.viewer.body_name = "trunk_base"
    cfg.episode_length_s = EPISODE_LENGTH_S

    # ── actions: all 14 servos, centred on the sideways stance (head/neck on HOME) ──
    a0 = cfg.actions["joint_pos"]
    cfg.actions["joint_pos"] = tk.StanceJointPositionActionCfg(
        entity_name=a0.entity_name, actuator_names=a0.actuator_names, scale=ACTION_SCALE, use_default_offset=True,
        stance_key=stance_key, stance_path=stance_path)

    # ── command: zero-padded twist slot (tiny non-zero ranges, dead-weights rule) ──
    tw = cfg.commands["twist"]
    assert isinstance(tw, UniformVelocityCommandCfg)
    tw.rel_standing_envs, tw.rel_heading_envs = 0.0, 0.0
    tw.heading_command = False
    tw.ranges.heading = None
    tw.ranges.lin_vel_x = (-0.01, 0.01)
    tw.ranges.lin_vel_y = (-0.01, 0.01)
    tw.ranges.ang_vel_z = (-0.01, 0.01)
    tw.debug_vis = False

    # ── observations: teacher's 61D family terms (noise / delays / IMU misalignment / bias), then board + trick ──
    family = ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions", "command", "head_command",
              "body_command")
    critic_family = ("base_lin_vel",) + family[:6] + ("foot_air_time", "foot_contact", "foot_contact_forces") + family[6:]
    for grp, names in (("actor", family), ("critic", critic_family)):
        terms = cfg.observations[grp].terms
        terms.clear()
        for n_ in names:
            terms[n_] = deepcopy(teacher.observations[grp].terms[n_])
        terms["board"] = ObservationTermCfg(func=tk.board_obs)
        terms["trick"] = ObservationTermCfg(func=tk.trick_obs)

    # ── events: teacher DR stack minus the push-stance machinery, own truck DR, own spawn LAST ──
    cfg.events = deepcopy(teacher.events)
    for k in ("skate_mode_setup", "skate_slope_setup", "reset_skate", "board_truck_stiffness"):
        cfg.events.pop(k, None)
    if not ENABLE_PUSHES:
        cfg.events.pop("push_robot", None)
    ff = cfg.events["foot_friction"].params["asset_cfg"]
    ff.geom_names = ("left_foot_collision", "right_foot_collision", r"^(left|right)_sole_deck_s[0-3]_collision$")
    cfg.events["board_rolling"].params["ranges"] = BOARD_ROLLING_RANGE
    cfg.events["board_truck_stiffness"] = EventTermCfg(
        func=dr.joint_stiffness, mode="reset",
        params={"asset_cfg": SceneEntityCfg("board", joint_names=(r"^passive_(front|rear)_truck$",)),
                "operation": "abs", "ranges": TRUCK_K_RANGE, "shared_random": True})
    cfg.events["reset_trick"] = EventTermCfg(
        func=tk.reset_trick_state, mode="reset",
        params={"trick_probs": dict(trick_probs or TRICK_PROBS_ALL), "speeds": dict(speeds or SPAWN_SPEEDS),
                "stance_key": stance_key, "stance_path": stance_path or str(tk.STANCE_JSON), "axle_x": axle_x,
                "dir_mode": dir_mode, "alt_spawns": dict(alt_spawns or {})})

    # ── rewards ──
    keep = {k: cfg.rewards[k] for k in ("dof_pos_limits", "action_rate_l2")}
    cfg.rewards.clear()
    cfg.rewards.update(keep)
    cfg.rewards["action_rate_l2"].weight = ACTION_RATE_STAGES[0]["weight"]
    R = RewardTermCfg
    cfg.rewards.update({
        "upright": R(func=tk.trunk_upright, weight=1.0),
        "on_deck": R(func=tk.on_deck, weight=0.5),
        "fall": R(func=tk.trick_fall, weight=-10.0),
        "body_on_board": R(func=tk.contact_cost, weight=-1.0, params={"sensor_name": "body_board"}),
        "head_pose": R(func=tk.joint_dev_l2, weight=-0.5, params={"joint_names": HEAD_JOINTS}),
        "ride": R(func=tk.ride_track, weight=3.0),
        "stop": R(func=tk.stop_track, weight=3.0),
        "stop_tail": R(func=tk.stop_tail_drag, weight=0.5),
        "kickturn": R(func=tk.kickturn_track, weight=3.0),
        "kickturn_done": R(func=tk.kickturn_done, weight=3.0),
        "tictac": R(func=tk.tictac_track, weight=3.0),
        "manual": R(func=tk.manual_in, weight=5.0),
        "manual_shape": R(func=tk.manual_shape, weight=1.0),
        "nose_lift": R(func=tk.nose_lift, weight=nose_lift_w),
        "manual_strike": R(func=tk.manual_strike, weight=-1.0),
    })

    # ── terminations ──
    T = TerminationTermCfg
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.terminations["fell_over"].params["limit_angle"] = math.radians(60.0)
    cfg.terminations.update({
        "trunk_low": T(func=mdp.root_height_below_minimum, params={"minimum_height": 0.083}),
        "board_lost": T(func=tk.board_lost),
        "board_flipped": T(func=tk.board_flipped),
        "foot_on_floor": T(func=tk.foot_on_floor),
        "nan_state": T(func=microduck_mdp.skate_state_is_nan, params={"sensor_names": ALL_SENSORS}),
    })

    # ── curriculum: only the smoothness tax ramp (introduced after skill discovery) ──
    cfg.curriculum = {
        "action_rate": CurriculumTermCfg(func=microduck_mdp.reward_weight,
                                         params={"reward_name": "action_rate_l2", "weight_stages": ACTION_RATE_STAGES}),
    }
    return cfg


def make_microduck_skate_trick_ppo_runner_cfg(experiment_name: str = "skate_trick") -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True,
                            distribution_cfg={"class_name": "GaussianDistribution", "init_std": INIT_STD,
                                              "std_type": "scalar"}),
        critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
        algorithm=PpoWithSymmetryCfg(symmetry_cfg=None, value_loss_coef=1.0, use_clipped_value_loss=True,
                                     clip_param=0.2, entropy_coef=0.005, num_learning_epochs=5, num_mini_batches=4,
                                     learning_rate=1.0e-3, schedule="adaptive", gamma=0.99, lam=0.95, desired_kl=0.01,
                                     max_grad_norm=1.0),
        experiment_name=experiment_name,
        save_interval=250,
        num_steps_per_env=24,
        max_iterations=3000,
    )


# ── forward-stance variant (2026-11-19): tail tricks on the wide double-kick ──
FWD_STANCE_JSON = str(_data_path("stance_poses_fwd.json"))
# reverse-curriculum: start some tail-trick episodes in the CoM-behind-the-axle pose
FWD_ALT_SPAWNS = {"manual": ("tail_back", 0.5), "kickturn": ("tail_back", 0.3), "tictac": ("tail_back", 0.3)}
FWD_TRICK_PROBS = dict(ride=0.2, stop=0.15, kickturn=0.25, tictac=0.15, manual=0.25)


def make_microduck_skate_trick_fwd_env_cfg(play: bool = False, trick_probs: dict | None = None,
                                           stance_key: str = "tail") -> ManagerBasedRlEnvCfg:
    from micro_skater.robot.skateboard_variants import DKF_AXLE_X, MICRODUCK_SKATEBOARD_DOUBLEKICK_FWD_CFG
    return make_microduck_skate_trick_env_cfg(play=play, trick_probs=trick_probs or FWD_TRICK_PROBS,
                                              board_cfg=MICRODUCK_SKATEBOARD_DOUBLEKICK_FWD_CFG, stance_key=stance_key,
                                              stance_path=FWD_STANCE_JSON, axle_x=DKF_AXLE_X, dir_mode="plus",
                                              alt_spawns=FWD_ALT_SPAWNS, nose_lift_w=0.5)
