# micro-skater: part of the author's skateboard add-on for microduck_rl. Apache-2.0 (see LICENSE).
"""microduck_rl's robot constants plus the skateboard, the skate robot and the opt-in servo realism.

Every name of ``mjlab_microduck.robot.microduck_constants`` is re-exported first (imported, not copied), so the task
configs keep one namespace. The robot meshes come from the installed microduck_rl package.
"""
from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import mujoco
import mjlab_microduck.robot.microduck_constants as _upstream
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

globals().update({_k: _v for _k, _v in vars(_upstream).items() if not (_k.startswith("__") and _k.endswith("__"))})

from micro_skater.actuator.friction_dr_bam import FrictionDRBamActuatorCfg  # noqa: E402

_ASSETS_DIR: Path = Path(__file__).resolve().parent / "assets"

# Drop-deck cruiser skateboard for the Skate task (kingpin trucks on stiff torsion "bushings", rolling-friction wheels,
# all joints passive_*). The robot stands on a low 9.4 x 11 cm platform (geom deck_collision); ramps rise to 7 cm wide
# raised ends over the trucks and 18 deg kicktails (deck_*_collision). Overall 22.4 cm long, 61 g. The body origin is
# the platform centre.
MICRODUCK_SKATEBOARD_XML: Path = _ASSETS_DIR / "skateboard.xml"
SKATEBOARD_DECK_CENTER_Z = 0.0140    # board body origin (platform centre) above the floor
SKATEBOARD_DECK_TOP_Z = 0.0160       # standing platform top
SKATEBOARD_WHEEL_RADIUS = 0.010
SKATEBOARD_DECK_HALF_LEN = 0.047     # standing platform half length (x)
SKATEBOARD_DECK_HALF_WID = 0.055     # standing platform half width (y)
SKATEBOARD_WHEELBASE = 0.135         # truck to truck (x)
SKATEBOARD_WHEEL_TRACK = 0.090       # wheel centre to wheel centre (y); wheels 8 mm wide
SKATEBOARD_END_HALF_WID = 0.035      # raised ends and kicktails, half width (y)
SKATEBOARD_END_TOP_Z = 0.0260        # raised ends (over the trucks) top
SKATEBOARD_HALF_LEN = 0.112          # nose/tail tip from the origin (x)


def get_skateboard_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICRODUCK_SKATEBOARD_XML))


# Skate v6: flat box "deck soles" under each mesh sole, fitted to the hull's flat bottom (within 0.3 mm, tilt 0.07 deg)
# in the ankle body frame. The ~5k-vertex mesh hulls rest on a 6 x 4 mm patch at the 10 um scale: on the deck boxes
# their contact set flickered between 1 and 4 points on 41-55 % of the steps and sank 1.75 mm into the 4 mm deck in a
# lean (the on-board "collision jitter"). The boxes collide with the deck only (SKATE_COLLISION bitmask); the mesh
# soles keep every other contact (floor, robot, wheels).
SKATE_SOLE_BOXES = {
    # 1 mm flat patch (45.6 x 34 mm): the 0.3 mm patch (42 x 31 mm) cost skate_v5d half its riders (support area)
    "left": dict(body="ankle_left", pos=(-0.00701, -0.02284, -0.01408),
                 quat=(-0.002002, 0.002540, 0.737474, 0.675368), size=(0.02280, 0.01701, 0.00100)),
    "right": dict(body="ankle_right", pos=(0.00701, -0.02283, -0.01402),
                  quat=(0.675364, -0.737475, -0.002970, -0.002456), size=(0.02280, 0.01696, 0.00100)),
}


def get_skate_robot_spec() -> mujoco.MjSpec:
    """The groundcontact robot plus the massless deck-contact sole boxes (<side>_sole_deck_collision)."""
    spec = _upstream.get_standup_spec()
    for side, b in SKATE_SOLE_BOXES.items():
        g = spec.body(b["body"]).add_geom()
        g.name = f"{side}_sole_deck_collision"
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = b["size"]
        g.pos = b["pos"]
        g.quat = b["quat"]
        g.mass = 0.0
        g.group = 3
        g.rgba = (0.9, 0.2, 0.2, 0.4)
    return spec


# -- Servo realism (opt-in through env vars; unset = microduck_rl's BAM model exactly) ---------------------------------
# Current limit: the robot's runtime sets the XL330 firmware current limit to 1000 mA while BAM's XL330 default is
# 1.75 A. MICRODUCK_BAM_MAX_CURRENT=<amps> overrides the BAM current limiter.
# Datasheet joint speed: BAM's back-EMF uses ke = kt = 0.366, i.e. no-load ~2.73 rad/s per V; the XL330-M288-T
# datasheet says 103 rpm at 5.0 V = 2.157 rad/s per V. MICRODUCK_BAM_DATASHEET_SPEED=1 sets ke_eff = 1/2.157 (a number
# > 1 overrides the rad/s-per-V value). Stall torque and the current clamp are unchanged.
_BAM_ACTUATOR_KWARGS = dict(_upstream._BAM_ACTUATOR_KWARGS)
if os.environ.get("MICRODUCK_BAM_MAX_CURRENT"):
    _BAM_ACTUATOR_KWARGS["max_current"] = float(os.environ["MICRODUCK_BAM_MAX_CURRENT"])
XL330_M288_NOLOAD_RAD_S_PER_V = 103.0 * 2.0 * 3.141592653589793 / 60.0 / 5.0   # 2.1572
_ds = (os.environ.get("MICRODUCK_BAM_DATASHEET_SPEED") or "").strip()
if _ds.lower() not in ("", "0", "0.0", "false", "no", "off"):
    try:
        _ds_v = float(_ds)
    except ValueError:   # "true" / "on" -> the datasheet value
        _ds_v = 1.0
    _BAM_ACTUATOR_KWARGS["noload_speed_per_volt"] = _ds_v if _ds_v > 1.0 else XL330_M288_NOLOAD_RAD_S_PER_V
actuators = FrictionDRBamActuatorCfg(**_BAM_ACTUATOR_KWARGS)

# The stand-up robot with this package's actuator (same model; adds the realism options and the current log).
MICRODUCK_STANDUP_ROBOT_CFG = dataclasses.replace(
    _upstream.MICRODUCK_STANDUP_ROBOT_CFG,
    articulation=EntityArticulationInfoCfg(actuators=(actuators,), soft_joint_pos_limit_factor=0.9),
)

# Skate v6 collision bitmask (a contact needs (contype_a & conaffinity_b) | (contype_b & conaffinity_a)):
#   mesh soles      contype 0, conaffinity 1 -> floor (1/1), robot parts, wheels (1/1); NOT the deck (4/5)
#   box deck soles  contype 4, conaffinity 4 -> the deck (4/5) and each other only
#   deck geoms      contype 4, conaffinity 5 (SKATEBOARD_V6_COLLISION) -> box soles, floor, robot body
# Bit 4, not 2: the trunk's power_support mesh (class self_collision_only) is 2 / 2, and on bit 2 it collided with the
# deck and the box soles. Box soles get the foot contact parameters (condim 3, priority 1, friction 1.0): left to the
# ".*_collision" fallback they would be condim 1, i.e. frictionless.
SKATE_DECK_BIT = 4
_SOLE_DECK = r"^(left|right)_sole_deck_collision$"
_SOLE_MESH = r"^(left|right)_foot_collision$"
SKATE_COLLISION = CollisionCfg(
    geom_names_expr=[".*_collision"],
    contype={_SOLE_DECK: SKATE_DECK_BIT, _SOLE_MESH: 0, ".*_collision": 1},
    conaffinity={_SOLE_DECK: SKATE_DECK_BIT, _SOLE_MESH: 1, ".*_collision": 1},
    condim={_SOLE_DECK: 3, _SOLE_MESH: 3, ".*_collision": 1},
    priority={_SOLE_DECK: 1, _SOLE_MESH: 1},
    friction={_SOLE_DECK: (1.0,), _SOLE_MESH: (1.0,)},
)

MICRODUCK_SKATE_ROBOT_CFG = EntityCfg(
    spec_fn=get_skate_robot_spec,
    init_state=_upstream.HOME_FRAME,
    collisions=(SKATE_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

# Floating, articulated, unactuated skateboard prop. Placed each episode by the reset events; collisions come from the
# XML (wheels are condim-6 cylinders with rolling friction, the deck is grip tape).
MICRODUCK_SKATEBOARD_CFG = EntityCfg(
    spec_fn=get_skateboard_spec,
    init_state=EntityCfg.InitialStateCfg(
        pos=(0.0, 0.0, SKATEBOARD_DECK_CENTER_Z),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
)

# Skate v6: deck geoms on SKATE_DECK_BIT so only the box deck soles touch them (see SKATE_COLLISION). Everything else
# keeps its XML collision values (disable_other_geoms=False); condim / friction restate the XML.
SKATEBOARD_V6_COLLISION = CollisionCfg(
    geom_names_expr=[r"^deck.*_collision$"],
    contype=SKATE_DECK_BIT,
    conaffinity=SKATE_DECK_BIT | 1,
    condim=3,
    priority=0,
    friction=(1.2, 0.005, 0.0001),
    disable_other_geoms=False,
)
MICRODUCK_SKATEBOARD_V6_CFG = EntityCfg(
    spec_fn=get_skateboard_spec,
    init_state=MICRODUCK_SKATEBOARD_CFG.init_state,
    collisions=(SKATEBOARD_V6_COLLISION,),
)
