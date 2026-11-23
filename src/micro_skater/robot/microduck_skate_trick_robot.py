# micro-skater: part of the author's skateboard add-on for microduck_rl. Apache-2.0 (see LICENSE).
"""Robot cfg for the double-kick TRICK env (2026-11-19): MICRODUCK_SKATE_ROBOT_CFG with the sole-to-deck contact made of
four small corner SPHERES per sole instead of the thin box.

Why: in MuJoCo Warp (mjwarp, mjlab 1.3) the thin sole box (2 mm) against the deck box drops contacts at some relative
yaws -- in an isolated two-body test it fell straight through the deck at 0 / 30 / 90 deg yaw while CPU MuJoCo held it
at 0.8 mm. In the sideways stance (sole yawed 90 deg on the deck)
the soles sank through the deck within ~10 steps. Sphere-box is analytic and matched CPU MuJoCo at every yaw.

The box (<side>_sole_deck_collision) is kept, collision-disabled, as the sole frame for observations; the spheres are
<side>_sole_deck_s{0..3}_collision with the foot contact parameters (condim 3, priority 1, friction 1.0).
"""

from __future__ import annotations

import mujoco
import numpy as np

from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

from micro_skater.robot.microduck_constants import (
    HOME_FRAME,
    MICRODUCK_SKATE_ROBOT_CFG,
    SKATE_DECK_BIT,
    SKATE_SOLE_BOXES,
    get_skate_robot_spec,
)

SOLE_SPHERE_R = 0.0015
_SOLE_BOX = r"^(left|right)_sole_deck_collision$"
_SOLE_SPH = r"^(left|right)_sole_deck_s[0-3]_collision$"
_SOLE_MESH = r"^(left|right)_foot_collision$"


def get_skate_trick_robot_spec() -> mujoco.MjSpec:
    spec = get_skate_robot_spec()
    r = SOLE_SPHERE_R
    for side, b in SKATE_SOLE_BOXES.items():
        body = spec.body(b["body"])
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, np.asarray(b["quat"], dtype=float))
        R = R.reshape(3, 3)
        hx, hy, hz = b["size"]
        k = 0
        for sx in (-1, 1):
            for sy in (-1, 1):
                g = body.add_geom()
                g.name = f"{side}_sole_deck_s{k}_collision"
                g.type = mujoco.mjtGeom.mjGEOM_SPHERE
                g.size = (r, 0, 0)
                g.pos = np.asarray(b["pos"], float) + R @ np.array([sx * (hx - r), sy * (hy - r), -hz + r])
                g.mass = 0.0
                g.group = 3
                g.rgba = (0.9, 0.2, 0.2, 0.6)
                k += 1
    return spec


SKATE_TRICK_COLLISION = CollisionCfg(
    geom_names_expr=[".*_collision"],
    contype={_SOLE_BOX: 0, _SOLE_SPH: SKATE_DECK_BIT, _SOLE_MESH: 0, ".*_collision": 1},
    conaffinity={_SOLE_BOX: 0, _SOLE_SPH: SKATE_DECK_BIT, _SOLE_MESH: 1, ".*_collision": 1},
    condim={_SOLE_BOX: 3, _SOLE_SPH: 3, _SOLE_MESH: 3, ".*_collision": 1},
    priority={_SOLE_SPH: 1, _SOLE_MESH: 1},
    friction={_SOLE_SPH: (1.0,), _SOLE_MESH: (1.0,)},
)

MICRODUCK_SKATE_TRICK_ROBOT_CFG = EntityCfg(
    spec_fn=get_skate_trick_robot_spec,
    init_state=HOME_FRAME,
    collisions=(SKATE_TRICK_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=MICRODUCK_SKATE_ROBOT_CFG.articulation.actuators,
        soft_joint_pos_limit_factor=0.9,
    ),
)
