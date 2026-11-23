# micro-skater: part of the author's skateboard add-on for microduck_rl. Apache-2.0 (see LICENSE).
"""Skateboard model variants (2026-11-06). The stock board (microduck_constants.MICRODUCK_SKATEBOARD_V6_CFG,
robot/assets/skateboard.xml) is untouched; every variant is a NEW model file + entity cfg here.

WIDE ENDS (robot/assets/skateboard_wide.xml, generated):
the ramps, raised ends and kicktails, front AND rear, widened 7 -> 11 cm (= the standing platform), collision and
visual. Everything else (platform, trucks, wheels, total mass 61 g) is the stock board. The deck mass is spread over
the larger slab volume (lighter print per cm^3), so the deck body's inertia grows: diag (kg m^2) 2.96e-5 / 1.60e-4 /
1.86e-4 -> 4.04e-5 / 1.94e-4 / 2.31e-4 (+36 / +21 / +24 %), deck CoM 1.5 mm higher (5.75 -> 7.24 mm above the body
origin). Same-density print instead: board 75.8 g (--constant-density, not used).
Why: the manual / kick-turn need both soles on the raised rear end over the rear truck: on the
7 cm end each sole overlaps it by a 1-2 cm strip only.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
from mjlab.entity import EntityCfg

from micro_skater.robot.microduck_constants import (
    MICRODUCK_SKATEBOARD_CFG,
    SKATEBOARD_V6_COLLISION,
)

_ROBOT_DIR = Path(__file__).parent / "assets"
MICRODUCK_SKATEBOARD_WIDE_XML: Path = _ROBOT_DIR / "skateboard_wide.xml"
SKATEBOARD_WIDE_END_HALF_WID = 0.055      # raised ends + kicktails + ramps, half width (stock 0.035)
SKATEBOARD_REAR_AXLE_X = -0.0675          # deck frame (origin = platform centre), WHEELBASE / 2
SKATEBOARD_AXLE_Z = 0.010 - 0.014         # axle height in the deck body frame (wheel radius - body origin height)
SKATEBOARD_END_SLAB_X = (-0.079, -0.059)  # raised rear end slab (flat, top 26 mm), deck frame x range


def get_skateboard_wide_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICRODUCK_SKATEBOARD_WIDE_XML))


# Same collision setup as the stock v6 board (deck geoms on SKATE_DECK_BIT: only the box deck soles, the floor and the
# robot body touch them), same initial state.
MICRODUCK_SKATEBOARD_WIDE_V6_CFG = EntityCfg(
    spec_fn=get_skateboard_wide_spec,
    init_state=MICRODUCK_SKATEBOARD_CFG.init_state,
    collisions=(SKATEBOARD_V6_COLLISION,),
)


# ── DOUBLE-KICK (2026-11-19, trick work) ──────────────────────────────────────────────────────────────────────────────
# robot/assets/skateboard_doublekick.xml:
# a popsicle board for a SIDEWAYS rider (one sole over each truck). 21.1 x 7.0 cm, flat deck top 24 mm (body origin
# 22 mm), flat between x = +-0.067, 4 cm kicks at 20 deg (tips 34 mm up, tail strike ~34-37 deg nose-up), wheelbase
# 11.0 cm (axles at x = +-0.055, z = -0.012 in the body frame), track 8 cm, 20 mm wheels, bushing 1.0 N m/rad (its
# armature is sized for <= 1.0: do not randomize the trucks stiffer), 57 g. Sites deck_top, nose_tip (+x), tail_tip (-x).
MICRODUCK_SKATEBOARD_DOUBLEKICK_XML: Path = _ROBOT_DIR / "skateboard_doublekick.xml"
DK_DECK_CENTER_Z = 0.022
DK_DECK_TOP_Z = 0.024
DK_FLAT_HALF_LEN = 0.067
DK_DECK_HALF_WID = 0.035
DK_HALF_LEN = 0.1053          # kick tips
DK_AXLE_X = 0.055             # +-, deck frame
DK_AXLE_Z = -0.012            # deck body frame (wheel radius 0.010 - origin 0.022)
DK_WHEEL_RADIUS = 0.010
DK_TRUCK_K_MAX = 1.0


def get_skateboard_doublekick_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICRODUCK_SKATEBOARD_DOUBLEKICK_XML))


MICRODUCK_SKATEBOARD_DOUBLEKICK_CFG = EntityCfg(
    spec_fn=get_skateboard_doublekick_spec,
    init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, DK_DECK_CENTER_Z)),
    collisions=(SKATEBOARD_V6_COLLISION,),
)


# Forward-stance double-kick (2026-11-19): wider deck / longer flat for a rider facing the nose with feet side by side
# Generated with DK_* env
# overrides; same deck height as the sideways board.
MICRODUCK_SKATEBOARD_DOUBLEKICK_FWD_XML: Path = _ROBOT_DIR / "skateboard_doublekick_fwd.xml"


def _dkf_axle_x() -> float:
    try:
        import xml.etree.ElementTree as ET
        for b in ET.parse(MICRODUCK_SKATEBOARD_DOUBLEKICK_FWD_XML).iter("body"):
            if b.get("name") == "rear_truck":
                return abs(float(b.get("pos").split()[0]))
    except (OSError, ET.ParseError):
        pass
    return DK_AXLE_X


DKF_AXLE_X = _dkf_axle_x()


def get_skateboard_doublekick_fwd_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICRODUCK_SKATEBOARD_DOUBLEKICK_FWD_XML))


MICRODUCK_SKATEBOARD_DOUBLEKICK_FWD_CFG = EntityCfg(
    spec_fn=get_skateboard_doublekick_fwd_spec,
    init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, DK_DECK_CENTER_Z)),
    collisions=(SKATEBOARD_V6_COLLISION,),
)


# ── sequence board (2026-11-19): shaped fishtail kicktail for push -> manual -> half-pipe ──────────────────────────────
# 21.3 cm, 12.4 cm
# tail / 9.2 cm nose, deck top 24 mm (deck origin 0.022 like the double-kicks), axles +-0.045, 16 mm wheels, 83.8 g.
MICRODUCK_SKATEBOARD_SEQUENCE_XML: Path = _ROBOT_DIR / "skateboard_sequence.xml"


def get_skateboard_sequence_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICRODUCK_SKATEBOARD_SEQUENCE_XML))


MICRODUCK_SKATEBOARD_SEQUENCE_CFG = EntityCfg(
    spec_fn=get_skateboard_sequence_spec,
    init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, DK_DECK_CENTER_Z)),
    collisions=(SKATEBOARD_V6_COLLISION,),
)
