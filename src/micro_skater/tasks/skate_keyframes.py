# micro-skater: part of the author's skateboard add-on for microduck_rl. Apache-2.0 (see LICENSE).
"""Static keyframes for the Skate task on the drop-deck cruiser (skateboard.xml, 2026-11-02).

Solved offline (SLSQP IK over trunk pose and
the 10 leg joints, then a contact LP for the minimum peak joint torque, servo limit 0.96 N m) and dropped onto the
board in MuJoCo by (training collision robot, XML servos holding the pose with the LP feed-forward).
Frame: standing-platform centre at (0, 0), platform top 16 mm above the floor (9.4 x 11 cm), robot facing +x.
The LEFT foot rides; the RIGHT foot pushes. ``pos`` is the trunk_base free-joint position, ``rpy`` its
roll/pitch/yaw (rad). Head joints stay at HOME. A held pose falls after 0.5-1 s without active balance (even HOME
on flat ground does), so the sim check judges the initial contacts only.

GLIDE       both soles side by side on the platform, CoM over the centreline, facing along the board.
            Peak static torque 0.051 N m, trunk yaw -0.0 deg, hip yaws -0/+0 deg, board-foot load 0.50 W; sim 0.3 s loads (W): left->deck_collision 0.50, right->deck_collision 0.50.
PUSH_MID    board sole near the centreline, push sole on the floor level with it.
            Peak static torque 0.060 N m, trunk yaw -9.3 deg, hip yaws -9/+4 deg, board-foot load 0.26 W, max push accel 2.05 m/s^2 at 0.5 N m; sim 0.3 s loads (W): left->deck_collision 0.40, right->floor 0.59.
PUSH_FRONT  push sole 4 cm ahead: the start of a stroke.
            Peak static torque 0.043 N m, trunk yaw +10.0 deg, hip yaws +10/-5 deg, board-foot load 0.29 W, max push accel 1.93 m/s^2 at 0.5 N m; sim 0.3 s loads (W): left->deck_collision 0.35, right->floor 0.65.
PUSH_BACK   push sole 6 cm behind: the end of a stroke.
            Peak static torque 0.054 N m, trunk yaw -10.0 deg, hip yaws -10/+5 deg, board-foot load 0.51 W, max push accel 2.38 m/s^2 at 0.5 N m; sim 0.3 s loads (W): left->deck_collision 0.37, right->floor 0.66.
MOUNT       board sole near the centreline, push sole set down on the platform's right edge.
            Peak static torque 0.045 N m, trunk yaw -15.0 deg, hip yaws -15/-15 deg, board-foot load 0.51 W; sim 0.3 s loads (W): left->deck_collision 0.64, right->deck_collision 0.39.
            Marginal: only 17% of the push sole's flat patch is over the platform (the soles cannot come closer than about 7.9 cm side by side), trunk roll at its 12 deg bound and yaw -15 deg; LP with the push contacts clipped to the deck edge.
DISMOUNT    NOT INCLUDED: no static solution. With the push foot in the air beside the deck, a held one-foot stance needs about 20 deg of trunk tilt to bring the CoM over the stance sole, violating by 7-9 mm at 12 deg even with the board foot at y 2-4 cm and 3 cm of allowed deck-load offset; with the push sole touching the floor beside the deck the soles would have to be 11-12 cm apart (violation 1.3 mm at y 3 cm, 4.8 mm at the 4.2 cm glide spot). The glide -> push step is dynamic, like a walking step.

Push -> glide needs the board foot to shuffle out from near the centreline (|y| < 1 cm) to its glide spot
(y about +4.2 cm) and the trunk to shift about 4.7 cm toward it; glide -> push the reverse.
"""

SKATE_KEYFRAMES = {
    "GLIDE": dict(
        pos=(+0.0007, -0.0001, 0.1361),
        rpy=(-0.0002, -0.0075, -0.0005),
        joints={"left_hip_yaw": -0.0011, "left_hip_roll": -0.0871, "left_hip_pitch": -0.4448, "left_knee": -0.1940, "left_ankle": +0.2584, "right_hip_yaw": +0.0002, "right_hip_roll": +0.0875, "right_hip_pitch": +0.4452, "right_knee": +0.1943, "right_ankle": -0.2584},
    ),
    "PUSH_MID": dict(
        pos=(+0.0110, -0.0472, 0.1227),
        rpy=(+0.0184, +0.0103, -0.1616),
        joints={"left_hip_yaw": -0.1610, "left_hip_roll": -0.1071, "left_hip_pitch": -0.0148, "left_knee": +0.2573, "left_ankle": +0.2648, "right_hip_yaw": +0.0754, "right_hip_roll": +0.0698, "right_hip_pitch": +0.5270, "right_knee": +0.3542, "right_ankle": -0.1610},
    ),
    "PUSH_FRONT": dict(
        pos=(+0.0220, -0.0481, 0.1243),
        rpy=(+0.0076, +0.0061, +0.1745),
        joints={"left_hip_yaw": +0.1752, "left_hip_roll": -0.0937, "left_hip_pitch": -0.0267, "left_knee": +0.1154, "left_ankle": +0.1347, "right_hip_yaw": -0.0877, "right_hip_roll": +0.0792, "right_hip_pitch": +0.6966, "right_knee": +0.4090, "right_ankle": -0.2821},
    ),
    "PUSH_BACK": dict(
        pos=(-0.0364, -0.0512, 0.1224),
        rpy=(+0.1178, +0.0160, -0.1745),
        joints={"left_hip_yaw": -0.1762, "left_hip_roll": -0.2061, "left_hip_pitch": -0.7249, "left_knee": +0.2287, "left_ankle": +0.9584, "right_hip_yaw": +0.0856, "right_hip_roll": -0.0287, "right_hip_pitch": +0.4358, "right_knee": +0.4558, "right_ankle": +0.0461},
    ),
    "MOUNT": dict(
        pos=(+0.0048, -0.0200, 0.1349),
        rpy=(-0.2094, -0.0869, -0.2618),
        joints={"left_hip_yaw": -0.2702, "left_hip_roll": +0.1374, "left_hip_pitch": -0.2445, "left_knee": -0.0885, "left_ankle": +0.1844, "right_hip_yaw": -0.2650, "right_hip_roll": +0.3119, "right_hip_pitch": +0.6383, "right_knee": +0.3551, "right_ankle": -0.3128},
    ),
}
