# Hardware realism: making the simulation pessimistic on purpose

← [back to README](../README.md)

A policy that only works in a "perfect" simulator usually fails on a real robot. That gap is called the
**sim-to-real gap**. This project aimed to keep the simulated robot **no stronger, faster or luckier** than the real
one. The settings below were on for all four final policies. Every claim about the real hardware comes from datasheets
or the robot's runtime configuration, and none has been bench-tested here yet.

## The real robot, as modelled

| Item | Value used in simulation | Source |
|---|---|---|
| Servos | 14 × Dynamixel XL330-M288 | Matched the servo model's torque constant and resistance to the datasheet |
| Battery | 6.5–8.2 V per episode, with sag under load (the real pack is a 2S LiPo, 6.6–8.2 V) | Robot runtime config |
| Servo current limit | **1.0 A** (the earlier default model allowed 1.75 A) | Robot runtime config |
| Servo speed | Datasheet no-load speed, scaled with battery voltage (~2.16 rad/s per volt) | XL330-M288 datasheet |
| Control delay | 15–30 ms per actuator, randomized | — |
| Mass | ×0.98–1.10 per episode (0.72–0.81 kg); the real robot is ~0.80 kg | The real robot is heavier than the CAD |

The servo model is [BAM](https://github.com/Rhoban/bam). It models the motor's back-EMF, friction and current limit,
so joint speed and torque come from physics, not from a fixed cap.

## Domain randomization (every episode is a slightly different robot)

- Center-of-mass offsets of the trunk and of the head, ±3 mm each, fixed (no curriculum) in the four phase tasks.
  They do **not** keep microduck_rl's CoM curriculum, which widens the trunk range from 3 to 15 mm and the head range
  from 3 to 10 mm. The walker and runner keep it; the v7f skater ends at ±10 mm (trunk) / ±6 mm (head).
- Motor armature and friction (±10 %).
- Encoder bias (±0.015 rad) and an IMU mounting error of up to 6°.
- Foot and deck friction, wheel rolling resistance (×0.3–1.0) and truck stiffness.
- Sensor noise and one-step sensor lag.

## Costs that exist only because of real hardware

| Cost | Why |
|---|---|
| **Servo heat** (estimated I²R copper loss) | A pose held hard (a "squeeze") can overheat small servos in a few minutes |
| **Joint-limit margins** | The robot's runtime does **not** stop joints at the CAD limits, so the policy has to |
| **Knee side** | The knee *can* bend backwards mechanically, but it looks wrong and stresses the leg |
| **Hip-yaw target error** | At the 1.0 A cap, a servo told to reach an impossible angle just stalls hot. Heat alone gave no gradient there |
| **Current curriculum** | The push policy was eased from 1.4 A down to 1.0 A, so it didn't collapse when the cap tightened |

## What is still NOT realistic, or not deployable

- **Privileged observations.** About 35 of the 114 inputs are things the robot can't sense: board state, foot
  positions on the deck and the half-pipe position. Deployment needs a **student policy** that uses only real sensors.
  The robot's runtime also expects a 61-input policy, centred on its HOME pose and with its own filters.
- **The 1.0 A limit may not be enforced** in the servos' current operating mode. Bench-test a stalled servo or switch
  modes before relying on it.
- **Both push policies** spend about 10–13 % of their frames slightly (more than 0.5°) past a joint's soft stop. This
  is measured in the final chain eval (seeds 3 and 4): p16 in the first push 9.9–10.8 %, p17e during the foot push
  9.5–12.8 %. (The 14.5 % → 8.5 % in the training history is a different number: the p17e training metric.) The hip
  yaws still saturate during the foot push.
- **No backlash** (gear play) in these policies.
- **The board doesn't exist yet.** It is a custom 21 cm fishtail with a 12.4 cm wide tail (~84 g), designed in simulation.

An independent review in the project rated the result **"sim-stage OK, not deployable"**. That is an accurate summary.
