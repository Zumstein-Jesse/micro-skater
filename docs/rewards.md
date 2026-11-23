# Reward terms, phase by phase

← [back to README](../README.md)

**How to read this page.**

- **+** is a reward and **−** is a cost.
- Weights are the **final curriculum values**, read from the task code: many terms are raised gradually during training
  (a *curriculum*), and the values here are where each curriculum ends. The `params/env.yaml` saved next to each
  checkpoint is the config as resolved at **iteration 0** of its final run, so many weights there are lower, or 0 for
  terms a curriculum switches on later.
- The **Code** column names the function behind each term: `module.function` in `src/micro_skater/tasks/` (mjlab's
  own terms are marked "mjlab"). The term names are the keys under `rewards:` in the env config.
- The weights are only here so you can compare terms within one phase. Don't read much into the exact numbers.
- "Potential" terms pay for *getting closer* to a goal. The banked pay is **taken back** if the episode then fails, so
  only a finished transition keeps it. This is called potential-based shaping with charge-back.
- All policies run at **50 Hz**. Each training run uses **2048 parallel environments**, and every final policy is a
  **resumed** checkpoint, not trained from scratch.

> Source: the task code in [`src/micro_skater/tasks/`](../src/micro_skater/tasks/)
> (`microduck_skate_seq_{push,manual,ramp}_env_cfg.py` for the term lists, weights and curricula; `skate_sequence.py`,
> `skate_seq_ramp.py` and `mdp.py` for the term functions), checked against the saved configs
> ([`checkpoints/<name>/params/env.yaml`](../checkpoints/README.md)) for the term lists and functions.

## Shared by every phase

| Term | Code | +/− | Purpose |
|---|---|---|---|
| `upright` | `skate_trick.trunk_upright` / `skate_seq_ramp.seq_upright` | +1 | Trunk upright |
| `on_deck` | `skate_sequence.seq_on_deck` | +0.5 | Board foot on the deck |
| `ride`, `ride_wheels` | `skate_trick.ride_track`, `skate_sequence.ride_wheels_down` | +3 / +1 | Clean riding with all wheels down |
| `posture` | `skate_sequence.posture` / `skate_seq_ramp.posture` | + | Near a sensible riding pose |
| `fall` | `skate_sequence.seq_fall_push` (push), `.seq_fall_stage` (manual), `.seq_fall` (half-pipe), `.p17_fall_push` (foot push) | −10 (one-time) | Falling, losing the board or flipping it. Priced as a real one-time 10, not scaled by the time step |
| `body_on_board` | `skate_trick.contact_cost` | −1 | Trunk or knees touching the board |
| `head_pose`, `head_steady` | `skate_trick.joint_dev_l2`, `skate_sequence.head_steady` | − | Head near neutral and steady |
| `board_wobble` | `skate_sequence.board_wobble` | −0.2 | Board rocking on its trucks |
| `dof_pos_limits` | mjlab `joint_pos_limits` | −1 | Joints at their limits (mjlab standard term) |
| `action_rate_l2` | mjlab `action_rate_l2` | −0.1 to −0.2 | Jerky, buzzing actions |
| `skate_heat` | `mdp.skate_heat_cost` | −0.3 per W | Estimated servo copper loss (I²R, smoothed over 2 s) above 0.65 W. It stands in for "don't cook the servos" |

**Episode endings (terminations):**

- trunk tilted more than 60°, or the trunk too low;
- board lost or flipped;
- the board foot on the floor;
- going over the half-pipe coping;
- a riding-tilt guard;
- the end of the half-pipe.

## 1. Push and step-on (p16)

**Task.** Start with one foot on the board, kick until a "step on now" command, then step the push foot onto the deck
and settle into the stance the manual policy expects.

**Success.** Both feet on the deck for at least 0.3 s, all wheels down, and the hand-off stance held for 0.2 s.

**Spawns.** Starts are mixed:

- the real start;
- mid-push;
- mid-step-on states solved with inverse kinematics;
- a "bridge ladder" of recorded states. It is weighted toward rungs where success is about 50 %, so the hardest
  solvable starts get the most practice.

The servo current cap was lowered **1.4 → 1.2 → 1.1 → 1.0 A**. Each step came once 70 % of the episodes from the real
start succeeded, or at a fixed deadline (an iteration count) if that share was not reached in time.

| Group | Term | Code | +/− | Purpose | Failure it fixed |
|---|---|---|---|---|---|
| Task | `push_progress` | `skate_sequence.push_progress` | +10 | Speed gained by real kicks (potential) | — |
| | `push_speed` | `skate_sequence.push_speed` | +2 | Rolling speed with the board foot on the deck | — |
| | `push_step_on` | `skate_sequence.push_step_on` | +8 one-time | Finishing the step-on | It paid every step, so quick taps onto the deck farmed it |
| | `push_step_path` | `skate_sequence.push_step_path` | +20 potential | Foot up and over the deck edge, plus a weight shift | A straight pull jammed the foot into the deck edge |
| | `settle_stance_path` | `skate_sequence.settle_stance_path` | +0.25 potential | Toward the manual policy's hand-off stance | The manual survived only 6–31 % of hand-offs |
| | `glide_stance` | `skate_sequence.glide_stance` | +1 | Push foot in its spot after the step-on | — |
| | `push_time` | `skate_sequence.phase_time` | −1 | Time spent in PUSH | Parking in the push stance |
| | `cmd_wait` | `skate_sequence.cmd_wait` | −1 per s | Ignoring the step-on command | Refusing used to be free (the deadline was too far away to matter) |
| | `p15_speed_deficit` | `skate_sequence.p15_speed_deficit` | −1.5 | Falling behind a slowly rising target speed | A 1.9 s stall with the foot planted |
| | `heading_line` | `skate_sequence.heading_line_cost` | −0.6 | Drifting off the line to the half-pipe | It turned 61° left and the chain missed the pipe |
| Style | `push_stroke` | `skate_sequence.push_stroke` | −1.2 | Short or shallow kicks | Tapping the floor at 4–7 Hz |
| | `push_cadence`, `p12_cadence` | `skate_sequence.push_cadence` | −1 / −1.5 | Touchdowns at least ~0.8 s apart | A rushed 0.38 s cadence |
| | `push_stroke_shape`, `p16_stance_time` | `skate_sequence.push_stroke_shape`, `skate_sequence.p16_stance_time` | −1 / −0.4 | Long planted contact and a long backward sweep | 60–80 ms "dab" contacts |
| | `push_foot_height`, `p16_foot_lift`, `p13_hover` | `skate_sequence.push_foot_height`, `skate_sequence.p16_foot_lift`, `skate_sequence.p13_hover` | −1.5 / −0.75 / −2 | A low swing, no foot parked mid-air | A "chicken kick" 6 cm up |
| | `p13_trunk_hold`, `push_trunk_floor` | `skate_sequence.p13_trunk_hold`, `skate_sequence.push_trunk_floor` | −1 / −2 | Keep the crouch height | Slowly sinking into a deep squat |
| | `p14_stepon_pop`, `p13_stepon_pose`, `step_lurch`, `step_clearance` | `skate_sequence.p14_stepon_pop`, `skate_sequence.p13_stepon_pose`, `skate_sequence.step_lurch`, `skate_sequence.step_clearance` | −1.5 / −1.5 / −2 / −4 | A calm step-on that clears the deck edge | A "jack-in-the-box" stand-up |
| | head terms (`push_head_level`, `p12_head_steady`, `p14_head_yaw`, `settle_head_pose`, …) | `skate_sequence.head_level`, `skate_sequence.p12_head_steady`, `skate_sequence.p14_head_yaw`, `skate_sequence.settle_head_pose` | − | Head level and still | The head swung as a counterweight |
| Realism | `p16_knee_side` | `skate_sequence.p16_knee_side` | −0.27 | Knee bending the wrong way | The push knee reversed in 41–50 % of kicks |
| | `p16_leg_limits` | `skate_sequence.p16_leg_limits` | −0.5 | Leg joints within 3° of their mechanical stops | The robot's runtime doesn't enforce the stops |
| | `p16_neck` | `skate_sequence.p16_neck` | −0.25 | Craning the neck forward | Neck pitched down to −0.77 rad at 1.0 A |

Smaller terms: `push_overspeed`, `push_swing_foot`, `push_head_rate`, `p12_trunk_steady`, `p12_swing_sole`,
`p15_land_roll`, `tail_strike`.

## 2. Manual (m6)

**Task.** Pop the nose, hold 9–14° nose-up for 1.5–2.5 s, set the nose down, and roll on toward the pipe. The hold
timer is "leaky": time outside the band drains it twice as fast.

**Spawns.**

- 55 % gliding, 35 % already in a manual, 10 % at the exit, at 0.6–1.2 m/s.
- Half of the gliding starts are **recorded hand-off states** from the push policy.
- Trained at 1.0 A from the start.

| Group | Term | Code | +/− | Purpose | Failure it fixed |
|---|---|---|---|---|---|
| Task | `manual` | `skate_trick.manual_in` | +4 | Nose up 5–22° on the back wheels | — |
| | `manual_shape` | `skate_sequence.manual_quality` | +4 | Peak pay at 11–13°, held still | A 3–4° hover got almost full pay |
| | `manual_hold_done` | `skate_sequence.manual_hold_done` | +8 one-time | Completing the hold | Stretching the hold out earned more than finishing |
| | `speed_loss` | `skate_sequence.manual_speed_loss` | −4 | Losing speed beyond normal coasting | The board stopped short of the pipe at 1.0 A |
| | `heading_line` | `skate_sequence.heading_line_cost` | −0.6 | Ride toward the pipe | — |
| Style | `pop_rate`, `pitch_rate` | `skate_sequence.manual_pitch_rate` | −30 / −10 | Slow pop, no snapping | The nose snapped up at more than 200°/s |
| | `setdown_rate` | `skate_sequence.setdown_rate` | −2 | Gentle set-down | Slammed nose |
| | `manual_over` | `skate_sequence.manual_over` | −3 | Staying under 14° | Holds drifted up to 14–16° |
| | head terms (`head_level`, `head_rate_pop`, `m5_head_tilt`, `m5_head_dev`) | `skate_sequence.head_level`, `skate_sequence.head_rate`, `skate_sequence.head_level`, `skate_sequence.settle_head_pose` | − | A bounded head "catch" for balance, not a ban | Face to the sky, neck bent 60° |
| | `m5_trunk_band`, `stance_width` | `microduck_skate_seq_manual_env_cfg.m5_trunk_band`, `skate_sequence.stance_width` | −2 / −0.3 | Reasonable back-lean and stance | A bow-legged crouch |
| Realism | `m6_knee_side`, `m6_leg_limits`, `m6_neck` | `skate_sequence.p16_knee_side`, `skate_sequence.p16_leg_limits`, `microduck_skate_seq_manual_env_cfg.m6_neck` | −0.27 / −0.5 / −0.25 | Same as push | — |
| | `tail_strike` | `skate_sequence.seq_tail_strike` | −3 | Tail hitting the ground | — |

## 3. Half-pipe (r10, fine-tuned from r9)

**Task.** Carve into the pipe (8 cm walls), ride the walls back and forth, and allow the foot-push hand-over on the
flat bottom.

**Success.** 10 s in the pipe or 12 wall passes. Rolling back out of the entry end counts as a failure.

**Spawns.**

- 40 % at the pipe entry, 35 % on the flat bottom, 25 % on a wall.
- 45 % of the entry starts are recorded manual→ramp hand-offs.
- r10 adds 35 % recorded states from just **after a foot push**, when control comes back to the ramp policy.

| Group | Term | Code | +/− | Purpose | Failure it fixed |
|---|---|---|---|---|---|
| Task | `ramp_apex` | `skate_seq_ramp.ramp_apex` | +6 | Each wall top, only if the walls alternate | The first apex detector could be farmed |
| | `ramp_lateral` | `skate_sequence.ramp_lateral` | +2 | Moving across the pipe | — |
| | `r7_retain`, `r7_keep` | `skate_seq_ramp.r7_retain`, `skate_seq_ramp.r7_keep` | +14 / +3 | Keeping energy and height per wall visit, reduced if the head folds | r6 "pumped" by folding its head |
| | `r7_pump`, `r7_pump_dir` | `skate_seq_ramp.r7_pump`, `skate_seq_ramp.r7_pump_dir` | + | Pumping work. Still in the config, but leg pumping was later shown to be infeasible on this robot | — |
| | `ramp_track` | `skate_seq_ramp.ramp_track` | +1.5 | Following the carve command on the flat | r1 never turned |
| | `r9_gain`, `r9_reach` | `skate_seq_ramp.r9_gain`, `skate_seq_ramp.r9_reach` | +3 / +1 | Height gained by a flat-bottom foot push | No way to keep speed |
| | `ramp_park`, `exit_time` | `skate_seq_ramp.ramp_park`, `skate_seq_ramp.exit_time` | −0.5 / −2 | No parking on the flat, no dawdling before the pipe | Waiting outside paid better than riding |
| | `r9_drag`, `r9_late` | `skate_seq_ramp.r9_drag`, `skate_seq_ramp.r9_late` | −1 / −3 | Short floor contact, feet back on before the wall | — |
| Style | `r7_head`, `r6_head`, `r5_head_hold` | `skate_seq_ramp.r7_head`, `skate_seq_ramp.r6_head`, `skate_seq_ramp.r5_head_hold` | − | Head level and still on the walls | Head fold / whip |
| | `r10_head_low` | `skate_seq_ramp.r10_head_low` | −1 | Head at least 15 cm above the deck | Squatting after the foot push |
| | `r6_trunk_band` | `skate_seq_ramp.r6_trunk_band` | −0.5 | Trunk height band | A frozen 7 cm squat |
| Realism | `r10_jlim` | `skate_seq_ramp.r10_jlim` | −0.5 | Joint-limit margin on the hips | Hips pushed past their stops |
| | `r10_hy_heat`, `r10_hy_err` | `skate_seq_ramp.r10_hy_heat`, `skate_seq_ramp.r10_hy_err` | −0.3 per W / −0.1 | Hip-yaw heat, and asking for hip angles the servo can actually reach | Hip servos stalled at the 1.0 A cap 99 % of the time. A heat cost alone had no gradient at the cap |
| | `r9_knee` | `skate_seq_ramp.r9_knee` | −1 | Knee on its normal side | "Bird knee" |

## 4. Foot push between the walls (p17e)

**Task.** This is the **push task again**. The robot starts *riding* (both feet on), makes **one** kick off the floor,
steps back on, and hands back to the half-pipe policy.

**Success.** A legal kick (it must add speed), then the riding stance held for 0.2 s. On the pipe, the deadline is
3 s.

**Spawns.**

- A **stance bridge** (`alpha`, 0 → 1) blends the push policy's own pre-kick states into the half-pipe policy's real
  riding states, one rung at a time.
- 70 % of pipe starts are real states recorded from full chain runs, at their true speeds of 0.25–0.75 m/s.
- On the pipe the policy is shown "floor-equivalent" position inputs, exactly as the chain will feed them.

**Changes from p16:**

| Term | Code | +/− | Purpose | Failure it fixed |
|---|---|---|---|---|
| `p17_glide_tax` | `skate_sequence.p17_glide_tax` | −2 per s | Kick now instead of gliding past the chance | Refusing only cost something at a far-off deadline |
| `p17_stance_path` | `skate_sequence.p17_stance_path` | +0.25 potential | Return to the riding stance | The ramp policy needs its own stance back |
| `fall` = `p17_fall_push` | `skate_sequence.p17_fall_push` | − | Falling mid-kick costs about as much as missing the deadline | — |
| `p16_leg_limits` | `skate_sequence.p16_leg_limits` | −0.75, margin 1°, cap 40 | Stay inside the stops | p17d pushed the right hip about 5° past its stop. The old cap gave no gradient beyond 3° |
| `p16_knee_side` | `skate_sequence.p16_knee_side` | −0.45 | No reversed push knee | The knee reversed in 21 of 24 kicks |

Heading and hand-off-stance terms are switched off for these episodes.

## Lessons the reward code teaches

1. **A failed attempt must not cost more than refusing.** Otherwise the policy learns not to try.
2. **Discounting changes behaviour.** With γ = 0.99 at 50 Hz, a reward 1 s away is worth about 60 %. A cost 6–8 s away
   is nearly free, so refusals need a price **now**.
3. **No jackpots.** Pay for tracking a slowly rising target, not for raw speed, so rushing earns nothing.
4. **Pay one-time events once.** Otherwise they get farmed.
5. **Potentials with charge-back.** Guide the robot toward a goal, but take the pay back if it then fails.
6. **Costs need a gradient where the behaviour is.** A capped or saturated cost can't push the policy any further.
7. **Measure feasibility first.** Leg pumping was rewarded for several versions before measurements showed it can't
   work on this robot.
8. **Some "ugly" motion is balance.** Allow a bounded head catch instead of banning it.
9. **Scale matters (Froude scaling).** The leg is about 10 cm long against about 90 cm for a human. Time scales with
   √(0.10/0.9) ≈ 0.33, so a calm human-looking kick has to be slower than "natural" for the robot.
