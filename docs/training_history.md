# Training history: how p1 became p17e

← [back to README](../README.md)

The run names are short codes:

- **p**: push (later also the foot push);
- **m**: manual;
- **r**: ramp / half-pipe;
- **s**: an early single-policy attempt.

A higher number means a later fine-tune of the same policy. Nearly every step was "watch the video, measure the
failure, change one thing".

All numbers are from simulation. The whole sequence took about 3 days of training on a single GPU (an RTX 3090).

## 0. Before: a board the robot can actually ride

- An earlier project taught the Microduck to ride and push on a plain board. The **manual** on that board turned out
  to be **geometrically impossible**: the ends were too narrow for the feet, and there is no ankle roll joint.
- A new board was designed for this sequence. It is a 21 cm fishtail with a 12.4 cm wide tail, 16 mm wheels and a
  weight of about 84 g. A feasibility study ran on the CPU before any training.
- The project also used a mini half-pipe made of flat box facets. A smooth heightfield broke CPU MuJoCo, so facets were
  used instead.

## 1. One policy for everything (s1 → s2e): failed in an instructive way

A single policy saw a phase indicator and had to do the whole run.

- It learned to push at about 2 strokes per second, but it **never stepped on the board**. The manual came right after
  the step-on and was risky, so pushing forever paid better.
- **Lesson:** in a serial chain, a weak later phase discourages the transition into it. So the project split the run
  into specialists early.

## 2. Three specialists (2026-11-20)

All three use the same 114 observations and 14 actions, so the chain can switch between them on any step.

### Push (p1 → p15)

- **p1–p5** never stepped on. Measurements showed the step-on needs a whole-body move that the push stance can't start
  from. The reverse curriculum was also spawning impossible states, with feet inside the deck.
- **Fixes:**
  - "stepfix": mid-step starts solved with whole-body inverse kinematics;
  - a "bridge ladder" of recorded states that walks back toward the real start;
  - a **landing rule**: after the command, the next foot landing must be on the deck;
  - removing a cost that made a failed attempt dearer than refusing.
- **p8** reached **98 % step-on from rest**.
- **p9–p15** were polish:
  - a slowly rising speed target, so rushing doesn't pay;
  - a deliberate stroke cadence, about 0.8 s between touchdowns;
  - head and trunk steadiness;
  - holding a straight line to the pipe;
  - ending in **exactly the stance the manual policy expects**. The first hand-offs survived only 6–31 %.

### Manual (m1 → m5)

- **m2** reached **97 %**: pop at about 107°/s, hold 1.5 s.
- Video reviews flagged head nodding and a short hold, so **m3–m5** added head and pop-speed costs.
- They also trained on **recorded hand-off states** from the push policy.

### Half-pipe (r1 → r8)

- **r1** never turned, and its apex detector was broken.
- **r2** latched the apex and counted riding back out of the entry as a failure.
- **r5–r8** tried to learn **pumping**. r6 "pumped" by folding its head.
- Measurements showed why:
  - the head and neck are **38 % of the robot's mass**;
  - a leg crouch moves the centre of mass by only 0.3–0.6 cm;
  - a scripted pump lost energy.
- **Pumping was dropped.** The walls were lowered from 12 cm to 8 cm, and the plan became a **foot push** on the flat
  bottom.

## 3. Hardware realism pass (2026-11-21)

These changes went in:

- current limit 1.75 A → **1.0 A**, matching the robot's runtime setting;
- datasheet speed-vs-voltage;
- servo heat;
- mass randomization;
- costs for joint limits and reversed knees.

Results:

- **p16** (push) eased into 1.0 A through a current curriculum (1.4 → 1.0 A). Under the same settings, the old p15
  succeeded 0 % from the start; p16 succeeded 32/32.
- **m6** (manual) was retrained at 1.0 A.
- The new bottleneck: at 1.0 A the robot reached the pipe at only ~0.62 m/s.

## 4. The foot push (r9 → p17e)

- **r9, r9b, r9c**: teaching the *half-pipe* policy to kick failed. It never learned a stroke from states that came
  from a different policy.
- **p17 → p17c**: the *push* policy already knows how to kick, so it learned to kick **from the riding stance**. A
  "stance bridge" moved its starting states step by step (alpha 0 → 1) from its own pre-kick pose to the half-pipe
  policy's riding pose.
- **p17d** trained on **real hand-over states recorded from full chain runs**, at their true low speeds of
  0.20–0.53 m/s.
  - Chain completion on held-out seeds rose from **8/64 to 23/64**.
  - A review then caught a new failure: after the foot push, the half-pipe policy sometimes **squatted** down to its
    joint limits.
- **r10** fine-tuned the half-pipe policy on states recorded right after a foot push.
  - It added a head-height cost and a hip-servo stall cost.
  - Ramp squats dropped from 23 to 1, and time with the hip servos at the current cap fell from about 99 % to 20 %.
- **p17e** added stricter joint-limit and knee costs to the foot push. Its training metric for time past the limits
  (`seq_p17e_past`: the share of training steps with a leg joint more than 0.5° past its stop) fell from
  about 14.5 % to 8.5 %, with no loss of success. In the final chain eval the same measure is 9.5–12.8 %; see
  [realism.md](realism.md).

## 5. Final evaluation (2026-11-22)

| | |
|---|---|
| Policies | push p16 · manual m6 · half-pipe r10 · foot push p17e |
| Settings | full realism, 8 cm walls, foot-push window from 0.20 m/s |
| Chain complete (6 batches × 32 runs; seeds 3, 4, 5, 5, 5, 5) | 13, 9, 13, 16, 8, 11 → **70 / 192 ≈ 36 %** |
| Run shown in the videos | the top-scoring run of the batch with **11 / 32** |

The same seed can give a different count on a rerun (for example 16 vs 8), because GPU simulation isn't bit-exact
from run to run.

Independent reviews (technical, realism, viewer) signed off on the result as **"sim-stage OK, not deployable"**. See
the limitations in the README.

## Process notes that helped

- **Every long training run had an independent review first.** A second agent or person checked the config and the
  plan before GPU time was spent.
- **Claims are made only from evals.** Each eval reports a per-phase "funnel" (how many runs reach each phase), plus
  videos of a typical run and the best run.
- **Held-out seeds.** Evaluate on random seeds the policy never trained on. p17d's chain hand-over bank was recorded
  from seeds 0–2, so the final eval used seeds 3–5.
