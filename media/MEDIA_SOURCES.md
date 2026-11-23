# Where each media file came from

All media is **simulation**. The Isaac Sim clips are kinematic replays of MuJoCo runs, with no physics in Isaac. GIFs
were made with ffmpeg in one pass with `palettegen` and `paletteuse` (`stats_mode=diff`, bayer dither, lanczos
scaling), except `hero.gif` (256 colours, Floyd–Steinberg dither, see its row).

Abbreviations for the source videos and recordings (none of them is shipped except where noted):

| Short name | Source |
|---|---|
| STD | The final standard (real-time) 4x5 cut of the Isaac Sim replay (40.2 s, 2026-11-22, 4-pass render, with bloopers; the half-pipe is shortened to the part the run uses, for the render only). Made with `tools/isaac_render/chain_cine.py` from the demo run below. Shipped as `full_run_standard_4x5.mp4` |
| CINE | The final cinematic 4x5 cut of the same replay (49.8 s, 4-pass render; slow motion and a 3× speed-up at ~14 s are badged; bloopers from 28.6 s; half-pipe shortened for the render as in STD). Shipped as `full_run_cinematic_4x5.mp4` |
| MJ-demo | The MuJoCo video of the demo run that the chain eval writes (`chain_demo.mp4`; 1280×720, 50 fps, 25.3 s, with HUD) |
| MJ-kick | The MuJoCo eval video of the double-kick trick policy (r2b, iteration 2249) that warm-started the sequence: the best of 64 robots doing a kick-turn (1280×720, 8.0 s) |
| TAKE-hero | A recorded session of the keyboard play sim: one continuous MuJoCo simulation, qpos saved every 50 Hz step, plus the compiled play-park model |
| TAKE-wall | A second play-sim recording (blooper: walks into the platform wall, falls at 6.84 s) |
| TAKE-drop | A third play-sim recording (blooper: rides over the platform edge, falls at 5.10 s) |

The demo run is eval batch `final_p17e8439_s5d`, env 11 (the top-scoring run of that batch, which had 11/32 complete).

| File | Source | Cut (start s + duration s) | Size / fps |
|---|---|---|---|
| hero.gif | CINE montage (push, manual, half-pipe, foot push) | 2.5–3.85, 7.8–9.6, 18.6–20.8, 24.3–25.9 | 400 px, 6 fps, 256 colours (`stats_mode=diff`), Floyd–Steinberg dither |
| phase1_push.gif | STD | 2.0 + 4.27 | 480 px, 10 fps |
| phase2_manual.gif | STD | 6.27 + 3.2 | 480 px, 10 fps |
| phase3_halfpipe.gif | STD | 16.03 + 3.4 | 480 px, 10 fps |
| phase4_footpush_slowmo.gif | CINE | 23.4 + 3.7 (0.25× slow motion, badged) | 480 px, 10 fps |
| blooper_first_push.gif | STD | 22.6 + 3.7 (eval ep 13) | 320 px, 10 fps |
| blooper_foot_push.gif | STD | 26.3 + 3.7 (eval ep 3) | 320 px, 10 fps |
| blooper_wall_ride.gif | STD | 30.0 + 3.7 (eval ep 31) | 320 px, 10 fps |
| mujoco_training_view.gif | MJ-demo | 2.0 + 6.0 | 480 px, 10 fps |
| early_v2_paddling.gif | A MuJoCo eval render of skater v2 at iteration 500 | 2.0 + 4.0 | 360 px |
| mj_walk.gif | TAKE-hero, walk from standstill (cmd 0.4 m/s, ends at the start of a left turn) | 2.0–5.4 s | 480×600, 10 fps, side view |
| mj_run.gif | TAKE-hero, Shift-run loop on the platform (0.72 m/s, curving left) | 6.4–10.0 s | 480×600, 10 fps, side view |
| mj_walkup_stepon.gif | TAKE-hero, E pressed at 35.94 s: walk-up next to the board, step on (on board at 38.52 s) | 35.6–39.3 s | 480×600, 10 fps, fixed 3/4 front camera |
| mj_push_ride.gif | TAKE-hero, push off on the platform (39.32 s), down its 6° ramp (42.82 s), onto the floor (~45 s, up to ~2 m/s) | 39.2–45.4 s | 480×600, 10 fps, side-rear chase |
| mj_carve.gif | TAKE-hero, carving on the floor: right turn then left (1.9 → 1.4 m/s) | 45.8–50.6 s | 480×600, 10 fps, rear chase |
| mj_step_off.gif | TAKE-hero, brake (52.30 s), Q pressed (53.16 s), step off (back on foot 54.42 s), stand | 52.2–56.0 s | 480×600, 10 fps, fixed front camera |
| mj_blooper_wall.gif | TAKE-wall, walks into the side of the platform and falls (fall at 6.84 s) | 3.6–8.6 s | 480×600, 10 fps, side view (heading frozen at 6.4 s) |
| mj_blooper_skate_drop.gif | TAKE-drop, pushes off, rolls over the corner of the 20 cm high platform and falls (5.10 s) | 1.5–7.0 s | 480×600, 10 fps, side view (heading frozen at 4.8 s) |
| mj_chain_manual.gif | MJ-demo, ride → manual (nose up ~11–12°) → set-down | 4.5 + 3.1 | 480×270, 10 fps |
| mj_chain_halfpipe.gif | MJ-demo, carve into the half-pipe, wall ride 1, foot push, wall ride 2 | 14.0 + 7.4 | 480×270, 10 fps |
| mj_trick_kickturn.gif | MJ-kick, standing kick-turn: the board is turned 90° (heading error +88° → 0°) | 0.0 + 4.5 | 480×270, 10 fps |
| full_run_standard_4x5.mp4 | STD, whole video | — | 720×900, x264 CRF 25, yuv420p, faststart, no audio |
| full_run_cinematic_4x5.mp4 | CINE, whole video | — | 720×900, x264 CRF 25, yuv420p, faststart, no audio |

All STD / CINE files above were cut from the final renders. Every cut was re-located by frame matching against the
preview renders it was first made from: all of them sit at the same timestamps (the cinematic cut only grew by the
bloopers inserted after 28.6 s).

The `mj_*` files from TAKE-* are new MuJoCo renders made with `tools/render_takes_mujoco.py`: it loads a take, sets
qpos for each 50 Hz step, calls `mj_forward` (no physics is stepped; the take already is the simulation) and renders
with `mujoco.Renderer` from a free camera that follows the robot trunk (look-at smoothing 0.15 s, heading smoothing
0.8–2.5 s). Far pixels are painted with a sky gradient (the model has no skybox) and distance fog is on. Frames go
to ffmpeg at 25 fps in real time (no speed-up), then the GIF pass below resamples to 10 fps. Example:

```
python tools/render_takes_mujoco.py --npz <take>.npz --model <play park>.mjb --out mj_walk.mp4 \
    --start 2.0 --end 5.4 --width 480 --height 600 --fps 25 --distance 0.8 --elevation -12 --follow-heading --azimuth 90
```

Camera settings per file (all `--width 480 --height 600 --fps 25`):

| File | Camera arguments |
|---|---|
| mj_walk | `--distance 0.8 --elevation -12 --follow-heading --azimuth 90` |
| mj_run | `--distance 0.9 --elevation -12 --follow-heading --azimuth 90` |
| mj_walkup_stepon | `--distance 1.0 --elevation -20 --azimuth 150 --target mid` |
| mj_push_ride | `--distance 1.1 --elevation -15 --follow-heading --azimuth 70 --target board --smooth-heading 1.0` |
| mj_carve | `--distance 0.9 --elevation -18 --follow-heading --azimuth 25 --target board --smooth-heading 2.5` |
| mj_step_off | `--distance 1.0 --elevation -15 --azimuth 180 --target mid` |
| mj_blooper_wall | `--distance 1.0 --elevation -15 --follow-heading --azimuth 90 --freeze-heading-after 6.4 --smooth-heading 1.5` |
| mj_blooper_skate_drop | `--distance 1.1 --elevation -15 --follow-heading --azimuth 90 --freeze-heading-after 4.8 --smooth-heading 1.5` |

Segment times come from the take's own event list (`--list` prints it). The hero events are: walk 2.00 s, run 5.88 s,
run end 22.00 s, walk to board 28.90 s, E 35.94 s, on board 38.52 s, push 39.32 s, drop-in 42.82 s, floor 44.96 s,
carve 45.96 s, brake 52.30 s, Q 53.16 s, off board 54.44 s, run away 55.64 s, end 63.76 s.

GIF command:

```
ffmpeg -ss START -t DUR -i SRC -filter_complex "[0:v]fps=10,scale=480:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle" OUT.gif
```
