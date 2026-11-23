# micro-skater tools/isaac_render: the author's Isaac Sim replay pipeline. Apache-2.0 (see LICENSE).
"""Composite native 9x16 / 4x5 chain renders (edit_cine.py timelines) into the final mp4s.

    python compose_cine.py --frames DIR --timeline T.npz --out video.mp4 [--grade cine|none]
    python compose_cine.py --frames DIR --timeline T.npz --still K,K,... --still-dir D

Overlays (everything else is the render):
  - the phase label, bottom-left (Push / Step on / Ride / Manual / Set down / Carve in / Half-pipe / Foot push), from
    the timeline's per-frame phase (MuJoCo stage + owner + contact logs); hidden on the title / end frames
  - a small "SIMULATED · MuJoCo → Isaac Sim replay" tag top-left on every live frame
  - "SLOW MOTION k×" badge top-right on EVERY frame played slower than real time (k = the slowest speed of that ramp),
    "SPEED-UP k×" on every frame played faster (k = the fastest)
  - title card over the held first frame, the honest end card (edit_cine.py card_lines) over the held last frame
  - --grade cine (default): a gentle filmic grade (soft S-curve, warm highlights / cool shadows, vignette) on the
    rendered pixels only; no content is added or removed
The 9x16 layout keeps text out of the Shorts / Reels UI (top ~150 px, bottom ~420 px, right button rail).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONTS = os.environ.get("FONTS_DIR", os.path.join(os.environ.get("WINDIR", "C:" + "\\Windows"), "Fonts"))
ORANGE = (255, 138, 31)
PROTECTED: set[str] = set()   # output file names this pipeline must never overwrite

ap = argparse.ArgumentParser()
ap.add_argument("--frames", required=True)
ap.add_argument("--timeline", required=True)
ap.add_argument("--out", default=None)
ap.add_argument("--crf", type=int, default=17)
ap.add_argument("--grade", choices=["cine", "none"], default="cine")
ap.add_argument("--still", default=None)
ap.add_argument("--still-dir", default=None)
ap.add_argument("--range", default=None, help="a:b, encode only these frames (test renders)")
a = ap.parse_args()
if a.out:
    assert os.path.basename(a.out) not in PROTECTED, "protected file name"

tl = np.load(a.timeline)
fps = int(tl["fps"])
shots = json.loads(str(tl["shots"]))
N = len(tl["shot"])
starts = np.cumsum([0] + [s["frames"] for s in shots])
FMT = str(tl["format"])
V = FMT == "9x16"
W, H = (1080, 1920) if V else (1080, 1350)
NAMES = json.loads(str(tl["phase_names"]))
PHASE = tl["phase"].astype(int)
SPEED = tl["speed"].astype(float)
S = 1.0


def F(name, size):
    for n in ([name] if isinstance(name, str) else name):
        p = os.path.join(FONTS, n)
        if os.path.exists(p):
            return ImageFont.truetype(p, int(size))
    return ImageFont.load_default()


F_BLACK = lambda s: F(["seguibl.ttf", "segoeuib.ttf", "arialbd.ttf"], s)
F_BOLD = lambda s: F(["segoeuib.ttf", "arialbd.ttf"], s)
F_SEMI = lambda s: F(["seguisb.ttf", "segoeui.ttf", "arial.ttf"], s)
F_REG = lambda s: F(["segoeui.ttf", "arial.ttf"], s)


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


# ── grade ────────────────────────────────────────────────────────────────────
_yy, _xx = np.mgrid[0:H, 0:W]
_r = np.hypot((_xx - W / 2) / (W / 2), (_yy - H / 2) / (H / 2))
VIG = (1.0 - 0.20 * np.clip((_r - 0.55) / 0.85, 0, 1) ** 1.6)[..., None].astype(np.float32)
del _yy, _xx, _r


def grade(im):
    if a.grade == "none":
        return im
    x = np.asarray(im, np.float32) / 255.0
    lum = (0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2])[..., None]
    # soft S-curve around mid grey (contrast +12 %), toes / shoulders kept
    y = x + 0.12 * (x - 0.5) * (1 - np.abs(2 * x - 1))
    # split tone: warm highlights, slightly cool shadows
    hi = np.clip((lum - 0.45) / 0.55, 0, 1)
    lo = np.clip((0.45 - lum) / 0.45, 0, 1)
    y = y + hi * np.array([0.030, 0.012, -0.020], np.float32) + lo * np.array([-0.012, 0.0, 0.018], np.float32)
    y = y * VIG
    return Image.fromarray((np.clip(y, 0, 1) * 255 + 0.5).astype(np.uint8))


# ── overlays ─────────────────────────────────────────────────────────────────
def shadowed_text(layer, xy, text, font, fill, alpha, anchor="la", shadow=6):
    if alpha <= 0 or not text:
        return
    sh = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    ImageDraw.Draw(sh).text((xy[0] + 2, xy[1] + 3), text, font=font, fill=(0, 0, 0, int(170 * alpha)), anchor=anchor)
    sh = sh.filter(ImageFilter.GaussianBlur(shadow))
    layer.alpha_composite(sh)
    ImageDraw.Draw(layer).text(xy, text, font=font, fill=fill[:3] + (int(255 * alpha),), anchor=anchor)


def vignette(layer, alpha, strength=150):
    g = np.zeros((H, W, 4), np.uint8)
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.hypot((xx - W / 2) / (W / 2), (yy - H / 2) / (H / 2))
    g[..., 3] = (np.clip(0.35 + 0.65 * r, 0, 1) * strength * alpha).astype(np.uint8)
    layer.alpha_composite(Image.fromarray(g, "RGBA"))


def shade(layer, alpha, y0, y1, strength, top=True):
    """vertical gradient band (dark), strongest at the top (top=True) or bottom edge"""
    g = np.zeros((H, W, 4), np.uint8)
    yy = np.arange(H)[:, None].astype(float)
    u = (y1 - yy) / (y1 - y0) if top else (yy - y0) / (y1 - y0)
    g[..., 3] = (np.clip(u, 0, 1) ** 1.3 * strength * alpha).astype(np.uint8)
    layer.alpha_composite(Image.fromarray(g, "RGBA"))


TAG_Y = 172 if V else 36
LABEL_Y = H - (440 if V else 120)           # baseline of the phase label (9x16: above the Shorts caption zone)
# safe placement (per shot): the label / tag go where the robot + board are not (edit_cine subj_bbox, NDC)
LABEL_POS = {"bl": LABEL_Y, "br": LABEL_Y, "tl": TAG_Y + (190 if V else 150)}
TAG_POS = {"tl": (44, TAG_Y), "bot": (44, LABEL_Y + (70 if V else 52)), "tr": (W - 44 - 560, TAG_Y + 66)}
LABEL_W, LABEL_H, TAG_W, TAG_H = 406, 96, 600, 56


BADGE_W, BADGE_H = 340, 62
BADGE_Y = {"tr": TAG_Y - 2, "tl": TAG_Y - 2, "tl2": TAG_Y + 64, "br": LABEL_Y - 58}


def badge_xy(pos, w):
    """top-left of the badge text for a placement: right-aligned (tr / br) or left-aligned (tl / tl2)"""
    return (W - 48 - w if pos in ("tr", "br") else 60), BADGE_Y[pos]


def badge_rect(pos):
    x = W - 48 - (BADGE_W - 32) if pos in ("tr", "br") else 60
    y = BADGE_Y[pos]
    return (x - 16, y - 8, x + BADGE_W - 16, y + BADGE_H - 8)


def label_rect(pos):
    by = LABEL_POS[pos]
    x0 = 40 if pos != "br" else W - 40 - LABEL_W
    return (x0, by - LABEL_H + 16, x0 + LABEL_W, by + 22)


def tag_rect(pos):
    x, y = TAG_POS[pos]
    return (x - 14, y - 12, x + TAG_W, y + TAG_H - 12)


def inter(r, q):
    return max(0.0, min(r[2], q[2]) - max(r[0], q[0])) * max(0.0, min(r[3], q[3]) - max(r[1], q[1]))


def plan_overlays():
    """per shot: (label pos, tag pos) minimising the overlap with the robot + board over the shot (the default is
    kept unless it overlaps); returns {shot index: (label, tag)} and a printable report"""
    if "subj_bbox" not in tl.files:
        return {}, []
    bb = tl["subj_bbox"]
    px = np.stack([(bb[:, 0] + 1) / 2 * W, (1 - bb[:, 3]) / 2 * H, (bb[:, 2] + 1) / 2 * W, (1 - bb[:, 1]) / 2 * H], 1)
    plan, rep = {}, []
    for si, s in enumerate(shots):
        if s.get("card"):
            continue
        ks = range(starts[si], starts[si + 1])
        best = None
        for lp, pen_l in (("bl", 0.0),):          # the label stays bottom-left (the brief); shots are framed for it
            for tp, pen_t in (("tl", 0.0), ("bot", 0.02), ("tr", 0.03)):
                if lp == "tl" and tp == "tl":
                    continue
                lr, tr = label_rect(lp), tag_rect(tp)
                if inter(lr, tr) > 0:
                    continue
                ol = np.mean([inter(lr, px[k]) for k in ks]) / (LABEL_W * LABEL_H)
                ot = np.mean([inter(tr, px[k]) for k in ks]) / (TAG_W * TAG_H)
                sc = ol + 0.5 * ot + pen_l + pen_t
                if best is None or sc < best[0] - 1e-9:
                    best = (sc, lp, tp, ol, ot)
        plan[si] = best[1:3]
        rep.append(f"{s['name']:12s} label {best[1]:3s} ({best[3]:.0%} on robot)  tag {best[2]:3s} ({best[4]:.0%})")
    return plan, rep


def tag(layer, alpha=0.85, pos="tl"):
    d = ImageDraw.Draw(layer)
    fb, f = F_BOLD(27), F_SEMI(27)
    x, y = TAG_POS[pos]
    head, rest = "SIMULATED", ""        # r3 (author: fewer words): the one-word tag
    wh = d.textlength(head, font=fb)
    bb = d.textbbox((x + 22 + wh, y), rest, font=f) if rest else d.textbbox((x + 20, y), head, font=fb)
    d.rounded_rectangle((x - 12, bb[1] - 9, bb[2] + 14, bb[3] + 9), radius=12, fill=(0, 0, 0, int(120 * alpha)))
    cy = (bb[1] + bb[3]) / 2
    d.ellipse((x, cy - 5, x + 10, cy + 5), fill=ORANGE + (int(230 * alpha),))
    d.text((x + 20, y), head, font=fb, fill=ORANGE + (int(255 * alpha),))
    d.text((x + 22 + wh, y), rest, font=f, fill=(255, 255, 255, int(225 * alpha)))


def slow_badge(layer, k, alpha, pos="tr"):
    if alpha <= 0:
        return
    d = ImageDraw.Draw(layer)
    f = F_BOLD(34)
    text = f"SLOW MOTION  {k:g}×"
    bb = d.textbbox((0, 0), text, font=f)
    w = bb[2] - bb[0]
    x, y = badge_xy(pos, w)
    d.rounded_rectangle((x - 16, y - 8, x + w + 16, y + (bb[3] - bb[1]) + 16), radius=10,
                        fill=ORANGE + (int(225 * alpha),))
    d.text((x, y - bb[1] + 2), text, font=f, fill=(20, 20, 20, int(255 * alpha)))


def fast_badge(layer, k, alpha, pos="tr"):
    if alpha <= 0:
        return
    d = ImageDraw.Draw(layer)
    f = F_BOLD(34)
    text = f"SPEED-UP  {k:g}×"
    bb = d.textbbox((0, 0), text, font=f)
    w = bb[2] - bb[0]
    x, y = badge_xy(pos, w)
    d.rounded_rectangle((x - 16, y - 8, x + w + 16, y + (bb[3] - bb[1]) + 16), radius=10,
                        fill=(245, 245, 245, int(225 * alpha)))
    d.text((x, y - bb[1] + 2), text, font=f, fill=(20, 20, 20, int(255 * alpha)))


def phase_label(layer, text, alpha, slide, pos="bl", accent=ORANGE):
    if alpha <= 0 or not text:
        return
    by = LABEL_POS[pos]
    if pos == "tl":
        shade(layer, alpha * 0.8, by - 200, by + 90, 100, top=True)
    elif not V:
        shade(layer, alpha * 0.9, by - 170, by + 120, 120, top=False)
    else:
        shade(layer, alpha * 0.9, by - 200, by + 260, 110, top=False)
    f = F_BOLD(64 if V else 58)
    d = ImageDraw.Draw(layer)
    tw = d.textlength(text, font=f)
    x0 = 58 if pos != "br" else W - 58 - 26 - tw
    x = x0 + (1 - slide) * 18
    y = by + (1 - slide) * 10
    bb = d.textbbox((x + 26, y), text, font=f, anchor="ls")
    d.rounded_rectangle((x, bb[1] + 4, x + 9, bb[3] + 2), radius=4, fill=accent + (int(255 * alpha),))
    shadowed_text(layer, (x + 26, y), text, f, (255, 255, 255), alpha, anchor="ls", shadow=7)


RED = (235, 64, 52)


def card_bloopers(layer, alpha):
    """the BLOOPERS title over the first blooper's opening second"""
    if alpha <= 0:
        return
    shade(layer, alpha, 0, H * 0.5, 140)
    cy = 520 if V else H * 0.17
    shadowed_text(layer, (W / 2, cy), "BLOOPERS", F_BLACK(150 if V else 132), (255, 255, 255), alpha, anchor="mm",
                  shadow=10)


def stamp(layer, alpha, u, k=None):
    """the freeze-frame stamp: a red 'NOPE' that punches in (u: 0 -> 1 over its first 0.2 s), placed above the
    robot + board (subj_bbox) if there is room, else below it (above the label), else top"""
    if alpha <= 0:
        return
    sc = 1.35 - 0.35 * ease(u)
    f = F_BLACK(int(150 * sc))
    tmp = Image.new("RGBA", (W, 420), (0, 0, 0, 0))
    d = ImageDraw.Draw(tmp)
    bb = d.textbbox((W / 2, 210), "NOPE", font=f, anchor="mm")
    d.rounded_rectangle((bb[0] - 34, bb[1] - 24, bb[2] + 34, bb[3] + 24), radius=22, outline=RED + (int(255 * alpha),),
                        width=12)
    d.text((W / 2, 210), "NOPE", font=f, anchor="mm", fill=RED + (int(255 * alpha),))
    tmp = tmp.rotate(8, resample=Image.BICUBIC, center=(W / 2, 210))
    half = 125
    top_lim, bot_lim = TAG_Y + 70 + half, LABEL_Y - 110 - half
    yc = 720 if V else H * 0.30
    if k is not None and "subj_bbox" in tl.files:
        b = tl["subj_bbox"][k]
        y_top, y_bot = (1 - b[3]) / 2 * H, (1 - b[1]) / 2 * H
        if y_top - 30 - half >= top_lim:
            yc = y_top - 30 - half
        elif y_bot + 30 + half <= bot_lim:
            yc = y_bot + 30 + half
        else:
            yc = top_lim
    layer.alpha_composite(tmp, (0, int(yc - 210)))


def card_title(layer, alpha):
    if alpha <= 0:
        return
    vignette(layer, alpha, 110)
    shade(layer, alpha, 0, H * 0.55, 150)
    cy = 520 if V else H * 0.17
    shadowed_text(layer, (W / 2, cy), "MICRODUCK", F_BLACK(138 if V else 126), (255, 255, 255), alpha, anchor="mm",
                  shadow=10)
    sub = str(tl["title_sub"]) if "title_sub" in tl.files else "push → manual → half-pipe → foot-push"
    shadowed_text(layer, (W / 2, cy + 112), sub, F_SEMI(56), ORANGE, alpha, anchor="mm")


def card_end(layer, alpha):
    """the end card: 3 lines (>= 40 px) + 1 footnote (28 px) centred over the blurred, darkened last frame
    (compose() does the blur / darken)"""
    if alpha <= 0:
        return
    style = {"l1": (F_BOLD, 48, (255, 255, 255)), "l2": (F_SEMI, 42, ORANGE), "l3": (F_SEMI, 40, (245, 245, 245)),
             "foot": (F_REG, 28, (222, 222, 222))}
    lines = json.loads(str(tl["card_lines"])) if "card_lines" in tl.files else []
    gap = {"l1": 1.75, "l2": 1.7, "l3": 2.3, "foot": 1.4}
    tot = sum(style[kd][1] * gap[kd] for _, kd in lines if kd in style)
    y = (H * (0.42 if V else 0.47)) - tot / 2
    meas = ImageDraw.Draw(layer)
    for text, kind in lines:
        if kind not in style:
            continue
        fn, sz, col = style[kind]
        while meas.textlength(text, font=fn(sz)) > W - 60 and sz > (40 if kind != "foot" else 28):
            sz -= 1
        shadowed_text(layer, (W / 2, y), text, fn(sz), col, alpha, anchor="mm", shadow=4)
        y += style[kind][1] * gap[kind]


# slow-motion runs: each contiguous run of frames under real time is badged with the slowest speed of that run
SLOWK = np.zeros(N)
k = 0
while k < N:
    if 0 < SPEED[k] < 0.99:
        j = k
        while j + 1 < N and 0 < SPEED[j + 1] < 0.99:
            j += 1
        SLOWK[k:j + 1] = round(float(SPEED[k:j + 1].min()), 2)
        k = j + 1
    else:
        k += 1
# speed-up runs: each contiguous run of frames faster than real time is badged with the fastest speed of that run
FASTK = np.zeros(N)
k = 0
while k < N:
    if SPEED[k] > 1.01:
        j = k
        while j + 1 < N and SPEED[j + 1] > 1.01:
            j += 1
        FASTK[k:j + 1] = round(float(SPEED[k:j + 1].max()), 1)
        k = j + 1
    else:
        k += 1
# phase label change times (for the fade-in), per frame: frames since the label last changed
SINCE = np.zeros(N)
for k in range(1, N):
    SINCE[k] = 0 if PHASE[k] != PHASE[k - 1] else SINCE[k - 1] + 1


PLAN, PLAN_REP = plan_overlays()


def plan_badges():
    """per shot with a speed badge: the placement (top-right first) clear of the robot + board, the tag and the label"""
    out, rep = {}, []
    if "subj_bbox" not in tl.files:
        return out, rep
    bb = tl["subj_bbox"]
    px = np.stack([(bb[:, 0] + 1) / 2 * W, (1 - bb[:, 3]) / 2 * H, (bb[:, 2] + 1) / 2 * W, (1 - bb[:, 1]) / 2 * H], 1)
    for si, s in enumerate(shots):
        ks = [k for k in range(starts[si], starts[si + 1]) if SLOWK[k] > 0 or FASTK[k] > 0]
        if not ks or s.get("card"):
            continue
        lp, tp = PLAN.get(si, ("bl", "tl"))
        best = None
        for bp, pen in (("tr", 0.0), ("tl", 0.02), ("tl2", 0.03), ("br", 0.04)):
            r = badge_rect(bp)
            if inter(r, tag_rect(tp)) > 0 or inter(r, label_rect(lp)) > 0:
                continue
            ov = float(np.mean([inter(r, px[k]) for k in ks])) / (BADGE_W * BADGE_H)
            if best is None or ov + pen < best[0]:
                best = (ov + pen, bp, ov)
        out[si] = best[1]
        rep.append(f"{s['name']:12s} badge {best[1]:3s} ({best[2]:.0%} on robot)")
    return out, rep
print("overlay placement:", *PLAN_REP, sep="\n  ")
# frames since the label last changed text or moved (a placement change at a cut re-runs the fade-in, so the label
# never jumps across the frame mid-phase)
AGE = np.zeros(N)
for k in range(1, N):
    s0, s1 = int(tl["shot"][k - 1]), int(tl["shot"][k])
    moved = s0 != s1 and (PLAN.get(s0, ("bl", "tl"))[0] != PLAN.get(s1, ("bl", "tl"))[0]
                          or shots[s1].get("blooper") is not None)
    AGE[k] = 0 if (PHASE[k] != PHASE[k - 1] or moved) else AGE[k - 1] + 1


BPLAN, BPLAN_REP = plan_badges()
print("badge placement:", *BPLAN_REP, sep="\n  ")


def overlay(k):
    si = int(tl["shot"][k])
    s = shots[si]
    fi = k - starts[si]
    t, T = fi / fps, s["frames"] / fps
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    card = s.get("card")
    if card == "title":
        card_title(layer, ease((t - 0.1) / 0.45) * ease((T - t - 0.1) / 0.4))
    elif card == "end":
        card_end(layer, ease((t - 0.2) / 0.6))
    live = not s.get("hold")
    bl = s.get("blooper")
    if bl is not None:
        if bl == 0:
            card_bloopers(layer, ease((t - 0.05) / 0.3) * ease((1.5 - t) / 0.35))
        if SPEED[k] == 0:                       # the freeze: the step the eval ended this run
            k0 = k - int(np.sum(SPEED[starts[si]:k + 1] == 0)) + 1
            stamp(layer, 1.0, (k - k0) / (0.2 * fps), k)
    if live and PHASE[k] >= 0:
        u = AGE[k] / (0.25 * fps)
        if bl is not None and bl == 0 and t < 1.5:
            u = 0.0
        phase_label(layer, NAMES[PHASE[k]], ease(u * 1.4) if u < 1 else 1.0, ease(u), PLAN.get(si, ("bl", "tl"))[0],
                    accent=RED if bl is not None else ORANGE)
    if live and SLOWK[k] > 0:
        slow_badge(layer, SLOWK[k], min(1.0, (1.0 - SPEED[k]) / 0.06), BPLAN.get(si, "tr"))
    if live and FASTK[k] > 0:
        fast_badge(layer, FASTK[k], min(1.0, (SPEED[k] - 1.0) / 0.12), BPLAN.get(si, "tr"))
    if card is None:
        tag(layer, 0.85, PLAN.get(si, ("bl", "tl"))[1])
    return layer


FRAME_DIRS = a.frames.split(",")         # r3: several independent render passes of the same frames, averaged
                                         # (the real-time renderer's per-frame noise / shimmer drops ~1/sqrt(passes))


def load(k):
    acc = None
    for d_ in FRAME_DIRS:
        im = Image.open(os.path.join(d_, f"frame_{k:05d}.png")).convert("RGB")
        assert im.size == (W, H), (k, im.size)
        x = np.asarray(im, np.float32)
        acc = x if acc is None else acc + x
    im = Image.fromarray(np.clip(acc / len(FRAME_DIRS) + 0.5, 0, 255).astype(np.uint8))
    return grade(im).convert("RGBA")


# dips to black (0.25 s each side) into the bloopers and into the end card after them
DIPS = []
_bl = [si for si, s in enumerate(shots) if s.get("blooper") is not None]
if _bl:
    DIPS = [int(starts[_bl[0]]), int(starts[_bl[-1] + 1])]


def dip(k):
    v = 1.0
    for b in DIPS:
        v = min(v, ease(abs(k + 0.5 - b) / (0.25 * fps)))
    return v


def compose(k):
    im = load(k)
    si = int(tl["shot"][k])
    if shots[si].get("card") == "end":            # end card: blur + darken the held frame under the text
        t_ = (k - starts[si]) / fps
        aa = ease((t_ - 0.05) / 0.6)
        if aa > 0:
            bg = im.filter(ImageFilter.GaussianBlur(16))
            bg = Image.blend(bg, Image.new("RGBA", im.size, (8, 8, 10, 255)), 0.55)
            im = Image.blend(im, bg, aa)
    im.alpha_composite(overlay(k))
    fade = ease((k + 1) / 4.0) * ease((N - 1 - k) / (0.9 * fps)) * dip(k)
    if fade < 1:
        im = Image.blend(Image.new("RGBA", im.size, (0, 0, 0, 255)), im, fade)
    return im.convert("RGB")


if a.still:
    os.makedirs(a.still_dir, exist_ok=True)
    tagn = os.path.splitext(os.path.basename(a.timeline))[0].replace("timeline_", "")
    for k in map(int, a.still.split(",")):
        compose(k).save(os.path.join(a.still_dir, f"comp_{tagn}_{k:05d}.png"))
    print("stills ->", a.still_dir)
    raise SystemExit

R0, R1 = (map(int, a.range.split(":")) if a.range else (0, N))
R1 = min(R1, N)
missing = [k for k in range(R0, R1) for d_ in FRAME_DIRS if not os.path.exists(os.path.join(d_, f"frame_{k:05d}.png"))]
if missing:
    raise SystemExit(f"{len(missing)} frames missing, first {missing[:5]}")
ff = shutil.which("ffmpeg")
enc = subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                        "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", str(a.crf),
                        "-profile:v", "high", "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out],
                       stdin=subprocess.PIPE)
for k in range(R0, R1):
    enc.stdin.write(np.asarray(compose(k)).tobytes())
    if k % 300 == 0:
        print(f"compose {k}/{N}", flush=True)
enc.stdin.close()
assert enc.wait() == 0
print("wrote", a.out, f"{(R1 - R0) / fps:.1f} s")
