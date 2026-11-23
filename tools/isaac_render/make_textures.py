# micro-skater tools/isaac_render: the author's Isaac Sim replay pipeline. Apache-2.0 (see LICENSE).
"""Procedural textures for the skatepark (PIL): graffiti mural, chain-link fence, skate stickers, wheel / deck art.
    python make_textures.py   (any python with PIL + numpy)"""
import math
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

TEX = os.environ.get("SKATEPARK_TEX", os.path.join("assets", "skatepark", "tex"))
OUT = os.path.join(TEX, "gen")
os.makedirs(OUT, exist_ok=True)
FONTS = os.environ.get("FONTS_DIR", os.path.join(os.environ.get("WINDIR", "C:" + "\\Windows"), "Fonts"))


def font(names, size):
    for n in names:
        p = os.path.join(FONTS, n)
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def spray(draw_img, xy, r, color, n=400, rng=random):
    """Soft spray-paint blob."""
    layer = Image.new("RGBA", draw_img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    x, y = xy
    for _ in range(n):
        a = rng.uniform(0, 2 * math.pi)
        rr = abs(rng.gauss(0, r / 2.2))
        px, py = x + rr * math.cos(a), y + rr * math.sin(a)
        s = rng.uniform(1, 3.5)
        d.ellipse((px - s, py - s, px + s, py + s), fill=color + (rng.randint(60, 160),))
    layer = layer.filter(ImageFilter.GaussianBlur(1.2))
    draw_img.alpha_composite(layer)


def mural(path, text, w=4096, h=1024, seed=3):
    rng = random.Random(seed)
    base = Image.open(os.path.join(TEX, "Concrete_Block", "Concrete_Block_BaseColor.png")).convert("RGBA")
    bg = Image.new("RGBA", (w, h))
    for x in range(0, w, base.width):
        for y in range(0, h, base.height):
            bg.paste(base, (x, y))
    img = bg
    # background colour fields (big sprayed shapes)
    fields = [(25, 110, 190), (230, 70, 40), (250, 200, 40), (40, 170, 120), (140, 60, 170)]
    for i in range(14):
        c = fields[i % len(fields)]
        spray(img, (rng.uniform(0, w), rng.uniform(0.2 * h, 0.95 * h)), rng.uniform(150, 380), c, n=22000, rng=rng)
    # bubble letters
    f = font(["seguibl.ttf", "impact.ttf", "arialbd.ttf"], int(h * 0.48))
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    bbox = d.textbbox((0, 0), text, font=f)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x0, y0 = (w - tw) // 2 - bbox[0], int(h * 0.42) - th // 2 - bbox[1]
    for dx, dy, col, sw in ((26, 26, (20, 20, 30, 255), 34), (0, 0, (15, 15, 20, 255), 30), (0, 0, (255, 255, 255, 255), 16)):
        d.text((x0 + dx, y0 + dy), text, font=f, fill=col, stroke_width=sw, stroke_fill=col)
    # gradient fill
    fill = Image.new("RGBA", (w, h))
    fa = np.zeros((h, w, 4), np.uint8)
    t = np.linspace(0, 1, h)[:, None]
    top, bot = np.array([255, 214, 40]), np.array([255, 70, 120])
    fa[..., :3] = (top * (1 - t) + bot * t)[:, None, :].repeat(w, 1).reshape(h, w, 3) if False else \
        (top[None, None, :] * (1 - t[..., None]) + bot[None, None, :] * t[..., None]).repeat(w, 1).astype(np.uint8)
    fa[..., 3] = 255
    fill = Image.fromarray(fa, "RGBA")
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).text((x0, y0), text, font=f, fill=255)
    layer.paste(fill, (0, 0), mask)
    # highlights
    hl = Image.new("L", (w, h), 0)
    ImageDraw.Draw(hl).text((x0 - 8, y0 - 10), text, font=f, fill=255)
    hl = Image.composite(hl, Image.new("L", (w, h), 0), mask.point(lambda v: 255 - v))
    img.alpha_composite(layer)
    # small tags
    tf = font(["segoescb.ttf", "segoepr.ttf", "comicbd.ttf"], 90)
    td = ImageDraw.Draw(img)
    tags = ("mjlab", "PPO", "sim2real?", "duck life", "kickflip soon", "MuJoCo")
    for i, tag in enumerate(tags):
        x = 120 + i * (w - 400) / len(tags) + rng.uniform(-40, 40)
        td.text((x, rng.uniform(0.74 * h, 0.84 * h)), tag, font=tf,
                fill=(rng.randint(0, 60), rng.randint(0, 60), rng.randint(0, 80), 230))
    # paint drips from the bottom of the letters
    for _ in range(45):
        x = rng.uniform(x0 + 60, x0 + tw - 60)
        col = 0
        ys = [y for y in range(int(y0 + bbox[1] + th * 0.6), int(y0 + bbox[1] + th + 40)) if mask.getpixel((int(x), y)) > 0]
        if not ys:
            continue
        y = max(ys)
        L = rng.uniform(25, 120)
        td.line((x, y, x, y + L), fill=(255, 110 + rng.randint(-30, 30), 80, 235), width=rng.randint(4, 8))
        td.ellipse((x - 5, y + L - 5, x + 5, y + L + 5), fill=(255, 110, 80, 235))
    g = np.array(img).astype(np.float32)
    yy = np.linspace(0, 1, h)[:, None, None]
    g[..., :3] *= (0.85 + 0.15 * (1 - yy ** 3))          # darker toward the bottom (dirt)
    img = Image.fromarray(np.clip(g, 0, 255).astype(np.uint8), "RGBA").convert("RGB")
    img.save(path)


def chainlink(path, n=512, cells=4, wire=7):
    """Diamond chain-link mesh, RGBA (alpha = wire)."""
    img = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    s = n / cells
    col = (170, 172, 168, 255)
    for i in range(-cells, 2 * cells + 1):
        x = i * s
        d.line((x, 0, x + n, n), fill=col, width=wire)
        d.line((x, n, x + n, 0), fill=col, width=wire)
    img.save(path)


def solid(path, rgb, n=8):
    Image.new("RGB", (n, n), rgb).save(path)


def deck_art(path, w=1024, h=512):
    """Bottom graphic of the deck (seen when it flips / in falls)."""
    img = Image.new("RGB", (w, h), (240, 200, 60))
    d = ImageDraw.Draw(img)
    for i in range(0, w, 64):
        d.polygon([(i, 0), (i + 32, 0), (i - 200, h), (i - 232, h)], fill=(230, 80, 50))
    f = font(["seguibl.ttf", "impact.ttf"], 150)
    d.text((w // 2, h // 2), "DUCK", font=f, fill=(20, 20, 30), anchor="mm", stroke_width=6, stroke_fill=(255, 255, 255))
    img.save(path)


if __name__ == "__main__":
    mural(os.path.join(OUT, "mural.png"), "MICRODUCK")
    chainlink(os.path.join(OUT, "chainlink.png"))
    deck_art(os.path.join(OUT, "deck_art.png"))
    print("wrote", os.listdir(OUT))
