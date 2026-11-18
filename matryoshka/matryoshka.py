#!/usr/bin/env python3
"""
Matryoshka opening animation for Instagram (1080x1920, 30 fps).

Every doll is painted from two texture images that you can replace:
    textures/doll1_body.png  ... doll5_body.png   (whole doll front, 620x1000 canvas)
    textures/doll1_face.png  ... doll5_face.png   (face, fitted into the face oval)

See textures/template.png for where the silhouette, face oval and split line sit.

Usage:
    python matryoshka.py placeholders   # (re)create placeholder textures + template
    python matryoshka.py render         # render matryoshka.mp4
    python matryoshka.py preview        # write a few still frames to preview/

Requires: Python 3.9+, Pillow, numpy, ffmpeg on PATH.
"""
import json
import math
import os
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(HERE, "textures")

# ---------------------------------------------------------------- settings ---
DEFAULT_CONFIG = {
    "output": "matryoshka.mp4",
    "width": 1080,
    "height": 1920,
    "fps": 30,
    "num_dolls": 5,
    "title": "MATRYOSHKA",
    "subtitle": "One inside the other",
    "caption": "Each one holds the next.",
    "labels": ["", "", "", "", ""],       # text under each doll in the final row
    "background_top": "#1d1b2e",
    "background_bottom": "#3a2340",
    "text_color": "#f4e9d8",
    "font": None,                          # path to a .ttf, or null for automatic
}

# Doll sprite master size (texture canvas size)
SW, SH = 620, 1000
SPLIT = 0.52          # split line, fraction of height from the top
FACE = dict(cx=0.5, cy=0.215, rx=0.215, ry=0.125)   # fractions of SW / SH


def load_config():
    cfg = dict(DEFAULT_CONFIG)
    p = os.path.join(HERE, "config.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            cfg.update(json.load(f))
    return cfg


def font(cfg, size, bold=True):
    cands = [cfg.get("font")] if cfg.get("font") else []
    cands += [
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "C:/Windows/Fonts/georgiab.ttf" if bold else "C:/Windows/Fonts/georgia.ttf",
        "/System/Library/Fonts/Supplemental/Georgia Bold.ttf" if bold else
        "/System/Library/Fonts/Supplemental/Georgia.ttf",
        "/Library/Fonts/Arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for c in cands:
        if c and os.path.exists(c):
            return ImageFont.truetype(c, size)
    return ImageFont.load_default(size)


# ---------------------------------------------------------------- geometry ---
# Half-width profile (fraction of sprite width) down the doll, top -> bottom.
PROFILE = [
    (0.000, 0.00), (0.012, 0.15), (0.045, 0.26), (0.10, 0.325), (0.18, 0.35),
    (0.27, 0.34), (0.345, 0.30), (0.39, 0.295), (0.46, 0.35), (0.58, 0.44),
    (0.72, 0.49), (0.84, 0.495), (0.93, 0.465), (0.985, 0.42), (1.0, 0.40),
]


def _catmull(points, n=400):
    pts = np.array(points, float)
    pts = np.vstack([pts[0], pts, pts[-1]])
    out = []
    for i in range(1, len(pts) - 2):
        p0, p1, p2, p3 = pts[i - 1], pts[i], pts[i + 1], pts[i + 2]
        for t in np.linspace(0, 1, n // (len(pts) - 3), endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t +
                              (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
                              (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(pts[-2])
    return np.array(out)


def silhouette_polygon(w, h):
    prof = _catmull(PROFILE)
    right = [(w / 2 + hw * w, y * h) for y, hw in prof]
    left = [(w / 2 - hw * w, y * h) for y, hw in prof[::-1]]
    return right + left


def silhouette_mask(w, h, ss=3):
    m = Image.new("L", (w * ss, h * ss), 0)
    ImageDraw.Draw(m).polygon(silhouette_polygon(w * ss, h * ss), fill=255)
    return m.resize((w, h), Image.LANCZOS)


def face_box(w, h):
    cx, cy = FACE["cx"] * w, FACE["cy"] * h
    rx, ry = FACE["rx"] * w, FACE["ry"] * h
    return (cx - rx, cy - ry, cx + rx, cy + ry)


def ellipse_mask(size, box, ss=3):
    w, h = size
    m = Image.new("L", (w * ss, h * ss), 0)
    ImageDraw.Draw(m).ellipse([c * ss for c in box], fill=255)
    return m.resize((w, h), Image.LANCZOS)


def cover(img, w, h):
    """Scale + center-crop img to exactly w x h."""
    img = img.convert("RGBA")
    s = max(w / img.width, h / img.height)
    img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.LANCZOS)
    l, t = (img.width - w) // 2, (img.height - h) // 2
    return img.crop((l, t, l + w, t + h))


# ----------------------------------------------------- doll sprite builder ---
def build_doll(i):
    """Return (top_sprite, bottom_sprite, full_sprite) RGBA at SW x SH."""
    body = Image.open(os.path.join(TEX, f"doll{i}_body.png"))
    face = Image.open(os.path.join(TEX, f"doll{i}_face.png"))
    canvas = cover(body, SW, SH)

    fb = face_box(SW, SH)
    fw, fh = round(fb[2] - fb[0]), round(fb[3] - fb[1])
    fimg = cover(face, fw, fh)
    fmask = ellipse_mask((fw, fh), (0, 0, fw, fh))
    fimg.putalpha(Image.fromarray(np.minimum(np.array(fimg.getchannel("A")),
                                             np.array(fmask))))
    canvas.alpha_composite(fimg, (round(fb[0]), round(fb[1])))
    # thin rim around the face
    d = ImageDraw.Draw(canvas)
    d.ellipse(fb, outline=(60, 30, 20, 140), width=4)

    # cylindrical shading + highlight (gives the lacquered-wood look)
    arr = np.array(canvas).astype(np.float32)
    x = np.linspace(-1, 1, SW)
    shade = 0.62 + 0.38 * np.sqrt(np.clip(1 - x ** 2, 0, 1))
    hl = np.exp(-((x + 0.38) / 0.09) ** 2) * 0.22
    arr[..., :3] = arr[..., :3] * shade[None, :, None] + 255 * hl[None, :, None]
    arr = np.clip(arr, 0, 255)
    sil = np.array(silhouette_mask(SW, SH)).astype(np.float32) / 255
    arr[..., 3] = arr[..., 3] * sil
    full = Image.fromarray(arr.astype(np.uint8))

    # seam line
    sy = round(SPLIT * SH)
    seam = Image.new("RGBA", (SW, SH), 0)
    ImageDraw.Draw(seam).line([(0, sy), (SW, sy)], fill=(40, 20, 10, 170), width=5)
    seam.putalpha(Image.fromarray((np.array(seam.getchannel("A")) * sil).astype(np.uint8)))
    full.alpha_composite(seam)

    top = full.copy()
    top.paste((0, 0, 0, 0), (0, sy, SW, SH))
    bottom = full.copy()
    bottom.paste((0, 0, 0, 0), (0, 0, SW, sy))
    return top, bottom, full


def rim_sprite():
    """Dark inner opening ellipse at the split line (drawn behind inner doll)."""
    prof = _catmull(PROFILE)
    hw = np.interp(SPLIT, prof[:, 0], prof[:, 1]) * SW
    rh = hw * 0.22
    img = Image.new("RGBA", (SW, round(rh * 2) + 4), 0)
    ImageDraw.Draw(img).ellipse((SW / 2 - hw + 6, 2, SW / 2 + hw - 6, rh * 2 + 2),
                                fill=(25, 12, 8, 255))
    return img, rh


# ------------------------------------------------------------ placeholders ---
PALETTE = [
    ((178, 34, 40), (230, 176, 52), (250, 236, 210)),
    ((32, 80, 150), (220, 70, 60), (240, 232, 214)),
    ((34, 120, 80), (240, 196, 60), (246, 238, 220)),
    ((120, 50, 130), (90, 170, 200), (246, 234, 222)),
    ((220, 140, 30), (170, 40, 40), (252, 240, 220)),
]


def flower(d, cx, cy, r, petal, center):
    for k in range(6):
        a = k * math.pi / 3
        px, py = cx + math.cos(a) * r * 0.62, cy + math.sin(a) * r * 0.62
        d.ellipse((px - r * 0.42, py - r * 0.42, px + r * 0.42, py + r * 0.42), fill=petal)
    d.ellipse((cx - r * 0.34, cy - r * 0.34, cx + r * 0.34, cy + r * 0.34), fill=center)


def make_placeholders(n):
    os.makedirs(TEX, exist_ok=True)
    for i in range(1, n + 1):
        base, accent, light = PALETTE[(i - 1) % len(PALETTE)]
        ss = 2
        W, H = SW * ss, SH * ss
        img = Image.new("RGBA", (W, H), base + (255,))
        d = ImageDraw.Draw(img)
        # headscarf band around face
        fb = [c * ss for c in face_box(SW, SH)]
        pad = 34 * ss
        d.ellipse((fb[0] - pad, fb[1] - pad, fb[2] + pad, fb[3] + pad), fill=accent)
        d.ellipse((fb[0] - pad + 14 * ss, fb[1] - pad + 14 * ss,
                   fb[2] + pad - 14 * ss, fb[3] + pad - 14 * ss), fill=base)
        # scarf knot
        kx, ky = W / 2, fb[3] + pad * 0.9
        d.polygon([(kx, ky), (kx - 60 * ss, ky + 70 * ss), (kx - 10 * ss, ky + 60 * ss)], fill=accent)
        d.polygon([(kx, ky), (kx + 60 * ss, ky + 70 * ss), (kx + 10 * ss, ky + 60 * ss)], fill=accent)
        d.ellipse((kx - 18 * ss, ky - 14 * ss, kx + 18 * ss, ky + 18 * ss), fill=accent)
        # apron
        ax0, ay0, ax1, ay1 = W * 0.17, H * 0.47, W * 0.83, H * 0.97
        d.ellipse((ax0, ay0, ax1, ay1), fill=light)
        d.ellipse((ax0 + 10 * ss, ay0 + 10 * ss, ax1 - 10 * ss, ay1 - 10 * ss), outline=accent, width=6 * ss)
        flower(d, W / 2, H * 0.70, 95 * ss, base, accent)
        for dx, dy, r in [(-120, 90, 45), (120, 90, 45), (-95, -70, 34), (95, -70, 34), (0, 190, 40)]:
            flower(d, W / 2 + dx * ss, H * 0.70 + dy * ss, r * ss, accent, base)
        # leaves
        for sx in (-1, 1):
            d.ellipse((W / 2 + sx * 150 * ss - 30 * ss, H * 0.79 - 12 * ss,
                       W / 2 + sx * 150 * ss + 30 * ss, H * 0.79 + 12 * ss), fill=(60, 130, 60))
        # hem dots
        for k in range(14):
            x = W * (0.06 + 0.88 * k / 13)
            d.ellipse((x - 9 * ss, H * 0.985 - 9 * ss, x + 9 * ss, H * 0.985 + 9 * ss), fill=accent)
        img.resize((SW, SH), Image.LANCZOS).save(os.path.join(TEX, f"doll{i}_body.png"))

        # generic placeholder face (not anyone in particular)
        F = 600
        f = Image.new("RGBA", (F, F), (250, 222, 196, 255))
        d = ImageDraw.Draw(f)
        d.pieslice((-60, -330, F + 60, 230), 0, 180, fill=(110, 62, 30))       # hair
        d.line((F / 2, 0, F / 2, 225), fill=(80, 42, 20), width=8)            # parting
        for ex in (F * 0.34, F * 0.66):
            d.ellipse((ex - 34, 300 - 40, ex + 34, 300 + 40), fill=(255, 255, 255))
            d.ellipse((ex - 22, 300 - 22, ex + 22, 300 + 22), fill=(40, 70, 130))
            d.ellipse((ex - 9, 300 - 9, ex + 9, 300 + 9), fill=(10, 10, 10))
            d.arc((ex - 48, 300 - 62, ex + 48, 300 + 10), 200, 340, fill=(40, 20, 10), width=7)
        for cx in (F * 0.25, F * 0.75):
            d.ellipse((cx - 55, 410 - 40, cx + 55, 410 + 40), fill=(238, 120, 120))
        d.ellipse((F / 2 - 10, 370 - 8, F / 2 + 10, 370 + 8), fill=(220, 150, 130))
        d.chord((F / 2 - 60, 420, F / 2 + 60, 500), 0, 180, fill=(200, 40, 60))
        d.text((F / 2, F - 40), str(i), fill=(150, 110, 90), anchor="mm",
               font=ImageFont.load_default(40))
        f.save(os.path.join(TEX, f"doll{i}_face.png"))

    # template showing where everything goes
    t = Image.new("RGB", (SW, SH), (245, 245, 245))
    d = ImageDraw.Draw(t)
    d.polygon(silhouette_polygon(SW, SH), outline=(30, 30, 30), width=3)
    d.ellipse(face_box(SW, SH), outline=(200, 40, 40), width=3)
    sy = SPLIT * SH
    d.line((0, sy, SW, sy), fill=(40, 90, 200), width=2)
    ft = ImageFont.load_default(20)
    d.text((SW / 2, FACE["cy"] * SH), "face oval\n(doll#_face.png)", fill=(200, 40, 40), anchor="mm", font=ft, align="center")
    d.text((10, sy - 26), "split line (doll opens here)", fill=(40, 90, 200), font=ft)
    d.text((10, 10), f"doll#_body.png canvas: {SW} x {SH}", fill=(30, 30, 30), font=ft)
    t.save(os.path.join(TEX, "template.png"))
    print(f"Placeholders written to {TEX}")


# --------------------------------------------------------------- animation ---
def ease(t):            # smooth in-out
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def ease_out_back(t, s=1.6):
    t = min(max(t, 0.0), 1.0) - 1
    return t * t * ((s + 1) * t + s) + 1


def seg(t, a, b):
    return 0.0 if t <= a else 1.0 if t >= b else (t - a) / (b - a)


def lerp(a, b, t):
    return a + (b - a) * t


class Scene:
    def __init__(self, cfg):
        self.cfg = cfg
        self.W, self.H = cfg["width"], cfg["height"]
        self.n = cfg["num_dolls"]
        self.dolls = [build_doll(i + 1) for i in range(self.n)]
        self.rim, self.rim_h = rim_sprite()
        self.bg = self._background()
        self._cache = {}

        # stage (big doll in the middle)
        self.stage_h = round(self.H * 0.37)
        self.stage_cx = self.W / 2
        self.stage_base = self.H * 0.65
        # row along the bottom, largest -> smallest
        ratio = 0.8
        margin, gap = self.W * 0.06, self.W * 0.02
        ws = [ratio ** k for k in range(self.n)]
        w0 = (self.W - 2 * margin - gap * (self.n - 1)) / sum(ws)
        self.row_h = [w0 * ws[k] * SH / SW for k in range(self.n)]
        xs, x = [], margin
        for k in range(self.n):
            xs.append(x + w0 * ws[k] / 2)
            x += w0 * ws[k] + gap
        self.row_cx = xs
        self.row_base = self.H * 0.885
        self.final_base = self.H * 0.70      # row's base after the outro move
        self.f_title = font(cfg, 92)
        self.f_sub = font(cfg, 44, bold=False)
        self.f_cap = font(cfg, 56)
        self.f_lab = font(cfg, 34)

        # timeline (seconds)
        self.t_intro = 1.4
        self.t_open = 2.7
        self.t_last = 1.2
        self.t_outro = 3.2
        self.duration = self.t_intro + self.t_open * (self.n - 1) + self.t_last + self.t_outro

    def _background(self):
        top = np.array(Image.new("RGB", (1, 1), self.cfg["background_top"]).getpixel((0, 0)), float)
        bot = np.array(Image.new("RGB", (1, 1), self.cfg["background_bottom"]).getpixel((0, 0)), float)
        y = np.linspace(0, 1, self.H)[:, None, None]
        arr = top * (1 - y) + bot * y
        x = np.linspace(-1, 1, self.W)[None, :, None]
        yy = np.linspace(-1, 1, self.H)[:, None, None]
        arr = arr * (1 - 0.35 * (x ** 2 + 0.5 * yy ** 2))
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        # floor glow
        glow = Image.new("L", (self.W, self.H), 0)
        ImageDraw.Draw(glow).ellipse((-200, self.H * 0.58, self.W + 200, self.H * 1.15), fill=60)
        glow = glow.filter(ImageFilter.GaussianBlur(120))
        img.paste((255, 210, 160), (0, 0), glow)
        return img

    # ---- drawing helpers
    def _scaled(self, img, key, h):
        h = max(2, int(round(h)))
        k = (key, h)
        if k not in self._cache:
            w = max(2, int(round(img.width * h / SH)))
            hh = max(2, int(round(img.height * h / SH)))
            self._cache[k] = img.resize((w, hh), Image.LANCZOS if h < 400 else Image.BILINEAR)
            if len(self._cache) > 400:
                self._cache.pop(next(iter(self._cache)))
        return self._cache[k]

    def paste(self, frame, img, key, h, cx, top, angle=0.0, pivot=None, alpha=1.0):
        """Paste sprite scaled to doll-height h with its top-centre at (cx, top)."""
        s = self._scaled(img, key, h)
        if alpha < 1:
            s = s.copy()
            s.putalpha(s.getchannel("A").point(lambda v: int(v * alpha)))
        if angle:
            # rotate around pivot (given in sprite px of the scaled sprite)
            px, py = pivot
            R = int(math.hypot(max(px, s.width - px), max(py, s.height - py))) + 2
            pad = Image.new("RGBA", (2 * R, 2 * R), 0)
            pad.paste(s, (int(R - px), int(R - py)))
            pad = pad.rotate(angle, resample=Image.BICUBIC, center=(R, R))
            x0 = cx - s.width / 2 + px - R
            y0 = top + py - R
            frame.paste(pad, (int(round(x0)), int(round(y0))), pad)
        else:
            frame.paste(s, (int(round(cx - s.width / 2)), int(round(top))), s)

    def shadow(self, frame, cx, base, h, alpha=1.0):
        w = h * SW / SH * 0.95
        sh = Image.new("L", (int(w * 1.4), int(h * 0.16)), 0)
        ImageDraw.Draw(sh).ellipse((w * 0.2, h * 0.03, w * 1.2, h * 0.13), fill=int(110 * alpha))
        sh = sh.filter(ImageFilter.GaussianBlur(max(2, h * 0.02)))
        frame.paste((0, 0, 0), (int(cx - sh.width / 2), int(base - h * 0.08)), sh)

    def draw_whole(self, frame, i, h, cx, base, alpha=1.0):
        self.shadow(frame, cx, base, h, alpha)
        self.paste(frame, self.dolls[i][2], ("full", i), h, cx, base - h, alpha=alpha)

    def draw_text(self, frame, t):
        d = ImageDraw.Draw(frame)
        col = self.cfg["text_color"]
        a = ease(seg(t, 0.2, 1.0))
        if a > 0 and self.cfg.get("title"):
            y = lerp(self.H * 0.075, self.H * 0.062, a)
            layer = Image.new("RGBA", frame.size, 0)
            dl = ImageDraw.Draw(layer)
            rgb = Image.new("RGB", (1, 1), col).getpixel((0, 0))
            dl.text((self.W / 2, y), self.cfg["title"], font=self.f_title, anchor="mm",
                    fill=rgb + (int(255 * a),))
            if self.cfg.get("subtitle"):
                dl.text((self.W / 2, y + 80), self.cfg["subtitle"], font=self.f_sub, anchor="mm",
                        fill=rgb + (int(200 * a),))
            frame.paste(layer, (0, 0), layer)

    # ---- one frame
    def frame(self, t):
        f = self.bg.copy()
        n, Hs = self.n, self.stage_h
        cx0, base0 = self.stage_cx, self.stage_base

        t_open0 = self.t_intro
        t_last0 = t_open0 + self.t_open * (n - 1)
        t_outro0 = t_last0 + self.t_last

        # how far into the outro row-lift we are
        o = ease(seg(t, t_outro0 + 0.3, t_outro0 + 1.4))
        row_base = lerp(self.row_base, self.final_base, o)
        row_scale = lerp(1.0, 1.0, o)

        # dolls already placed in the row
        if t < t_last0:
            placed = int(max(0, (t - t_open0) // self.t_open))   # fully moved
        else:
            placed = n - 1 + (1 if t >= t_outro0 else 0)

        for k in range(min(placed, n)):
            self.draw_whole(f, k, self.row_h[k] * row_scale, self.row_cx[k], row_base)

        if t < t_open0:
            # intro: first doll pops in
            p = ease_out_back(seg(t, 0.0, 0.9))
            if p > 0.01:
                self.draw_whole(f, 0, Hs * p, cx0, base0)
        elif t < t_last0:
            k = int((t - t_open0) // self.t_open)
            u = (t - t_open0) - k * self.t_open
            self.open_step(f, k, u)
        elif t < t_outro0:
            # last (solid) doll flies to its row slot
            k = n - 1
            p = ease(seg(t - t_last0, 0.25, 1.05))
            h = lerp(Hs, self.row_h[k], p)
            cx = lerp(cx0, self.row_cx[k], p)
            base = lerp(base0, self.row_base, p) - math.sin(p * math.pi) * 120
            self.draw_whole(f, k, h, cx, base)

        # labels + caption
        if t >= t_outro0:
            d = ImageDraw.Draw(f)
            a = ease(seg(t, t_outro0 + 1.0, t_outro0 + 1.8))
            rgb = Image.new("RGB", (1, 1), self.cfg["text_color"]).getpixel((0, 0))
            layer = Image.new("RGBA", f.size, 0)
            dl = ImageDraw.Draw(layer)
            for k, lab in enumerate(self.cfg.get("labels", [])[:n]):
                if lab:
                    dl.text((self.row_cx[k], row_base + 50), lab, font=self.f_lab, anchor="mm",
                            fill=rgb + (int(255 * a),))
            if self.cfg.get("caption"):
                dl.text((self.W / 2, self.final_base + 190), self.cfg["caption"], font=self.f_cap,
                        anchor="mm", fill=rgb + (int(255 * a),))
            f.paste(layer, (0, 0), layer)

        self.draw_text(f, t)
        # fade in/out
        fade = min(ease(seg(t, 0, 0.35)), 1 - ease(seg(t, self.duration - 0.4, self.duration)))
        if fade < 1:
            f = Image.blend(Image.new("RGB", f.size, (0, 0, 0)), f, fade)
        return f

    def open_step(self, f, k, u):
        """Doll k (on stage) opens; doll k+1 comes out; k goes to the row."""
        Hs, cx0, base0 = self.stage_h, self.stage_cx, self.stage_base
        Ws = Hs * SW / SH
        top_k, bot_k, full_k = self.dolls[k]
        inner = k + 1
        Hi = Hs * 0.8
        split_y = base0 - Hs * (1 - SPLIT)

        # phases
        lift = ease(seg(u, 0.0, 0.65))          # top lifts off
        rise = ease(seg(u, 0.35, 1.05))         # inner rises out
        close = ease(seg(u, 1.05, 1.5))         # top goes back on
        slide = ease(seg(u, 1.05, 1.55))        # outer slides left, inner lands right
        fly = ease(seg(u, 1.65, 2.55))          # outer -> row, inner -> stage

        # outer doll position
        ocx = cx0 - Ws * 0.48 * slide
        obase = base0
        oh = Hs
        if fly > 0:
            ocx = lerp(cx0 - Ws * 0.48, self.row_cx[k], fly)
            obase = lerp(base0, self.row_base, fly) - math.sin(fly * math.pi) * 140
            oh = lerp(Hs, self.row_h[k], fly)

        # inner doll position
        rise_h = Hs * 0.62
        icx = cx0
        ibase = base0 - Hs * 0.035 - rise_h * rise
        if slide > 0:
            land_cx = cx0 + Ws * 0.56
            icx = lerp(cx0, land_cx, slide)
            ibase = lerp(base0 - Hs * 0.035 - rise_h, base0, slide) - math.sin(slide * math.pi) * 60
        ih = Hi
        if fly > 0:
            icx = lerp(cx0 + Ws * 0.56, cx0, fly)
            ih = lerp(Hi, Hs, fly)
            ibase = base0

        # top-half transform (relative to closed position)
        lift_amt = lift * (1 - close)
        t_dx = Ws * 0.60 * lift_amt
        t_dy = -Hs * 0.40 * lift_amt
        t_ang = -24 * lift_amt
        wobble = math.sin(u * 9) * 2.0 * lift_amt

        out = u >= 1.05   # inner fully out -> draw it in front
        sc = oh / Hs

        if not out:
            # 1) dark opening, 2) inner doll, 3) outer bottom, 4) outer top
            self.shadow(f, ocx, obase, oh)
            rim = self._scaled(self.rim, ("rim",), oh)
            ry = split_y - rim.height / 2
            f.paste(rim, (int(ocx - rim.width / 2), int(ry)), rim)
            self.paste(f, self.dolls[inner][2], ("full", inner), ih, icx, ibase - ih)
            self.paste(f, bot_k, ("bot", k), oh, ocx, obase - oh)
            pv = (SW * sc * Hs / SH / 2 * 0 + self._scaled(top_k, ("top", k), oh).width / 2,
                  SPLIT * oh)
            self.paste(f, top_k, ("top", k), oh, ocx + t_dx, obase - oh + t_dy,
                       angle=t_ang + wobble, pivot=pv)
        else:
            if close < 1:
                self.shadow(f, ocx, obase, oh)
                self.paste(f, bot_k, ("bot", k), oh, ocx, obase - oh)
                pv = (self._scaled(top_k, ("top", k), oh).width / 2, SPLIT * oh)
                self.paste(f, top_k, ("top", k), oh, ocx + t_dx, obase - oh + t_dy,
                           angle=t_ang, pivot=pv)
            else:
                self.draw_whole(f, k, oh, ocx, obase)
            self.shadow(f, icx, ibase, ih, alpha=ease(seg(u, 1.1, 1.5)))
            self.paste(f, self.dolls[inner][2], ("full", inner), ih, icx, ibase - ih)


# -------------------------------------------------------------- commands ---
def check_textures(n):
    missing = [p for i in range(1, n + 1) for p in (f"doll{i}_body.png", f"doll{i}_face.png")
               if not os.path.exists(os.path.join(TEX, p))]
    if missing:
        print("Missing textures – creating placeholders for all dolls first.")
        make_placeholders(n)


def render(cfg):
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found on PATH")
    check_textures(cfg["num_dolls"])
    sc = Scene(cfg)
    fps = cfg["fps"]
    frames = int(round(sc.duration * fps))
    out = os.path.join(HERE, cfg["output"])
    tmp = out[:-4] + ".partial.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{sc.W}x{sc.H}", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", tmp]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(frames):
        p.stdin.write(sc.frame(i / fps).tobytes())
        if i % fps == 0:
            print(f"\r{i}/{frames} frames", end="", flush=True)
    p.stdin.close()
    if p.wait() != 0:
        sys.exit("ffmpeg failed")
    os.replace(tmp, out)
    print(f"\nDone: {out}  ({sc.duration:.1f} s)")


def preview(cfg, times=None):
    check_textures(cfg["num_dolls"])
    sc = Scene(cfg)
    d = os.path.join(HERE, "preview")
    os.makedirs(d, exist_ok=True)
    times = times or [1.2, sc.t_intro + 0.8, sc.t_intro + 1.3, sc.t_intro + 2.1,
                      sc.t_intro + sc.t_open * 2 + 0.9, sc.duration - 1.0]
    for t in times:
        sc.frame(t).save(os.path.join(d, f"t{t:05.2f}.jpg"), quality=90)
    print(f"Preview frames in {d}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "render"
    cfg = load_config()
    if cmd == "placeholders":
        make_placeholders(cfg["num_dolls"])
    elif cmd == "preview":
        preview(cfg, [float(x) for x in sys.argv[2:]] or None)
    else:
        render(cfg)
