"""A deterministic, offline provider that draws flat illustrations with Pillow.

It exists so the whole pipeline (cost cap, retries, QA rejection, regeneration,
post-processing, manifest) can be run and tested without paying for images.
It can also be told to inject known defects, so the QA step has something real
to catch:

- garbled_text      the required text is drawn with letters swapped and mangled
- extra_fingers     the waving hand gets seven fingers instead of five
- off_brand_colours the brand palette is replaced with clashing colours
- missing_text      the required text is left out
- fail:N            the first N calls for that attempt raise a transient error

The picture it draws is picked from keywords in the prompt (cup, hand, leaf),
so a brief written for a real model still gets a sensible mock image.
"""
from __future__ import annotations

import io
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from .base import GenerationRequest, GenerationResult, ImageProvider, TransientProviderError

DEFECTS = {"garbled_text", "extra_fingers", "off_brand_colours", "missing_text"}
CLASH = ["#FF00CC", "#39FF14", "#7B00FF"]

_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
]


def load_font(size: int):
    for p in _FONT_CANDIDATES:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # very old Pillow
        return ImageFont.load_default()


def _hex(c: str) -> Tuple[int, int, int]:
    c = c.lstrip("#")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def garble(text: str, rng: random.Random) -> str:
    """Mangle text the way image models do: swapped, doubled and wrong letters."""
    chars = list(text)
    letters = [i for i, ch in enumerate(chars) if ch.isalpha()]
    if len(letters) >= 2:  # swap one neighbouring pair
        k = rng.randrange(len(letters) - 1)
        i, j = letters[k], letters[k + 1]
        chars[i], chars[j] = chars[j], chars[i]
    if len(letters) >= 4:  # and swap one letter for a wrong one
        k = rng.choice(letters[len(letters) // 2:])
        chars[k] = "Q" if chars[k] != "Q" else "X"
    return "".join(chars)


def parse_defect_specs(specs: List[str]) -> Dict[Tuple[str, int], List[str]]:
    """'hero:garbled_text' -> {('hero', 1): ['garbled_text']}; '@2' targets attempt 2."""
    plan: Dict[Tuple[str, int], List[str]] = {}
    for spec in specs or []:
        if ":" not in spec:
            raise ValueError(f"defect spec {spec!r} must look like asset_id:defect[@attempt]")
        asset_id, rest = spec.split(":", 1)
        attempt = 1
        if "@" in rest:
            rest, a = rest.rsplit("@", 1)
            attempt = int(a)
        if rest not in DEFECTS and not rest.startswith("fail:"):
            raise ValueError(f"unknown defect {rest!r}; choose from {sorted(DEFECTS)} or fail:N")
        plan.setdefault((asset_id, attempt), []).append(rest)
    return plan


class MockProvider(ImageProvider):
    name = "mock"

    def __init__(self, cost_per_image: float = 0.0, defects: Optional[List[str]] = None):
        self.cost_per_image = float(cost_per_image)
        self.plan = parse_defect_specs(defects or [])
        self.calls = 0
        self._failures_left: Dict[Tuple[str, int], int] = {}
        for key, items in self.plan.items():
            for d in items:
                if d.startswith("fail:"):
                    self._failures_left[key] = int(d.split(":", 1)[1])

    def estimate_cost(self, request: GenerationRequest) -> float:
        return self.cost_per_image

    def defects_for(self, request: GenerationRequest) -> List[str]:
        return [d for d in self.plan.get((request.asset_id, request.attempt), []) if d in DEFECTS]

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls += 1
        key = (request.asset_id, request.attempt)
        if self._failures_left.get(key, 0) > 0:
            self._failures_left[key] -= 1
            raise TransientProviderError("mock: simulated 503 from provider", status_code=503)
        seed = request.seed if request.seed is not None else 0
        defects = self.defects_for(request)
        img = self.draw(request, defects, seed)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return GenerationResult(image_bytes=buf.getvalue(), cost_usd=self.cost_per_image, seed=seed,
                                meta={"injected_defects": defects})

    # ---- drawing -------------------------------------------------------
    def draw(self, req: GenerationRequest, defects: List[str], seed: int) -> Image.Image:
        rng = random.Random(seed)
        w, h = req.width, req.height
        colours = list(req.brand_colours) or ["#1F6F5C", "#F2B134", "#F5EFE0"]
        if "off_brand_colours" in defects:
            colours = CLASH
        bg, accent = _hex(colours[0]), _hex(colours[1 % len(colours)])
        light = _hex(colours[2 % len(colours)]) if len(colours) > 2 else (245, 239, 224)

        img = Image.new("RGB", (w, h), bg)
        d = ImageDraw.Draw(img)
        s = min(w, h)
        cx, cy = w // 2, int(h * 0.42)
        # sun disc behind the subject
        r = int(s * 0.30)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=accent)

        p = req.prompt.lower()
        if "hand" in p:
            self._hand(d, cx - int(s * 0.12), cy, s, light, bg, 7 if "extra_fingers" in defects else 5)
            self._cup(d, cx + int(s * 0.16), cy + int(s * 0.06), int(s * 0.6), light, bg, rng)
        elif "leaf" in p:
            self._leaf(d, cx, cy, s, light, bg)
        else:
            if "extra_fingers" in defects:
                self._hand(d, cx - int(s * 0.12), cy, s, light, bg, 7)
            self._cup(d, cx, cy, s, light, bg, rng)

        text = req.required_text
        if text and "missing_text" not in defects:
            if "garbled_text" in defects:
                text = garble(text, rng)
            font = load_font(max(12, int(s * 0.11)))
            box = d.textbbox((0, 0), text, font=font)
            tw = box[2] - box[0]
            while tw > w * 0.9 and font.size > 12:
                font = load_font(font.size - 4)
                box = d.textbbox((0, 0), text, font=font)
                tw = box[2] - box[0]
            ty = int(h * 0.80)
            d.text(((w - tw) // 2 - box[0], ty), text, font=font, fill=light)
        return img

    @staticmethod
    def _cup(d, cx, cy, s, fg, bg, rng):
        cw, ch = int(s * 0.26), int(s * 0.22)
        x0, y0 = cx - cw // 2, cy - ch // 2 + int(s * 0.04)
        d.rounded_rectangle([x0, y0, x0 + cw, y0 + ch], radius=int(s * 0.04), fill=fg)
        hr = int(s * 0.06)
        d.ellipse([x0 + cw - hr // 2, y0 + ch // 4, x0 + cw + hr, y0 + ch // 4 + hr * 1.5],
                  outline=fg, width=max(3, int(s * 0.018)))
        d.rounded_rectangle([x0 - int(s * 0.05), y0 + ch, x0 + cw + int(s * 0.05), y0 + ch + int(s * 0.025)],
                            radius=int(s * 0.012), fill=fg)
        for i in (-1, 0, 1):
            sx = cx + i * int(s * 0.07)
            pts = []
            for k in range(20):
                t = k / 19
                pts.append((sx + math.sin(t * math.pi * 2) * s * 0.015, y0 - int(s * 0.03) - t * s * 0.12))
            d.line(pts, fill=fg, width=max(2, int(s * 0.012)))

    @staticmethod
    def _hand(d, cx, cy, s, fg, bg, fingers: int):
        """A cartoon open hand: round palm, fingers fanned out so each one is countable."""
        pr = int(s * 0.085)
        pcx, pcy = cx, cy + int(s * 0.07)
        fw = max(6, int(s * 0.032))
        fl = int(s * 0.15)
        n_up = fingers - 1  # the last one is the thumb
        spread = 18 if n_up <= 4 else 15
        start = -spread * (n_up - 1) / 2
        for i in range(n_up):
            ang = math.radians(start + i * spread)
            ex = pcx + math.sin(ang) * (pr + fl)
            ey = pcy - math.cos(ang) * (pr + fl)
            d.line([(pcx, pcy), (ex, ey)], fill=fg, width=fw)
            d.ellipse([ex - fw / 2, ey - fw / 2, ex + fw / 2, ey + fw / 2], fill=fg)
        ang = math.radians(-80)
        ex, ey = pcx + math.sin(ang) * (pr + fl * 0.7), pcy - math.cos(ang) * (pr + fl * 0.7) + int(s * 0.03)
        d.line([(pcx, pcy + int(s * 0.03)), (ex, ey)], fill=fg, width=fw)
        d.ellipse([ex - fw / 2, ey - fw / 2, ex + fw / 2, ey + fw / 2], fill=fg)
        d.ellipse([pcx - pr, pcy - pr, pcx + pr, pcy + pr], fill=fg)
        d.rectangle([pcx - int(pr * 0.6), pcy + int(pr * 0.5), pcx + int(pr * 0.6), pcy + int(pr * 1.6)], fill=fg)

    @staticmethod
    def _leaf(d, cx, cy, s, fg, bg):
        lw, lh = int(s * 0.20), int(s * 0.34)
        d.ellipse([cx - lw // 2, cy - lh // 2, cx + lw // 2, cy + lh // 2], fill=(76, 140, 74))
        d.line([(cx, cy - lh // 2 + 8), (cx, cy + lh // 2 + int(s * 0.05))], fill=fg, width=max(3, int(s * 0.012)))
        for k in range(1, 4):
            y = cy - lh // 2 + k * lh // 4
            d.line([(cx, y), (cx - lw // 3, y - lh // 10)], fill=fg, width=max(2, int(s * 0.007)))
            d.line([(cx, y), (cx + lw // 3, y - lh // 10)], fill=fg, width=max(2, int(s * 0.007)))
