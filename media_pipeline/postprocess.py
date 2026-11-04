"""Resize to each target size, export WebP (and PNG), optionally watermark."""
from __future__ import annotations

import io
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw, ImageOps

from .providers.mock import load_font


def open_image(image_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(image_bytes))
    img.load()
    return img.convert("RGB")


def fit(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    """Scale and centre-crop to exactly `size` (no stretching, no letterboxing)."""
    return ImageOps.fit(img, size, method=Image.LANCZOS, centering=(0.5, 0.5))


def watermark(img: Image.Image, text: str, opacity: int = 110) -> Image.Image:
    """Small semi-transparent text in the bottom-right corner."""
    base = img.convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    font = load_font(max(10, base.size[0] // 40))
    box = d.textbbox((0, 0), text, font=font)
    tw, th = box[2] - box[0], box[3] - box[1]
    margin = max(6, base.size[0] // 60)
    x, y = base.size[0] - tw - margin - box[0], base.size[1] - th - margin - box[1]
    d.text((x + 1, y + 1), text, font=font, fill=(0, 0, 0, opacity))
    d.text((x, y), text, font=font, fill=(255, 255, 255, opacity))
    return Image.alpha_composite(base, layer).convert("RGB")


def export_asset(image_bytes: bytes, sizes: List[Tuple[int, int]], out_dir: Path, stem: str,
                 watermark_text: Optional[str] = None, webp_quality: int = 85,
                 also_png: bool = False) -> List[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    src = open_image(image_bytes)
    files = []
    for w, h in sizes:
        img = fit(src, (w, h))
        if watermark_text:
            img = watermark(img, watermark_text)
        path = out_dir / f"{stem}_{w}x{h}.webp"
        img.save(path, format="WEBP", quality=webp_quality, method=6)
        files.append({"path": str(path), "width": w, "height": h, "format": "webp",
                      "bytes": path.stat().st_size})
        if also_png:
            p2 = out_dir / f"{stem}_{w}x{h}.png"
            img.save(p2, format="PNG", optimize=True)
            files.append({"path": str(p2), "width": w, "height": h, "format": "png",
                          "bytes": p2.stat().st_size})
    return files
