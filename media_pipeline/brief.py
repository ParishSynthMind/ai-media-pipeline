"""The brief: what to make, for which brand, in which sizes.

A brief is a YAML or JSON file. It is validated up front so a typo in a hex
colour or a size fails in a second, not after money has been spent on the
first five images.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import yaml

HEX_RE = re.compile(r"^#(?:[0-9a-fA-F]{6})$")
SIZE_RE = re.compile(r"^(\d{2,5})x(\d{2,5})$")
ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class BriefError(ValueError):
    """The brief file is missing something or has a bad value."""


@dataclass
class Asset:
    id: str
    prompt: str
    sizes: List[Tuple[int, int]]
    count: int = 1
    required_text: Optional[str] = None
    negative_prompt: Optional[str] = None

    @property
    def generation_size(self) -> Tuple[int, int]:
        """The largest target size; the model renders this shape, then we resize."""
        return max(self.sizes, key=lambda s: s[0] * s[1])


@dataclass
class Brief:
    brand_name: str
    brand_colours: List[str]
    style_notes: str
    assets: List[Asset]
    watermark: Optional[str] = None
    source: Optional[str] = None
    extra: dict = field(default_factory=dict)


def parse_size(value) -> Tuple[int, int]:
    m = SIZE_RE.match(str(value).strip().lower())
    if not m:
        raise BriefError(f"size {value!r} must look like 1200x628")
    w, h = int(m.group(1)), int(m.group(2))
    if not (16 <= w <= 8192 and 16 <= h <= 8192):
        raise BriefError(f"size {value!r} is outside 16..8192 pixels")
    return w, h


def brief_from_dict(data: dict, source: Optional[str] = None) -> Brief:
    if not isinstance(data, dict):
        raise BriefError("the brief must be a mapping at the top level")
    problems: List[str] = []

    brand = data.get("brand") or {}
    if not isinstance(brand, dict):
        raise BriefError("'brand' must be a mapping with name, colours and style_notes")
    name = str(brand.get("name") or "").strip()
    if not name:
        problems.append("brand.name is required")
    colours = brand.get("colours") or brand.get("colors") or []
    if not isinstance(colours, list) or not colours:
        problems.append("brand.colours must be a non-empty list of hex colours like '#1F6F5C'")
        colours = []
    clean_colours = []
    for c in colours:
        c = str(c).strip()
        if not HEX_RE.match(c):
            problems.append(f"brand colour {c!r} is not a 6-digit hex colour like '#1F6F5C'")
        else:
            clean_colours.append(c.upper())
    style = str(brand.get("style_notes") or "").strip()

    raw_assets = data.get("assets")
    if not isinstance(raw_assets, list) or not raw_assets:
        problems.append("assets must be a non-empty list")
        raw_assets = []

    assets: List[Asset] = []
    seen = set()
    for i, a in enumerate(raw_assets):
        where = f"assets[{i}]"
        if not isinstance(a, dict):
            problems.append(f"{where} must be a mapping")
            continue
        aid = str(a.get("id") or "").strip()
        if not ID_RE.match(aid):
            problems.append(f"{where}.id {aid!r} must be lowercase letters, digits, - or _")
        elif aid in seen:
            problems.append(f"{where}.id {aid!r} is used twice")
        seen.add(aid)
        prompt = str(a.get("prompt") or "").strip()
        if not prompt:
            problems.append(f"{where}.prompt is required")
        raw_sizes = a.get("sizes") or ([a["size"]] if a.get("size") else [])
        sizes = []
        if not raw_sizes:
            problems.append(f"{where} needs size or sizes")
        for s in raw_sizes:
            try:
                sizes.append(parse_size(s))
            except BriefError as e:
                problems.append(f"{where}: {e}")
        count = a.get("count", 1)
        if not isinstance(count, int) or isinstance(count, bool) or not (1 <= count <= 20):
            problems.append(f"{where}.count must be a whole number from 1 to 20")
            count = 1
        text = a.get("text")
        text = str(text).strip() if text not in (None, "") else None
        if text and len(text) > 60:
            problems.append(f"{where}.text is over 60 characters; image models garble long text")
        neg = a.get("negative_prompt")
        assets.append(Asset(id=aid, prompt=prompt, sizes=sizes, count=count,
                            required_text=text, negative_prompt=str(neg) if neg else None))

    if problems:
        raise BriefError("brief has problems:\n- " + "\n- ".join(problems))

    known = {"brand", "assets", "watermark"}
    return Brief(brand_name=name, brand_colours=clean_colours, style_notes=style,
                 assets=assets, watermark=(str(data["watermark"]) if data.get("watermark") else None),
                 source=source, extra={k: v for k, v in data.items() if k not in known})


def load_brief(path) -> Brief:
    path = Path(path)
    if not path.exists():
        raise BriefError(f"brief file not found: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        if path.suffix.lower() == ".json":
            data = json.loads(text)
        else:
            data = yaml.safe_load(text)
    except (json.JSONDecodeError, yaml.YAMLError) as e:
        raise BriefError(f"could not read {path.name}: {e}") from e
    return brief_from_dict(data, source=str(path))
