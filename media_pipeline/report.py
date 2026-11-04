"""Contact sheet: every attempt in a run, with its QA verdict, on one PNG."""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from .providers.mock import load_font

THUMB = 300
CAPTION_H = 170
PAD = 20
GREEN, RED, GREY = (34, 139, 34), (200, 40, 40), (110, 110, 110)


def make_contact_sheet(run_dir, out_path=None, title=None) -> Path:
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    cells = []
    for entry in manifest["assets"]:
        for a in entry["attempts"]:
            cells.append((entry, a))
    cols = max(1, min(4, len(cells)))
    rows = max(1, (len(cells) + cols - 1) // cols)
    head = 90
    W = PAD + cols * (THUMB + PAD)
    H = head + rows * (THUMB + CAPTION_H + PAD) + PAD
    sheet = Image.new("RGB", (W, H), (250, 250, 248))
    d = ImageDraw.Draw(sheet)
    f_title, f_b, f_s = load_font(26), load_font(17), load_font(14)
    t = manifest["totals"]
    heading = title or f"{manifest['brief']['brand_name']}: run {manifest['run_id']}"
    d.text((PAD, 18), heading, font=f_title, fill=(20, 20, 20))
    d.text((PAD, 54), f"provider {manifest['provider']} | QA {manifest['qa_model']} | "
                      f"{t.get('accepted', 0)} accepted, {t.get('failed', 0)} failed | "
                      f"total ${t.get('total_cost_usd', 0):.4f} (images ${t.get('provider_cost_usd', 0):.4f}, "
                      f"QA ${t.get('qa_cost_usd', 0):.4f}) | {manifest['status']}", font=f_s, fill=GREY)
    for i, (entry, a) in enumerate(cells):
        x = PAD + (i % cols) * (THUMB + PAD)
        y = head + (i // cols) * (THUMB + CAPTION_H + PAD)
        if a.get("raw_file") and (run_dir / a["raw_file"]).exists():
            im = Image.open(run_dir / a["raw_file"]).convert("RGB")
            im = ImageOps.contain(im, (THUMB, THUMB))
            bg = Image.new("RGB", (THUMB, THUMB), (230, 230, 230))
            bg.paste(im, ((THUMB - im.width) // 2, (THUMB - im.height) // 2))
            sheet.paste(bg, (x, y))
        else:
            d.rectangle([x, y, x + THUMB, y + THUMB], fill=(230, 230, 230))
            d.text((x + 10, y + THUMB // 2), "no image", font=f_b, fill=GREY)
        qa = a.get("qa") or {}
        if a.get("accepted"):
            label, colour = "ACCEPTED", GREEN
        elif qa:
            label, colour = "REJECTED", RED
        else:
            label, colour = "ERROR", GREY
        d.rectangle([x, y, x + THUMB, y + THUMB], outline=colour, width=5)
        cy = y + THUMB + 8
        d.text((x, cy), f"{entry['asset_id']} v{entry['variant']} attempt {a['attempt']}", font=f_b, fill=(20, 20, 20))
        score = f" score {qa.get('score')}" if qa else ""
        d.text((x, cy + 22), f"{label}{score}", font=f_b, fill=colour)
        flags = []
        if qa:
            if qa.get("garbled_text"):
                flags.append("garbled text")
            if qa.get("anatomy_issues"):
                flags.append("anatomy")
            if not qa.get("brand_colours_present"):
                flags.append("off-brand")
            if not qa.get("matches_prompt"):
                flags.append("off-prompt")
        why = ", ".join(flags) if flags else ""
        detail = (qa.get("reasons") or [""])[0] if qa and not a.get("accepted") else (qa.get("alt_text") or "")
        if a.get("error") and not qa:
            detail = a["error"]
        lines = ([why] if why else []) + textwrap.wrap(detail, 38)[:4]
        for k, line in enumerate(lines[:5]):
            d.text((x, cy + 46 + k * 18), line, font=f_s, fill=GREY)
    out_path = Path(out_path) if out_path else run_dir / "contact_sheet.png"
    sheet.save(out_path)
    return out_path
