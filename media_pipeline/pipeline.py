"""The batch run: generate, check, regenerate with feedback, post-process, record.

Order of work for every image:

1. Build the prompt (asset prompt + brand style + colours + required text +
   feedback from any rejected attempt).
2. Cost cap: estimated generation + QA cost is checked BEFORE the call. If it
   would push the run over the cap, the run stops cleanly and the manifest
   says why. The cap is also re-checked before every retry.
3. Generate, retrying transient errors (rate limits, 5xx, timeouts) with
   exponential backoff. An empty wallet or a bad key stops the whole run.
4. Claude QA. A rejection feeds its reasons into the next attempt's prompt,
   up to --max-attempts. After that the asset is marked failed.
5. Accepted images are resized to every target size and saved as WebP, with
   an optional watermark, and the alt text is stored.

manifest.json is rewritten after every attempt, so a crash or Ctrl-C leaves
an accurate record of what was made and what was spent.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from . import __version__
from .brief import Asset, Brief
from .postprocess import export_asset
from .providers.base import (
    AuthenticationError,
    ContentRejectedError,
    GenerationRequest,
    ImageProvider,
    InsufficientFundsError,
    ProviderError,
)
from .retry import call_with_retries

log = logging.getLogger(__name__)

EXT = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}


class CostCapReached(Exception):
    pass


class Ledger:
    def __init__(self, max_cost: float):
        self.max_cost = float(max_cost)
        self.provider = 0.0
        self.qa = 0.0

    @property
    def total(self) -> float:
        return self.provider + self.qa

    def can_afford(self, estimate: float) -> bool:
        return self.total + estimate <= self.max_cost + 1e-9

    def require(self, estimate: float, what: str) -> None:
        if not self.can_afford(estimate):
            raise CostCapReached(
                f"stopped before {what}: spent ${self.total:.4f}, next call estimated "
                f"${estimate:.4f}, cap ${self.max_cost:.2f}")


def build_prompt(brief: Brief, asset: Asset, feedback: str = "") -> str:
    parts = [asset.prompt.rstrip(".") + "."]
    if brief.style_notes:
        parts.append(f"Style: {brief.style_notes.rstrip('.')}.")
    parts.append(f"Use the {brief.brand_name} brand palette: {', '.join(brief.brand_colours)}.")
    if asset.required_text:
        parts.append(f'The image must include the exact text "{asset.required_text}", spelled correctly, '
                     "in large, clean, legible letters. No other text.")
    if feedback:
        parts.append(f"Fix these problems from the previous attempt: {feedback}.")
    return " ".join(parts)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ManifestWriter:
    def __init__(self, path: Path, manifest: dict):
        self.path = path
        self.manifest = manifest

    def write(self) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)


def run_pipeline(brief: Brief, provider: ImageProvider, qa, out_root, max_cost: float,
                 max_attempts: int = 3, retries: int = 3, base_delay: float = 1.0,
                 base_seed: int = 1000, run_id: Optional[str] = None, also_png: bool = False,
                 sleep: Callable[[float], None] = time.sleep) -> dict:
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(out_root) / run_id
    raw_dir, final_dir = run_dir / "raw", run_dir / "final"
    raw_dir.mkdir(parents=True, exist_ok=True)

    ledger = Ledger(max_cost)
    manifest = {
        "run_id": run_id,
        "tool_version": __version__,
        "started_at": _now(),
        "finished_at": None,
        "status": "running",
        "stop_reason": None,
        "brief": {"source": brief.source, "brand_name": brief.brand_name,
                  "brand_colours": brief.brand_colours, "watermark": brief.watermark},
        "provider": provider.name,
        "qa_model": getattr(qa, "model", None),
        "settings": {"max_cost_usd": max_cost, "max_attempts": max_attempts, "retries": retries,
                     "min_score": getattr(qa, "min_score", None)},
        "totals": {},
        "assets": [],
    }
    writer = ManifestWriter(run_dir / "manifest.json", manifest)

    def update_totals():
        entries = manifest["assets"]
        calls = [a for e in entries for a in e["attempts"]]
        manifest["totals"] = {
            "provider_cost_usd": round(ledger.provider, 6),
            "qa_cost_usd": round(ledger.qa, 6),
            "total_cost_usd": round(ledger.total, 6),
            "generation_calls": sum(1 for a in calls if a.get("generated")),
            "qa_calls": sum(1 for a in calls if a.get("qa")),
            "accepted": sum(1 for e in entries if e["status"] == "accepted"),
            "failed": sum(1 for e in entries if e["status"] == "failed"),
            "not_run": sum(1 for e in entries if e["status"] == "not_run"),
        }
        writer.write()

    def finish(status: str, reason: Optional[str] = None) -> dict:
        manifest["status"] = status
        manifest["stop_reason"] = reason
        manifest["finished_at"] = _now()
        update_totals()
        return manifest

    # Every planned image is listed up front, so a stopped run shows what was never made.
    for asset in brief.assets:
        for v in range(asset.count):
            manifest["assets"].append({
                "asset_id": asset.id, "variant": v + 1, "status": "not_run",
                "required_text": asset.required_text,
                "target_sizes": [f"{w}x{h}" for w, h in asset.sizes],
                "attempts": [], "files": [], "alt_text": None,
            })
    update_totals()

    idx = 0
    for asset in brief.assets:
        gw, gh = asset.generation_size
        for v in range(asset.count):
            entry = manifest["assets"][idx]
            idx += 1
            entry["status"] = "in_progress"
            feedback = ""
            for attempt in range(1, max_attempts + 1):
                prompt = build_prompt(brief, asset, feedback)
                req = GenerationRequest(prompt=prompt, width=gw, height=gh,
                                        seed=base_seed + v * 100 + attempt,
                                        negative_prompt=asset.negative_prompt,
                                        brand_colours=brief.brand_colours,
                                        required_text=asset.required_text,
                                        asset_id=asset.id, attempt=attempt)
                rec = {"attempt": attempt, "started_at": _now(), "prompt": prompt, "seed": req.seed,
                       "provider": provider.name, "generated": False, "cost_usd": 0.0,
                       "qa": None, "qa_cost_usd": 0.0, "accepted": False, "error": None, "raw_file": None}
                entry["attempts"].append(rec)
                try:
                    estimate = provider.estimate_cost(req) + qa.estimate_cost()
                    rec["estimated_cost_usd"] = round(estimate, 6)
                    ledger.require(estimate, f"{asset.id} attempt {attempt}")

                    def before_retry(n, err, _req=req, _rec=rec, _what=f"{asset.id} attempt {attempt}"):
                        _rec.setdefault("retries", []).append({"after_error": str(err), "status": err.status_code})
                        ledger.require(provider.estimate_cost(_req) + qa.estimate_cost(), f"retry {n} of {_what}")

                    result = call_with_retries(lambda _req=req: provider.generate(_req), retries=retries,
                                               base_delay=base_delay, sleep=sleep, before_retry=before_retry)
                except CostCapReached as e:
                    rec["error"] = str(e)
                    entry["status"] = "not_run" if attempt == 1 else "failed"
                    log.warning("%s", e)
                    return finish("stopped_cost_cap", str(e))
                except (AuthenticationError, InsufficientFundsError) as e:
                    rec["error"] = f"{type(e).__name__}: {e}"
                    rec["http_status"] = e.status_code
                    entry["status"] = "failed"
                    log.error("%s", e)
                    return finish("aborted_provider", f"{type(e).__name__} (HTTP {e.status_code}): {e}")
                except ContentRejectedError as e:
                    charged = float(getattr(e, "cost_usd", 0.0) or 0.0)
                    ledger.provider += charged
                    rec.update(error=f"provider rejected: {e}", cost_usd=charged, http_status=e.status_code)
                    feedback = ("the provider's safety filter rejected the last attempt; "
                                "keep the scene simple and brand-safe")
                    update_totals()
                    continue
                except ProviderError as e:
                    rec.update(error=f"provider error after retries: {e}", http_status=e.status_code)
                    update_totals()
                    continue

                ledger.provider += result.cost_usd
                rec.update(generated=True, cost_usd=round(result.cost_usd, 6), seed=result.seed,
                           provider_meta=result.meta)
                if result.provider_prompt and result.provider_prompt != prompt:
                    rec["provider_prompt"] = result.provider_prompt
                raw_path = raw_dir / f"{asset.id}_v{v + 1}_a{attempt}.{EXT.get(result.mime_type, 'png')}"
                raw_path.write_bytes(result.image_bytes)
                rec["raw_file"] = str(raw_path.relative_to(run_dir))
                update_totals()

                try:
                    ledger.require(qa.estimate_cost(), f"QA of {asset.id} attempt {attempt}")
                except CostCapReached as e:
                    rec["error"] = str(e)
                    entry["status"] = "failed"
                    return finish("stopped_cost_cap", str(e))
                verdict = qa.review(result.image_bytes, asset.prompt, brief.brand_name,
                                    brief.brand_colours, asset.required_text)
                ledger.qa += verdict.cost_usd
                rec["qa"] = verdict.to_dict()
                rec["qa_cost_usd"] = verdict.cost_usd
                rec["accepted"] = verdict.passed
                log.info("%s v%d attempt %d: %s (score %s) %s", asset.id, v + 1, attempt,
                         "ACCEPTED" if verdict.passed else "REJECTED", verdict.score, "; ".join(verdict.reasons))
                if verdict.passed:
                    files = export_asset(result.image_bytes, asset.sizes, final_dir, f"{asset.id}_v{v + 1}",
                                         watermark_text=brief.watermark, also_png=also_png)
                    for f in files:
                        f["path"] = str(Path(f["path"]).relative_to(run_dir))
                    entry.update(status="accepted", files=files, alt_text=verdict.alt_text or None,
                                 accepted_attempt=attempt)
                    update_totals()
                    break
                feedback = verdict.feedback(asset.required_text)
                rec["feedback_for_next_attempt"] = feedback or None
                update_totals()
            else:
                entry["status"] = "failed"
                entry["failure_reason"] = f"no attempt passed QA in {max_attempts} tries"
                update_totals()
            if entry["status"] == "in_progress":  # loop ended via `continue` on the final attempt
                entry["status"] = "failed"
                update_totals()

    return finish("completed")
