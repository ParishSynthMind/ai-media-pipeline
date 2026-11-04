"""Command line: `media-pipeline run brief.yaml --provider mock --max-cost 0.50`."""
from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional

from .brief import BriefError, load_brief
from .pipeline import run_pipeline
from .providers import IdeogramProvider, MockProvider
from .providers.base import AuthenticationError
from .qa import DEFAULT_MODEL, ClaudeQA, SkipQA
from .report import make_contact_sheet
from .secrets import get_secret


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="media-pipeline", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="generate every asset in a brief")
    r.add_argument("brief", help="brief file (.yaml, .yml or .json)")
    r.add_argument("--provider", choices=["mock", "ideogram"], default="mock")
    r.add_argument("--max-cost", type=float, required=True, help="hard cap in USD for the whole run (images + QA)")
    r.add_argument("--max-attempts", type=int, default=3, help="generations per image before it is marked failed")
    r.add_argument("--retries", type=int, default=3, help="retries per call on rate limits / 5xx")
    r.add_argument("--out", default="out", help="output folder (a sub-folder is made per run)")
    r.add_argument("--run-id", default=None)
    r.add_argument("--seed", type=int, default=1000, help="base seed (each attempt gets its own seed)")
    r.add_argument("--no-qa", action="store_true", help="skip Claude QA and accept every image")
    r.add_argument("--qa-model", default=DEFAULT_MODEL)
    r.add_argument("--min-score", type=int, default=7, help="lowest QA score (1-10) that can pass")
    r.add_argument("--png", action="store_true", help="also export PNG next to each WebP")
    r.add_argument("--speed", choices=["turbo", "default", "quality"], default="turbo",
                   help="Ideogram rendering speed (turbo is cheapest)")
    r.add_argument("--inject-defect", action="append", default=[], metavar="ASSET:DEFECT[@ATTEMPT]",
                   help="mock only: garbled_text, extra_fingers, off_brand_colours, missing_text or fail:N")
    r.add_argument("--mock-cost", type=float, default=0.0, help="mock only: simulated USD per image")
    r.add_argument("--contact-sheet", action="store_true", help="write contact_sheet.png for the run")
    r.add_argument("-v", "--verbose", action="store_true")

    c = sub.add_parser("contact-sheet", help="make a contact sheet PNG from a finished run folder")
    c.add_argument("run_dir")
    c.add_argument("--out", default=None)
    c.add_argument("--title", default=None)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpx2", "urllib3", "anthropic", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.command == "contact-sheet":
        print(make_contact_sheet(args.run_dir, args.out, args.title))
        return 0

    try:
        brief = load_brief(args.brief)
    except BriefError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if args.provider == "mock":
        provider = MockProvider(cost_per_image=args.mock_cost, defects=args.inject_defect)
    else:
        if args.inject_defect:
            print("error: --inject-defect only works with --provider mock", file=sys.stderr)
            return 2
        try:
            provider = IdeogramProvider(get_secret("IDEOGRAM_API_KEY") or "", rendering_speed=args.speed)
        except AuthenticationError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2

    if args.no_qa:
        qa = SkipQA()
    else:
        if not get_secret("ANTHROPIC_API_KEY"):
            print("error: set ANTHROPIC_API_KEY (or use --no-qa)", file=sys.stderr)
            return 2
        qa = ClaudeQA(model=args.qa_model, min_score=args.min_score)

    manifest = run_pipeline(brief, provider, qa, args.out, max_cost=args.max_cost,
                            max_attempts=args.max_attempts, retries=args.retries,
                            base_seed=args.seed, run_id=args.run_id, also_png=args.png)
    run_dir = f"{args.out}/{manifest['run_id']}"
    t = manifest["totals"]
    print(f"\nRun {manifest['run_id']}: {manifest['status']}")
    if manifest["stop_reason"]:
        print(f"  reason: {manifest['stop_reason']}")
    for e in manifest["assets"]:
        tries = len(e["attempts"])
        print(f"  {e['asset_id']:<16} v{e['variant']}  {e['status']:<10} attempts {tries}"
              + (f"  alt: {e['alt_text']}" if e.get("alt_text") else ""))
    print(f"  cost: images ${t['provider_cost_usd']:.4f} + QA ${t['qa_cost_usd']:.4f} = "
          f"${t['total_cost_usd']:.4f} (cap ${args.max_cost:.2f})")
    print(f"  manifest: {run_dir}/manifest.json")
    if args.contact_sheet:
        print(f"  contact sheet: {make_contact_sheet(run_dir)}")
    return 0 if manifest["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
