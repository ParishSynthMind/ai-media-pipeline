import json
from pathlib import Path

from PIL import Image

from media_pipeline.pipeline import build_prompt, run_pipeline
from media_pipeline.providers import MockProvider

from .conftest import GOOD, FakeQA, make_brief, verdict


def test_cost_cap_stops_before_overspend(tmp_path):
    # each image: $0.03 generation + $0.02 QA estimate = $0.05 checked before the call
    provider = MockProvider(cost_per_image=0.03)
    qa = FakeQA(cost=0.01, estimate=0.02)
    m = run_pipeline(make_brief(n_assets=5), provider, qa, tmp_path, max_cost=0.12, sleep=lambda s: None)
    assert m["status"] == "stopped_cost_cap"
    assert "cap $0.12" in m["stop_reason"]
    assert m["totals"]["total_cost_usd"] <= 0.12
    assert provider.calls == 2  # third call would have been estimated at 0.08 + 0.05 > 0.12
    statuses = [a["status"] for a in m["assets"]]
    assert statuses == ["accepted", "accepted", "not_run", "not_run", "not_run"]
    on_disk = json.loads((tmp_path / m["run_id"] / "manifest.json").read_text())
    assert on_disk["status"] == "stopped_cost_cap"


def test_cost_cap_counts_regenerations(tmp_path):
    provider = MockProvider(cost_per_image=0.03)
    qa = FakeQA(script=[verdict(garbled_text=True)], cost=0.01, estimate=0.02)
    m = run_pipeline(make_brief(), provider, qa, tmp_path, max_cost=0.10, max_attempts=5)
    assert m["status"] == "stopped_cost_cap"
    assert provider.calls == 2 and m["totals"]["total_cost_usd"] <= 0.10
    assert m["assets"][0]["status"] == "failed"


def test_zero_cap_makes_no_calls(tmp_path):
    provider = MockProvider(cost_per_image=0.03)
    m = run_pipeline(make_brief(), provider, FakeQA(), tmp_path, max_cost=0.0)
    assert provider.calls == 0 and m["status"] == "stopped_cost_cap"
    assert m["assets"][0]["status"] == "not_run"


def test_rejection_regenerates_with_feedback(tmp_path):
    qa = FakeQA(script=[verdict(garbled_text=True, text_seen="FRSEH ROAST", score=4,
                                reasons=["headline misspelled"]), GOOD])
    m = run_pipeline(make_brief(), MockProvider(), qa, tmp_path, max_cost=1.0)
    entry = m["assets"][0]
    assert entry["status"] == "accepted" and entry["accepted_attempt"] == 2
    a1, a2 = entry["attempts"]
    assert a1["accepted"] is False and a1["qa"]["garbled_text"] is True
    assert "Fix these problems" not in a1["prompt"]
    assert "Fix these problems" in a2["prompt"]
    assert 'must read exactly "FRESH ROAST"' in a2["prompt"]
    assert "headline misspelled" in a2["prompt"]
    assert a1["seed"] != a2["seed"]


def test_gives_up_after_max_attempts_and_moves_on(tmp_path):
    bad = verdict(anatomy_issues=True, reasons=["seven fingers"])
    qa = FakeQA(script=[bad, bad, bad, GOOD])
    m = run_pipeline(make_brief(n_assets=2), MockProvider(), qa, tmp_path, max_cost=1.0, max_attempts=3)
    first, second = m["assets"]
    assert first["status"] == "failed" and len(first["attempts"]) == 3
    assert "3 tries" in first["failure_reason"]
    assert first["files"] == []
    assert second["status"] == "accepted"
    assert m["status"] == "completed"
    assert m["totals"]["accepted"] == 1 and m["totals"]["failed"] == 1


def test_low_score_alone_is_a_rejection(tmp_path):
    qa = FakeQA(script=[verdict(score=5), GOOD], min_score=7)
    m = run_pipeline(make_brief(), MockProvider(), qa, tmp_path, max_cost=1.0)
    assert m["assets"][0]["accepted_attempt"] == 2


def test_manifest_contents(tmp_path):
    qa = FakeQA(script=[verdict(brand_colours_present=False), GOOD], cost=0.0123)
    m = run_pipeline(make_brief(count=2, watermark="TB"), MockProvider(cost_per_image=0.03), qa,
                     tmp_path, max_cost=1.0, run_id="r1")
    run = tmp_path / "r1"
    disk = json.loads((run / "manifest.json").read_text())
    assert disk == m
    assert m["provider"] == "mock" and m["qa_model"] == "fake-qa"
    assert m["brief"]["brand_colours"] == ["#1F6F5C", "#F2B134", "#F5EFE0"]
    assert m["settings"]["max_cost_usd"] == 1.0 and m["settings"]["max_attempts"] == 3
    assert [e["variant"] for e in m["assets"]] == [1, 2]
    t = m["totals"]
    assert t["generation_calls"] == 3 and t["qa_calls"] == 3
    assert abs(t["provider_cost_usd"] - 0.09) < 1e-9
    assert abs(t["qa_cost_usd"] - 3 * 0.0123) < 1e-9
    assert abs(t["total_cost_usd"] - (0.09 + 3 * 0.0123)) < 1e-9
    e1 = m["assets"][0]
    att = e1["attempts"][0]
    for key in ("attempt", "prompt", "seed", "provider", "cost_usd", "qa", "qa_cost_usd", "accepted", "raw_file",
                "estimated_cost_usd", "started_at"):
        assert key in att
    assert att["qa"]["brand_colours_present"] is False
    assert (run / att["raw_file"]).exists()
    assert e1["alt_text"] == GOOD["alt_text"]
    assert [f["format"] for f in e1["files"]] == ["webp", "webp"]
    for f in e1["files"]:
        assert not Path(f["path"]).is_absolute()
        with Image.open(run / f["path"]) as im:
            assert im.format == "WEBP" and im.size == (f["width"], f["height"])
    assert m["finished_at"] and m["status"] == "completed"


def test_build_prompt_includes_brand_and_text():
    b = make_brief()
    p = build_prompt(b, b.assets[0])
    assert "Test Brand" in p and "#1F6F5C" in p and '"FRESH ROAST"' in p and "flat vector" in p
