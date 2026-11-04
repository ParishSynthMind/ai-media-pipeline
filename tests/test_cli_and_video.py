import json

import pytest

from media_pipeline.cli import main
from media_pipeline.providers import NotImplementedVideoProvider
from media_pipeline.providers.base import GenerationRequest
from media_pipeline.providers.mock import MockProvider, garble, parse_defect_specs
from media_pipeline.providers.video import VideoRequest


def test_cli_mock_run_no_qa(tmp_path, capsys):
    code = main(["run", "examples/brief.yaml", "--provider", "mock", "--max-cost", "0.5", "--no-qa",
                 "--out", str(tmp_path), "--run-id", "cli", "--contact-sheet"])
    assert code == 0
    m = json.loads((tmp_path / "cli" / "manifest.json").read_text())
    assert m["status"] == "completed" and m["totals"]["accepted"] == 3
    assert (tmp_path / "cli" / "contact_sheet.png").exists()
    assert (tmp_path / "cli" / "final" / "story_v1_1080x1920.webp").exists()
    assert "completed" in capsys.readouterr().out


def test_cli_bad_brief_exit_2(tmp_path, capsys):
    bad = tmp_path / "b.yaml"
    bad.write_text("brand: {name: X, colours: [red]}\nassets: []\n")
    assert main(["run", str(bad), "--max-cost", "1", "--no-qa"]) == 2
    assert "brief has problems" in capsys.readouterr().err


def test_cli_defects_only_for_mock(capsys, monkeypatch):
    monkeypatch.setenv("IDEOGRAM_API_KEY", "x")
    assert main(["run", "examples/brief.yaml", "--provider", "ideogram", "--max-cost", "1", "--no-qa",
                 "--inject-defect", "story:garbled_text"]) == 2


def test_cli_needs_anthropic_key_for_qa(monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_KEYCHAIN_SERVICE", raising=False)
    assert main(["run", "examples/brief.yaml", "--max-cost", "1"]) == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_mock_defect_plan_and_determinism():
    plan = parse_defect_specs(["a:garbled_text", "a:extra_fingers@2", "b:fail:3"])
    assert plan[("a", 1)] == ["garbled_text"] and plan[("a", 2)] == ["extra_fingers"]
    with pytest.raises(ValueError):
        parse_defect_specs(["a:purple"])
    p = MockProvider(defects=["a:garbled_text"])
    r1 = GenerationRequest(prompt="coffee cup", width=300, height=300, seed=5, required_text="FRESH ROAST",
                           brand_colours=["#1F6F5C", "#F2B134"], asset_id="a", attempt=1)
    first = p.generate(r1)
    assert first.meta["injected_defects"] == ["garbled_text"]
    assert p.generate(r1).image_bytes == first.image_bytes  # same seed, same picture
    r2 = GenerationRequest(**{**r1.__dict__, "attempt": 2})
    assert p.generate(r2).meta["injected_defects"] == []


def test_garble_changes_text():
    import random
    out = garble("FRESH ROAST", random.Random(1))
    assert out != "FRESH ROAST" and len(out) == len("FRESH ROAST")


def test_video_is_an_honest_stub():
    v = NotImplementedVideoProvider()
    with pytest.raises(NotImplementedError, match="docs/VIDEO.md"):
        v.submit(VideoRequest(prompt="x", width=1080, height=1920))
    with pytest.raises(NotImplementedError):
        v.estimate_cost(VideoRequest(prompt="x", width=1, height=1))
