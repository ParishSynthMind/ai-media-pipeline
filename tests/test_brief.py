import json

import pytest

from media_pipeline.brief import BriefError, brief_from_dict, load_brief, parse_size


def base(**asset):
    a = {"id": "hero", "prompt": "a cup", "sizes": ["1200x628"]}
    a.update(asset)
    return {"brand": {"name": "B", "colours": ["#112233"]}, "assets": [a]}


def test_example_yaml_loads():
    b = load_brief("examples/brief.yaml")
    assert b.brand_name == "Juniper & Ash Coffee Co."
    assert b.brand_colours == ["#1F6F5C", "#F2B134", "#F5EFE0"]
    assert [a.id for a in b.assets] == ["instagram-post", "web-banner", "story"]
    assert b.assets[0].sizes == [(1080, 1080), (600, 600)]
    assert b.assets[1].required_text == "WELCOME BACK"
    assert b.watermark == "Juniper & Ash"


def test_example_json_loads():
    b = load_brief("examples/brief.json")
    assert b.assets[0].id == "square" and b.assets[0].count == 1


def test_generation_size_is_largest_target():
    b = brief_from_dict(base(sizes=["600x314", "1200x628"]))
    assert b.assets[0].generation_size == (1200, 628)


def test_single_size_key_and_colours_normalised():
    d = base()
    del d["assets"][0]["sizes"]
    d["assets"][0]["size"] = "800x800"
    d["brand"]["colours"] = ["#aabbcc"]
    b = brief_from_dict(d)
    assert b.assets[0].sizes == [(800, 800)] and b.brand_colours == ["#AABBCC"]


@pytest.mark.parametrize("mutate, message", [
    (lambda d: d["brand"].update(colours=["teal"]), "not a 6-digit hex"),
    (lambda d: d["brand"].update(colours=[]), "non-empty list"),
    (lambda d: d["brand"].update(name=""), "brand.name"),
    (lambda d: d["assets"][0].update(sizes=["big"]), "must look like 1200x628"),
    (lambda d: d["assets"][0].update(sizes=["99999x10"]), "outside 16..8192"),
    (lambda d: d["assets"][0].update(prompt=""), "prompt is required"),
    (lambda d: d["assets"][0].update(count=0), "count"),
    (lambda d: d["assets"][0].update(id="Bad Id!"), "lowercase"),
    (lambda d: d["assets"][0].update(text="x" * 61), "over 60"),
    (lambda d: d["assets"].append(dict(d["assets"][0])), "used twice"),
    (lambda d: d.update(assets=[]), "non-empty list"),
])
def test_validation_errors(mutate, message):
    d = base()
    mutate(d)
    with pytest.raises(BriefError) as e:
        brief_from_dict(d)
    assert message in str(e.value)


def test_all_problems_reported_together():
    d = base(prompt="", sizes=["nope"])
    d["brand"]["colours"] = ["#zzz"]
    with pytest.raises(BriefError) as e:
        brief_from_dict(d)
    text = str(e.value)
    assert "prompt is required" in text and "nope" in text and "#zzz" in text


def test_missing_and_malformed_files(tmp_path):
    with pytest.raises(BriefError, match="not found"):
        load_brief(tmp_path / "nope.yaml")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(BriefError, match="could not read"):
        load_brief(bad)
    good = tmp_path / "good.json"
    good.write_text(json.dumps(base()))
    assert load_brief(good).assets[0].id == "hero"


def test_parse_size():
    assert parse_size(" 1080X1920 ") == (1080, 1920)
