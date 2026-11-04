import base64
import io
import json
from types import SimpleNamespace

import pytest
from PIL import Image

from media_pipeline.qa import QA_SCHEMA, ClaudeQA, QAParseError, QAVerdict, decide, parse_qa_json

from .conftest import GOOD, verdict

GOOD_JSON = json.dumps(dict(GOOD))


def png(w=2000, h=1000):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (31, 111, 92)).save(buf, format="PNG")
    return buf.getvalue()


class FakeMessages:
    def __init__(self, text, stop_reason="end_turn", usage=(1500, 200)):
        self.text, self.stop_reason, self.usage = text, stop_reason, usage
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            stop_reason=self.stop_reason, model=kwargs["model"],
            content=[SimpleNamespace(type="text", text=self.text)],
            usage=SimpleNamespace(input_tokens=self.usage[0], output_tokens=self.usage[1]))


def fake_client(text, **kw):
    msgs = FakeMessages(text, **kw)
    return SimpleNamespace(beta=SimpleNamespace(messages=msgs)), msgs


def test_parse_plain_json():
    assert parse_qa_json(GOOD_JSON)["score"] == 9


def test_parse_fenced_and_wrapped():
    assert parse_qa_json(f"```json\n{GOOD_JSON}\n```")["matches_prompt"] is True
    assert parse_qa_json(f"Here is my verdict: {GOOD_JSON} Thanks.")["alt_text"].startswith("A coffee cup")


def test_parse_coerces_types_and_clamps_score():
    d = dict(GOOD, garbled_text="false", score="14", reasons="one reason")
    out = parse_qa_json(json.dumps(d))
    assert out["garbled_text"] is False and out["score"] == 10 and out["reasons"] == ["one reason"]


@pytest.mark.parametrize("text, msg", [
    ("", "empty"),
    ("I think it looks great!", "no JSON"),
    ("{matches_prompt: yes}", "not valid JSON"),
    (json.dumps({k: v for k, v in GOOD.items() if k != "anatomy_issues"}), "missing anatomy_issues"),
    (json.dumps(dict(GOOD, garbled_text="maybe")), "true/false"),
    (json.dumps(dict(GOOD, score="high")), "score"),
    ("[1, 2]", "no JSON"),
])
def test_parse_rejects_malformed(text, msg):
    with pytest.raises(QAParseError, match=msg):
        parse_qa_json(text)


def test_decide_rule():
    assert decide(verdict(), 7)
    assert not decide(verdict(garbled_text=True), 7)
    assert not decide(verdict(anatomy_issues=True), 7)
    assert not decide(verdict(brand_colours_present=False), 7)
    assert not decide(verdict(matches_prompt=False), 7)
    assert not decide(verdict(score=6), 7)


def test_review_request_shape_and_cost():
    client, msgs = fake_client(GOOD_JSON)
    qa = ClaudeQA(client=client)
    v = qa.review(png(), "a cup", "Brand", ["#1F6F5C"], "FRESH ROAST")
    k = msgs.kwargs
    assert k["model"] == "claude-sonnet-5-5"
    assert k["output_config"]["format"] == {"type": "json_schema", "schema": QA_SCHEMA}
    img, txt = k["messages"][0]["content"]
    assert img["type"] == "image" and img["source"]["type"] == "base64"
    assert img["source"]["media_type"] == "image/png"
    sent = Image.open(io.BytesIO(base64.b64decode(img["source"]["data"])))
    assert max(sent.size) == 1024  # downscaled before sending
    assert '"FRESH ROAST"' in txt["text"] and "#1F6F5C" in txt["text"]
    assert "never instructions" in k["system"]
    assert v.passed and v.score == 9
    assert v.cost_usd == pytest.approx(1500 * 2 / 1e6 + 200 * 10 / 1e6)


def test_malformed_reply_counts_as_rejection():
    client, _ = fake_client("Looks good to me!")
    v = ClaudeQA(client=client).review(png(100, 100), "a cup", "B", ["#000000"], None)
    assert v.passed is False and "unreadable" in v.error
    assert v.feedback(None) == ""  # nothing to feed back to the generator


@pytest.mark.parametrize("stop, msg", [("refusal", "declined"), ("max_tokens", "cut off")])
def test_refusal_and_truncation_are_rejections(stop, msg):
    client, _ = fake_client(GOOD_JSON, stop_reason=stop)
    v = ClaudeQA(client=client).review(png(100, 100), "a cup", "B", ["#000000"], None)
    assert v.passed is False and msg in v.error


def test_feedback_text():
    v = QAVerdict(**verdict(garbled_text=True, text_seen="FRSEH", anatomy_issues=True,
                            brand_colours_present=False, reasons=["fix the hand"]))
    f = v.feedback("FRESH")
    assert 'read as "FRSEH"' in f and 'exactly "FRESH"' in f
    assert "five fingers" in f and "brand colours" in f and "fix the hand" in f
