import io
import json

import pytest
import responses
from PIL import Image

from media_pipeline.pipeline import run_pipeline
from media_pipeline.providers import IdeogramProvider
from media_pipeline.providers.base import (
    AuthenticationError,
    ContentRejectedError,
    GenerationRequest,
    InsufficientFundsError,
    ProviderError,
    TransientProviderError,
)
from media_pipeline.providers.ideogram import nearest_aspect_ratio

from .conftest import FakeQA, make_brief

URL = "https://api.ideogram.ai/v2/image/generate/ideogram-3"
IMG_URL = "https://ideogram.ai/api/images/ephemeral/abc.png"
QUOTE = {"object": "price_quote", "billing_identifier": "ideogram_v3_generation", "quantity": 1,
         "usd_micros": 30000, "credit_millis": 3000, "qualifier": "exact"}


def png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (31, 111, 92)).save(buf, format="PNG")
    return buf.getvalue()


def req(**kw):
    base = dict(prompt="a coffee cup", width=1200, height=628, seed=42, negative_prompt="blurry",
                brand_colours=["#1F6F5C", "#F2B134"], required_text="HI")
    base.update(kw)
    return GenerationRequest(**base)


def field(body: bytes, name: str):
    marker = f'name="{name}"\r\n\r\n'.encode()
    if marker not in body:
        return None
    return body.split(marker, 1)[1].split(b"\r\n", 1)[0].decode()


def is_dry_run(call):
    return "dry_run=true" in call.request.url


def add_quote():
    responses.add(responses.POST, URL, json=QUOTE, status=200,
                  match=[responses.matchers.query_param_matcher({"dry_run": "true"})])


def add_generate(status=200, body=None):
    body = body if body is not None else {"generation_id": "g1", "seed": 42, "data": [
        {"prompt": "a coffee cup, flat", "resolution": "1312x736", "is_image_safe": True, "seed": 42, "url": IMG_URL}]}
    responses.add(responses.POST, URL, json=body, status=status,
                  match=[responses.matchers.query_param_matcher({})])


@responses.activate
def test_generate_request_shape_and_download():
    add_quote()
    add_generate()
    responses.add(responses.GET, IMG_URL, body=png_bytes(), status=200, content_type="image/png")
    p = IdeogramProvider("test-key")
    result = p.generate(req())

    quote_call, gen_call, dl_call = responses.calls
    assert is_dry_run(quote_call) and not is_dry_run(gen_call)
    for call in (quote_call, gen_call):
        assert call.request.headers["Api-Key"] == "test-key"
        assert call.request.headers["Content-Type"].startswith("multipart/form-data; boundary=")
    body = gen_call.request.body
    assert field(body, "prompt") == "a coffee cup"
    assert field(body, "rendering_speed") == "turbo"
    assert field(body, "aspect_ratio") == "2x1"
    assert field(body, "num_images") == "1"
    assert field(body, "seed") == "42"
    assert field(body, "negative_prompt") == "blurry"
    assert field(body, "magic_prompt") == "off"
    palette = json.loads(field(body, "color_palette"))
    assert [m["color_hex"] for m in palette["members"]] == ["#1F6F5C", "#F2B134"]
    assert dl_call.request.url == IMG_URL
    assert result.image_bytes == png_bytes()
    assert result.cost_usd == pytest.approx(0.03)
    assert result.seed == 42 and result.mime_type == "image/png"
    assert result.provider_prompt == "a coffee cup, flat"
    assert result.meta["generation_id"] == "g1"


@responses.activate
def test_quote_is_cached():
    add_quote()
    p = IdeogramProvider("k")
    assert p.estimate_cost(req()) == pytest.approx(0.03)
    assert p.estimate_cost(req()) == pytest.approx(0.03)
    assert len(responses.calls) == 1


@responses.activate
def test_quote_failure_uses_fallback_price():
    responses.add(responses.POST, URL, status=500, json={})
    assert IdeogramProvider("k", rendering_speed="quality").estimate_cost(req()) == pytest.approx(0.09)


@pytest.mark.parametrize("status, body, exc", [
    (402, {"error": "Insufficient credits", "reject_reason": "insufficient_funds"}, InsufficientFundsError),
    (401, {"detail": "bad key"}, AuthenticationError),
    (422, {"error": "unsafe prompt"}, ContentRejectedError),
    (429, {"error": "slow down", "reject_reason": "inflight_limit"}, TransientProviderError),
    (503, {}, TransientProviderError),
    (400, {"error": "bad aspect"}, ProviderError),
])
@responses.activate
def test_error_mapping(status, body, exc):
    add_quote()
    add_generate(status=status, body=body)
    with pytest.raises(exc) as e:
        IdeogramProvider("k").generate(req())
    assert e.value.status_code == status


@responses.activate
def test_402_message_is_clear_and_not_retryable():
    add_quote()
    add_generate(status=402, body={"error": "Insufficient credits", "reject_reason": "insufficient_funds"})
    with pytest.raises(InsufficientFundsError) as e:
        IdeogramProvider("k").generate(req())
    assert e.value.reject_reason == "insufficient_funds"
    assert e.value.retryable is False
    assert "Add credits" in str(e.value)


@responses.activate
def test_unsafe_image_is_rejected_and_costed():
    add_quote()
    add_generate(body={"generation_id": "g", "seed": 1, "data": [
        {"prompt": "p", "resolution": "1024x1024", "is_image_safe": False, "seed": 1, "url": None}]})
    with pytest.raises(ContentRejectedError) as e:
        IdeogramProvider("k").generate(req())
    assert e.value.cost_usd == pytest.approx(0.03)


@responses.activate
def test_expired_download_link_is_an_error():
    add_quote()
    add_generate()
    responses.add(responses.GET, IMG_URL, status=403)
    with pytest.raises(ProviderError, match="download"):
        IdeogramProvider("k").generate(req())


@responses.activate
def test_pipeline_aborts_cleanly_on_402(tmp_path):
    add_quote()
    add_generate(status=402, body={"error": "Insufficient credits", "reject_reason": "insufficient_funds"})
    m = run_pipeline(make_brief(n_assets=2), IdeogramProvider("k"), FakeQA(), tmp_path, max_cost=1.0,
                     sleep=lambda s: None)
    assert m["status"] == "aborted_provider"
    assert "HTTP 402" in m["stop_reason"]
    assert m["assets"][0]["attempts"][0]["http_status"] == 402
    assert m["assets"][1]["status"] == "not_run"
    assert m["totals"]["total_cost_usd"] == 0
    assert (tmp_path / m["run_id"] / "manifest.json").exists()


def test_missing_key_and_bad_speed():
    with pytest.raises(AuthenticationError):
        IdeogramProvider("")
    with pytest.raises(ValueError):
        IdeogramProvider("k", rendering_speed="warp")


@pytest.mark.parametrize("w,h,ratio", [(1080, 1080, "1x1"), (1080, 1920, "9x16"), (1200, 628, "2x1"),
                                       (1600, 900, "16x9"), (1080, 1350, "4x5")])
def test_nearest_aspect_ratio(w, h, ratio):
    assert nearest_aspect_ratio(w, h) == ratio
