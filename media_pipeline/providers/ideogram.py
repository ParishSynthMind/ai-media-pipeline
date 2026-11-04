"""Ideogram 3.0 image provider (Ideogram API v2).

Endpoint: POST https://api.ideogram.ai/v2/image/generate/ideogram-3
Auth:     Api-Key header
Body:     multipart/form-data
Result:   data[].url is a temporary link, so the image is downloaded at once.

Cost is known before spending anything: the same endpoint with ?dry_run=true
validates and prices a request without generating or billing it. The quote is
cached per rendering speed. If the quote call fails, a fallback price table is
used (prices as quoted by the API on 2 November 2026).
"""
from __future__ import annotations

import json
import logging
from typing import Dict, Optional

import requests

from .base import (
    AuthenticationError,
    ContentRejectedError,
    GenerationRequest,
    GenerationResult,
    ImageProvider,
    InsufficientFundsError,
    ProviderError,
    TransientProviderError,
)

log = logging.getLogger(__name__)

BASE_URL = "https://api.ideogram.ai"
GENERATE_PATH = "/v2/image/generate/ideogram-3"
SPEEDS = ("turbo", "default", "quality")
# USD per image, from the API's own dry_run quotes on 2026-11-02.
FALLBACK_PRICES = {"turbo": 0.03, "default": 0.06, "quality": 0.09}
ASPECT_RATIOS = ["1x3", "3x1", "1x2", "2x1", "9x16", "16x9", "10x16", "16x10",
                 "2x3", "3x2", "3x4", "4x3", "4x5", "5x4", "1x1"]


def nearest_aspect_ratio(width: int, height: int) -> str:
    target = width / height

    def ratio(a: str) -> float:
        x, y = a.split("x")
        return int(x) / int(y)

    return min(ASPECT_RATIOS, key=lambda a: abs(ratio(a) - target))


def _error_body(resp: requests.Response) -> dict:
    try:
        body = resp.json()
        return body if isinstance(body, dict) else {}
    except ValueError:
        return {}


def raise_for_ideogram_status(resp: requests.Response) -> None:
    """Turn an Ideogram error response into the pipeline's error types."""
    code = resp.status_code
    if code < 400:
        return
    body = _error_body(resp)
    reason = body.get("reject_reason")
    msg = body.get("error") or body.get("message") or resp.text[:200] or resp.reason
    if code == 401:
        raise AuthenticationError(f"Ideogram rejected the API key (401): {msg}", status_code=401)
    if code == 402 or reason == "insufficient_funds":
        raise InsufficientFundsError(
            f"Ideogram API wallet cannot pay for this ({code}, {reason or 'payment required'}). "
            "Add credits in the Ideogram API dashboard under Billing.",
            status_code=code, reject_reason=reason)
    if code == 422:
        raise ContentRejectedError(f"Ideogram safety check rejected the prompt (422): {msg}", status_code=422)
    if code == 429:
        raise TransientProviderError(f"Ideogram rate limit (429, {reason or 'too many requests'})", status_code=429)
    if code >= 500:
        raise TransientProviderError(f"Ideogram server error ({code})", status_code=code)
    raise ProviderError(f"Ideogram request failed ({code}): {msg}", status_code=code)


class IdeogramProvider(ImageProvider):
    name = "ideogram"

    def __init__(self, api_key: str, rendering_speed: str = "turbo", magic_prompt: str = "off",
                 style_type: str = "design", use_brand_palette: bool = True,
                 session: Optional[requests.Session] = None, base_url: str = BASE_URL,
                 timeout: float = 120.0):
        if not api_key:
            raise AuthenticationError("no Ideogram API key (set IDEOGRAM_API_KEY)")
        if rendering_speed not in SPEEDS:
            raise ValueError(f"rendering_speed must be one of {SPEEDS}")
        self._api_key = api_key
        self.rendering_speed = rendering_speed
        self.magic_prompt = magic_prompt
        self.style_type = style_type
        self.use_brand_palette = use_brand_palette
        self.session = session or requests.Session()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._quotes: Dict[str, float] = {}

    @property
    def _headers(self) -> dict:
        return {"Api-Key": self._api_key}

    def build_form(self, req: GenerationRequest) -> dict:
        """Multipart fields. Values are (None, str) tuples so requests sends multipart, not urlencoded."""
        fields = {
            "prompt": req.prompt,
            "rendering_speed": self.rendering_speed,
            "aspect_ratio": nearest_aspect_ratio(req.width, req.height),
            "num_images": "1",
            "magic_prompt": self.magic_prompt,
            "style_type": self.style_type,
        }
        if req.seed is not None:
            fields["seed"] = str(int(req.seed) % 2147483648)
        if req.negative_prompt:
            fields["negative_prompt"] = req.negative_prompt
        if self.use_brand_palette and req.brand_colours:
            members = []
            for i, c in enumerate(req.brand_colours[:5]):
                members.append({"color_hex": c, "color_weight": round(max(0.05, 1.0 - i * 0.25), 2)})
            fields["color_palette"] = json.dumps({"members": members})
        return {k: (None, v) for k, v in fields.items()}

    def estimate_cost(self, request: GenerationRequest) -> float:
        speed = self.rendering_speed
        if speed in self._quotes:
            return self._quotes[speed]
        try:
            resp = self.session.post(f"{self.base_url}{GENERATE_PATH}", params={"dry_run": "true"},
                                     headers=self._headers, files=self.build_form(request), timeout=30)
            raise_for_ideogram_status(resp)
            quote = resp.json()
            price = int(quote["usd_micros"]) / 1_000_000 / max(1, int(quote.get("quantity", 1)))
            self._quotes[speed] = price
            log.info("Ideogram quote: %s image = $%.4f (%s)", speed, price, quote.get("qualifier"))
        except (AuthenticationError, InsufficientFundsError):
            raise
        except (requests.RequestException, ProviderError, KeyError, ValueError, TypeError) as e:
            price = FALLBACK_PRICES[speed]
            log.warning("Ideogram price quote failed (%s); using fallback $%.2f", e, price)
            self._quotes[speed] = price
        return self._quotes[speed]

    def generate(self, request: GenerationRequest) -> GenerationResult:
        price = self.estimate_cost(request)
        try:
            resp = self.session.post(f"{self.base_url}{GENERATE_PATH}", headers=self._headers,
                                     files=self.build_form(request), timeout=self.timeout)
        except (requests.Timeout, requests.ConnectionError) as e:
            raise TransientProviderError(f"Ideogram network error: {e}") from e
        raise_for_ideogram_status(resp)
        body = resp.json()
        data = body.get("data") or []
        if not data:
            raise ProviderError("Ideogram returned no images", status_code=resp.status_code)
        item = data[0]
        if not item.get("is_image_safe", True) or not item.get("url"):
            # Ideogram bills generations that complete, so count the cost even though there is no image.
            err = ContentRejectedError("Ideogram marked the image unsafe and returned no URL")
            err.cost_usd = price
            raise err
        try:
            img = self.session.get(item["url"], timeout=60)
        except (requests.Timeout, requests.ConnectionError) as e:
            raise ProviderError(f"could not download the Ideogram image: {e}") from e
        if img.status_code != 200 or not img.content:
            raise ProviderError(f"could not download the Ideogram image ({img.status_code})",
                                status_code=img.status_code)
        mime = img.headers.get("Content-Type", "image/png").split(";")[0].strip() or "image/png"
        return GenerationResult(image_bytes=img.content, cost_usd=price, seed=item.get("seed"),
                                mime_type=mime, provider_prompt=item.get("prompt"),
                                meta={"resolution": item.get("resolution"),
                                      "generation_id": body.get("generation_id"),
                                      "rendering_speed": self.rendering_speed})
