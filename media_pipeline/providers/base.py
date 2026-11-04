"""The provider interface every image (or, later, video) backend implements."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class GenerationRequest:
    prompt: str
    width: int
    height: int
    seed: Optional[int] = None
    negative_prompt: Optional[str] = None
    brand_colours: List[str] = field(default_factory=list)
    required_text: Optional[str] = None
    # Which asset / attempt this is. Real providers ignore it; the mock uses it
    # to decide where to inject a defect.
    asset_id: str = ""
    attempt: int = 1


@dataclass
class GenerationResult:
    image_bytes: bytes
    cost_usd: float
    seed: Optional[int]
    mime_type: str = "image/png"
    provider_prompt: Optional[str] = None  # the prompt the provider says it used
    meta: dict = field(default_factory=dict)


class ProviderError(Exception):
    """A provider call failed. `retryable` says whether trying again could help."""

    retryable = False

    def __init__(self, message: str, status_code: Optional[int] = None, retryable: Optional[bool] = None):
        super().__init__(message)
        self.status_code = status_code
        if retryable is not None:
            self.retryable = retryable


class TransientProviderError(ProviderError):
    """Rate limit, timeout, 5xx: worth retrying with backoff."""

    retryable = True


class AuthenticationError(ProviderError):
    """Bad or revoked API key. Stops the run: every later call would fail too."""

    fatal = True


class InsufficientFundsError(ProviderError):
    """The provider wallet is empty (HTTP 402). Stops the run."""

    fatal = True

    def __init__(self, message: str, status_code: Optional[int] = 402, reject_reason: Optional[str] = None):
        super().__init__(message, status_code=status_code, retryable=False)
        self.reject_reason = reject_reason


class ContentRejectedError(ProviderError):
    """The provider refused this prompt or image (safety check). Counts as a failed attempt."""


class ImageProvider(ABC):
    name = "base"

    @abstractmethod
    def estimate_cost(self, request: GenerationRequest) -> float:
        """Cost in USD of one generate() call, known BEFORE the call is made."""

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Make one image. Raise a ProviderError subclass on failure."""
