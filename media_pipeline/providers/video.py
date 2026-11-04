"""Video is an extension point, not a working feature.

No video provider is implemented or tested in this repo. This module fixes
the shape a video provider must have so it can plug into the same cost cap,
retry and manifest machinery. See docs/VIDEO.md.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VideoRequest:
    prompt: str
    width: int
    height: int
    duration_seconds: float = 5.0
    seed: Optional[int] = None
    start_image: Optional[bytes] = None  # for image-to-video
    asset_id: str = ""
    attempt: int = 1


@dataclass
class VideoResult:
    video_bytes: bytes
    cost_usd: float
    seed: Optional[int]
    mime_type: str = "video/mp4"
    meta: dict = field(default_factory=dict)


class VideoProvider(ABC):
    name = "video-base"

    @abstractmethod
    def estimate_cost(self, request: VideoRequest) -> float:
        """USD for one clip, known before the call (video is usually priced per second)."""

    @abstractmethod
    def submit(self, request: VideoRequest) -> str:
        """Start a render job and return its job id (video APIs are asynchronous)."""

    @abstractmethod
    def poll(self, job_id: str) -> Optional[VideoResult]:
        """Return the finished clip, or None if still rendering. Raise ProviderError on failure."""


class NotImplementedVideoProvider(VideoProvider):
    name = "video-not-implemented"

    def _no(self):
        raise NotImplementedError(
            "No video provider is implemented. Write a VideoProvider subclass "
            "(see docs/VIDEO.md); nothing in this repo generates video today.")

    def estimate_cost(self, request: VideoRequest) -> float:
        self._no()

    def submit(self, request: VideoRequest) -> str:
        self._no()

    def poll(self, job_id: str) -> Optional[VideoResult]:
        self._no()
