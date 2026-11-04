import pytest

from media_pipeline.pipeline import run_pipeline
from media_pipeline.providers import MockProvider
from media_pipeline.providers.base import ProviderError, TransientProviderError
from media_pipeline.retry import backoff_delays, call_with_retries

from .conftest import FakeQA, make_brief


def test_retries_transient_then_succeeds():
    calls, slept = [], []

    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise TransientProviderError("503", status_code=503)
        return "ok"

    assert call_with_retries(fn, retries=3, base_delay=1.0, sleep=slept.append) == "ok"
    assert len(calls) == 3 and len(slept) == 2
    assert slept[1] > slept[0]  # exponential backoff


def test_gives_up_after_retries():
    slept = []

    def fn():
        raise TransientProviderError("429", status_code=429)

    with pytest.raises(TransientProviderError):
        call_with_retries(fn, retries=2, sleep=slept.append)
    assert len(slept) == 2


def test_non_retryable_is_not_retried():
    calls = []

    def fn():
        calls.append(1)
        raise ProviderError("400 bad request", status_code=400)

    with pytest.raises(ProviderError):
        call_with_retries(fn, retries=5, sleep=lambda s: None)
    assert len(calls) == 1


def test_backoff_delays_capped():
    d = backoff_delays(8, base=1.0, cap=10.0)
    assert d[0] < 1.6 and max(d) <= 10.5


def test_pipeline_recovers_from_transient_provider_errors(tmp_path):
    provider = MockProvider(defects=["asset-0:fail:2"])
    m = run_pipeline(make_brief(), provider, FakeQA(), tmp_path, max_cost=1.0, sleep=lambda s: None)
    entry = m["assets"][0]
    assert entry["status"] == "accepted"
    assert len(entry["attempts"]) == 1
    assert len(entry["attempts"][0]["retries"]) == 2
    assert entry["attempts"][0]["retries"][0]["status"] == 503


def test_pipeline_attempt_fails_when_retries_exhausted(tmp_path):
    provider = MockProvider(defects=["asset-0:fail:9"])
    m = run_pipeline(make_brief(), provider, FakeQA(), tmp_path, max_cost=1.0, max_attempts=2,
                     retries=1, sleep=lambda s: None)
    entry = m["assets"][0]
    # attempt 1 exhausts its retries; attempt 2 has its own (empty) failure plan and succeeds
    assert "after retries" in entry["attempts"][0]["error"]
    assert entry["status"] == "accepted" and entry["accepted_attempt"] == 2
