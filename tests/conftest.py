from typing import List, Optional

import pytest

from media_pipeline.brief import brief_from_dict
from media_pipeline.qa import QAVerdict, decide

GOOD = dict(matches_prompt=True, garbled_text=False, anatomy_issues=False, brand_colours_present=True,
            score=9, reasons=[], text_seen="FRESH ROAST", alt_text="A coffee cup with the words FRESH ROAST.")


def verdict(**over) -> dict:
    v = dict(GOOD)
    v.update(over)
    return v


class FakeQA:
    """Returns scripted verdicts in order (the last one repeats), records every call."""

    model = "fake-qa"

    def __init__(self, script: Optional[List[dict]] = None, min_score: int = 7, cost: float = 0.01,
                 estimate: float = 0.02):
        self.script = list(script or [GOOD])
        self.min_score = min_score
        self.cost = cost
        self.estimate = estimate
        self.calls = []

    def estimate_cost(self):
        return self.estimate

    def review(self, image_bytes, prompt, brand_name, brand_colours, required_text):
        self.calls.append({"prompt": prompt, "bytes": len(image_bytes), "required_text": required_text})
        fields = self.script[min(len(self.calls) - 1, len(self.script) - 1)]
        return QAVerdict(**fields, passed=decide(fields, self.min_score), cost_usd=self.cost, model=self.model)


def make_brief(n_assets=1, count=1, text="FRESH ROAST", watermark=None, sizes=("400x400", "200x200")):
    return brief_from_dict({
        "brand": {"name": "Test Brand", "colours": ["#1F6F5C", "#F2B134", "#F5EFE0"],
                  "style_notes": "flat vector"},
        "watermark": watermark,
        "assets": [{"id": f"asset-{i}", "prompt": "a coffee cup", "text": text,
                    "sizes": list(sizes), "count": count} for i in range(n_assets)],
    })


@pytest.fixture
def no_sleep():
    slept = []
    return slept.append, slept
