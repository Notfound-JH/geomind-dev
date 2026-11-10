from __future__ import annotations

from typing import Any


class BaseEvaluator:
    key = "always_success"

    def evaluate(self, episode_trace: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "success": True,
            "score": 1.0,
            "trace_len": len(episode_trace),
        }
