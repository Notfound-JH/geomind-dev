from __future__ import annotations

from typing import Any


VARIANT_OVERLAYS: dict[str, dict[str, Any]] = {
    "base": {},
    "sem": {"meta.channel": "semantic"},
    "semantic": {"meta.channel": "semantic"},
    "visual": {"meta.channel": "visual"},
    "execution": {"meta.channel": "execution"},
}


def overlay_for_variant(variant: str) -> dict[str, Any]:
    return dict(VARIANT_OVERLAYS.get(str(variant).strip().lower(), {}))
