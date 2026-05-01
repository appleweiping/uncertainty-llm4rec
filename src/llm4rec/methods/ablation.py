"""Config-driven ablation controls for Phase 6 OursMethod."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DEFAULT_COMPONENTS = {
    "uncertainty_policy": True,
    "grounding_check": True,
    "candidate_normalized_confidence": True,
    "popularity_adjustment": True,
    "echo_risk_guard": True,
    "fallback": True,
    "fallback_only": False,
}

VARIANT_DISABLED_COMPONENTS = {
    "full": [],
    "no_uncertainty": ["uncertainty_policy"],
    "no_grounding": ["grounding_check"],
    "no_candidate_normalized_confidence": ["candidate_normalized_confidence"],
    "no_popularity_adjustment": ["popularity_adjustment"],
    "no_echo_guard": ["echo_risk_guard"],
    "no_fallback": ["fallback"],
    "fallback_only": [
        "generation_acceptance",
        "uncertainty_policy",
        "grounding_check",
        "candidate_normalized_confidence",
        "popularity_adjustment",
        "echo_risk_guard",
    ],
}


@dataclass(frozen=True, slots=True)
class AblationSettings:
    variant: str
    components: dict[str, bool]
    disabled_components: list[str]

    def enabled(self, component: str) -> bool:
        return bool(self.components.get(component, False))


def resolve_ablation_settings(method_config: dict[str, Any]) -> AblationSettings:
    params = dict(method_config.get("params") or {})
    ablation = dict(params.get("ablation") or {})
    variant = str(ablation.get("variant") or "full")
    if variant not in VARIANT_DISABLED_COMPONENTS:
        raise ValueError(f"unknown OursMethod ablation variant: {variant}")

    components = dict(DEFAULT_COMPONENTS)
    components.update({key: bool(value) for key, value in dict(params.get("components") or {}).items()})
    explicit_disabled = [str(value) for value in ablation.get("disabled_components") or []]
    disabled = _unique([*VARIANT_DISABLED_COMPONENTS[variant], *explicit_disabled])

    for component in disabled:
        if component in components:
            components[component] = False
    if variant == "fallback_only":
        components["fallback"] = True
        components["fallback_only"] = True

    return AblationSettings(
        variant=variant,
        components=components,
        disabled_components=disabled,
    )


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output
