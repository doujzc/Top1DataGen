"""Controlled multi-turn synthesis primitives for Top1 training data."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import hashlib
import json
import random
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlsplit

from top1_data_gen.data import Top1DataError, normalize_messages, read_jsonl


SYNTHESIS_VERSION = "top1_controlled_multiturn_v1"
DIRECTNESS_AUDIT_VERSION = 3
CONTRAST_AUDIT_VERSION = 1
CONTRAST_NATURAL_LINK_AUDIT_VERSION = 2
DIALOGUE_QUALITY_AUDIT_VERSION = 1
PLAN_FIDELITY_AUDIT_VERSION = 1
PLAN_FIDELITY_CONSENSUS_AUDIT_VERSION = 2
OBSERVED_AXIS_PLAN_FIDELITY_AUDIT_VERSION = 3
STRICT_JSON_SCHEMA_PROTOCOL = "top1_strict_json_schema_v1"
OBSERVED_PHENOMENA = (
    "single_turn",
    "intent_change",
    "progressive_reveal",
    "contextual_follow_up",
    "clarification_revision",
    "assistant_distractor",
    "rambling",
    "other",
)
QUALITY_FIELDS = (
    "coherent",
    "natural",
    "current_target_identifiable",
    "policy_boundary_respected",
    "no_candidate_name_leakage",
)
STRICT_DIALOGUE_QUALITY_FIELDS = (
    "single_turn_standalone",
    "assistant_no_fabricated_personal_experience",
    "assistant_no_unsupported_factual_claims",
    "entity_and_facts_consistent",
    "conversation_context_consistent",
)
DIALOGUE_QUALITY_AUDIT_FIELDS = (
    "opening_context_valid",
    "temporal_context_consistent",
    "assistant_no_fabricated_personal_experience",
    "assistant_no_unsupported_factual_claims",
    "entity_and_facts_consistent",
    "natural_dialogue",
)
NON_SWITCH_WEIGHTS = (
    ("progressive_reveal", 18),
    ("contextual_follow_up", 14),
    ("clarification_revision", 10),
    ("assistant_distractor", 8),
    ("rambling", 4),
)

PHENOMENON_INSTRUCTIONS = {
    "single_turn": (
        "只生成一条 user 消息，不生成 assistant。请求必须自然、完整且明确属于 "
        "target_candidate；同时在主题、表面措辞或任务形式上靠近 contrast_candidate，"
        "但不能真的同时包含两个待执行目标，也不能泄漏候选英文名。唯一例外是 content_axis "
        "定义本身明确要求生成固有无意义或信息不足的输入，此时可按该定义缺少任务对象或动作，"
        "但仍不得用缺失先行词的指代假装存在历史。"
    ),
    "intent_change": (
        "前面的用户轮次保持 source_candidate 目标；最后一轮直接提出 target_candidate "
        "的新需求。最后一轮必须整条消息都只表达新需求：不得回应或评价上一轮，不得致谢、确认，"
        "也不得用连接语宣布或暗示切换、取消、顺带提问或换题。直接从新需求本身说起；新旧主题可以"
        "完全无关，这种突然跨领域且没有过渡的表达正是预期，不要为了显得自然补写过渡语。"
    ),
    "progressive_reveal": (
        "所有用户轮次属于同一 target_candidate。用户逐轮补充约束或细节；最后一轮必须显式重述"
        "核心对象和动作，单独阅读也能判明当前目标；不得仅用‘它’‘这个’‘那’‘呢’‘还会吗’等"
        "指代、省略、确认或纯短追问承接历史，否则属于 contextual_follow_up。"
    ),
    "contextual_follow_up": (
        "所有用户轮次属于同一 target_candidate。最后一轮必须使用自然的省略、指代、确认或短追问，"
        "需要结合紧邻历史才能完整理解；不能变成无法恢复对象的残缺输入。"
    ),
    "clarification_revision": (
        "所有用户轮次属于同一 target_candidate。最后一轮自然纠正或修订先前的对象属性、条件或"
        "期望结果，但不改变候选类别。"
    ),
    "assistant_distractor": (
        "用户目标始终属于 target_candidate；某一条 assistant 消息出现合理的误解、无关建议或错误"
        "侧重点，最后一轮用户直接澄清真实目标。不要让干扰内容改变最终目标。"
    ),
    "rambling": (
        "用户在保持 target_candidate 目标的同时加入生活化背景或无关细节；最后一轮回到清晰可判的"
        "当前请求，冗余信息不能成为另一个待执行任务。"
    ),
}


@dataclass(frozen=True)
class DialogueBlueprint:
    """A deterministic semantic plan rendered by an LLM later."""

    scenario_id: str
    phenomenon: str
    target_candidate_name: str
    source_candidate_name: str | None
    user_turn_count: int
    seed: int
    contrast_candidate_name: str | None = None
    content_axis: str | None = None
    source_content_axis: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON representation."""

        payload = {
            "scenario_id": self.scenario_id,
            "phenomenon": self.phenomenon,
            "target_candidate_name": self.target_candidate_name,
            "source_candidate_name": self.source_candidate_name,
            "user_turn_count": self.user_turn_count,
            "seed": self.seed,
            "turn_plan": _turn_plan(self),
        }
        if self.contrast_candidate_name is not None:
            payload["contrast_candidate_name"] = self.contrast_candidate_name
        if self.content_axis is not None:
            payload["content_axis"] = self.content_axis
        if self.source_content_axis is not None:
            payload["source_content_axis"] = self.source_content_axis
        return payload


@dataclass(frozen=True)
class ModelCall:
    """One parsed OpenAI-compatible chat completion."""

    content: str
    usage: Mapping[str, Any]
    finish_reason: str | None
    elapsed_seconds: float


def _exact_object_schema(properties: Mapping[str, Any]) -> dict[str, Any]:
    """Build an exact JSON object schema with every property required."""

    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(properties),
        "additionalProperties": False,
    }


def _response_ids(values: Iterable[str]) -> tuple[str, ...]:
    ids = tuple(values)
    if (
        not ids
        or any(not isinstance(value, str) or not value for value in ids)
        or len(set(ids)) != len(ids)
    ):
        raise Top1DataError("response schema requires distinct non-empty IDs")
    return ids


def _string_enum(values: Iterable[str], *, context: str) -> dict[str, Any]:
    items = tuple(values)
    if (
        not items
        or any(not isinstance(value, str) or not value for value in items)
        or len(set(items)) != len(items)
    ):
        raise Top1DataError(f"{context} requires distinct non-empty strings")
    return {"type": "string", "enum": list(items)}


def _batch_response_schema(
    response_field: str,
    item_schema: Mapping[str, Any],
    batch_size: int,
) -> dict[str, Any]:
    return _exact_object_schema(
        {
            response_field: {
                "type": "array",
                "items": dict(item_schema),
                "minItems": batch_size,
                "maxItems": batch_size,
            }
        }
    )


def generation_response_schema(
    blueprints: Sequence[DialogueBlueprint],
) -> dict[str, Any]:
    """Build the strict structured-output schema for one generation batch."""

    ids = _response_ids(blueprint.scenario_id for blueprint in blueprints)
    max_messages = max(2 * blueprint.user_turn_count - 1 for blueprint in blueprints)
    message_schema = _exact_object_schema(
        {
            "role": {"type": "string", "enum": ["user", "assistant"]},
            "content": {"type": "string", "minLength": 1},
        }
    )
    sample_schema = _exact_object_schema(
        {
            "scenario_id": _string_enum(ids, context="generation scenario IDs"),
            "messages": {
                "type": "array",
                "items": message_schema,
                "minItems": 1,
                "maxItems": max_messages,
            },
            "scenario_summary": {"type": "string"},
        }
    )
    return _batch_response_schema("samples", sample_schema, len(ids))


def judgment_response_schema(
    assigned_ids: Iterable[str],
    candidate_names: Sequence[str],
    *,
    quality_fields: Sequence[str] = QUALITY_FIELDS,
) -> dict[str, Any]:
    """Build the strict structured-output schema for a blind-judge batch."""

    ids = _response_ids(assigned_ids)
    candidate_schema = _string_enum(candidate_names, context="candidate names")
    quality_names = tuple(quality_fields)
    if not quality_names or len(set(quality_names)) != len(quality_names):
        raise Top1DataError("quality fields must be distinct and non-empty")
    quality_schema = _exact_object_schema(
        {field: {"type": "boolean"} for field in quality_names}
    )
    item_schema = _exact_object_schema(
        {
            "scenario_id": _string_enum(ids, context="judgment scenario IDs"),
            "predicted_candidate_name": candidate_schema,
            "observed_phenomenon": _string_enum(
                OBSERVED_PHENOMENA,
                context="observed phenomena",
            ),
            "observed_source_candidate_name": {
                "anyOf": [candidate_schema, {"type": "null"}]
            },
            "intent_change_is_direct": {"type": "boolean"},
            "quality": quality_schema,
            "issues": {"type": "array", "items": {"type": "string"}},
        }
    )
    return _batch_response_schema("judgments", item_schema, len(ids))


def directness_response_schema(assigned_ids: Iterable[str]) -> dict[str, Any]:
    """Build the strict structured-output schema for directness auditing."""

    ids = _response_ids(assigned_ids)
    item_schema = _exact_object_schema(
        {
            "scenario_id": _string_enum(ids, context="directness scenario IDs"),
            "contains_only_new_request": {"type": "boolean"},
            "references_previous_exchange": {"type": "boolean"},
            "uses_transition_or_acknowledgment": {"type": "boolean"},
            "direct_final_request": {"type": "boolean"},
            "has_switch_meta_language": {"type": "boolean"},
            "reason": {"type": "string", "minLength": 1},
        }
    )
    return _batch_response_schema("audits", item_schema, len(ids))


def dialogue_quality_response_schema(
    assigned_ids: Iterable[str],
) -> dict[str, Any]:
    """Build the strict structured-output schema for dialogue-quality auditing."""

    ids = _response_ids(assigned_ids)
    item_schema = _exact_object_schema(
        {
            "scenario_id": _string_enum(
                ids,
                context="dialogue-quality scenario IDs",
            ),
            **{
                field: {"type": "boolean"}
                for field in DIALOGUE_QUALITY_AUDIT_FIELDS
            },
            "reason": {"type": "string", "minLength": 1},
        }
    )
    return _batch_response_schema("audits", item_schema, len(ids))


def contrast_response_schema(
    assigned_ids: Iterable[str],
    *,
    require_natural_link: bool = False,
) -> dict[str, Any]:
    """Build the strict structured-output schema for contrast auditing."""

    ids = _response_ids(assigned_ids)
    fields = (
        "target_semantics_present",
        "contrast_is_plausible",
        "target_preferred_over_contrast",
        "single_current_goal",
        "natural_expression",
    )
    if require_natural_link:
        fields += ("contrast_link_natural",)
    item_schema = _exact_object_schema(
        {
            "scenario_id": _string_enum(ids, context="contrast scenario IDs"),
            **{field: {"type": "boolean"} for field in fields},
            "reason": {"type": "string", "minLength": 1},
        }
    )
    return _batch_response_schema("audits", item_schema, len(ids))


def plan_fidelity_response_schema(
    assigned_ids: Iterable[str],
    *,
    observed_axis_catalogs: Mapping[
        str, Mapping[str, Sequence[str] | None]
    ]
    | None = None,
) -> dict[str, Any]:
    """Build the strict structured-output schema for plan-fidelity auditing."""

    ids = _response_ids(assigned_ids)
    properties: dict[str, Any] = {
        "scenario_id": _string_enum(ids, context="plan-fidelity scenario IDs"),
        "target_axis_realized": {"type": "boolean"},
        "target_candidate_respected": {"type": "boolean"},
        "source_axis_realized": {"type": "boolean"},
        "source_candidate_respected": {"type": "boolean"},
        "phenomenon_realized": {"type": "boolean"},
        "no_extra_current_goal": {"type": "boolean"},
    }
    reason_min_length = 1
    if observed_axis_catalogs is not None:
        if set(observed_axis_catalogs) != set(ids):
            raise Top1DataError(
                "observed-axis response schema must cover exactly the batch IDs"
            )
        target_axes: list[str] = []
        source_axes: list[str] = []
        for scenario_id in ids:
            catalog = observed_axis_catalogs[scenario_id]
            raw_target_axes = catalog.get("target_content_axes")
            raw_source_axes = catalog.get("source_content_axes")
            if (
                not isinstance(raw_target_axes, Sequence)
                or isinstance(raw_target_axes, (str, bytes))
                or not raw_target_axes
            ):
                raise Top1DataError(
                    "observed-axis response schema requires target axes"
                )
            for axis in raw_target_axes:
                if not isinstance(axis, str) or not axis:
                    raise Top1DataError(
                        "observed-axis response schema has an invalid target axis"
                    )
                if axis not in target_axes:
                    target_axes.append(axis)
            if raw_source_axes is not None:
                if (
                    not isinstance(raw_source_axes, Sequence)
                    or isinstance(raw_source_axes, (str, bytes))
                    or not raw_source_axes
                ):
                    raise Top1DataError(
                        "observed-axis response schema has invalid source axes"
                    )
                for axis in raw_source_axes:
                    if not isinstance(axis, str) or not axis:
                        raise Top1DataError(
                            "observed-axis response schema has an invalid source axis"
                        )
                    if axis not in source_axes:
                        source_axes.append(axis)
        properties.update(
            {
                "observed_target_content_axis": _string_enum(
                    target_axes,
                    context="observed target axes",
                ),
                "observed_source_content_axis": (
                    {
                        "anyOf": [
                            _string_enum(source_axes, context="observed source axes"),
                            {"type": "null"},
                        ]
                    }
                    if source_axes
                    else {"type": "null"}
                ),
                "target_axis_unambiguous": {"type": "boolean"},
                "source_axis_unambiguous": {"type": "boolean"},
            }
        )
        reason_min_length = 20
    properties["reason"] = {
        "type": "string",
        "minLength": reason_min_length,
    }
    return _batch_response_schema(
        "audits",
        _exact_object_schema(properties),
        len(ids),
    )


def _slug(value: str) -> str:
    return "".join(character.lower() if character.isalnum() else "_" for character in value).strip("_")


def _turn_plan(blueprint: DialogueBlueprint) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for turn in range(1, blueprint.user_turn_count + 1):
        candidate = blueprint.target_candidate_name
        behavior = blueprint.phenomenon
        if blueprint.phenomenon == "intent_change" and turn < blueprint.user_turn_count:
            candidate = str(blueprint.source_candidate_name)
            behavior = "source_context"
        elif blueprint.phenomenon == "intent_change":
            behavior = "direct_intent_change"
        plan.append(
            {
                "user_turn": turn,
                "candidate_name": candidate,
                "behavior": behavior,
            }
        )
    return plan


def build_dialogue_blueprints(
    candidate_names: Sequence[str],
    *,
    target_count: int = 800,
    intent_change_per_pair: int = 10,
    seed: int = 20260818,
    synthesis_version: str = SYNTHESIS_VERSION,
    single_turn_per_candidate: int = 0,
    single_turn_confusions: Mapping[str, Sequence[str]] | None = None,
    single_turn_axis_confusions: (
        Mapping[str, Mapping[str, Sequence[str]]] | None
    ) = None,
    multi_turn_user_counts: Sequence[int] = (3, 4, 5),
    content_axes: Mapping[str, Sequence[str]] | None = None,
    content_axis_allowed_phenomena: (
        Mapping[str, Mapping[str, Sequence[str]]] | None
    ) = None,
) -> list[DialogueBlueprint]:
    """Build a balanced, plan-first set of controlled dialogue scenarios."""

    names = tuple(candidate_names)
    if len(names) < 2 or len(set(names)) != len(names):
        raise Top1DataError("synthesis requires at least two unique candidates")
    if intent_change_per_pair < 0:
        raise Top1DataError("intent_change_per_pair cannot be negative")
    if single_turn_per_candidate < 0:
        raise Top1DataError("single_turn_per_candidate cannot be negative")
    turn_counts = tuple(multi_turn_user_counts)
    if not turn_counts or any(
        not isinstance(value, int) or isinstance(value, bool) or value < 2
        for value in turn_counts
    ):
        raise Top1DataError("multi_turn_user_counts must contain integers of at least 2")
    if not synthesis_version or any(
        not (character.isalnum() or character in {"_", "-"})
        for character in synthesis_version
    ):
        raise Top1DataError(
            "synthesis_version must contain only letters, numbers, '_' or '-'"
        )
    confusion_map = single_turn_confusions or {}
    content_axis_map = content_axes or {}
    if single_turn_per_candidate and single_turn_axis_confusions is None:
        if set(confusion_map) != set(names):
            raise Top1DataError(
                "single_turn_confusions must cover exactly the candidate registry"
            )
        for target, confusions in confusion_map.items():
            if not confusions:
                raise Top1DataError(
                    f"single_turn_confusions[{target!r}] cannot be empty"
                )
            if any(value not in names or value == target for value in confusions):
                raise Top1DataError(
                    f"single_turn_confusions[{target!r}] contains an invalid candidate"
                )
    if content_axis_map:
        if set(content_axis_map) != set(names):
            raise Top1DataError("content_axes must cover exactly the candidate registry")
        for target, axes in content_axis_map.items():
            if (
                not isinstance(axes, Sequence)
                or isinstance(axes, (str, bytes))
                or not axes
                or any(not isinstance(axis, str) or not axis.strip() for axis in axes)
            ):
                raise Top1DataError(f"content_axes[{target!r}] must contain non-empty strings")
            normalized_axes = tuple(axis.strip() for axis in axes)
            if len(set(normalized_axes)) != len(normalized_axes):
                raise Top1DataError(f"content_axes[{target!r}] contains duplicate axes")
    axis_confusion_map = _normalize_single_turn_axis_confusions(
        names,
        content_axis_map,
        single_turn_axis_confusions,
    )
    allowed_phenomena = _normalize_content_axis_allowed_phenomena(
        names,
        content_axis_map,
        content_axis_allowed_phenomena,
    )
    single_rows = len(names) * single_turn_per_candidate
    pair_rows = len(names) * (len(names) - 1) * intent_change_per_pair
    required_rows = single_rows + pair_rows
    if target_count < required_rows:
        raise Top1DataError(
            f"target_count={target_count} is smaller than the {required_rows} "
            "required single-turn and intent-change rows"
        )

    rng = random.Random(seed)
    plans: list[DialogueBlueprint] = []

    for target in names:
        confusions = tuple(confusion_map.get(target, ()))
        for index in range(1, single_turn_per_candidate + 1):
            plans.append(
                DialogueBlueprint(
                    scenario_id=(
                        f"{synthesis_version}_single_turn_"
                        f"{_slug(target)}_{index:03d}"
                    ),
                    phenomenon="single_turn",
                    target_candidate_name=target,
                    source_candidate_name=None,
                    user_turn_count=1,
                    seed=rng.randrange(2**31),
                    contrast_candidate_name=(
                        confusions[(index - 1) % len(confusions)]
                        if confusions
                        else None
                    ),
                )
            )
    for source in names:
        for target in names:
            if source == target:
                continue
            for index in range(1, intent_change_per_pair + 1):
                plans.append(
                    DialogueBlueprint(
                        scenario_id=(
                            f"{synthesis_version}_intent_change_"
                            f"{_slug(source)}_to_{_slug(target)}_{index:03d}"
                        ),
                        phenomenon="intent_change",
                        target_candidate_name=target,
                        source_candidate_name=source,
                        user_turn_count=rng.choice(turn_counts),
                        seed=rng.randrange(2**31),
                    )
                )

    weighted_cycle = [
        phenomenon
        for phenomenon, weight in NON_SWITCH_WEIGHTS
        for _ in range(weight)
    ]
    per_key_count: Counter[tuple[str, str]] = Counter()
    remaining = target_count - len(plans)
    if single_turn_per_candidate:
        base_quota, extra = divmod(remaining, len(names))
        for target_index, target in enumerate(names):
            quota = base_quota + int(target_index < extra)
            phenomenon_sequence = _weighted_phenomenon_allocation(quota)
            for phenomenon in phenomenon_sequence:
                key = (phenomenon, target)
                per_key_count[key] += 1
                plans.append(
                    DialogueBlueprint(
                        scenario_id=(
                            f"{synthesis_version}_{phenomenon}_"
                            f"{_slug(target)}_{per_key_count[key]:03d}"
                        ),
                        phenomenon=phenomenon,
                        target_candidate_name=target,
                        source_candidate_name=None,
                        user_turn_count=rng.choice(turn_counts),
                        seed=rng.randrange(2**31),
                    )
                )
    else:
        for offset in range(remaining):
            target = names[offset % len(names)]
            candidate_sequence_index = offset // len(names)
            phenomenon = weighted_cycle[candidate_sequence_index % len(weighted_cycle)]
            key = (phenomenon, target)
            per_key_count[key] += 1
            plans.append(
                DialogueBlueprint(
                    scenario_id=(
                        f"{synthesis_version}_{phenomenon}_"
                        f"{_slug(target)}_{per_key_count[key]:03d}"
                    ),
                    phenomenon=phenomenon,
                    target_candidate_name=target,
                    source_candidate_name=None,
                    user_turn_count=rng.choice(turn_counts),
                    seed=rng.randrange(2**31),
                )
            )

    if content_axis_map:
        plans = _assign_content_axes(
            plans,
            names,
            content_axis_map,
            allowed_phenomena,
        )
    if axis_confusion_map is not None:
        plans = _assign_single_turn_axis_confusions(
            plans,
            axis_confusion_map,
        )
    if len({plan.scenario_id for plan in plans}) != len(plans):
        raise Top1DataError("duplicate synthesis scenario IDs")
    rng.shuffle(plans)
    return plans


def _weighted_phenomenon_allocation(count: int) -> list[str]:
    """Allocate all non-switch phenomena proportionally for one candidate."""

    if count <= 0:
        return []
    total_weight = sum(weight for _, weight in NON_SWITCH_WEIGHTS)
    allocations = {
        phenomenon: count * weight // total_weight
        for phenomenon, weight in NON_SWITCH_WEIGHTS
    }
    remainder = count - sum(allocations.values())
    ranked = sorted(
        NON_SWITCH_WEIGHTS,
        key=lambda item: (count * item[1]) % total_weight,
        reverse=True,
    )
    for phenomenon, _ in ranked[:remainder]:
        allocations[phenomenon] += 1
    return [
        phenomenon
        for phenomenon, _ in NON_SWITCH_WEIGHTS
        for _ in range(allocations[phenomenon])
    ]


def _normalize_single_turn_axis_confusions(
    candidate_names: Sequence[str],
    content_axes: Mapping[str, Sequence[str]],
    configured: Mapping[str, Mapping[str, Sequence[str]]] | None,
) -> dict[str, dict[str, tuple[str, ...]]] | None:
    """Validate an optional exact axis-level single-turn contrast registry."""

    if configured is None:
        return None
    if not isinstance(configured, Mapping):
        raise Top1DataError("single_turn_axis_confusions must be an object")
    if not content_axes:
        raise Top1DataError(
            "single_turn_axis_confusions requires a non-empty content_axes mapping"
        )

    names = tuple(candidate_names)
    expected_candidates = set(names)
    configured_candidates = set(configured)
    if configured_candidates != expected_candidates:
        missing = sorted(expected_candidates - configured_candidates)
        unknown = sorted(configured_candidates - expected_candidates)
        details: list[str] = []
        if missing:
            details.append("missing candidates: " + ", ".join(missing))
        if unknown:
            details.append("unknown candidates: " + ", ".join(unknown))
        raise Top1DataError(
            "single_turn_axis_confusions must cover exactly the candidate registry"
            + (" (" + "; ".join(details) + ")" if details else "")
        )

    normalized: dict[str, dict[str, tuple[str, ...]]] = {}
    for target in names:
        raw_axis_map = configured[target]
        if not isinstance(raw_axis_map, Mapping):
            raise Top1DataError(
                f"single_turn_axis_confusions[{target!r}] must be an object"
            )
        expected_axes = {
            str(axis).strip() for axis in content_axes.get(target, ())
        }
        configured_axes = set(raw_axis_map)
        if configured_axes != expected_axes:
            missing = sorted(expected_axes - configured_axes)
            unknown = sorted(configured_axes - expected_axes)
            details = []
            if missing:
                details.append("missing axes: " + ", ".join(missing))
            if unknown:
                details.append("unknown axes: " + ", ".join(unknown))
            raise Top1DataError(
                f"single_turn_axis_confusions[{target!r}] must cover exactly "
                "the configured content axes"
                + (" (" + "; ".join(details) + ")" if details else "")
            )

        normalized[target] = {}
        for axis in content_axes[target]:
            normalized_axis = str(axis).strip()
            raw_confusions = raw_axis_map[normalized_axis]
            if (
                not isinstance(raw_confusions, Sequence)
                or isinstance(raw_confusions, (str, bytes))
                or not raw_confusions
                or any(
                    not isinstance(value, str) or not value.strip()
                    for value in raw_confusions
                )
            ):
                raise Top1DataError(
                    f"single_turn_axis_confusions[{target!r}]"
                    f"[{normalized_axis!r}] must contain non-empty candidate names"
                )
            confusions = tuple(value.strip() for value in raw_confusions)
            if len(set(confusions)) != len(confusions):
                raise Top1DataError(
                    f"single_turn_axis_confusions[{target!r}]"
                    f"[{normalized_axis!r}] contains duplicate candidates"
                )
            invalid = [
                value
                for value in confusions
                if value not in expected_candidates or value == target
            ]
            if invalid:
                raise Top1DataError(
                    f"single_turn_axis_confusions[{target!r}]"
                    f"[{normalized_axis!r}] contains an unknown or self candidate: "
                    + ", ".join(invalid)
                )
            normalized[target][normalized_axis] = confusions
    return normalized


def _normalize_content_axis_allowed_phenomena(
    candidate_names: Sequence[str],
    content_axes: Mapping[str, Sequence[str]],
    configured: Mapping[str, Mapping[str, Sequence[str]]] | None,
) -> dict[str, dict[str, frozenset[str]]]:
    """Validate optional axis restrictions and expand them to a complete map."""

    if configured is not None and not isinstance(configured, Mapping):
        raise Top1DataError("content_axis_allowed_phenomena must be an object")
    overrides = configured or {}
    if overrides and not content_axes:
        raise Top1DataError(
            "content_axis_allowed_phenomena requires a non-empty content_axes mapping"
        )
    unknown_candidates = set(overrides) - set(candidate_names)
    if unknown_candidates:
        raise Top1DataError(
            "content_axis_allowed_phenomena contains unknown candidates: "
            + ", ".join(sorted(unknown_candidates))
        )

    all_phenomena = (
        "single_turn",
        "intent_change",
        *(phenomenon for phenomenon, _ in NON_SWITCH_WEIGHTS),
    )
    allowed_values = set(all_phenomena)
    result: dict[str, dict[str, frozenset[str]]] = {}
    for candidate in candidate_names:
        axes = tuple(str(axis).strip() for axis in content_axes.get(candidate, ()))
        raw_rules = overrides.get(candidate, {})
        if not isinstance(raw_rules, Mapping):
            raise Top1DataError(
                f"content_axis_allowed_phenomena[{candidate!r}] must be an object"
            )
        unknown_axes = set(raw_rules) - set(axes)
        if unknown_axes:
            raise Top1DataError(
                f"content_axis_allowed_phenomena[{candidate!r}] contains unknown axes: "
                + ", ".join(sorted(unknown_axes))
            )
        result[candidate] = {}
        for axis in axes:
            raw_phenomena = raw_rules.get(axis, all_phenomena)
            if (
                not isinstance(raw_phenomena, Sequence)
                or isinstance(raw_phenomena, (str, bytes))
                or not raw_phenomena
                or any(not isinstance(value, str) for value in raw_phenomena)
            ):
                raise Top1DataError(
                    f"content_axis_allowed_phenomena[{candidate!r}][{axis!r}] "
                    "must contain non-empty phenomenon names"
                )
            phenomena = tuple(value.strip() for value in raw_phenomena)
            if len(set(phenomena)) != len(phenomena):
                raise Top1DataError(
                    f"content_axis_allowed_phenomena[{candidate!r}][{axis!r}] "
                    "contains duplicate phenomena"
                )
            invalid = set(phenomena) - allowed_values
            if invalid:
                raise Top1DataError(
                    f"content_axis_allowed_phenomena[{candidate!r}][{axis!r}] "
                    "contains invalid phenomena: "
                    + ", ".join(sorted(invalid))
                )
            result[candidate][axis] = frozenset(phenomena)
    return result


def _assign_single_turn_axis_confusions(
    plans: Sequence[DialogueBlueprint],
    axis_confusions: Mapping[str, Mapping[str, Sequence[str]]],
) -> list[DialogueBlueprint]:
    """Assign balanced axis-compatible contrasts before the final plan shuffle."""

    assigned = list(plans)
    for target, axis_map in axis_confusions.items():
        for axis, confusions in axis_map.items():
            indices = [
                index
                for index, plan in enumerate(assigned)
                if plan.phenomenon == "single_turn"
                and plan.target_candidate_name == target
                and plan.content_axis == axis
            ]
            choices = tuple(confusions)
            for offset, index in enumerate(indices):
                assigned[index] = replace(
                    assigned[index],
                    contrast_candidate_name=choices[offset % len(choices)],
                )
    return assigned


def _balanced_axis_assignments(
    phenomena: Sequence[str],
    axes: Sequence[str],
    allowed_by_axis: Mapping[str, frozenset[str]],
    *,
    context: str,
) -> tuple[str, ...]:
    """Assign axes deterministically while honoring restrictions and balance."""

    if not phenomena:
        return ()
    axis_order = {axis: index for index, axis in enumerate(axes)}
    choices = [
        tuple(axis for axis in axes if phenomenon in allowed_by_axis[axis])
        for phenomenon in phenomena
    ]
    for phenomenon, allowed_axes in zip(phenomena, choices):
        if not allowed_axes:
            raise Top1DataError(
                f"no content axis allows phenomenon {phenomenon!r} for {context}"
            )

    assignments: list[str | None] = [None] * len(phenomena)
    counts: Counter[str] = Counter()
    # Allocate the most constrained rows first. Least-used-axis tie-breaking then
    # lets flexible rows fill any deficit left by restricted axes.
    allocation_order = sorted(
        range(len(phenomena)),
        key=lambda index: (len(choices[index]), index),
    )
    for index in allocation_order:
        axis = min(
            choices[index],
            key=lambda value: (counts[value], axis_order[value]),
        )
        assignments[index] = axis
        counts[axis] += 1

    active_axes = tuple(
        axis
        for axis in axes
        if any(phenomenon in allowed_by_axis[axis] for phenomenon in phenomena)
    )
    active_counts = [counts[axis] for axis in active_axes]
    if active_counts and max(active_counts) - min(active_counts) > 1:
        raise Top1DataError(
            f"content axes cannot be balanced within one row for {context}: "
            + ", ".join(f"{axis}={counts[axis]}" for axis in active_axes)
        )
    return tuple(str(axis) for axis in assignments)


def _assign_content_axes(
    plans: Sequence[DialogueBlueprint],
    candidate_names: Sequence[str],
    content_axes: Mapping[str, Sequence[str]],
    allowed_phenomena: Mapping[str, Mapping[str, frozenset[str]]],
) -> list[DialogueBlueprint]:
    """Apply balanced target and source axes to an immutable blueprint plan."""

    assigned = list(plans)
    for candidate in candidate_names:
        axes = tuple(str(axis).strip() for axis in content_axes[candidate])
        target_indices = [
            index
            for index, plan in enumerate(assigned)
            if plan.target_candidate_name == candidate
        ]
        target_assignments = _balanced_axis_assignments(
            [assigned[index].phenomenon for index in target_indices],
            axes,
            allowed_phenomena[candidate],
            context=f"target candidate {candidate}",
        )
        for index, axis in zip(target_indices, target_assignments):
            assigned[index] = replace(assigned[index], content_axis=axis)
        target_counts = Counter(target_assignments)
        missing_axes = [axis for axis in axes if not target_counts[axis]]
        if missing_axes:
            raise Top1DataError(
                f"target candidate {candidate} does not cover content axes: "
                + ", ".join(missing_axes)
            )

        source_indices = [
            index
            for index, plan in enumerate(assigned)
            if plan.phenomenon == "intent_change"
            and plan.source_candidate_name == candidate
        ]
        source_assignments = _balanced_axis_assignments(
            ["intent_change"] * len(source_indices),
            axes,
            allowed_phenomena[candidate],
            context=f"source candidate {candidate}",
        )
        for index, axis in zip(source_indices, source_assignments):
            assigned[index] = replace(assigned[index], source_content_axis=axis)
    return assigned


def load_taxonomy_descriptions(
    path: str | Path,
    candidate_names: Sequence[str],
) -> dict[str, dict[str, str]]:
    """Load concise and extended definitions from the reviewed LabelDesc data."""

    names = tuple(candidate_names)
    result = {name: {} for name in names}
    for row in read_jsonl(path):
        candidate = row.get("target_candidate_name")
        description_type = row.get("description_type")
        messages = row.get("messages")
        if candidate not in result or description_type not in {
            "concise_definition",
            "extended_definition",
        }:
            continue
        normalized = normalize_messages(messages)
        if len(normalized) != 1:
            raise Top1DataError("LabelDesc definitions must be single-turn")
        result[str(candidate)][str(description_type)] = normalized[0]["content"]

    missing = [
        f"{candidate}:{description_type}"
        for candidate in names
        for description_type in ("concise_definition", "extended_definition")
        if description_type not in result[candidate]
    ]
    if missing:
        raise Top1DataError("missing synthesis taxonomy definitions: " + ", ".join(missing))
    return result


def taxonomy_prompt(descriptions: Mapping[str, Mapping[str, str]]) -> str:
    """Render the closed candidate taxonomy for generators and blind judges."""

    sections: list[str] = []
    for candidate, values in descriptions.items():
        sections.extend(
            (
                f"### {candidate}",
                str(values["concise_definition"]),
                str(values["extended_definition"]),
            )
        )
    return "\n\n".join(sections)


def validate_content_axis_definitions(
    content_axes: Mapping[str, Sequence[str]] | None,
    content_axis_definitions: Mapping[str, Mapping[str, str]] | None,
    candidate_names: Sequence[str],
) -> dict[str, dict[str, str]] | None:
    """Validate optional axis definitions against the exact configured axis shape."""

    if content_axis_definitions is None:
        return None
    if not isinstance(content_axes, Mapping) or not content_axes:
        raise Top1DataError(
            "content_axis_definitions requires a non-empty content_axes mapping"
        )
    if not isinstance(content_axis_definitions, Mapping):
        raise Top1DataError("content_axis_definitions must be an object")
    expected_candidates = set(candidate_names)
    if set(content_axes) != expected_candidates:
        raise Top1DataError("content_axes must cover exactly the candidate registry")
    if set(content_axis_definitions) != expected_candidates:
        raise Top1DataError(
            "content_axis_definitions must cover exactly the candidate registry"
        )

    normalized: dict[str, dict[str, str]] = {}
    for candidate in candidate_names:
        raw_axes = content_axes[candidate]
        if (
            not isinstance(raw_axes, Sequence)
            or isinstance(raw_axes, (str, bytes))
            or not raw_axes
            or any(not isinstance(axis, str) or not axis.strip() for axis in raw_axes)
        ):
            raise Top1DataError(
                f"content_axes[{candidate!r}] must contain non-empty strings"
            )
        axes = tuple(axis.strip() for axis in raw_axes)
        if len(set(axes)) != len(axes):
            raise Top1DataError(f"content_axes[{candidate!r}] contains duplicate axes")
        raw_definitions = content_axis_definitions[candidate]
        if not isinstance(raw_definitions, Mapping):
            raise Top1DataError(
                f"content_axis_definitions[{candidate!r}] must be an object"
            )
        if set(raw_definitions) != set(axes):
            raise Top1DataError(
                f"content_axis_definitions[{candidate!r}] must define exactly "
                "the configured content axes"
            )
        normalized[candidate] = {}
        for axis in axes:
            definition = raw_definitions[axis]
            if not isinstance(definition, str) or not definition.strip():
                raise Top1DataError(
                    f"content_axis_definitions[{candidate!r}][{axis!r}] "
                    "must be a non-empty string"
                )
            normalized[candidate][axis] = definition.strip()
    return normalized


def validate_content_axis_priority(
    content_axes: Mapping[str, Sequence[str]] | None,
    content_axis_priority: Mapping[str, Sequence[str]] | None,
    candidate_names: Sequence[str],
) -> dict[str, tuple[str, ...]] | None:
    """Validate that each optional priority is an exact axis permutation."""

    if content_axis_priority is None:
        return None
    if not isinstance(content_axes, Mapping) or not content_axes:
        raise Top1DataError(
            "content_axis_priority requires a non-empty content_axes mapping"
        )
    if not isinstance(content_axis_priority, Mapping):
        raise Top1DataError("content_axis_priority must be an object")
    expected_candidates = set(candidate_names)
    if set(content_axes) != expected_candidates:
        raise Top1DataError("content_axes must cover exactly the candidate registry")
    if set(content_axis_priority) != expected_candidates:
        raise Top1DataError(
            "content_axis_priority must cover exactly the candidate registry"
        )

    normalized: dict[str, tuple[str, ...]] = {}
    for candidate in candidate_names:
        raw_axes = content_axes[candidate]
        raw_priority = content_axis_priority[candidate]
        if (
            not isinstance(raw_axes, Sequence)
            or isinstance(raw_axes, (str, bytes))
            or not raw_axes
            or any(not isinstance(axis, str) or not axis.strip() for axis in raw_axes)
        ):
            raise Top1DataError(
                f"content_axes[{candidate!r}] must contain non-empty strings"
            )
        if (
            not isinstance(raw_priority, Sequence)
            or isinstance(raw_priority, (str, bytes))
            or not raw_priority
            or any(
                not isinstance(axis, str) or not axis.strip()
                for axis in raw_priority
            )
        ):
            raise Top1DataError(
                f"content_axis_priority[{candidate!r}] must contain axis strings"
            )
        axes = tuple(axis.strip() for axis in raw_axes)
        priority = tuple(axis.strip() for axis in raw_priority)
        if len(priority) != len(axes) or len(set(priority)) != len(priority):
            raise Top1DataError(
                f"content_axis_priority[{candidate!r}] must be a full permutation"
            )
        if set(priority) != set(axes):
            raise Top1DataError(
                f"content_axis_priority[{candidate!r}] must be a full permutation"
            )
        normalized[candidate] = priority
    return normalized


def generation_messages(
    blueprints: Sequence[DialogueBlueprint],
    taxonomy: str,
    *,
    boundary_guidance: str | None = None,
    content_axis_definitions: Mapping[str, Mapping[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Build one structured batch request for the language realizer."""

    plans = []
    for blueprint in blueprints:
        payload = blueprint.to_dict()
        payload["phenomenon_instruction"] = PHENOMENON_INSTRUCTIONS[blueprint.phenomenon]
        if content_axis_definitions is not None:
            target_definitions = content_axis_definitions.get(
                blueprint.target_candidate_name
            )
            if not isinstance(target_definitions, Mapping):
                raise Top1DataError(
                    "content_axis_definitions does not cover a planned target candidate"
                )
            if blueprint.content_axis not in target_definitions:
                raise Top1DataError(
                    "content_axis_definitions does not cover a planned target axis"
                )
            payload["content_axis_definition"] = str(
                target_definitions[blueprint.content_axis]
            )
            if blueprint.source_candidate_name is not None:
                source_definitions = content_axis_definitions.get(
                    blueprint.source_candidate_name
                )
                if not isinstance(source_definitions, Mapping):
                    raise Top1DataError(
                        "content_axis_definitions does not cover a planned source candidate"
                    )
                if blueprint.source_content_axis not in source_definitions:
                    raise Top1DataError(
                        "content_axis_definitions does not cover a planned source axis"
                    )
                payload["source_content_axis_definition"] = str(
                    source_definitions[blueprint.source_content_axis]
                )
        plans.append(payload)
    axis_guidance = ""
    if any(blueprint.content_axis is not None for blueprint in blueprints):
        axis_guidance = (
            "\n- content_axis 是当前目标必须采用的内容子场景；intent_change 的历史部分还必须采用 "
            "source_content_axis。若计划提供对应 definition，必须按定义实现，不能只根据轴名猜测。"
            "轴名和定义只是语义规划，不得原样写进对话。"
        )
    if boundary_guidance is None:
        boundary_guidance = """- ProductEcommerce 生成京东、淘宝等平台通常可推荐或购买的普通商品在购买前的搜索、品牌或型号选择、比较、价格优惠、性能评价、适用性判断、推荐或购买需求；即使已经给出具体型号或没有点名平台也属于它。药品、整车、房屋、服务和软件不属于它。
- ProductGeneral 不得生成普通电商商品的购买前比较、价格、优惠、性能评价或适用性判断；它只处理非普通电商消费对象，以及已有商品的使用、故障、售后和订单事务。
- StockQuery 只查询已经存在的股票或证券公开行情事实；未来预测、标的推荐、买卖和仓位决策属于 StockAdvice。"""
    system = f"""你是中文多轮对话数据生成器。你只负责把给定的结构化计划实现成自然对话，不得改变计划中的类别和轮次。

候选定义：
{taxonomy}

统一要求：
- 每个样本必须严格输出 2 * user_turn_count - 1 条 messages：索引从 1 起，奇数条是 user、偶数条是 assistant，第一条和最后一条都是 user。
- user 消息数必须等于计划中的 user_turn_count；不要加入 system 或 tool 消息，也绝对不要生成最后一条 user 之后的 assistant 消息或回答最终 user。
- 当前分类目标永远是最后一条 user 消息。历史要真实、有帮助，但不得泄漏英文候选名。
- intent_change 的最后一轮应当无需过渡就直接跨到新需求；新旧主题可以完全无关，不要添加致谢、承接、取消旧任务或宣布换题的文字。
- 对话使用简洁自然的中文，主题、实体、表达方式和 assistant 回复应有变化，避免批量模板腔。{axis_guidance}
{boundary_guidance}
- 输出必须是一个 JSON 对象，不要输出 Markdown 或解释。"""
    user = {
        "plans": plans,
        "output_schema": {
            "samples": [
                {
                    "scenario_id": "与输入完全一致",
                    "messages": [
                        {"role": "user|assistant", "content": "自然中文消息"}
                    ],
                    "scenario_summary": "不含候选英文名的一句场景摘要",
                }
            ]
        },
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def judgment_messages(
    samples: Sequence[Mapping[str, Any]],
    taxonomy: str,
    *,
    candidate_count: int = 7,
    allow_single_turn: bool = False,
    quality_fields: Sequence[str] = QUALITY_FIELDS,
) -> list[dict[str, str]]:
    """Build a blind classification and quality-review request."""

    phenomenon_rule_items = [
            "- intent_change：较早用户目标属于另一个候选，最后一轮不加过渡地直接提出不同候选的新目标；新旧主题可以完全无关。",
            "- progressive_reveal：同一目标逐轮补充信息；最后一轮仍显式包含核心对象与动作，单独阅读可判明当前目标。若末轮主要靠省略、指代、确认或短追问恢复对象，应判 contextual_follow_up。",
            "- contextual_follow_up：最后一轮有省略、指代、确认或短追问，必须结合历史理解。",
            "- clarification_revision：最后一轮修正先前条件或对象属性，但候选类别不变。",
            "- assistant_distractor：assistant 曾误解或引入无关方向，最后用户澄清原目标。",
            "- rambling：存在明显生活化冗余背景，但没有形成第二个待执行目标。",
            "- other：不符合以上任一种。",
    ]
    if allow_single_turn:
        phenomenon_rule_items.insert(
            1,
            "- single_turn：对话只有一条完整 user 请求，没有 assistant 或历史消息。",
        )
    phenomenon_rules = "\n".join(phenomenon_rule_items)
    allowed_phenomenon_count = 8 if allow_single_turn else 7
    candidate_rule = (
        "只允许七个英文候选名。"
        if candidate_count == 7 and not allow_single_turn
        else f"只允许候选定义中列出的 {candidate_count} 个英文候选名。"
    )
    privacy_rule = ""
    if "no_personal_data" in quality_fields:
        privacy_rule = (
            "no_personal_data 表示对话不含身份证号、手机号、精确住址、真实姓名、"
            "订单号、票号、病历号或其它可定位个人的标识；即使看似虚构也应判为 false。"
        )
    privacy_clause = f"{privacy_rule} " if privacy_rule else ""
    strict_quality_clause = ""
    if any(field in quality_fields for field in STRICT_DIALOGUE_QUALITY_FIELDS):
        strict_quality_clause = """

启用的严格对话质量字段定义：
- current_target_identifiable 在这里判断“应选哪个候选”能否确定，而不是要求具体任务对象一定可恢复；若 predicted_candidate_name=NoAvailable 且输入固有无意义或信息不足，那么只要无法支持任何其它候选这一边界清楚，本字段仍可填 true。
- single_turn_standalone：仅对 single_turn 判断。若 predicted_candidate_name=NoAvailable 且输入本身就是无意义或信息不足、无法恢复任何具体任务，可以因为任务本就不可恢复而不完整；除此以外，唯一一条 user 消息必须自身完整表达当前意图。若 predicted_candidate_name=ChitChat，问候、告别、情绪分享或兴趣闲聊本身就是完整意图，不强求同时出现对象和动作。任何 single_turn 都不能用“那个”“这趟”“刚才”“继续”等缺少先行词的表达假装承接不存在的历史。对其它现象固定填 true。
- assistant_no_fabricated_personal_experience：所有 assistant 消息都不得声称自己真实购买、使用、旅行、患病、品尝或亲历过某事；能力说明和明确的假设性表达不算个人经历。
- assistant_no_unsupported_factual_claims：必须逐条检查每条 assistant 消息中的每个外部可核验事实。不得凭空给出当前价格优惠、餐厅设施菜单、商品型号属性、设备错误码含义、历史或实时行情、票务班次余票和时长、政策条款与状态、旅行设施运营状态、检查结论或外部操作结果等需要来源或用户证据的具体事实；断言仅仅“看起来合理”、常见或可信不足以通过。只有稳定常识、明确引用用户已提供的信息、追问或清楚标明不确定性且不冒充事实的概括可以通过。
- entity_and_facts_consistent：人物、地点、日期、商品、证券、数量、预算、症状和其它事实必须在各轮保持一致；不得无说明地替换、混淆或增加与上下文矛盾的实体或事实。
- conversation_context_consistent：首条 user 消息不得假装存在更早但未提供的对话；同一连续会话的时间关系必须成立，例如刚完成多轮交互后不能又说“好久不见”。自然引用首条消息之后的可见历史不构成问题。若 predicted_candidate_name=NoAvailable 且单轮输入固有无意义或信息不足，只要没有缺失先行词或伪造历史，仍应填 true，不能仅因任务不完整而判 false。
以上字段必须逐项独立判断；任一处违反就将对应字段填 false。"""
    candidate_schema = (
        "七个候选之一"
        if candidate_count == 7 and not allow_single_turn
        else f"{candidate_count} 个候选之一"
    )
    phenomenon_schema = (
        "七种现象之一"
        if not allow_single_turn
        else f"{allowed_phenomenon_count} 种现象之一"
    )
    system = f"""你是独立的多轮 Top1 数据审计员。你看不到生成计划，必须仅根据对话盲判最后一条 user 消息。

候选定义：
{taxonomy}

判别规则：当前消息可独立理解时以当前消息为准；存在省略、指代、确认、修订时只使用必要历史；当前目标变化时选择新目标。{candidate_rule}

现象定义：
{phenomenon_rules}

质量字段必须是布尔值。{privacy_clause}intent_change_is_direct 仅在观察到 intent_change 时表示最后一轮是否直接表达新需求、没有回应、致谢、取消、承接或宣布换题等元话语；其它现象填写 true。IntentChange 的最后一轮无需与前文有主题关联，突然跨到无关领域正是预期，不得仅因换题突然或没有过渡而把 intent_change_is_direct 或任何自然度字段判为 false。若观察到 assistant_distractor，一次语言和角色仍连贯的有意误解、偏题或错误侧重点不得仅因此导致自然度、连贯性或其它质量字段为 false；其中的事实断言与其它质量问题仍须独立严格审查。若当前意图是 ChitChat，问候、告别、情绪分享或兴趣闲聊本身可以构成完整意图，不要求同时具备对象和动作。observed_source_candidate_name 仅在 intent_change 时填写较早用户目标，否则为 null。{strict_quality_clause}
issues 只记录对话自身客观存在的样本质量问题，例如表达不自然、事实矛盾、无依据事实、角色错位、候选名泄漏或缺失先行词；任务不被任何专用候选支持且因此正确预测为 NoAvailable，本身不是质量问题，不得写入 issues。没有客观质量问题时必须输出空数组。
只输出 JSON 对象，不要输出 Markdown。"""
    user = {
        "samples": [
            {
                "scenario_id": str(
                    sample.get("audit_id", sample["scenario_id"])
                ),
                "messages": sample["messages"],
            }
            for sample in samples
        ],
        "output_schema": {
            "judgments": [
                {
                    "scenario_id": "与输入完全一致",
                    "predicted_candidate_name": candidate_schema,
                    "observed_phenomenon": phenomenon_schema,
                    "observed_source_candidate_name": "候选名或null",
                    "intent_change_is_direct": True,
                    "quality": {field: True for field in quality_fields},
                    "issues": ["问题；没有则空数组"],
                }
            ]
        },
        "output_requirement": "每个输入 scenario_id 恰好输出一次，不得遗漏、重复或增加其它 ID。",
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def directness_messages(
    samples: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Build a strict, label-blind audit for direct IntentChange final turns."""

    system = """你是 IntentChange 最后一轮表达纯净度的极严格审计员。只判断表达方式，不判断候选标签。

唯一合格条件：最后一条 user 消息从新意图本身开始，整条消息只陈述新意图及其必要背景、条件或问题。问候、告别、情绪分享或兴趣闲聊本身可以构成完整新意图，不强求同时出现对象和动作。普通请求礼貌词（如“请”“麻烦”）可以属于新意图。

IntentChange 的最后一轮无需与前文有任何主题关联；直接突然跨到完全无关的领域正是预期。不得仅因换题突然、没有过渡或与历史主题无关而将任何字段判为不合格。

以下任一情况都必须判为不合格，即使后面的新需求非常清楚：
1. 回应、确认、致谢或评价上一轮回答；
2. 提及结束、取消、放弃、搁置或改变原任务；
3. 使用承接或切换话题的元话语引出新需求；
4. 把新需求说成“顺便”附带提出的事情，而不是直接提出。

不合格示例：“好的，我回头试试。对了，我想……”“谢谢。帮我……”“明白了，那查一下……”“顺便帮我……”“换个话题……”“那个不用了，帮我……”。合格示例：“我想买台轻薄本，预算七千，主要用于写代码。”“查一下宁德时代今天的收盘价和涨跌幅。”

仍需严格拒绝回应、感谢、取消、切换元话语、把新需求说成“顺便”以及依赖历史才能理解的新意图。审计完整对话只为定位最后一轮。不要输出或推断候选名。每个输入 scenario_id 恰好输出一次，只输出 JSON 对象。"""
    user = {
        "samples": [
            {
                "scenario_id": str(
                    sample.get("audit_id", sample["scenario_id"])
                ),
                "messages": sample["messages"],
            }
            for sample in samples
        ],
        "output_schema": {
            "audits": [
                {
                    "scenario_id": "与输入完全一致",
                    "contains_only_new_request": True,
                    "references_previous_exchange": False,
                    "uses_transition_or_acknowledgment": False,
                    "direct_final_request": True,
                    "has_switch_meta_language": False,
                    "reason": "一句简短理由",
                }
            ]
        },
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def dialogue_quality_messages(
    samples: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Build a label- and plan-blind strict dialogue-quality audit request."""

    system = """你是严格的对话质量审计员。你看不到候选标签、生成计划、内容轴或目标现象，只能依据完整对话判断质量；不要推断或输出任何标签。

逐项独立判断，任一处违反就填 false：
1. opening_context_valid：多轮对话的首条 user 必须能作为本段对话的自然开场，不能致谢、总结或声称“已经跟你说过”等方式假装存在未提供的更早对话。只有一条 user 时，若该消息本身就是无意义或信息不足、因而无法支持任何具体任务，只要没有“那个”“这趟”“刚才”“继续”等缺少先行词、假装承接不存在历史的表达，opening_context_valid 就应为 true，不能仅因任务固有不完整或无意义而判 false。问候、告别、情绪分享或兴趣闲聊本身也是完整自然的当前意图，不强求同时出现对象和动作。
2. temporal_context_consistent：同一连续会话内的时间关系必须成立；刚刚连续交互后又说“好久不见”等自相矛盾表达必须为 false。
3. assistant_no_fabricated_personal_experience：assistant 不得声称自己真实购买、使用、旅行、育儿、养宠、患病、品尝、感受或亲历某事。
4. assistant_no_unsupported_factual_claims：逐条检查每条 assistant 消息中的每一个外部可核验断言。assistant 不得凭空补写当前价格优惠、餐厅设施菜单、具体商品型号及属性、设备错误码含义、历史或实时行情、票务班次余票和时长、政策条款与状态、旅行设施运营状态、检查结论、交易收购或外部操作结果。断言仅仅看起来合理、常见或可信不构成依据；只有稳定常识、明确复述用户已给信息、追问、或清楚标明不确定性且不冒充事实的概括可以通过。
5. entity_and_facts_consistent：人物、地点、日期、商品、证券、市场、数量、预算、症状、时间顺序和其它事实在各轮必须一致；不得无说明替换实体、补造事实或陈述互相矛盾的能力与属性。
6. natural_dialogue：整段对话必须像真实自然交流；不得为制造分类边界硬塞无关背景，不得有生硬模板、重复结束、角色错位或不合常理的双任务拼接。但最后一条 user 不加过渡地突然切换到与前文完全无关的新需求，是合格的直接 IntentChange，不得仅因突然换题、没有主题关联或没有过渡而判 false。assistant 出现一次语言和角色仍连贯的有意误解、偏题或错误侧重点，是合格的 assistant_distractor 结构，也不得仅因这一次偏题而判 false；仍须独立审查其中的事实、角色和其它质量问题。

reason 必须具体指出通过依据或第一个失败点，不能只复述字段名。每个输入 scenario_id 恰好输出一次，只输出 JSON 对象。"""
    user = {
        "samples": [
            {
                "scenario_id": str(
                    sample.get("audit_id", sample["scenario_id"])
                ),
                "messages": sample["messages"],
            }
            for sample in samples
        ],
        "output_schema": {
            "audits": [
                {
                    "scenario_id": "与输入完全一致",
                    **{field: True for field in DIALOGUE_QUALITY_AUDIT_FIELDS},
                    "reason": "具体说明通过依据或第一个失败点",
                }
            ]
        },
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def contrast_messages(
    samples: Sequence[Mapping[str, Any]],
    taxonomy: str,
    *,
    require_natural_link: bool = False,
) -> list[dict[str, str]]:
    """Build a plan-aware audit for single-turn boundary examples."""

    natural_link_rule = ""
    if require_natural_link:
        natural_link_rule = """
6. contrast_link_natural：target 与 contrast 的混淆必须来自真实自然的语义重叠、对象边界或相邻交付物；不得为了靠近 contrast 而硬塞体检、股票、政务等与当前请求无关的背景或限定。"""
    insufficient_axis_rule = ""
    if require_natural_link:
        insufficient_axis_rule = """
特殊例外：若 target_candidate=NoAvailable，且计划提供的 content_axis_definition 明确要求输入本身固有无意义、不可理解或信息不足，则缺少可恢复任务本身就是 target 语义，target_semantics_present 不要求凭空存在动作和对象；但用缺失先行词的指代伪造历史仍必须拒绝。"""
    system = f"""你是单轮意图边界数据审计员。生成计划已经给出 target_candidate 和 contrast_candidate；你不负责重新标注，而要严格判断样本是否真的是有训练价值的边界例。

候选定义：
{taxonomy}

合格样本必须同时满足：
1. target_semantics_present：请求的最终动作、对象和交付物明确符合 target_candidate；
2. contrast_is_plausible：表面措辞、主题或任务形式确实容易让弱分类器误判为 contrast_candidate，而不是毫无关系的简单样本；
3. target_preferred_over_contrast：按候选边界，target_candidate 明确优于 contrast_candidate；
4. single_current_goal：只包含一个当前待执行目标，没有把两个类别的任务硬拼在一起；
5. natural_expression：像真实用户的一条自然请求，不为制造边界而解释标签规则。{natural_link_rule}

{insufficient_axis_rule}
任何一项不满足都必须判 false。每个输入 scenario_id 恰好输出一次，只输出 JSON 对象。"""
    audit_schema = {
        "scenario_id": "与输入完全一致",
        "target_semantics_present": True,
        "contrast_is_plausible": True,
        "target_preferred_over_contrast": True,
        "single_current_goal": True,
        "natural_expression": True,
        "reason": "一句简短理由",
    }
    if require_natural_link:
        audit_schema["contrast_link_natural"] = True
    user = {
        "samples": [
            {
                "scenario_id": str(sample["scenario_id"]),
                "messages": sample["messages"],
                "target_candidate_name": str(sample["target_candidate_name"]),
                "contrast_candidate_name": str(sample["contrast_candidate_name"]),
                **(
                    {
                        "content_axis": str(sample["content_axis"]),
                        "content_axis_definition": str(
                            sample["content_axis_definition"]
                        ),
                    }
                    if "content_axis" in sample
                    and "content_axis_definition" in sample
                    else {}
                ),
            }
            for sample in samples
        ],
        "output_schema": {
            "audits": [audit_schema]
        },
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def plan_fidelity_messages(
    samples: Sequence[Mapping[str, Any]],
    taxonomy: str,
    *,
    content_axis_definitions: Mapping[str, Mapping[str, str]] | None = None,
    content_axis_priority: Mapping[str, Sequence[str]] | None = None,
    require_observed_axis_match: bool = False,
) -> list[dict[str, str]]:
    """Build a plan-aware semantic-axis and final-action audit."""

    if require_observed_axis_match:
        if content_axis_definitions is None:
            raise Top1DataError(
                "observed-axis auditing requires content_axis_definitions"
            )
        if content_axis_priority is None:
            raise Top1DataError(
                "observed-axis auditing requires content_axis_priority"
            )
        observed_samples: list[dict[str, Any]] = []
        for sample in samples:
            target = str(sample["target_candidate_name"])
            source = sample.get("source_candidate_name")
            target_catalog = content_axis_definitions.get(target)
            target_priority = content_axis_priority.get(target)
            if not isinstance(target_catalog, Mapping) or not target_catalog:
                raise Top1DataError(
                    "observed-axis auditing has no target content-axis catalog"
                )
            if (
                not isinstance(target_priority, Sequence)
                or isinstance(target_priority, (str, bytes))
                or tuple(target_priority) == ()
                or any(
                    not isinstance(axis, str) or not axis
                    for axis in target_priority
                )
                or set(target_priority) != set(target_catalog)
                or len(tuple(target_priority)) != len(target_catalog)
            ):
                raise Top1DataError(
                    "observed-axis auditing has no valid target axis priority"
                )
            source_catalog: Mapping[str, str] | None = None
            source_priority: Sequence[str] | None = None
            if source is not None:
                source_catalog = content_axis_definitions.get(str(source))
                source_priority = content_axis_priority.get(str(source))
                if not isinstance(source_catalog, Mapping) or not source_catalog:
                    raise Top1DataError(
                        "observed-axis auditing has no source content-axis catalog"
                    )
                if (
                    not isinstance(source_priority, Sequence)
                    or isinstance(source_priority, (str, bytes))
                    or any(
                        not isinstance(axis, str) or not axis
                        for axis in source_priority
                    )
                    or set(source_priority) != set(source_catalog)
                    or len(tuple(source_priority)) != len(source_catalog)
                ):
                    raise Top1DataError(
                        "observed-axis auditing has no valid source axis priority"
                    )
            observed_samples.append(
                {
                    "scenario_id": str(
                        sample.get("audit_id", sample["scenario_id"])
                    ),
                    "messages": sample["messages"],
                    "target_candidate_name": target,
                    "source_candidate_name": source,
                    "planned_phenomenon": str(sample["planned_phenomenon"]),
                    "target_content_axis_catalog": {
                        str(axis): str(target_catalog[str(axis)])
                        for axis in target_priority
                    },
                    "target_content_axis_priority": list(target_priority),
                    "source_content_axis_catalog": (
                        {
                            str(axis): str(source_catalog[str(axis)])
                            for axis in source_priority or ()
                        }
                        if source_catalog is not None
                        else None
                    ),
                    "source_content_axis_priority": (
                        list(source_priority) if source_priority is not None else None
                    ),
                }
            )
        system = f"""你是对话生成计划的语义保真审计员。计划不会告诉你预期内容轴；你必须只根据对话，从给定候选的完整内容轴目录中独立观察最匹配的轴，不能迎合或猜测蓝图答案。

候选定义：
{taxonomy}

逐项严格判断：
- observed_target_content_axis：只依据最后一条 user 消息自身表达的当前请求，从 target_content_axis_catalog 选择唯一最匹配的轴键。最后一条消息自身必须落实该轴的动作、对象、属性或交付物；不能借开场、较早历史或 assistant 消息补充轴所需的对象、动作、属性或交付物。即使历史能消解一个指代，也只能确认实体一致性，不能把历史里的轴语义算作末轮证据。single_turn 只能看唯一一条 user；
- target_axis_unambiguous：先找出末轮真实落实的轴；若有多个，则严格按 target_content_axis_priority 选择排列最前的唯一主轴。证据足以确定真实轴集合并能按该规则得到唯一主轴时填 true；没有轴被真实落实或证据本身无法判断时填 false；
- observed_source_content_axis：仅对 intent_change，从较早 user 目标独立选择 source_content_axis_catalog 中唯一最匹配的轴键；非 intent_change 必须填 null；
- source_axis_unambiguous：intent_change 同样先找出历史真实落实的轴，再按 source_content_axis_priority 选择排列最前的唯一主轴；证据不足时填 false。非 intent_change 固定填 true；
- target_axis_realized：末轮自身确实清楚落实 observed_target_content_axis 才填 true；
- target_candidate_respected：按最终动作、对象和交付物，而不只是主题，当前请求明确属于 target_candidate；
- source_axis_realized：intent_change 的较早 user 目标确实落实 observed_source_content_axis；其它现象填 true；
- source_candidate_respected：intent_change 的较早 user 目标明确属于 source_candidate；其它现象填 true；
- phenomenon_realized：实际轮次与对话关系符合 planned_phenomenon。planned_phenomenon=progressive_reveal 时，若末轮仅靠指代、省略、确认或短追问恢复核心对象或动作，填 false；这种结构属于 contextual_follow_up；
- no_extra_current_goal：最后一轮没有硬拼另一个当前待执行目标。

若某个轴的定义本身明确表示输入固有无意义、不可理解或信息不足，那么末轮自身呈现的这种缺失可以作为落实该轴的正面证据；但缺少先行词的“那个”“这趟”“刚才”“继续”等伪承接不属于这种例外。目录中的轴名和定义只是可选集合，不是答案证据；priority 仅用于多个已落实轴之间选主轴，不能让未落实的高优先级轴胜出。reason 必须至少 20 个字符，具体说明末轮动作/对象（或上述固有不足）为何支持所观察轴，并在 intent_change 时同时说明历史轴。每个 scenario_id 恰好输出一次，只输出 JSON 对象。"""
        user = {
            "samples": observed_samples,
            "output_schema": {
                "audits": [
                    {
                        "scenario_id": "与输入完全一致",
                        "observed_target_content_axis": "target目录中的唯一轴键",
                        "observed_source_content_axis": "source目录中的唯一轴键或null",
                        "target_axis_unambiguous": True,
                        "source_axis_unambiguous": True,
                        "target_axis_realized": True,
                        "target_candidate_respected": True,
                        "source_axis_realized": True,
                        "source_candidate_respected": True,
                        "phenomenon_realized": True,
                        "no_extra_current_goal": True,
                        "reason": "至少20个字符的具体证据说明",
                    }
                ]
            },
        }
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ]

    system = f"""你是对话生成计划的语义保真审计员。计划标签不是让你迎合的答案；只要实际对话没有落实相应语义，就必须拒绝。

候选定义：
{taxonomy}

逐项严格判断：
- target_axis_realized：只能依据最后一条 user 消息自身表达的当前请求判断。该消息本身必须包含能够落实 content_axis 的动作、对象、属性或交付物；不能因为开场、较早历史或 assistant 消息曾出现该子场景就判 true。即使历史能消解末轮指代，也只能确认实体一致性，不能为末轮补充轴所需的对象、动作、属性或交付物。single_turn 只能查看唯一一条 user 消息；
- target_candidate_respected：按最终动作、对象和交付物，而不只是讨论主题，当前请求明确属于 target_candidate；
- source_axis_realized：仅对 intent_change，较早用户目标确实采用 source_content_axis；其它现象填 true；
- source_candidate_respected：仅对 intent_change，较早用户目标明确属于 source_candidate；其它现象填 true；
- phenomenon_realized：实际轮次与对话关系符合 planned_phenomenon。planned_phenomenon=progressive_reveal 时，若末轮仅靠指代、省略、确认或短追问恢复核心对象或动作，填 false；这种结构属于 contextual_follow_up；
- no_extra_current_goal：最后一轮没有硬拼另一个当前待执行目标。

轴名只是抽象主题，允许自然实现，不要求原词出现，但计划中的轴名、候选名和历史内容本身都不能作为 target_axis_realized 的证据。每个 scenario_id 恰好输出一次，只输出 JSON 对象。"""
    user = {
        "samples": [
            {
                "scenario_id": str(sample["scenario_id"]),
                "messages": sample["messages"],
                "target_candidate_name": str(sample["target_candidate_name"]),
                "source_candidate_name": sample.get("source_candidate_name"),
                "planned_phenomenon": str(sample["planned_phenomenon"]),
                "content_axis": str(sample["content_axis"]),
                "source_content_axis": sample.get("source_content_axis"),
            }
            for sample in samples
        ],
        "output_schema": {
            "audits": [
                {
                    "scenario_id": "与输入完全一致",
                    "target_axis_realized": True,
                    "target_candidate_respected": True,
                    "source_axis_realized": True,
                    "source_candidate_respected": True,
                    "phenomenon_realized": True,
                    "no_extra_current_goal": True,
                    "reason": "一句简短理由",
                }
            ]
        },
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def parse_json_object(
    content: str,
    *,
    strict: bool = False,
) -> dict[str, Any]:
    """Parse a JSON object, with legacy wrapper recovery when not strict."""

    text = content.strip()
    if not strict and text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1])
            if text.lstrip().startswith("json"):
                text = text.lstrip()[4:].lstrip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as initial_error:
        if strict:
            raise Top1DataError(
                f"invalid model JSON: {initial_error.msg}"
            ) from initial_error
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise Top1DataError("model response contains no JSON object")
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise Top1DataError(f"invalid model JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise Top1DataError("model response must be a JSON object")
    return payload


def _strict_response_shape_errors(
    payload: Mapping[str, Any],
    *,
    response_field: str,
    item_fields: Iterable[str],
    nested_fields: Mapping[str, Iterable[str]] | None = None,
) -> list[str]:
    """Return exact-object errors for a strict structured response."""

    issues: list[str] = []
    expected_root = {response_field}
    if set(payload) != expected_root:
        issues.append(
            f"response root must contain exactly {response_field}"
        )
    raw_items = payload.get(response_field)
    if not isinstance(raw_items, list):
        return issues
    expected_item = set(item_fields)
    nested = {
        key: set(values) for key, values in (nested_fields or {}).items()
    }
    for index, item in enumerate(raw_items):
        if not isinstance(item, Mapping):
            continue
        if set(item) != expected_item:
            issues.append(
                f"response.{response_field}[{index}] has unexpected or missing fields"
            )
        for field, expected_nested in nested.items():
            value = item.get(field)
            if isinstance(value, Mapping) and set(value) != expected_nested:
                issues.append(
                    f"response.{response_field}[{index}].{field} "
                    "has unexpected or missing fields"
                )
    return issues


def _strict_response_envelope_errors(
    items: Sequence[Any],
    expected_ids: Sequence[str],
    *,
    response_field: str,
) -> list[str]:
    """Return batch-level errors when a response list is not an exact ID envelope."""

    allowed_ids = set(expected_ids)
    counts: Counter[str] = Counter()
    issues: list[str] = []
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            issues.append(f"response.{response_field}[{index}] must be an object")
            continue
        scenario_id = item.get("scenario_id")
        if not isinstance(scenario_id, str):
            issues.append(
                f"response.{response_field}[{index}].scenario_id must be a string"
            )
            continue
        if scenario_id not in allowed_ids:
            issues.append(
                f"response.{response_field}[{index}] has an unexpected scenario_id"
            )
            continue
        counts[scenario_id] += 1
    for scenario_id in expected_ids:
        if counts[scenario_id] == 0:
            issues.append(f"expected scenario_id missing from response.{response_field}")
        elif counts[scenario_id] > 1:
            issues.append(f"duplicate scenario_id in response.{response_field}")
    return issues


def parse_generated_samples(
    content: str,
    assigned: Mapping[str, DialogueBlueprint],
    candidate_names: Sequence[str],
    *,
    strict_envelope: bool = False,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    """Parse and structurally validate one generation batch."""

    errors = {scenario_id: [] for scenario_id in assigned}
    try:
        payload = parse_json_object(content, strict=strict_envelope)
    except Top1DataError as exc:
        return {}, {scenario_id: [str(exc)] for scenario_id in assigned}
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list):
        return {}, {scenario_id: ["response.samples must be a list"] for scenario_id in assigned}
    if strict_envelope:
        shape_errors = _strict_response_shape_errors(
            payload,
            response_field="samples",
            item_fields=("scenario_id", "messages", "scenario_summary"),
        )
        for sample_index, raw_sample in enumerate(raw_samples):
            if not isinstance(raw_sample, Mapping):
                continue
            messages = raw_sample.get("messages")
            if not isinstance(messages, list):
                continue
            for message_index, message in enumerate(messages):
                if isinstance(message, Mapping) and set(message) != {"role", "content"}:
                    shape_errors.append(
                        f"response.samples[{sample_index}].messages[{message_index}] "
                        "has unexpected or missing fields"
                    )
        if shape_errors:
            return {}, {
                scenario_id: list(shape_errors) for scenario_id in assigned
            }
        envelope_errors = _strict_response_envelope_errors(
            raw_samples,
            tuple(assigned),
            response_field="samples",
        )
        if envelope_errors:
            return {}, {
                scenario_id: list(envelope_errors) for scenario_id in assigned
            }

    parsed: dict[str, dict[str, Any]] = {}
    for raw_sample in raw_samples:
        if not isinstance(raw_sample, Mapping):
            continue
        scenario_id = raw_sample.get("scenario_id")
        if not isinstance(scenario_id, str) or scenario_id not in assigned:
            continue
        if scenario_id in parsed:
            errors[scenario_id].append("duplicate scenario in generation response")
            continue
        messages = raw_sample.get("messages")
        sample_errors = validate_generated_messages(
            messages,
            assigned[scenario_id],
            candidate_names,
        )
        if sample_errors:
            errors[scenario_id].extend(sample_errors)
            continue
        parsed[scenario_id] = {
            "scenario_id": scenario_id,
            "messages": [dict(message) for message in messages],
            "scenario_summary": str(raw_sample.get("scenario_summary", "")).strip(),
        }
    for scenario_id in assigned:
        if scenario_id not in parsed and not errors[scenario_id]:
            errors[scenario_id].append("scenario missing from generation response")
    return parsed, errors


def validate_generated_messages(
    messages: Any,
    blueprint: DialogueBlueprint,
    candidate_names: Sequence[str],
) -> list[str]:
    """Apply label-independent structural quality gates."""

    issues: list[str] = []
    if not isinstance(messages, list):
        return ["messages must be a list"]
    expected_message_count = blueprint.user_turn_count * 2 - 1
    if len(messages) != expected_message_count:
        issues.append(
            f"expected {expected_message_count} alternating messages, got {len(messages)}"
        )
    total_characters = 0
    previous_content: str | None = None
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            issues.append(f"messages[{index}] must be an object")
            continue
        expected_role = "user" if index % 2 == 0 else "assistant"
        role = message.get("role")
        content = message.get("content")
        if role != expected_role:
            issues.append(f"messages[{index}] role must be {expected_role}")
        if not isinstance(content, str) or not content.strip():
            issues.append(f"messages[{index}] content must be non-empty")
            continue
        clean_content = content.strip()
        total_characters += len(clean_content)
        if len(clean_content) > 320:
            issues.append(f"messages[{index}] exceeds 320 characters")
        if previous_content == clean_content:
            issues.append("adjacent messages cannot be identical")
        previous_content = clean_content
        for candidate in candidate_names:
            if candidate in clean_content:
                issues.append(f"messages[{index}] leaks a candidate name")
                break
    if total_characters > 2_000:
        issues.append("conversation exceeds 2,000 characters")
    return issues


def parse_judgments(
    content: str,
    assigned_ids: Iterable[str],
    candidate_names: Sequence[str],
    *,
    quality_fields: Sequence[str] = QUALITY_FIELDS,
    require_issues_field: bool = False,
    strict_envelope: bool = False,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    """Parse one blind-judge response with strict structured fields."""

    ids = tuple(assigned_ids)
    errors = {scenario_id: [] for scenario_id in ids}
    try:
        payload = parse_json_object(content, strict=strict_envelope)
    except Top1DataError as exc:
        return {}, {scenario_id: [str(exc)] for scenario_id in ids}
    raw_judgments = payload.get("judgments")
    if not isinstance(raw_judgments, list):
        return {}, {scenario_id: ["response.judgments must be a list"] for scenario_id in ids}
    if strict_envelope:
        shape_errors = _strict_response_shape_errors(
            payload,
            response_field="judgments",
            item_fields=(
                "scenario_id",
                "predicted_candidate_name",
                "observed_phenomenon",
                "observed_source_candidate_name",
                "intent_change_is_direct",
                "quality",
                "issues",
            ),
            nested_fields={"quality": quality_fields},
        )
        if shape_errors:
            return {}, {
                scenario_id: list(shape_errors) for scenario_id in ids
            }
        envelope_errors = _strict_response_envelope_errors(
            raw_judgments,
            ids,
            response_field="judgments",
        )
        if envelope_errors:
            return {}, {
                scenario_id: list(envelope_errors) for scenario_id in ids
            }

    allowed_ids = set(ids)
    allowed_candidates = set(candidate_names)
    parsed: dict[str, dict[str, Any]] = {}
    for raw in raw_judgments:
        if not isinstance(raw, Mapping):
            continue
        scenario_id = raw.get("scenario_id")
        if not isinstance(scenario_id, str) or scenario_id not in allowed_ids:
            continue
        if scenario_id in parsed:
            parsed.pop(scenario_id)
            errors[scenario_id].append("duplicate scenario in judgment response")
            continue
        item_errors: list[str] = []
        predicted = raw.get("predicted_candidate_name")
        source = raw.get("observed_source_candidate_name")
        phenomenon = raw.get("observed_phenomenon")
        quality = raw.get("quality")
        if predicted not in allowed_candidates:
            item_errors.append("invalid predicted candidate")
        if source is not None and source not in allowed_candidates:
            item_errors.append("invalid observed source candidate")
        if phenomenon not in OBSERVED_PHENOMENA:
            item_errors.append("invalid observed phenomenon")
        if not isinstance(raw.get("intent_change_is_direct"), bool):
            item_errors.append("intent_change_is_direct must be boolean")
        if not isinstance(quality, Mapping):
            item_errors.append("quality must be an object")
        else:
            for field in quality_fields:
                if not isinstance(quality.get(field), bool):
                    item_errors.append(f"quality.{field} must be boolean")
        if require_issues_field and "issues" not in raw:
            item_errors.append("issues field is required")
        issues = raw.get("issues", [])
        if not isinstance(issues, list) or any(not isinstance(value, str) for value in issues):
            item_errors.append("issues must be a string list")
        if item_errors:
            errors[scenario_id].extend(item_errors)
            continue
        parsed[scenario_id] = {
            "scenario_id": scenario_id,
            "predicted_candidate_name": predicted,
            "observed_phenomenon": phenomenon,
            "observed_source_candidate_name": source,
            "intent_change_is_direct": raw["intent_change_is_direct"],
            "quality": {field: bool(quality[field]) for field in quality_fields},
            "issues": list(issues),
        }
    for scenario_id in ids:
        if scenario_id not in parsed and not errors[scenario_id]:
            errors[scenario_id].append("scenario missing from judgment response")
    return parsed, errors


def parse_directness_audits(
    content: str,
    assigned_ids: Iterable[str],
    *,
    strict_envelope: bool = False,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    """Parse strict IntentChange directness audit output."""

    ids = tuple(assigned_ids)
    errors = {scenario_id: [] for scenario_id in ids}
    try:
        payload = parse_json_object(content, strict=strict_envelope)
    except Top1DataError as exc:
        return {}, {scenario_id: [str(exc)] for scenario_id in ids}
    raw_audits = payload.get("audits")
    if not isinstance(raw_audits, list):
        return {}, {scenario_id: ["response.audits must be a list"] for scenario_id in ids}
    if strict_envelope:
        shape_errors = _strict_response_shape_errors(
            payload,
            response_field="audits",
            item_fields=(
                "scenario_id",
                "contains_only_new_request",
                "references_previous_exchange",
                "uses_transition_or_acknowledgment",
                "direct_final_request",
                "has_switch_meta_language",
                "reason",
            ),
        )
        if shape_errors:
            return {}, {
                scenario_id: list(shape_errors) for scenario_id in ids
            }
        envelope_errors = _strict_response_envelope_errors(
            raw_audits,
            ids,
            response_field="audits",
        )
        if envelope_errors:
            return {}, {
                scenario_id: list(envelope_errors) for scenario_id in ids
            }
    allowed_ids = set(ids)
    parsed: dict[str, dict[str, Any]] = {}
    for raw in raw_audits:
        if not isinstance(raw, Mapping):
            continue
        scenario_id = raw.get("scenario_id")
        if not isinstance(scenario_id, str) or scenario_id not in allowed_ids:
            continue
        if scenario_id in parsed:
            parsed.pop(scenario_id)
            errors[scenario_id].append("duplicate scenario in directness response")
            continue
        only_new = raw.get("contains_only_new_request")
        previous = raw.get("references_previous_exchange")
        transition = raw.get("uses_transition_or_acknowledgment")
        direct = raw.get("direct_final_request")
        meta = raw.get("has_switch_meta_language")
        reason = raw.get("reason")
        item_errors: list[str] = []
        if not isinstance(only_new, bool):
            item_errors.append("contains_only_new_request must be boolean")
        if not isinstance(previous, bool):
            item_errors.append("references_previous_exchange must be boolean")
        if not isinstance(transition, bool):
            item_errors.append("uses_transition_or_acknowledgment must be boolean")
        if not isinstance(direct, bool):
            item_errors.append("direct_final_request must be boolean")
        if not isinstance(meta, bool):
            item_errors.append("has_switch_meta_language must be boolean")
        if not isinstance(reason, str):
            item_errors.append("reason must be a string")
        if item_errors:
            errors[scenario_id].extend(item_errors)
            continue
        parsed[scenario_id] = {
            "scenario_id": scenario_id,
            "contains_only_new_request": only_new,
            "references_previous_exchange": previous,
            "uses_transition_or_acknowledgment": transition,
            "direct_final_request": direct,
            "has_switch_meta_language": meta,
            "reason": reason.strip(),
        }
    for scenario_id in ids:
        if scenario_id not in parsed and not errors[scenario_id]:
            errors[scenario_id].append("scenario missing from directness response")
    return parsed, errors


def parse_dialogue_quality_audits(
    content: str,
    assigned_ids: Iterable[str],
    *,
    strict_envelope: bool = False,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    """Parse a label- and plan-blind dialogue-quality audit response."""

    ids = tuple(assigned_ids)
    errors = {scenario_id: [] for scenario_id in ids}
    try:
        payload = parse_json_object(content, strict=strict_envelope)
    except Top1DataError as exc:
        return {}, {scenario_id: [str(exc)] for scenario_id in ids}
    raw_audits = payload.get("audits")
    if not isinstance(raw_audits, list):
        return {}, {
            scenario_id: ["response.audits must be a list"] for scenario_id in ids
        }
    if strict_envelope:
        shape_errors = _strict_response_shape_errors(
            payload,
            response_field="audits",
            item_fields=(
                "scenario_id",
                *DIALOGUE_QUALITY_AUDIT_FIELDS,
                "reason",
            ),
        )
        if shape_errors:
            return {}, {
                scenario_id: list(shape_errors) for scenario_id in ids
            }
        envelope_errors = _strict_response_envelope_errors(
            raw_audits,
            ids,
            response_field="audits",
        )
        if envelope_errors:
            return {}, {
                scenario_id: list(envelope_errors) for scenario_id in ids
            }
    allowed_ids = set(ids)
    parsed: dict[str, dict[str, Any]] = {}
    for raw in raw_audits:
        if not isinstance(raw, Mapping):
            continue
        scenario_id = raw.get("scenario_id")
        if not isinstance(scenario_id, str) or scenario_id not in allowed_ids:
            continue
        if scenario_id in parsed:
            parsed.pop(scenario_id)
            errors[scenario_id].append(
                "duplicate scenario in dialogue-quality response"
            )
            continue
        item_errors = [
            f"{field} must be boolean"
            for field in DIALOGUE_QUALITY_AUDIT_FIELDS
            if not isinstance(raw.get(field), bool)
        ]
        reason = raw.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            item_errors.append("reason must be a non-empty string")
        if item_errors:
            errors[scenario_id].extend(item_errors)
            continue
        parsed[scenario_id] = {
            "scenario_id": scenario_id,
            **{
                field: bool(raw[field])
                for field in DIALOGUE_QUALITY_AUDIT_FIELDS
            },
            "reason": reason.strip(),
        }
    for scenario_id in ids:
        if scenario_id not in parsed and not errors[scenario_id]:
            errors[scenario_id].append(
                "scenario missing from dialogue-quality response"
            )
    return parsed, errors


def parse_contrast_audits(
    content: str,
    assigned_ids: Iterable[str],
    *,
    require_natural_link: bool = False,
    strict_envelope: bool = False,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    """Parse strict single-turn boundary audit output."""

    ids = tuple(assigned_ids)
    errors = {scenario_id: [] for scenario_id in ids}
    try:
        payload = parse_json_object(content, strict=strict_envelope)
    except Top1DataError as exc:
        return {}, {scenario_id: [str(exc)] for scenario_id in ids}
    raw_audits = payload.get("audits")
    if not isinstance(raw_audits, list):
        return {}, {scenario_id: ["response.audits must be a list"] for scenario_id in ids}
    fields = (
        "target_semantics_present",
        "contrast_is_plausible",
        "target_preferred_over_contrast",
        "single_current_goal",
        "natural_expression",
    )
    if require_natural_link:
        fields += ("contrast_link_natural",)
    if strict_envelope:
        shape_errors = _strict_response_shape_errors(
            payload,
            response_field="audits",
            item_fields=("scenario_id", *fields, "reason"),
        )
        if shape_errors:
            return {}, {
                scenario_id: list(shape_errors) for scenario_id in ids
            }
        envelope_errors = _strict_response_envelope_errors(
            raw_audits,
            ids,
            response_field="audits",
        )
        if envelope_errors:
            return {}, {
                scenario_id: list(envelope_errors) for scenario_id in ids
            }
    allowed_ids = set(ids)
    parsed: dict[str, dict[str, Any]] = {}
    for raw in raw_audits:
        if not isinstance(raw, Mapping):
            continue
        scenario_id = raw.get("scenario_id")
        if not isinstance(scenario_id, str) or scenario_id not in allowed_ids:
            continue
        if scenario_id in parsed:
            parsed.pop(scenario_id)
            errors[scenario_id].append("duplicate scenario in contrast response")
            continue
        item_errors = [
            f"{field} must be boolean"
            for field in fields
            if not isinstance(raw.get(field), bool)
        ]
        reason = raw.get("reason")
        if not isinstance(reason, str):
            item_errors.append("reason must be a string")
        if item_errors:
            errors[scenario_id].extend(item_errors)
            continue
        parsed[scenario_id] = {
            "scenario_id": scenario_id,
            **{field: bool(raw[field]) for field in fields},
            "reason": reason.strip(),
        }
    for scenario_id in ids:
        if scenario_id not in parsed and not errors[scenario_id]:
            errors[scenario_id].append("scenario missing from contrast response")
    return parsed, errors


def parse_plan_fidelity_audits(
    content: str,
    assigned_ids: Iterable[str],
    *,
    observed_axis_catalogs: Mapping[str, Mapping[str, Sequence[str] | None]]
    | None = None,
    strict_envelope: bool = False,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    """Parse strict plan-fidelity audit output."""

    ids = tuple(assigned_ids)
    errors = {scenario_id: [] for scenario_id in ids}
    try:
        payload = parse_json_object(content, strict=strict_envelope)
    except Top1DataError as exc:
        return {}, {scenario_id: [str(exc)] for scenario_id in ids}
    raw_audits = payload.get("audits")
    if not isinstance(raw_audits, list):
        return {}, {scenario_id: ["response.audits must be a list"] for scenario_id in ids}
    fields = (
        "target_axis_realized",
        "target_candidate_respected",
        "source_axis_realized",
        "source_candidate_respected",
        "phenomenon_realized",
        "no_extra_current_goal",
    )
    if strict_envelope:
        observed_fields = (
            (
                "observed_target_content_axis",
                "observed_source_content_axis",
                "target_axis_unambiguous",
                "source_axis_unambiguous",
            )
            if observed_axis_catalogs is not None
            else ()
        )
        shape_errors = _strict_response_shape_errors(
            payload,
            response_field="audits",
            item_fields=("scenario_id", *fields, *observed_fields, "reason"),
        )
        if shape_errors:
            return {}, {
                scenario_id: list(shape_errors) for scenario_id in ids
            }
        envelope_errors = _strict_response_envelope_errors(
            raw_audits,
            ids,
            response_field="audits",
        )
        if envelope_errors:
            return {}, {
                scenario_id: list(envelope_errors) for scenario_id in ids
            }
    allowed_ids = set(ids)
    parsed: dict[str, dict[str, Any]] = {}
    for raw in raw_audits:
        if not isinstance(raw, Mapping):
            continue
        scenario_id = raw.get("scenario_id")
        if not isinstance(scenario_id, str) or scenario_id not in allowed_ids:
            continue
        if scenario_id in parsed:
            parsed.pop(scenario_id)
            errors[scenario_id].append("duplicate scenario in plan-fidelity response")
            continue
        item_errors = [
            f"{field} must be boolean"
            for field in fields
            if not isinstance(raw.get(field), bool)
        ]
        reason = raw.get("reason")
        if not isinstance(reason, str):
            item_errors.append("reason must be a string")
        observed_values: dict[str, Any] = {}
        if observed_axis_catalogs is not None:
            catalog = observed_axis_catalogs.get(scenario_id)
            if not isinstance(catalog, Mapping):
                item_errors.append("missing observed-axis catalog")
            else:
                target_axes = catalog.get("target_content_axes")
                source_axes = catalog.get("source_content_axes")
                if (
                    not isinstance(target_axes, Sequence)
                    or isinstance(target_axes, (str, bytes))
                    or not target_axes
                    or any(
                        not isinstance(axis, str) or not axis
                        for axis in target_axes
                    )
                ):
                    item_errors.append("invalid target observed-axis catalog")
                else:
                    observed_target = raw.get("observed_target_content_axis")
                    if observed_target not in set(target_axes):
                        item_errors.append(
                            "observed_target_content_axis must be in target catalog"
                        )
                    observed_values["observed_target_content_axis"] = observed_target
                observed_source = raw.get("observed_source_content_axis")
                if source_axes is None:
                    if observed_source is not None:
                        item_errors.append(
                            "observed_source_content_axis must be null for non-switch"
                        )
                elif (
                    not isinstance(source_axes, Sequence)
                    or isinstance(source_axes, (str, bytes))
                    or not source_axes
                    or any(
                        not isinstance(axis, str) or not axis
                        for axis in source_axes
                    )
                ):
                    item_errors.append("invalid source observed-axis catalog")
                elif observed_source not in set(source_axes):
                    item_errors.append(
                        "observed_source_content_axis must be in source catalog"
                    )
                observed_values["observed_source_content_axis"] = observed_source
                for field in (
                    "target_axis_unambiguous",
                    "source_axis_unambiguous",
                ):
                    if not isinstance(raw.get(field), bool):
                        item_errors.append(f"{field} must be boolean")
                    else:
                        observed_values[field] = bool(raw[field])
            if isinstance(reason, str) and len(reason.strip()) < 20:
                item_errors.append("reason must contain at least 20 characters")
        if item_errors:
            errors[scenario_id].extend(item_errors)
            continue
        parsed[scenario_id] = {
            "scenario_id": scenario_id,
            **{field: bool(raw[field]) for field in fields},
            **observed_values,
            "reason": reason.strip(),
        }
    for scenario_id in ids:
        if scenario_id not in parsed and not errors[scenario_id]:
            errors[scenario_id].append("scenario missing from plan-fidelity response")
    return parsed, errors


def combine_directness_audits(
    audits_by_model: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Combine independent directness judgments with reject-on-disagreement gates."""

    if len(audits_by_model) < 2:
        raise Top1DataError("at least two directness audits are required")
    items = list(audits_by_model.items())
    return {
        "contains_only_new_request": all(
            audit["contains_only_new_request"] is True for _, audit in items
        ),
        "references_previous_exchange": any(
            audit["references_previous_exchange"] is True for _, audit in items
        ),
        "uses_transition_or_acknowledgment": any(
            audit["uses_transition_or_acknowledgment"] is True
            for _, audit in items
        ),
        "direct_final_request": all(
            audit["direct_final_request"] is True for _, audit in items
        ),
        "has_switch_meta_language": any(
            audit["has_switch_meta_language"] is True for _, audit in items
        ),
        "reason": " | ".join(
            f"{model}: {audit['reason']}" for model, audit in items
        ),
        "model_audits": [
            {"model": model, "audit": dict(audit)} for model, audit in items
        ],
    }


def combine_dialogue_quality_audits(
    audits_by_model: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Combine independent dialogue-quality audits with unanimous gates."""

    if len(audits_by_model) < 2:
        raise Top1DataError("at least two dialogue-quality audits are required")
    items = list(audits_by_model.items())
    return {
        **{
            field: all(audit[field] is True for _, audit in items)
            for field in DIALOGUE_QUALITY_AUDIT_FIELDS
        },
        "reason": " | ".join(
            f"{model}: {audit['reason']}" for model, audit in items
        ),
        "model_audits": [
            {"model": model, "audit": dict(audit)} for model, audit in items
        ],
    }


def combine_contrast_audits(
    audits_by_model: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Combine independent boundary audits with reject-on-disagreement gates."""

    if len(audits_by_model) < 2:
        raise Top1DataError("at least two contrast audits are required")
    fields = (
        "target_semantics_present",
        "contrast_is_plausible",
        "target_preferred_over_contrast",
        "single_current_goal",
        "natural_expression",
    )
    items = list(audits_by_model.items())
    if all("contrast_link_natural" in audit for _, audit in items):
        fields += ("contrast_link_natural",)
    return {
        **{
            field: all(audit[field] is True for _, audit in items)
            for field in fields
        },
        "reason": " | ".join(
            f"{model}: {audit['reason']}" for model, audit in items
        ),
        "model_audits": [
            {"model": model, "audit": dict(audit)} for model, audit in items
        ],
    }


def combine_plan_fidelity_audits(
    audits_by_model: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Combine independent plan-fidelity audits with unanimous boolean gates."""

    if len(audits_by_model) < 2:
        raise Top1DataError("at least two plan-fidelity audits are required")
    fields = (
        "target_axis_realized",
        "target_candidate_respected",
        "source_axis_realized",
        "source_candidate_respected",
        "phenomenon_realized",
        "no_extra_current_goal",
    )
    items = list(audits_by_model.items())
    combined = {
        **{
            field: all(audit[field] is True for _, audit in items)
            for field in fields
        },
        "reason": " | ".join(
            f"{model}: {audit['reason']}" for model, audit in items
        ),
        "model_audits": [
            {"model": model, "audit": dict(audit)} for model, audit in items
        ],
    }
    if all(
        "observed_target_content_axis" in audit
        and "observed_source_content_axis" in audit
        and "target_axis_unambiguous" in audit
        and "source_axis_unambiguous" in audit
        for _, audit in items
    ):
        target_axes = {
            audit["observed_target_content_axis"] for _, audit in items
        }
        source_axes = {
            audit["observed_source_content_axis"] for _, audit in items
        }
        target_consensus = len(target_axes) == 1
        source_consensus = len(source_axes) == 1
        combined.update(
            {
                "observed_target_content_axis": (
                    next(iter(target_axes)) if target_consensus else None
                ),
                "observed_source_content_axis": (
                    next(iter(source_axes)) if source_consensus else None
                ),
                "target_axis_unambiguous": target_consensus
                and all(
                    audit["target_axis_unambiguous"] is True
                    for _, audit in items
                ),
                "source_axis_unambiguous": source_consensus
                and all(
                    audit["source_axis_unambiguous"] is True
                    for _, audit in items
                ),
            }
        )
    return combined


def acceptance_reasons(
    blueprint: DialogueBlueprint,
    labeler: Mapping[str, Any] | None,
    reviewer: Mapping[str, Any] | None,
    directness: Mapping[str, Any] | None = None,
    contrast: Mapping[str, Any] | None = None,
    plan_fidelity: Mapping[str, Any] | None = None,
    dialogue_quality: Mapping[str, Any] | None = None,
    *,
    quality_fields: Sequence[str] = QUALITY_FIELDS,
    require_plan_fidelity: bool = False,
    require_observed_axis_match: bool = False,
    require_dialogue_quality: bool = False,
    require_empty_judgment_issues: bool = False,
    require_both_judges_plan_match: bool = False,
    require_contrast_link_natural: bool = False,
) -> list[str]:
    """Return structured reasons preventing one sample from being accepted."""

    reasons: list[str] = []
    for role, judgment in (("labeler", labeler), ("reviewer", reviewer)):
        if judgment is None:
            reasons.append(f"missing_{role}_judgment")
            continue
        if judgment.get("predicted_candidate_name") != blueprint.target_candidate_name:
            reasons.append(f"{role}_label_mismatch")
        quality = judgment.get("quality")
        if not isinstance(quality, Mapping):
            reasons.append(f"{role}_quality_missing")
        else:
            for field in quality_fields:
                if quality.get(field) is not True:
                    reasons.append(f"{role}_quality_{field}")
        if require_empty_judgment_issues and judgment.get("issues") != []:
            reasons.append(f"{role}_issues_not_empty")

    if reviewer is not None:
        if reviewer.get("observed_phenomenon") != blueprint.phenomenon:
            reasons.append("reviewer_phenomenon_mismatch")
        if blueprint.phenomenon == "intent_change":
            if reviewer.get("observed_source_candidate_name") != blueprint.source_candidate_name:
                reasons.append("reviewer_source_candidate_mismatch")
            if reviewer.get("intent_change_is_direct") is not True:
                reasons.append("intent_change_not_direct")
    if require_both_judges_plan_match and labeler is not None:
        if labeler.get("observed_phenomenon") != blueprint.phenomenon:
            reasons.append("labeler_phenomenon_mismatch")
        if blueprint.phenomenon == "intent_change":
            if labeler.get("observed_source_candidate_name") != blueprint.source_candidate_name:
                reasons.append("labeler_source_candidate_mismatch")
            if labeler.get("intent_change_is_direct") is not True:
                reasons.append("labeler_intent_change_not_direct")
    if require_both_judges_plan_match and blueprint.phenomenon != "intent_change":
        for role, judgment in (("labeler", labeler), ("reviewer", reviewer)):
            if judgment is None:
                continue
            if judgment.get("observed_source_candidate_name") is not None:
                reasons.append(f"{role}_non_intent_source_candidate")
            if judgment.get("intent_change_is_direct") is not True:
                reasons.append(f"{role}_non_intent_directness")
    if blueprint.phenomenon == "intent_change":
        if directness is None:
            reasons.append("missing_directness_audit")
        else:
            if directness.get("contains_only_new_request") is not True:
                reasons.append("directness_not_only_new_request")
            if directness.get("references_previous_exchange") is not False:
                reasons.append("directness_references_previous_exchange")
            if directness.get("uses_transition_or_acknowledgment") is not False:
                reasons.append("directness_transition_or_acknowledgment")
            if directness.get("direct_final_request") is not True:
                reasons.append("directness_final_request_failed")
            if directness.get("has_switch_meta_language") is not False:
                reasons.append("directness_switch_meta_language")
    if blueprint.phenomenon == "single_turn":
        if contrast is None:
            reasons.append("missing_contrast_audit")
        else:
            for field in (
                "target_semantics_present",
                "contrast_is_plausible",
                "target_preferred_over_contrast",
                "single_current_goal",
                "natural_expression",
            ):
                if contrast.get(field) is not True:
                    reasons.append(f"contrast_{field}")
            if (
                require_contrast_link_natural
                and contrast.get("contrast_link_natural") is not True
            ):
                reasons.append("contrast_contrast_link_natural")
    if require_plan_fidelity:
        if plan_fidelity is None:
            reasons.append("missing_plan_fidelity_audit")
        else:
            for field in (
                "target_axis_realized",
                "target_candidate_respected",
                "source_axis_realized",
                "source_candidate_respected",
                "phenomenon_realized",
                "no_extra_current_goal",
            ):
                if plan_fidelity.get(field) is not True:
                    reasons.append(f"plan_fidelity_{field}")
            if require_observed_axis_match:
                if plan_fidelity.get("target_axis_unambiguous") is not True:
                    reasons.append("plan_fidelity_target_axis_unambiguous")
                if (
                    plan_fidelity.get("observed_target_content_axis")
                    != blueprint.content_axis
                ):
                    reasons.append("plan_fidelity_target_axis_mismatch")
                if plan_fidelity.get("source_axis_unambiguous") is not True:
                    reasons.append("plan_fidelity_source_axis_unambiguous")
                if (
                    plan_fidelity.get("observed_source_content_axis")
                    != blueprint.source_content_axis
                ):
                    reasons.append("plan_fidelity_source_axis_mismatch")
    if require_dialogue_quality:
        if dialogue_quality is None:
            reasons.append("missing_dialogue_quality_audit")
        else:
            for field in DIALOGUE_QUALITY_AUDIT_FIELDS:
                if dialogue_quality.get(field) is not True:
                    reasons.append(f"dialogue_quality_{field}")
    return reasons


def load_api_credentials(path: str | Path) -> tuple[str, str]:
    """Read base URL and API key without exposing either through CLI arguments."""

    values: dict[str, str] = {}
    for raw_line in Path(path).expanduser().read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        key, separator, value = raw_line.partition(":")
        if not separator or not value.strip():
            raise Top1DataError("credentials file must use key:value lines")
        values[key.strip()] = value.strip()
    base_url = values.get("base_url", "").rstrip("/")
    api_key = values.get("api_key", "")
    if not base_url or not api_key:
        raise Top1DataError("credentials file must define base_url and api_key")
    parsed = urlsplit(base_url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise Top1DataError("credentials base_url must use a valid HTTPS endpoint")
    return base_url, api_key


def json_schema_response_format(
    response_schema: Mapping[str, Any],
) -> dict[str, Any]:
    """Wrap one exact schema for an OpenAI-compatible structured response."""

    if not isinstance(response_schema, Mapping) or not response_schema:
        raise Top1DataError("response_schema must be a non-empty object")
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "top1_synthesis_response",
            "strict": True,
            "schema": dict(response_schema),
        },
    }


class OpenAICompatibleClient:
    """Small retrying JSON client that keeps synthesis dependency-free."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout_seconds: float = 180.0,
        request_attempts: int = 4,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.request_attempts = request_attempts

    def chat_json(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, str]],
        temperature: float,
        max_tokens: int,
        response_schema: Mapping[str, Any] | None = None,
        require_stop: bool = False,
    ) -> ModelCall:
        """Call `/chat/completions` and return the assistant JSON text."""

        response_format = (
            {"type": "json_object"}
            if response_schema is None
            else json_schema_response_format(response_schema)
        )
        payload = {
            "model": model,
            "messages": [dict(message) for message in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": response_format,
        }
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(1, self.request_attempts + 1):
            started = time.monotonic()
            try:
                def curl_quote(value: str) -> str:
                    return value.replace("\\", "\\\\").replace('"', '\\"')

                with tempfile.NamedTemporaryFile(
                    prefix="top1-synthesis-",
                    suffix=".json",
                ) as body_file:
                    body_file.write(encoded)
                    body_file.flush()
                    config = "\n".join(
                        (
                            "silent",
                            "show-error",
                            "fail",
                            f"max-time = {self.timeout_seconds}",
                            f'url = "{curl_quote(f"{self.base_url}/chat/completions")}"',
                            'request = "POST"',
                            f'header = "Authorization: Bearer {curl_quote(self._api_key)}"',
                            'header = "Content-Type: application/json"',
                            f'data-binary = "@{curl_quote(body_file.name)}"',
                        )
                    )
                    completed = subprocess.run(
                        ("curl", "--config", "-"),
                        input=config.encode("utf-8"),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=self.timeout_seconds + 10,
                        check=False,
                    )
                if completed.returncode != 0:
                    detail = completed.stderr.decode("utf-8", errors="replace").strip()
                    raise RuntimeError(
                        f"curl model request failed with exit {completed.returncode}: {detail[-300:]}"
                    )
                raw_payload = json.loads(completed.stdout.decode("utf-8"))
                choice = raw_payload["choices"][0]
                finish_reason = choice.get("finish_reason")
                if require_stop and finish_reason != "stop":
                    raise Top1DataError(
                        "model completion did not stop cleanly: "
                        f"finish_reason={finish_reason!r}"
                    )
                content = choice["message"]["content"]
                if not isinstance(content, str) or not content.strip():
                    raise Top1DataError("model returned empty completion content")
                if response_schema is not None:
                    try:
                        structured_content = json.loads(content.strip())
                    except json.JSONDecodeError as exc:
                        raise Top1DataError(
                            f"model returned invalid strict JSON: {exc.msg}"
                        ) from exc
                    if not isinstance(structured_content, dict):
                        raise Top1DataError(
                            "model strict JSON response must be an object"
                        )
                usage = raw_payload.get("usage")
                return ModelCall(
                    content=content,
                    usage=dict(usage) if isinstance(usage, Mapping) else {},
                    finish_reason=finish_reason,
                    elapsed_seconds=time.monotonic() - started,
                )
            except (
                subprocess.TimeoutExpired,
                RuntimeError,
                json.JSONDecodeError,
                KeyError,
                IndexError,
                Top1DataError,
            ) as exc:
                last_error = exc
            if attempt < self.request_attempts:
                time.sleep(min(2 ** (attempt - 1), 8))
        raise RuntimeError(f"model request failed after {self.request_attempts} attempts: {last_error}")


def content_sha256(value: str) -> str:
    """Hash prompt or endpoint text for reproducible manifests."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()
