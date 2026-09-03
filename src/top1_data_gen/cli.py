#!/usr/bin/env python3
"""Generate controlled, independently reviewed multi-turn Top1 training data."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from top1_data_gen.synthesis import (
    CONTRAST_AUDIT_VERSION,
    CONTRAST_NATURAL_LINK_AUDIT_VERSION,
    DIALOGUE_QUALITY_AUDIT_VERSION,
    DIRECTNESS_AUDIT_VERSION,
    OBSERVED_AXIS_PLAN_FIDELITY_AUDIT_VERSION,
    PLAN_FIDELITY_CONSENSUS_AUDIT_VERSION,
    PLAN_FIDELITY_AUDIT_VERSION,
    QUALITY_FIELDS,
    STRICT_JSON_SCHEMA_PROTOCOL,
    STRICT_DIALOGUE_QUALITY_FIELDS,
    DialogueBlueprint,
    ModelCall,
    OpenAICompatibleClient,
    acceptance_reasons,
    build_dialogue_blueprints,
    combine_contrast_audits,
    combine_dialogue_quality_audits,
    combine_directness_audits,
    combine_plan_fidelity_audits,
    content_sha256,
    contrast_response_schema,
    contrast_messages,
    dialogue_quality_response_schema,
    dialogue_quality_messages,
    directness_response_schema,
    directness_messages,
    generation_response_schema,
    generation_messages,
    json_schema_response_format,
    judgment_response_schema,
    judgment_messages,
    load_api_credentials,
    load_taxonomy_descriptions,
    parse_generated_samples,
    parse_contrast_audits,
    parse_dialogue_quality_audits,
    parse_directness_audits,
    parse_judgments,
    parse_plan_fidelity_audits,
    plan_fidelity_response_schema,
    plan_fidelity_messages,
    taxonomy_prompt,
    validate_content_axis_definitions,
    validate_content_axis_priority,
)
from top1_data_gen.data import (
    Top1DataError,
    load_candidate_names,
    normalize_messages,
    read_jsonl,
    sha256_file,
    validate_training_rows,
    write_json,
    write_jsonl,
)


DEFAULT_CONFIG = "configs/top1_synthesis_v2.json"
DEFAULT_CREDENTIALS = "credentials"
DEFAULT_OUTPUT_DIR = "data_top1/generated/top1_controlled_multiturn_v2"


class StageAPICircuitBreakerError(RuntimeError):
    """Abort an uncommitted synthesis round after a stage exhausts API retries."""

    def __init__(
        self,
        *,
        stage: str,
        failed_scenario_ids: Sequence[str],
        failed_batches: int,
        cancelled_scenario_ids: Sequence[str],
        raw_path: Path,
    ) -> None:
        self.stage = stage
        self.failed_scenario_ids = tuple(sorted(set(failed_scenario_ids)))
        self.failed_batches = failed_batches
        self.cancelled_scenario_ids = tuple(
            sorted(set(cancelled_scenario_ids))
        )
        self.raw_path = raw_path
        super().__init__(
            "API circuit breaker tripped: "
            f"stage={stage}, failed_scenarios={len(self.failed_scenario_ids)}, "
            f"failed_batches={failed_batches}, "
            f"cancelled_scenarios={len(self.cancelled_scenario_ids)}. "
            "The current synthesis round was not committed and sample attempts "
            f"were not consumed; raw records were preserved at {raw_path}."
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse independent synthesis arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--credentials-file", default=DEFAULT_CREDENTIALS)
    parser.add_argument(
        "--candidate-registry",
        default="configs/top1_candidates_v2.json",
    )
    parser.add_argument(
        "--taxonomy-data",
        default="data_top1/top1_labeldesc_v2.jsonl",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-count", type=int)
    parser.add_argument("--intent-change-per-pair", type=int)
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--max-sample-attempts", type=int)
    parser.add_argument("--generation-batch-size", type=int)
    parser.add_argument("--judgment-batch-size", type=int)
    scope_group = parser.add_mutually_exclusive_group()
    scope_group.add_argument(
        "--scenario-limit",
        type=int,
        help="process only this many planned scenarios in the current invocation",
    )
    scope_group.add_argument(
        "--axis-pilot-per-axis",
        type=int,
        help=(
            "process a deterministic pilot with this many scenarios per "
            "(target candidate, content axis) in the current invocation"
        ),
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="write the immutable manifest and plans without calling an LLM",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="exit successfully even if quality gates exhaust some scenario attempts",
    )
    return parser.parse_args(argv)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise Top1DataError(f"invalid JSON config: {path}") from exc
    if not isinstance(payload, dict):
        raise Top1DataError(f"JSON object required: {path}")
    return payload


def _append_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _chunks(values: Sequence[Any], size: int) -> list[list[Any]]:
    if size <= 0:
        raise Top1DataError("batch size must be positive")
    return [list(values[index : index + size]) for index in range(0, len(values), size)]


def _abort_round_on_api_failure_enabled(config: Mapping[str, Any]) -> bool:
    """Resolve the opt-in circuit breaker without changing legacy behavior."""

    value = config.get("abort_round_on_api_failure", False)
    if not isinstance(value, bool):
        raise Top1DataError("abort_round_on_api_failure must be a boolean")
    return value


def _pilot_phenomenon_bucket(plan: DialogueBlueprint) -> str:
    if plan.phenomenon in {"single_turn", "intent_change"}:
        return plan.phenomenon
    return "non_switch_multiturn"


def _select_axis_pilot(
    plans: Sequence[DialogueBlueprint],
    per_axis: int,
) -> list[DialogueBlueprint]:
    """Select a deterministic, phenomenon-stratified target-axis pilot."""

    if isinstance(per_axis, bool) or not isinstance(per_axis, int) or per_axis <= 0:
        raise Top1DataError("axis-pilot-per-axis must be a positive integer")
    grouped: dict[tuple[str, str], list[DialogueBlueprint]] = defaultdict(list)
    for plan in plans:
        axis = plan.content_axis
        if not isinstance(axis, str) or not axis.strip():
            raise Top1DataError(
                "axis pilot requires a non-empty content_axis on every plan"
            )
        grouped[(plan.target_candidate_name, axis.strip())].append(plan)
    if not grouped:
        raise Top1DataError("axis pilot requires at least one planned scenario")

    axes_by_target: dict[str, list[str]] = defaultdict(list)
    for target, axis in grouped:
        axes_by_target[target].append(axis)
    phenomenon_cycle = (
        "single_turn",
        "intent_change",
        "non_switch_multiturn",
    )
    selected: list[DialogueBlueprint] = []
    for target in sorted(axes_by_target):
        for axis_index, axis in enumerate(sorted(axes_by_target[target])):
            remaining = list(grouped[(target, axis)])
            if len(remaining) < per_axis:
                raise Top1DataError(
                    "axis pilot requested "
                    f"{per_axis} rows for ({target}, {axis}), but only "
                    f"{len(remaining)} are planned"
                )
            for offset in range(per_axis):
                # Rotate across axes so even per_axis=1 covers all three buckets.
                preferred = phenomenon_cycle[(axis_index + offset) % len(phenomenon_cycle)]
                match_index = next(
                    (
                        index
                        for index, plan in enumerate(remaining)
                        if _pilot_phenomenon_bucket(plan) == preferred
                    ),
                    0,
                )
                selected.append(remaining.pop(match_index))
    return selected


def _select_active_plans(
    plans: Sequence[DialogueBlueprint],
    *,
    scenario_limit: int | None,
    axis_pilot_per_axis: int | None,
) -> list[DialogueBlueprint]:
    """Apply an invocation-only scope without changing the immutable plan."""

    if scenario_limit is not None and axis_pilot_per_axis is not None:
        raise Top1DataError(
            "scenario-limit and axis-pilot-per-axis are mutually exclusive"
        )
    if scenario_limit is not None:
        if scenario_limit <= 0:
            raise Top1DataError("scenario-limit must be positive")
        return list(plans[:scenario_limit])
    if axis_pilot_per_axis is not None:
        return _select_axis_pilot(plans, axis_pilot_per_axis)
    return list(plans)


def _request_hash(
    messages: Sequence[Mapping[str, str]],
    *,
    stage: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    response_schema: Mapping[str, Any] | None = None,
    protocol: str = STRICT_JSON_SCHEMA_PROTOCOL,
    require_stop: bool = False,
) -> str:
    """Hash a legacy prompt or the complete strict-v2 request contract."""

    if response_schema is None:
        return content_sha256(
            json.dumps(messages, ensure_ascii=False, sort_keys=True)
        )
    if stage is None or model is None or temperature is None or max_tokens is None:
        raise Top1DataError(
            "strict request hashes require stage, model, temperature, and max_tokens"
        )
    payload = {
        "protocol": protocol,
        "stage": stage,
        "model": model,
        "messages": [dict(message) for message in messages],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": json_schema_response_format(response_schema),
        "require_stop": require_stop,
    }
    return content_sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _canonical_messages(messages: Sequence[Mapping[str, Any]]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (message["role"], message["content"])
        for message in normalize_messages(messages)
    )


def _run_model_batches(
    *,
    stage: str,
    batches: Sequence[Sequence[Any]],
    model: str,
    client: OpenAICompatibleClient,
    max_workers: int,
    temperature: float,
    max_tokens: int,
    build_messages: Callable[[Sequence[Any]], list[dict[str, str]]],
    build_response_schema: Callable[[Sequence[Any]], Mapping[str, Any]]
    | None = None,
    require_stop: bool = False,
    item_id: Callable[[Any], str],
    item_attempt: Callable[[Any], int],
    raw_path: Path,
    abort_on_api_failure: bool = False,
) -> list[dict[str, Any]]:
    """Run independent batches concurrently while logging only secret-free responses."""

    if not batches:
        return []
    prepared: list[
        tuple[
            int,
            Sequence[Any],
            list[dict[str, str]],
            Mapping[str, Any] | None,
            str,
        ]
    ] = []
    for index, batch in enumerate(batches, start=1):
        messages = build_messages(batch)
        response_schema = (
            build_response_schema(batch)
            if build_response_schema is not None
            else None
        )
        request_sha256 = _request_hash(
            messages,
            stage=stage,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_schema=response_schema,
            require_stop=require_stop,
        )
        prepared.append(
            (index, batch, messages, response_schema, request_sha256)
        )

    expected_by_hash = {
        request_sha256: {
            "scenario_ids": [item_id(item) for item in batch],
            "sample_attempts": {
                item_id(item): item_attempt(item) for item in batch
            },
        }
        for _, batch, _, _, request_sha256 in prepared
    }
    cached: list[dict[str, Any]] = []
    cached_hashes: set[str] = set()
    if raw_path.is_file():
        for record in read_jsonl(raw_path):
            request_sha256 = record.get("request_sha256")
            expected = expected_by_hash.get(str(request_sha256))
            if (
                expected is None
                or request_sha256 in cached_hashes
                or record.get("status") != "completed"
                or record.get("model") != model
                or record.get("scenario_ids") != expected["scenario_ids"]
            ):
                continue
            if record.get("sample_attempts") != expected["sample_attempts"]:
                continue
            cached_record = dict(record)
            cached_record["cache_hit"] = True
            cached.append(cached_record)
            cached_hashes.add(str(request_sha256))

    pending_prepared = [
        value
        for value in prepared
        if value[4] not in cached_hashes
    ]
    results: list[dict[str, Any]] = list(cached)
    if cached:
        print(f"[synthesis] {stage}: reused {len(cached)}/{len(batches)} cached batches")
    if not pending_prepared:
        return results
    completed = 0
    failed_scenario_ids: list[str] = []
    cancelled_scenario_ids: list[str] = []
    failed_batches = 0
    breaker_tripped = False
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for (
            batch_index,
            batch,
            messages,
            response_schema,
            request_sha256,
        ) in pending_prepared:
            request_arguments: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if response_schema is not None:
                request_arguments.update(
                    {
                        "response_schema": response_schema,
                        "require_stop": require_stop,
                    }
                )
            futures[executor.submit(client.chat_json, **request_arguments)] = (
                batch_index,
                batch,
                messages,
                response_schema,
                request_sha256,
            )
        for future in as_completed(futures):
            (
                batch_index,
                batch,
                messages,
                response_schema,
                request_sha256,
            ) = futures[future]
            identifiers = [item_id(item) for item in batch]
            record: dict[str, Any] = {
                "timestamp": _now(),
                "stage": stage,
                "batch_index": batch_index,
                "scenario_ids": identifiers,
                "sample_attempts": {
                    item_id(item): item_attempt(item) for item in batch
                },
                "model": model,
                "request_sha256": request_sha256,
            }
            if response_schema is not None:
                record["response_protocol"] = STRICT_JSON_SCHEMA_PROTOCOL
            if future.cancelled():
                record.update(
                    {
                        "status": "cancelled",
                        "error": "cancelled_by_api_circuit_breaker",
                    }
                )
                cancelled_scenario_ids.extend(identifiers)
            else:
                try:
                    call: ModelCall = future.result()
                    record.update(
                        {
                            "status": "completed",
                            "content": call.content,
                            "finish_reason": call.finish_reason,
                            "usage": dict(call.usage),
                            "elapsed_seconds": round(call.elapsed_seconds, 6),
                        }
                    )
                except Exception as exc:  # noqa: BLE001 - preserve API failure
                    record.update({"status": "failed", "error": str(exc)})
                    failed_scenario_ids.extend(identifiers)
                    failed_batches += 1
                    if abort_on_api_failure and not breaker_tripped:
                        breaker_tripped = True
                        for other_future in futures:
                            if other_future is not future:
                                other_future.cancel()
            _append_jsonl(raw_path, (record,))
            results.append(record)
            completed += 1
            done_total = len(cached) + completed
            if completed == len(pending_prepared) or completed % 10 == 0:
                print(f"[synthesis] {stage}: {done_total}/{len(batches)} batches", flush=True)
    if abort_on_api_failure and failed_scenario_ids:
        raise StageAPICircuitBreakerError(
            stage=stage,
            failed_scenario_ids=failed_scenario_ids,
            failed_batches=failed_batches,
            cancelled_scenario_ids=cancelled_scenario_ids,
            raw_path=raw_path,
        )
    return results


def _accepted_attempt(row: Mapping[str, Any]) -> int:
    synthesis = row.get("synthesis")
    if not isinstance(synthesis, Mapping) or not isinstance(synthesis.get("attempt"), int):
        raise Top1DataError("accepted record has no synthesis attempt")
    return int(synthesis["attempt"])


def _load_invalidations(path: Path) -> dict[str, int]:
    invalidated: dict[str, int] = {}
    if not path.is_file():
        return invalidated
    for row in read_jsonl(path):
        scenario_id = row.get("scenario_id")
        attempt = row.get("invalidated_attempt")
        if isinstance(scenario_id, str) and isinstance(attempt, int):
            invalidated[scenario_id] = max(invalidated.get(scenario_id, 0), attempt)
    return invalidated


def _load_accepted(
    path: Path,
    invalidation_path: Path,
) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    result: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        scenario_id = row.get("scenario_id")
        if not isinstance(scenario_id, str):
            raise Top1DataError("accepted record has no scenario_id")
        previous = result.get(scenario_id)
        if previous is None or _accepted_attempt(row) > _accepted_attempt(previous):
            result[scenario_id] = row
    invalidated = _load_invalidations(invalidation_path)
    return {
        scenario_id: row
        for scenario_id, row in result.items()
        if _accepted_attempt(row) > invalidated.get(scenario_id, 0)
    }


def _load_attempt_counts(path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not path.is_file():
        return counts
    for row in read_jsonl(path):
        scenario_id = row.get("scenario_id")
        attempt = row.get("attempt")
        if isinstance(scenario_id, str) and isinstance(attempt, int):
            counts[scenario_id] = max(counts[scenario_id], attempt)
    return counts


def _usage_summary(raw_directory: Path) -> dict[str, Any]:
    summary: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "calls": 0,
            "failed_calls": 0,
            "cancelled_batches": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    )
    for path in sorted(raw_directory.glob("*_responses.jsonl")):
        for row in read_jsonl(path):
            stage = str(row.get("stage", "unknown"))
            bucket = summary[stage]
            if row.get("status") == "cancelled":
                bucket["cancelled_batches"] += 1
                continue
            bucket["calls"] += 1
            if row.get("status") != "completed":
                bucket["failed_calls"] += 1
            usage = row.get("usage")
            if isinstance(usage, Mapping):
                for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    value = usage.get(field)
                    if isinstance(value, int):
                        bucket[field] += value
    return dict(sorted(summary.items()))


def _count(rows: Iterable[Mapping[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(field)) for row in rows).items()))


def _write_summary(
    *,
    path: Path,
    pipeline_version: str,
    plans: Sequence[DialogueBlueprint],
    accepted: Mapping[str, Mapping[str, Any]],
    attempt_path: Path,
    raw_directory: Path,
    train_path: Path,
    complete: bool,
) -> None:
    attempts = read_jsonl(attempt_path) if attempt_path.is_file() else []
    rejected_attempts = [row for row in attempts if row.get("status") != "accepted"]
    rejection_reasons = Counter(
        str(reason)
        for row in rejected_attempts
        for reason in row.get("reasons", [])
        if isinstance(reason, str)
    )
    accepted_rows = list(accepted.values())
    output: dict[str, Any] = {
        "path": str(train_path),
        "exists": train_path.is_file(),
    }
    if train_path.is_file():
        output["sha256"] = sha256_file(train_path)
    write_json(
        path,
        {
            "schema_version": 1,
            "pipeline_version": pipeline_version,
            "updated_at": _now(),
            "complete": complete,
            "planned_rows": len(plans),
            "accepted_rows": len(accepted_rows),
            "unresolved_rows": len(plans) - len(accepted_rows),
            "attempts": len(attempts),
            "rejected_attempts": len(rejected_attempts),
            "acceptance_rate_per_attempt": (
                len(accepted_rows) / len(attempts) if attempts else 0.0
            ),
            "candidate_counts": _count(accepted_rows, "target_candidate_name"),
            "phenomenon_counts": _count(accepted_rows, "conversation_phenomenon"),
            "source_candidate_counts": _count(
                [row for row in accepted_rows if row.get("source_candidate_name")],
                "source_candidate_name",
            ),
            "rejection_reason_counts": dict(sorted(rejection_reasons.items())),
            "model_usage": _usage_summary(raw_directory),
            "output": output,
        },
    )


def _manifest_signature(payload: Mapping[str, Any]) -> str:
    stable = {key: value for key, value in payload.items() if key != "created_at"}
    return content_sha256(json.dumps(stable, ensure_ascii=False, sort_keys=True))


def _prepare_run(
    *,
    output_directory: Path,
    pipeline_version: str,
    config: Mapping[str, Any],
    candidate_path: Path,
    taxonomy_path: Path,
    endpoint: str,
    plans: Sequence[DialogueBlueprint],
    taxonomy: str,
    implementation_paths: Mapping[str, Path] | None = None,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = output_directory / "manifest.json"
    plans_path = output_directory / "plans.jsonl"
    payload: dict[str, Any] = {
        "schema_version": 1,
        "pipeline_version": pipeline_version,
        "created_at": _now(),
        "config": dict(config),
        "inputs": {
            "candidate_registry": {
                "path": str(candidate_path),
                "sha256": sha256_file(candidate_path),
            },
            "taxonomy_data": {
                "path": str(taxonomy_path),
                "sha256": sha256_file(taxonomy_path),
            },
        },
        "endpoint_sha256": content_sha256(endpoint),
        "taxonomy_prompt_sha256": content_sha256(taxonomy),
        "planned_rows": len(plans),
    }
    if bool(config.get("record_implementation_hashes", False)):
        expected_paths = {
            "src/top1_data_gen/cli.py",
            "src/top1_data_gen/synthesis.py",
        }
        if implementation_paths is None or set(implementation_paths) != expected_paths:
            raise Top1DataError(
                "implementation hash recording requires the generator and synthesis sources"
            )
        payload["implementation"] = {
            name: {"path": name, "sha256": sha256_file(path)}
            for name, path in sorted(implementation_paths.items())
        }
    signature = _manifest_signature(payload)
    payload["run_signature"] = signature
    if manifest_path.is_file():
        existing = _read_json(manifest_path)
        if existing.get("run_signature") != signature:
            raise Top1DataError(
                f"output directory belongs to a different synthesis run: {output_directory}"
            )
    else:
        write_json(manifest_path, payload)

    expected_plans = [plan.to_dict() for plan in plans]
    if plans_path.is_file():
        if read_jsonl(plans_path) != expected_plans:
            raise Top1DataError("existing synthesis plans differ from the current run")
    else:
        write_jsonl(plans_path, expected_plans)


def _raw_record_by_ids(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    return {
        str(scenario_id): record
        for record in records
        for scenario_id in record.get("scenario_ids", [])
    }


def _generate_attempt(
    *,
    blueprints: Sequence[DialogueBlueprint],
    attempt_numbers: Mapping[str, int],
    taxonomy: str,
    candidate_names: Sequence[str],
    config: Mapping[str, Any],
    client: OpenAICompatibleClient,
    raw_directory: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    batches = _chunks(blueprints, int(config["generation_batch_size"]))
    strict_protocol = _strict_response_envelope_enabled(config)
    records = _run_model_batches(
        stage="generation",
        batches=batches,
        model=str(config["generator_model"]),
        client=client,
        max_workers=int(config["max_workers"]),
        temperature=float(config["generator_temperature"]),
        max_tokens=int(config["generation_max_tokens"]),
        build_messages=lambda batch: generation_messages(
            batch,
            taxonomy,
            boundary_guidance=(
                str(config["generation_boundary_guidance"])
                if "generation_boundary_guidance" in config
                else None
            ),
            content_axis_definitions=config.get("content_axis_definitions"),
        ),
        build_response_schema=(
            generation_response_schema if strict_protocol else None
        ),
        require_stop=strict_protocol,
        item_id=lambda item: item.scenario_id,
        item_attempt=lambda item: attempt_numbers[item.scenario_id],
        raw_path=raw_directory / "generation_responses.jsonl",
        abort_on_api_failure=_abort_round_on_api_failure_enabled(config),
    )
    generated: dict[str, dict[str, Any]] = {}
    errors = {blueprint.scenario_id: [] for blueprint in blueprints}
    blueprint_by_id = {blueprint.scenario_id: blueprint for blueprint in blueprints}
    for record in records:
        identifiers = [str(value) for value in record["scenario_ids"]]
        if record.get("status") != "completed":
            for scenario_id in identifiers:
                errors[scenario_id].append("generation_api_failure")
            continue
        assigned = {scenario_id: blueprint_by_id[scenario_id] for scenario_id in identifiers}
        parsed, parse_errors = parse_generated_samples(
            str(record["content"]),
            assigned,
            candidate_names,
            strict_envelope=_strict_response_envelope_enabled(config),
        )
        for scenario_id, values in parse_errors.items():
            errors[scenario_id].extend(values)
        for scenario_id, sample in parsed.items():
            sample["attempt"] = attempt_numbers[scenario_id]
            generated[scenario_id] = sample
    return generated, errors


def _quality_fields(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the blind-judge quality contract enabled for this run."""

    fields = QUALITY_FIELDS
    if bool(config.get("require_no_personal_data", False)):
        fields += ("no_personal_data",)
    if bool(config.get("require_strict_dialogue_quality", False)):
        fields += STRICT_DIALOGUE_QUALITY_FIELDS
    return fields


def _strict_response_envelope_enabled(config: Mapping[str, Any]) -> bool:
    """Keep legacy parsing permissive while making strict-v2 batches exact."""

    return bool(
        config.get("require_strict_dialogue_quality", False)
        or config.get("require_observed_axis_match", False)
    )


def _judge_attempt(
    *,
    stage: str,
    samples: Sequence[Mapping[str, Any]],
    taxonomy: str,
    candidate_names: Sequence[str],
    model: str,
    config: Mapping[str, Any],
    client: OpenAICompatibleClient,
    raw_directory: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    quality_fields = _quality_fields(config)
    strict_protocol = _strict_response_envelope_enabled(config)
    use_opaque_ids = bool(config.get("require_strict_dialogue_quality", False))
    request_samples = (
        [
            {**sample, "audit_id": f"blind_{index:06d}"}
            for index, sample in enumerate(samples, start=1)
        ]
        if use_opaque_ids
        else list(samples)
    )
    request_to_scenario = {
        str(
            sample["audit_id"] if use_opaque_ids else sample["scenario_id"]
        ): str(sample["scenario_id"])
        for sample in request_samples
    }
    batches = _chunks(request_samples, int(config["judgment_batch_size"]))
    records = _run_model_batches(
        stage=stage,
        batches=batches,
        model=model,
        client=client,
        max_workers=int(config["max_workers"]),
        temperature=float(config["judge_temperature"]),
        max_tokens=int(config["judgment_max_tokens"]),
        build_messages=lambda batch: judgment_messages(
            batch,
            taxonomy,
            candidate_count=len(candidate_names),
            allow_single_turn=int(config.get("single_turn_per_candidate", 0)) > 0,
            quality_fields=quality_fields,
        ),
        build_response_schema=(
            lambda batch: judgment_response_schema(
                (
                    str(
                        item["audit_id"]
                        if use_opaque_ids
                        else item["scenario_id"]
                    )
                    for item in batch
                ),
                candidate_names,
                quality_fields=quality_fields,
            )
        )
        if strict_protocol
        else None,
        require_stop=strict_protocol,
        item_id=lambda item: str(
            item["audit_id"] if use_opaque_ids else item["scenario_id"]
        ),
        item_attempt=lambda item: int(item["attempt"]),
        raw_path=raw_directory / f"{stage}_responses.jsonl",
        abort_on_api_failure=_abort_round_on_api_failure_enabled(config),
    )
    judgments: dict[str, dict[str, Any]] = {}
    errors = {str(sample["scenario_id"]): [] for sample in samples}
    for record in records:
        request_ids = [str(value) for value in record["scenario_ids"]]
        if record.get("status") != "completed":
            for request_id in request_ids:
                scenario_id = request_to_scenario[request_id]
                errors[scenario_id].append(f"{stage}_api_failure")
            continue
        parsed, parse_errors = parse_judgments(
            str(record["content"]),
            request_ids,
            candidate_names,
            quality_fields=quality_fields,
            require_issues_field=bool(
                config.get("require_empty_judgment_issues", False)
            ),
            strict_envelope=_strict_response_envelope_enabled(config),
        )
        for request_id, judgment in parsed.items():
            scenario_id = request_to_scenario[request_id]
            judgments[scenario_id] = {
                **judgment,
                "scenario_id": scenario_id,
            }
        for request_id, values in parse_errors.items():
            scenario_id = request_to_scenario[request_id]
            errors[scenario_id].extend(values)
    return judgments, errors


def _directness_attempt(
    *,
    samples: Sequence[Mapping[str, Any]],
    models: Sequence[str],
    config: Mapping[str, Any],
    client: OpenAICompatibleClient,
    raw_directory: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    if len(models) < 2 or len(set(models)) != len(models):
        raise Top1DataError("directness audit requires at least two distinct models")
    identifiers = [str(sample["scenario_id"]) for sample in samples]
    strict_protocol = _strict_response_envelope_enabled(config)
    use_opaque_ids = bool(config.get("require_strict_dialogue_quality", False))
    request_samples = (
        [
            {**sample, "audit_id": f"directness_{index:06d}"}
            for index, sample in enumerate(samples, start=1)
        ]
        if use_opaque_ids
        else list(samples)
    )
    request_to_scenario = {
        str(
            sample["audit_id"] if use_opaque_ids else sample["scenario_id"]
        ): str(sample["scenario_id"])
        for sample in request_samples
    }
    batches = _chunks(request_samples, int(config["judgment_batch_size"]))
    errors = {scenario_id: [] for scenario_id in identifiers}
    audits_by_model: dict[str, dict[str, dict[str, Any]]] = {}
    for index, model in enumerate(models):
        stage = "directness" if index == 0 else f"directness_crosscheck_{index}"
        records = _run_model_batches(
            stage=stage,
            batches=batches,
            model=model,
            client=client,
            max_workers=int(config["max_workers"]),
            temperature=float(config["judge_temperature"]),
            max_tokens=int(config["judgment_max_tokens"]),
            build_messages=directness_messages,
            build_response_schema=(
                lambda batch: directness_response_schema(
                    str(
                        item["audit_id"]
                        if use_opaque_ids
                        else item["scenario_id"]
                    )
                    for item in batch
                )
            )
            if strict_protocol
            else None,
            require_stop=strict_protocol,
            item_id=lambda item: str(
                item["audit_id"] if use_opaque_ids else item["scenario_id"]
            ),
            item_attempt=lambda item: int(item["attempt"]),
            raw_path=raw_directory / f"{stage}_responses.jsonl",
            abort_on_api_failure=_abort_round_on_api_failure_enabled(config),
        )
        model_audits: dict[str, dict[str, Any]] = {}
        for record in records:
            request_ids = [str(value) for value in record["scenario_ids"]]
            if record.get("status") != "completed":
                for request_id in request_ids:
                    scenario_id = request_to_scenario[request_id]
                    errors[scenario_id].append(f"{stage}_api_failure")
                continue
            parsed, parse_errors = parse_directness_audits(
                str(record["content"]),
                request_ids,
                strict_envelope=_strict_response_envelope_enabled(config),
            )
            for request_id, audit in parsed.items():
                scenario_id = request_to_scenario[request_id]
                model_audits[scenario_id] = {
                    **audit,
                    "scenario_id": scenario_id,
                }
            for request_id, values in parse_errors.items():
                scenario_id = request_to_scenario[request_id]
                errors[scenario_id].extend(f"{stage}:{value}" for value in values)
        audits_by_model[model] = model_audits

    consensus: dict[str, dict[str, Any]] = {}
    for scenario_id in identifiers:
        judgments = [audits_by_model[model].get(scenario_id) for model in models]
        if errors[scenario_id] or any(judgment is None for judgment in judgments):
            continue
        model_judgments = {
            model: judgment
            for model, judgment in zip(models, judgments)
            if judgment is not None
        }
        consensus[scenario_id] = {
            "scenario_id": scenario_id,
            **combine_directness_audits(model_judgments),
        }
    return consensus, errors


def _dialogue_quality_attempt(
    *,
    samples: Sequence[Mapping[str, Any]],
    models: Sequence[str],
    config: Mapping[str, Any],
    client: OpenAICompatibleClient,
    raw_directory: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    """Run two independent label- and plan-blind dialogue-quality audits."""

    if len(models) < 2 or len(set(models)) != len(models):
        raise Top1DataError(
            "dialogue-quality audit requires at least two distinct models"
        )
    identifiers = [str(sample["scenario_id"]) for sample in samples]
    strict_protocol = _strict_response_envelope_enabled(config)
    aliased_samples = [
        {**sample, "audit_id": f"dialogue_{index:06d}"}
        for index, sample in enumerate(samples, start=1)
    ]
    alias_to_scenario = {
        str(sample["audit_id"]): str(sample["scenario_id"])
        for sample in aliased_samples
    }
    batches = _chunks(aliased_samples, int(config["judgment_batch_size"]))
    errors = {scenario_id: [] for scenario_id in identifiers}
    audits_by_model: dict[str, dict[str, dict[str, Any]]] = {}
    for index, model in enumerate(models):
        stage = (
            "dialogue_quality"
            if index == 0
            else f"dialogue_quality_crosscheck_{index}"
        )
        records = _run_model_batches(
            stage=stage,
            batches=batches,
            model=model,
            client=client,
            max_workers=int(config["max_workers"]),
            temperature=float(config["judge_temperature"]),
            max_tokens=int(config["judgment_max_tokens"]),
            build_messages=dialogue_quality_messages,
            build_response_schema=(
                lambda batch: dialogue_quality_response_schema(
                    str(item["audit_id"]) for item in batch
                )
            )
            if strict_protocol
            else None,
            require_stop=strict_protocol,
            item_id=lambda item: str(item["audit_id"]),
            item_attempt=lambda item: int(item["attempt"]),
            raw_path=raw_directory / f"{stage}_responses.jsonl",
            abort_on_api_failure=_abort_round_on_api_failure_enabled(config),
        )
        model_audits: dict[str, dict[str, Any]] = {}
        for record in records:
            batch_aliases = [str(value) for value in record["scenario_ids"]]
            if record.get("status") != "completed":
                for audit_id in batch_aliases:
                    scenario_id = alias_to_scenario[audit_id]
                    errors[scenario_id].append(f"{stage}_api_failure")
                continue
            parsed, parse_errors = parse_dialogue_quality_audits(
                str(record["content"]),
                batch_aliases,
                strict_envelope=True,
            )
            for audit_id, audit in parsed.items():
                scenario_id = alias_to_scenario[audit_id]
                model_audits[scenario_id] = {
                    **audit,
                    "scenario_id": scenario_id,
                }
            for audit_id, values in parse_errors.items():
                scenario_id = alias_to_scenario[audit_id]
                errors[scenario_id].extend(
                    f"{stage}:{value}" for value in values
                )
        audits_by_model[model] = model_audits

    consensus: dict[str, dict[str, Any]] = {}
    for scenario_id in identifiers:
        judgments = [audits_by_model[model].get(scenario_id) for model in models]
        if errors[scenario_id] or any(judgment is None for judgment in judgments):
            continue
        consensus[scenario_id] = {
            "scenario_id": scenario_id,
            **combine_dialogue_quality_audits(
                {
                    model: judgment
                    for model, judgment in zip(models, judgments)
                    if judgment is not None
                }
            ),
        }
    return consensus, errors


def _contrast_attempt(
    *,
    samples: Sequence[Mapping[str, Any]],
    taxonomy: str,
    models: Sequence[str],
    config: Mapping[str, Any],
    client: OpenAICompatibleClient,
    raw_directory: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    """Audit whether planned single-turn contrasts are real and unambiguous."""

    if len(models) < 2 or len(set(models)) != len(models):
        raise Top1DataError("contrast audit requires at least two distinct models")
    strict_protocol = _strict_response_envelope_enabled(config)
    batches = _chunks(samples, int(config["judgment_batch_size"]))
    identifiers = [str(sample["scenario_id"]) for sample in samples]
    errors = {scenario_id: [] for scenario_id in identifiers}
    audits_by_model: dict[str, dict[str, dict[str, Any]]] = {}
    for index, model in enumerate(models):
        stage = "contrast" if index == 0 else f"contrast_crosscheck_{index}"
        records = _run_model_batches(
            stage=stage,
            batches=batches,
            model=model,
            client=client,
            max_workers=int(config["max_workers"]),
            temperature=float(config["judge_temperature"]),
            max_tokens=int(config["judgment_max_tokens"]),
            build_messages=lambda batch: contrast_messages(
                batch,
                taxonomy,
                require_natural_link=bool(
                    config.get("require_strict_dialogue_quality", False)
                ),
            ),
            build_response_schema=(
                lambda batch: contrast_response_schema(
                    (str(item["scenario_id"]) for item in batch),
                    require_natural_link=bool(
                        config.get("require_strict_dialogue_quality", False)
                    ),
                )
            )
            if strict_protocol
            else None,
            require_stop=strict_protocol,
            item_id=lambda item: str(item["scenario_id"]),
            item_attempt=lambda item: int(item["attempt"]),
            raw_path=raw_directory / f"{stage}_responses.jsonl",
            abort_on_api_failure=_abort_round_on_api_failure_enabled(config),
        )
        model_audits: dict[str, dict[str, Any]] = {}
        for record in records:
            batch_ids = [str(value) for value in record["scenario_ids"]]
            if record.get("status") != "completed":
                for scenario_id in batch_ids:
                    errors[scenario_id].append(f"{stage}_api_failure")
                continue
            parsed, parse_errors = parse_contrast_audits(
                str(record["content"]),
                batch_ids,
                require_natural_link=bool(
                    config.get("require_strict_dialogue_quality", False)
                ),
                strict_envelope=_strict_response_envelope_enabled(config),
            )
            model_audits.update(parsed)
            for scenario_id, values in parse_errors.items():
                errors[scenario_id].extend(f"{stage}:{value}" for value in values)
        audits_by_model[model] = model_audits

    consensus: dict[str, dict[str, Any]] = {}
    for scenario_id in identifiers:
        judgments = [audits_by_model[model].get(scenario_id) for model in models]
        if errors[scenario_id] or any(judgment is None for judgment in judgments):
            continue
        consensus[scenario_id] = {
            "scenario_id": scenario_id,
            **combine_contrast_audits(
                {
                    model: judgment
                    for model, judgment in zip(models, judgments)
                    if judgment is not None
                }
            ),
        }
    return consensus, errors


def _plan_fidelity_models(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Resolve legacy single-model or opt-in strict two-model plan auditing."""

    configured = config.get("plan_auditor_models")
    if configured is not None:
        if (
            not isinstance(configured, Sequence)
            or isinstance(configured, (str, bytes))
            or len(configured) != 2
            or any(not isinstance(model, str) or not model.strip() for model in configured)
        ):
            raise Top1DataError(
                "plan_auditor_models must contain exactly two non-empty model names"
            )
        models = tuple(model.strip() for model in configured)
        if len(set(models)) != 2:
            raise Top1DataError("plan_auditor_models must contain distinct models")
        return models

    primary_value = config.get("plan_auditor_model", config["reviewer_model"])
    if not isinstance(primary_value, str) or not primary_value.strip():
        raise Top1DataError("plan_auditor_model must be non-empty")
    primary = primary_value.strip()
    if not (
        bool(config.get("require_strict_dialogue_quality", False))
        or bool(config.get("require_observed_axis_match", False))
    ):
        return (primary,)

    strict_models: list[str] = []
    for value in (primary, config["reviewer_model"], config["labeler_model"]):
        if not isinstance(value, str) or not value.strip():
            raise Top1DataError("strict plan-fidelity models must be non-empty strings")
        model = value.strip()
        if model and model not in strict_models:
            strict_models.append(model)
    if len(strict_models) < 2:
        raise Top1DataError(
            "strict plan-fidelity audit requires two distinct configured models"
        )
    return tuple(strict_models[:2])


def _plan_fidelity_attempt(
    *,
    samples: Sequence[Mapping[str, Any]],
    taxonomy: str,
    models: Sequence[str],
    config: Mapping[str, Any],
    client: OpenAICompatibleClient,
    raw_directory: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    """Audit final action, semantic axes, source context, and phenomenon."""

    if (
        not models
        or isinstance(models, (str, bytes))
        or any(not isinstance(model, str) or not model.strip() for model in models)
        or len(set(models)) != len(models)
    ):
        raise Top1DataError("plan-fidelity audit models must be non-empty and distinct")
    require_observed_axis_match = bool(
        config.get("require_observed_axis_match", False)
    )
    strict_protocol = _strict_response_envelope_enabled(config)
    if require_observed_axis_match and len(models) < 2:
        raise Top1DataError("observed-axis audit requires at least two models")
    raw_axis_definitions = config.get("content_axis_definitions")
    raw_axis_priority = config.get("content_axis_priority")
    if require_observed_axis_match and not isinstance(
        raw_axis_definitions, Mapping
    ):
        raise Top1DataError(
            "require_observed_axis_match requires content_axis_definitions"
        )
    if require_observed_axis_match and not isinstance(raw_axis_priority, Mapping):
        raise Top1DataError(
            "require_observed_axis_match requires content_axis_priority"
        )
    axis_definitions = (
        raw_axis_definitions
        if isinstance(raw_axis_definitions, Mapping)
        else None
    )
    axis_priority = (
        raw_axis_priority if isinstance(raw_axis_priority, Mapping) else None
    )
    identifiers = [str(sample["scenario_id"]) for sample in samples]
    request_samples = (
        [
            {**sample, "audit_id": f"plan_{index:06d}"}
            for index, sample in enumerate(samples, start=1)
        ]
        if require_observed_axis_match
        else list(samples)
    )
    request_to_scenario = {
        str(
            sample["audit_id"]
            if require_observed_axis_match
            else sample["scenario_id"]
        ): str(sample["scenario_id"])
        for sample in request_samples
    }
    observed_axis_catalogs: dict[
        str, dict[str, Sequence[str] | None]
    ] | None = None
    if require_observed_axis_match:
        observed_axis_catalogs = {}
        for sample in request_samples:
            request_id = str(sample["audit_id"])
            target = str(sample["target_candidate_name"])
            source = sample.get("source_candidate_name")
            target_definitions = axis_definitions.get(target) if axis_definitions else None
            source_definitions = (
                axis_definitions.get(str(source))
                if axis_definitions and source is not None
                else None
            )
            if not isinstance(target_definitions, Mapping):
                raise Top1DataError(
                    "observed-axis audit has no target content-axis catalog"
                )
            if source is not None and not isinstance(source_definitions, Mapping):
                raise Top1DataError(
                    "observed-axis audit has no source content-axis catalog"
                )
            observed_axis_catalogs[request_id] = {
                "target_content_axes": tuple(target_definitions),
                "source_content_axes": (
                    tuple(source_definitions)
                    if isinstance(source_definitions, Mapping)
                    else None
                ),
            }
    batches = _chunks(request_samples, int(config["judgment_batch_size"]))
    errors = {scenario_id: [] for scenario_id in identifiers}
    audits_by_model: dict[str, dict[str, dict[str, Any]]] = {}
    multiple_models = len(models) > 1
    for index, model in enumerate(models):
        stage = "plan_fidelity" if index == 0 else f"plan_fidelity_crosscheck_{index}"
        records = _run_model_batches(
            stage=stage,
            batches=batches,
            model=model,
            client=client,
            max_workers=int(config["max_workers"]),
            temperature=float(config["judge_temperature"]),
            max_tokens=int(config["judgment_max_tokens"]),
            build_messages=lambda batch: plan_fidelity_messages(
                batch,
                taxonomy,
                content_axis_definitions=axis_definitions,
                content_axis_priority=axis_priority,
                require_observed_axis_match=require_observed_axis_match,
            ),
            build_response_schema=(
                lambda batch: plan_fidelity_response_schema(
                    (
                        str(
                            item["audit_id"]
                            if require_observed_axis_match
                            else item["scenario_id"]
                        )
                        for item in batch
                    ),
                    observed_axis_catalogs=(
                        {
                            str(item["audit_id"]): observed_axis_catalogs[
                                str(item["audit_id"])
                            ]
                            for item in batch
                        }
                        if observed_axis_catalogs is not None
                        else None
                    ),
                )
            )
            if strict_protocol
            else None,
            require_stop=strict_protocol,
            item_id=lambda item: str(
                item["audit_id"]
                if require_observed_axis_match
                else item["scenario_id"]
            ),
            item_attempt=lambda item: int(item["attempt"]),
            raw_path=raw_directory / f"{stage}_responses.jsonl",
            abort_on_api_failure=_abort_round_on_api_failure_enabled(config),
        )
        model_audits: dict[str, dict[str, Any]] = {}
        for record in records:
            request_ids = [str(value) for value in record["scenario_ids"]]
            if record.get("status") != "completed":
                for request_id in request_ids:
                    scenario_id = request_to_scenario[request_id]
                    error = "plan_fidelity_api_failure"
                    errors[scenario_id].append(
                        f"{stage}:{error}" if multiple_models else error
                    )
                continue
            parsed, parse_errors = parse_plan_fidelity_audits(
                str(record["content"]),
                request_ids,
                observed_axis_catalogs=(
                    {
                        request_id: observed_axis_catalogs[request_id]
                        for request_id in request_ids
                    }
                    if observed_axis_catalogs is not None
                    else None
                ),
                strict_envelope=_strict_response_envelope_enabled(config),
            )
            for request_id, audit in parsed.items():
                scenario_id = request_to_scenario[request_id]
                model_audits[scenario_id] = {
                    **audit,
                    "scenario_id": scenario_id,
                }
            for request_id, values in parse_errors.items():
                scenario_id = request_to_scenario[request_id]
                errors[scenario_id].extend(
                    f"{stage}:{value}" if multiple_models else value
                    for value in values
                )
        audits_by_model[model] = model_audits

    audits: dict[str, dict[str, Any]] = {}
    for scenario_id in identifiers:
        model_judgments = {
            model: audits_by_model[model].get(scenario_id) for model in models
        }
        if errors[scenario_id] or any(
            judgment is None for judgment in model_judgments.values()
        ):
            continue
        if not multiple_models:
            judgment = next(iter(model_judgments.values()))
            if judgment is not None:
                audits[scenario_id] = judgment
            continue
        audits[scenario_id] = {
            "scenario_id": scenario_id,
            **combine_plan_fidelity_audits(
                {
                    model: judgment
                    for model, judgment in model_judgments.items()
                    if judgment is not None
                }
            ),
        }
    return audits, errors


def _load_directness_records(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    records: dict[tuple[str, int], dict[str, Any]] = {}
    if not path.is_file():
        return records
    for row in read_jsonl(path):
        scenario_id = row.get("scenario_id")
        attempt = row.get("attempt")
        if (
            isinstance(scenario_id, str)
            and isinstance(attempt, int)
            and row.get("audit_version") == DIRECTNESS_AUDIT_VERSION
        ):
            records[(scenario_id, attempt)] = row
    return records


def _audit_existing_directness(
    *,
    accepted: dict[str, dict[str, Any]],
    directness_path: Path,
    invalidation_path: Path,
    models: Sequence[str],
    config: Mapping[str, Any],
    client: OpenAICompatibleClient,
    raw_directory: Path,
) -> dict[str, dict[str, Any]]:
    existing = _load_directness_records(directness_path)
    samples = []
    for row in accepted.values():
        if row.get("conversation_phenomenon") != "intent_change":
            continue
        attempt = _accepted_attempt(row)
        if (str(row["scenario_id"]), attempt) in existing:
            continue
        samples.append(
            {
                "scenario_id": str(row["scenario_id"]),
                "messages": row["messages"],
                "attempt": attempt,
            }
        )
    if not samples:
        return accepted
    print(f"[synthesis] strict directness backfill: {len(samples)} IntentChange rows", flush=True)
    audits, errors = _directness_attempt(
        samples=samples,
        models=models,
        config=config,
        client=client,
        raw_directory=raw_directory,
    )
    audit_records: list[dict[str, Any]] = []
    invalidations: list[dict[str, Any]] = []
    for sample in samples:
        scenario_id = str(sample["scenario_id"])
        audit = audits.get(scenario_id)
        audit_errors = errors[scenario_id]
        passed = (
            not audit_errors
            and audit is not None
            and audit.get("contains_only_new_request") is True
            and audit.get("references_previous_exchange") is False
            and audit.get("uses_transition_or_acknowledgment") is False
            and audit.get("direct_final_request") is True
            and audit.get("has_switch_meta_language") is False
        )
        audit_records.append(
            {
                "timestamp": _now(),
                "audit_version": DIRECTNESS_AUDIT_VERSION,
                "scenario_id": scenario_id,
                "attempt": int(sample["attempt"]),
                "source": "backfill",
                "passed": passed,
                "errors": audit_errors,
                "audit": audit,
            }
        )
        if not passed:
            invalidations.append(
                {
                    "timestamp": _now(),
                    "audit_version": DIRECTNESS_AUDIT_VERSION,
                    "scenario_id": scenario_id,
                    "invalidated_attempt": int(sample["attempt"]),
                    "reason": "strict_directness_backfill_failed",
                }
            )
    _append_jsonl(directness_path, audit_records)
    _append_jsonl(invalidation_path, invalidations)
    for row in invalidations:
        accepted.pop(str(row["scenario_id"]), None)
    print(
        f"[synthesis] strict directness backfill: "
        f"{len(samples) - len(invalidations)} passed, {len(invalidations)} invalidated",
        flush=True,
    )
    return accepted


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    candidate_path = Path(args.candidate_registry).expanduser().resolve()
    taxonomy_path = Path(args.taxonomy_data).expanduser().resolve()
    output_directory = Path(args.output_dir).expanduser().resolve()
    config = _read_json(config_path)
    pipeline_version = config.get("pipeline_version")
    if not isinstance(pipeline_version, str) or not pipeline_version.strip():
        raise Top1DataError("synthesis config requires a pipeline_version")
    pipeline_version = pipeline_version.strip()
    runtime_config = dict(config)
    _abort_round_on_api_failure_enabled(runtime_config)
    for argument, field in (
        (args.max_workers, "max_workers"),
        (args.max_sample_attempts, "max_sample_attempts"),
        (args.generation_batch_size, "generation_batch_size"),
        (args.judgment_batch_size, "judgment_batch_size"),
    ):
        if argument is not None:
            runtime_config[field] = argument
    for argument, field in (
        (args.target_count, "target_count"),
        (args.intent_change_per_pair, "intent_change_per_pair"),
    ):
        if argument is not None:
            config[field] = argument
    if int(config["target_count"]) <= 0:
        raise Top1DataError("target_count must be positive")
    if args.scenario_limit is not None and args.scenario_limit <= 0:
        raise Top1DataError("scenario-limit must be positive")
    if args.axis_pilot_per_axis is not None and args.axis_pilot_per_axis <= 0:
        raise Top1DataError("axis-pilot-per-axis must be positive")
    if (
        int(runtime_config["max_workers"]) <= 0
        or int(runtime_config["max_sample_attempts"]) <= 0
        or int(runtime_config["generation_batch_size"]) <= 0
        or int(runtime_config["judgment_batch_size"]) <= 0
    ):
        raise Top1DataError("worker and attempt counts must be positive")

    candidate_names = load_candidate_names(candidate_path)
    axis_definitions = validate_content_axis_definitions(
        config.get("content_axes"),
        config.get("content_axis_definitions"),
        candidate_names,
    )
    if axis_definitions is not None:
        runtime_config["content_axis_definitions"] = axis_definitions
    axis_priority = validate_content_axis_priority(
        config.get("content_axes"),
        config.get("content_axis_priority"),
        candidate_names,
    )
    if axis_priority is not None:
        runtime_config["content_axis_priority"] = axis_priority
    if bool(config.get("require_observed_axis_match", False)):
        if not bool(config.get("require_plan_fidelity", False)):
            raise Top1DataError(
                "require_observed_axis_match requires require_plan_fidelity"
            )
        if axis_definitions is None:
            raise Top1DataError(
                "require_observed_axis_match requires content_axis_definitions"
            )
        if axis_priority is None:
            raise Top1DataError(
                "require_observed_axis_match requires content_axis_priority"
            )
    if (
        bool(config.get("require_strict_dialogue_quality", False))
        and str(config["reviewer_model"]).strip()
        == str(config["labeler_model"]).strip()
    ):
        raise Top1DataError(
            "strict dialogue-quality audit requires distinct reviewer and labeler models"
        )
    descriptions = load_taxonomy_descriptions(taxonomy_path, candidate_names)
    taxonomy = taxonomy_prompt(descriptions)
    plans = build_dialogue_blueprints(
        candidate_names,
        target_count=int(config["target_count"]),
        intent_change_per_pair=int(config["intent_change_per_pair"]),
        seed=int(config["seed"]),
        synthesis_version=pipeline_version,
        single_turn_per_candidate=int(config.get("single_turn_per_candidate", 0)),
        single_turn_confusions=config.get("single_turn_confusions"),
        single_turn_axis_confusions=config.get("single_turn_axis_confusions"),
        multi_turn_user_counts=tuple(config.get("multi_turn_user_counts", (3, 4, 5))),
        content_axes=config.get("content_axes"),
        content_axis_allowed_phenomena=config.get(
            "content_axis_allowed_phenomena"
        ),
    )
    base_url, api_key = load_api_credentials(args.credentials_file)
    _prepare_run(
        output_directory=output_directory,
        pipeline_version=pipeline_version,
        config=config,
        candidate_path=candidate_path,
        taxonomy_path=taxonomy_path,
        endpoint=base_url,
        plans=plans,
        taxonomy=taxonomy,
        implementation_paths={
            "src/top1_data_gen/cli.py": Path(__file__).resolve(),
            "src/top1_data_gen/synthesis.py": (
                Path(__file__).resolve().with_name("synthesis.py")
            ),
        },
    )
    print(f"[synthesis] planned {len(plans)} controlled dialogues: {output_directory}")
    if args.plan_only:
        print("[synthesis] plan-only mode completed; no model calls were made")
        return

    raw_directory = output_directory / "raw"
    attempt_path = output_directory / "attempts.jsonl"
    accepted_path = output_directory / "accepted_records.jsonl"
    directness_path = output_directory / "directness_records.jsonl"
    invalidation_path = output_directory / "invalidated_records.jsonl"
    train_path = output_directory / "train.jsonl"
    rejected_path = output_directory / "rejected.jsonl"
    summary_path = output_directory / "summary.json"
    accepted = _load_accepted(accepted_path, invalidation_path)
    attempt_counts = _load_attempt_counts(attempt_path)
    plan_by_id = {plan.scenario_id: plan for plan in plans}
    active_plans = _select_active_plans(
        plans,
        scenario_limit=args.scenario_limit,
        axis_pilot_per_axis=args.axis_pilot_per_axis,
    )
    active_scenario_ids = {plan.scenario_id for plan in active_plans}
    if args.scenario_limit is not None or args.axis_pilot_per_axis is not None:
        scope_description = (
            f"first {args.scenario_limit} plans"
            if args.scenario_limit is not None
            else f"{args.axis_pilot_per_axis} per target content axis"
        )
        print(
            f"[synthesis] invocation scope ({scope_description}): "
            f"{len(active_scenario_ids)}/{len(plans)} scenarios",
            flush=True,
        )
    unknown_accepted = set(accepted) - set(plan_by_id)
    if unknown_accepted:
        raise Top1DataError("accepted records contain scenarios outside the immutable plan")
    client = OpenAICompatibleClient(
        base_url=base_url,
        api_key=api_key,
        timeout_seconds=float(config["request_timeout_seconds"]),
        request_attempts=int(config["request_attempts"]),
    )
    accepted = _audit_existing_directness(
        accepted=accepted,
        directness_path=directness_path,
        invalidation_path=invalidation_path,
        models=(str(config["reviewer_model"]), str(config["labeler_model"])),
        config=runtime_config,
        client=client,
        raw_directory=raw_directory,
    )
    accepted_conversations = {
        _canonical_messages(row["messages"]): scenario_id
        for scenario_id, row in accepted.items()
    }

    while True:
        pending = [
            plan
            for plan in plans
            if plan.scenario_id in active_scenario_ids
            and plan.scenario_id not in accepted
            and attempt_counts[plan.scenario_id]
            < int(runtime_config["max_sample_attempts"])
        ]
        if not pending:
            break
        attempt_numbers = {
            plan.scenario_id: attempt_counts[plan.scenario_id] + 1 for plan in pending
        }
        round_number = min(attempt_numbers.values())
        print(
            f"[synthesis] attempt round {round_number}: {len(pending)} pending, "
            f"{len(accepted)}/{len(plans)} accepted"
        )
        generated, generation_errors = _generate_attempt(
            blueprints=pending,
            attempt_numbers=attempt_numbers,
            taxonomy=taxonomy,
            candidate_names=candidate_names,
            config=runtime_config,
            client=client,
            raw_directory=raw_directory,
        )
        samples = list(generated.values())
        labeler, labeler_errors = _judge_attempt(
            stage="labeler",
            samples=samples,
            taxonomy=taxonomy,
            candidate_names=candidate_names,
            model=str(config["labeler_model"]),
            config=runtime_config,
            client=client,
            raw_directory=raw_directory,
        )
        reviewer, reviewer_errors = _judge_attempt(
            stage="reviewer",
            samples=samples,
            taxonomy=taxonomy,
            candidate_names=candidate_names,
            model=str(config["reviewer_model"]),
            config=runtime_config,
            client=client,
            raw_directory=raw_directory,
        )
        dialogue_quality: dict[str, dict[str, Any]] = {}
        dialogue_quality_errors = {
            str(sample["scenario_id"]): [] for sample in samples
        }
        if bool(runtime_config.get("require_strict_dialogue_quality", False)):
            dialogue_quality, dialogue_quality_errors = _dialogue_quality_attempt(
                samples=samples,
                models=(
                    str(config["reviewer_model"]),
                    str(config["labeler_model"]),
                ),
                config=runtime_config,
                client=client,
                raw_directory=raw_directory,
            )
        intent_change_samples = [
            sample
            for sample in samples
            if plan_by_id[str(sample["scenario_id"])].phenomenon == "intent_change"
        ]
        directness, directness_errors = _directness_attempt(
            samples=intent_change_samples,
            models=(str(config["reviewer_model"]), str(config["labeler_model"])),
            config=runtime_config,
            client=client,
            raw_directory=raw_directory,
        )
        single_turn_samples = []
        for sample in samples:
            blueprint = plan_by_id[str(sample["scenario_id"])]
            if blueprint.phenomenon != "single_turn":
                continue
            target_axis_definitions = runtime_config.get(
                "content_axis_definitions", {}
            )
            target_axis_definition = None
            if isinstance(target_axis_definitions, Mapping):
                candidate_axis_definitions = target_axis_definitions.get(
                    blueprint.target_candidate_name
                )
                if isinstance(candidate_axis_definitions, Mapping):
                    target_axis_definition = candidate_axis_definitions.get(
                        blueprint.content_axis
                    )
            single_turn_samples.append(
                {
                    **sample,
                    "target_candidate_name": blueprint.target_candidate_name,
                    "contrast_candidate_name": blueprint.contrast_candidate_name,
                    **(
                        {
                            "content_axis": blueprint.content_axis,
                            "content_axis_definition": target_axis_definition,
                        }
                        if blueprint.content_axis is not None
                        and isinstance(target_axis_definition, str)
                        else {}
                    ),
                }
            )
        contrast, contrast_errors = _contrast_attempt(
            samples=single_turn_samples,
            taxonomy=taxonomy,
            models=(str(config["reviewer_model"]), str(config["labeler_model"])),
            config=runtime_config,
            client=client,
            raw_directory=raw_directory,
        )
        plan_fidelity: dict[str, dict[str, Any]] = {}
        plan_fidelity_errors = {str(sample["scenario_id"]): [] for sample in samples}
        if bool(runtime_config.get("require_plan_fidelity", False)):
            plan_fidelity_samples = []
            for sample in samples:
                blueprint = plan_by_id[str(sample["scenario_id"])]
                plan_fidelity_samples.append(
                    {
                        **sample,
                        "target_candidate_name": blueprint.target_candidate_name,
                        "source_candidate_name": blueprint.source_candidate_name,
                        "planned_phenomenon": blueprint.phenomenon,
                        "content_axis": blueprint.content_axis,
                        "source_content_axis": blueprint.source_content_axis,
                    }
                )
            plan_fidelity, plan_fidelity_errors = _plan_fidelity_attempt(
                samples=plan_fidelity_samples,
                taxonomy=taxonomy,
                models=_plan_fidelity_models(runtime_config),
                config=runtime_config,
                client=client,
                raw_directory=raw_directory,
            )
        _append_jsonl(
            directness_path,
            (
                {
                    "timestamp": _now(),
                    "audit_version": DIRECTNESS_AUDIT_VERSION,
                    "scenario_id": str(sample["scenario_id"]),
                    "attempt": int(sample["attempt"]),
                    "source": "generation_attempt",
                    "passed": (
                        not directness_errors[str(sample["scenario_id"])]
                        and directness.get(str(sample["scenario_id"]), {}).get(
                            "contains_only_new_request"
                        )
                        is True
                        and directness.get(str(sample["scenario_id"]), {}).get(
                            "references_previous_exchange"
                        )
                        is False
                        and directness.get(str(sample["scenario_id"]), {}).get(
                            "uses_transition_or_acknowledgment"
                        )
                        is False
                        and directness.get(str(sample["scenario_id"]), {}).get(
                            "direct_final_request"
                        )
                        is True
                        and directness.get(str(sample["scenario_id"]), {}).get(
                            "has_switch_meta_language"
                        )
                        is False
                    ),
                    "errors": directness_errors[str(sample["scenario_id"])],
                    "audit": directness.get(str(sample["scenario_id"])),
                }
                for sample in intent_change_samples
            ),
        )

        attempt_records: list[dict[str, Any]] = []
        newly_accepted: list[dict[str, Any]] = []
        round_conversations: dict[tuple[tuple[str, str], ...], str] = {}
        for blueprint in pending:
            scenario_id = blueprint.scenario_id
            reasons = list(generation_errors[scenario_id])
            sample = generated.get(scenario_id)
            labeler_judgment = labeler.get(scenario_id)
            reviewer_judgment = reviewer.get(scenario_id)
            directness_judgment = directness.get(scenario_id)
            contrast_judgment = contrast.get(scenario_id)
            plan_fidelity_judgment = plan_fidelity.get(scenario_id)
            dialogue_quality_judgment = dialogue_quality.get(scenario_id)
            if sample is not None:
                reasons.extend(labeler_errors[scenario_id])
                reasons.extend(reviewer_errors[scenario_id])
                if bool(
                    runtime_config.get("require_strict_dialogue_quality", False)
                ):
                    reasons.extend(dialogue_quality_errors[scenario_id])
                if blueprint.phenomenon == "intent_change":
                    reasons.extend(directness_errors[scenario_id])
                if blueprint.phenomenon == "single_turn":
                    reasons.extend(contrast_errors[scenario_id])
                if bool(runtime_config.get("require_plan_fidelity", False)):
                    reasons.extend(plan_fidelity_errors[scenario_id])
                reasons.extend(
                    acceptance_reasons(
                        blueprint,
                        labeler_judgment,
                        reviewer_judgment,
                        directness_judgment,
                        contrast_judgment,
                        plan_fidelity_judgment,
                        dialogue_quality_judgment,
                        quality_fields=_quality_fields(runtime_config),
                        require_plan_fidelity=bool(
                            runtime_config.get("require_plan_fidelity", False)
                        ),
                        require_observed_axis_match=bool(
                            runtime_config.get("require_observed_axis_match", False)
                        ),
                        require_dialogue_quality=bool(
                            runtime_config.get(
                                "require_strict_dialogue_quality", False
                            )
                        ),
                        require_empty_judgment_issues=bool(
                            runtime_config.get(
                                "require_empty_judgment_issues", False
                            )
                        ),
                        require_both_judges_plan_match=bool(
                            runtime_config.get(
                                "require_strict_dialogue_quality", False
                            )
                        )
                        or bool(
                            runtime_config.get(
                                "require_empty_judgment_issues", False
                            )
                        ),
                        require_contrast_link_natural=bool(
                            runtime_config.get(
                                "require_strict_dialogue_quality", False
                            )
                        ),
                    )
                )
                canonical = _canonical_messages(sample["messages"])
                if canonical in accepted_conversations or canonical in round_conversations:
                    reasons.append("duplicate_conversation")
            reasons = list(dict.fromkeys(reasons))
            status = "accepted" if not reasons and sample is not None else "rejected"
            attempt_record: dict[str, Any] = {
                "timestamp": _now(),
                "scenario_id": scenario_id,
                "attempt": attempt_numbers[scenario_id],
                "status": status,
                "reasons": reasons,
                "planned_target_candidate_name": blueprint.target_candidate_name,
                "planned_source_candidate_name": blueprint.source_candidate_name,
                "planned_phenomenon": blueprint.phenomenon,
                "labeler": labeler_judgment,
                "reviewer": reviewer_judgment,
                "directness": directness_judgment,
            }
            if bool(runtime_config.get("require_strict_dialogue_quality", False)):
                attempt_record["dialogue_quality"] = dialogue_quality_judgment
                attempt_record[
                    "dialogue_quality_audit_version"
                ] = DIALOGUE_QUALITY_AUDIT_VERSION
            if blueprint.phenomenon == "single_turn":
                attempt_record["contrast"] = contrast_judgment
            if bool(runtime_config.get("require_plan_fidelity", False)):
                attempt_record["plan_fidelity"] = plan_fidelity_judgment
            if sample is not None:
                attempt_record["messages"] = sample["messages"]
            attempt_records.append(attempt_record)
            attempt_counts[scenario_id] = attempt_numbers[scenario_id]
            if status != "accepted" or sample is None:
                continue
            row: dict[str, Any] = {
                "id": scenario_id,
                "dataset_version": pipeline_version,
                "source_type": (
                    "llm_controlled_dialogue"
                    if int(config.get("single_turn_per_candidate", 0)) > 0
                    else "llm_controlled_multiturn"
                ),
                "scenario_id": scenario_id,
                "conversation_phenomenon": blueprint.phenomenon,
                "messages": sample["messages"],
                "target_candidate_name": blueprint.target_candidate_name,
                "synthesis": {
                    "blueprint_seed": blueprint.seed,
                    "attempt": attempt_numbers[scenario_id],
                    "generator_model": config["generator_model"],
                    "labeler_model": config["labeler_model"],
                    "reviewer_model": config["reviewer_model"],
                    "labeler_predicted_candidate_name": labeler_judgment[
                        "predicted_candidate_name"
                    ],
                    "reviewer_predicted_candidate_name": reviewer_judgment[
                        "predicted_candidate_name"
                    ],
                    "directness_audit": directness_judgment,
                },
            }
            if bool(runtime_config.get("require_plan_fidelity", False)):
                row["synthesis"][
                    "plan_fidelity_audit_version"
                ] = (
                    OBSERVED_AXIS_PLAN_FIDELITY_AUDIT_VERSION
                    if bool(
                        runtime_config.get("require_observed_axis_match", False)
                    )
                    else (
                        PLAN_FIDELITY_CONSENSUS_AUDIT_VERSION
                        if isinstance(plan_fidelity_judgment, Mapping)
                        and isinstance(
                            plan_fidelity_judgment.get("model_audits"), list
                        )
                        else PLAN_FIDELITY_AUDIT_VERSION
                    )
                )
                row["synthesis"]["plan_fidelity_audit"] = plan_fidelity_judgment
            if bool(runtime_config.get("require_strict_dialogue_quality", False)):
                row["synthesis"][
                    "dialogue_quality_audit_version"
                ] = DIALOGUE_QUALITY_AUDIT_VERSION
                row["synthesis"][
                    "dialogue_quality_audit"
                ] = dialogue_quality_judgment
            if blueprint.phenomenon == "single_turn":
                row["synthesis"]["contrast_audit_version"] = (
                    CONTRAST_NATURAL_LINK_AUDIT_VERSION
                    if bool(
                        runtime_config.get(
                            "require_strict_dialogue_quality", False
                        )
                    )
                    else CONTRAST_AUDIT_VERSION
                )
                row["synthesis"]["contrast_audit"] = contrast_judgment
            if blueprint.source_candidate_name is not None:
                row["source_candidate_name"] = blueprint.source_candidate_name
            if blueprint.contrast_candidate_name is not None:
                row["contrast_candidate_name"] = blueprint.contrast_candidate_name
            if blueprint.content_axis is not None:
                row["content_axis"] = blueprint.content_axis
            if blueprint.source_content_axis is not None:
                row["source_content_axis"] = blueprint.source_content_axis
            newly_accepted.append(row)
            canonical = _canonical_messages(sample["messages"])
            round_conversations[canonical] = scenario_id

        _append_jsonl(attempt_path, attempt_records)
        _append_jsonl(accepted_path, newly_accepted)
        for row in newly_accepted:
            scenario_id = str(row["scenario_id"])
            accepted[scenario_id] = row
            accepted_conversations[_canonical_messages(row["messages"])] = scenario_id
        ordered_rows = [accepted[plan.scenario_id] for plan in plans if plan.scenario_id in accepted]
        if ordered_rows:
            validate_training_rows(ordered_rows, candidate_names, source=pipeline_version)
        write_jsonl(train_path, ordered_rows)
        _write_summary(
            path=summary_path,
            pipeline_version=pipeline_version,
            plans=plans,
            accepted=accepted,
            attempt_path=attempt_path,
            raw_directory=raw_directory,
            train_path=train_path,
            complete=len(accepted) == len(plans),
        )
        print(
            f"[synthesis] round completed: +{len(newly_accepted)} accepted; "
            f"total {len(accepted)}/{len(plans)}"
        )

    unresolved = [plan for plan in plans if plan.scenario_id not in accepted]
    active_unresolved = [
        plan for plan in unresolved if plan.scenario_id in active_scenario_ids
    ]
    ordered_rows = [accepted[plan.scenario_id] for plan in plans if plan.scenario_id in accepted]
    if ordered_rows:
        validate_training_rows(ordered_rows, candidate_names, source=pipeline_version)
    write_jsonl(train_path, ordered_rows)
    last_rejection: dict[str, dict[str, Any]] = {}
    if attempt_path.is_file():
        for row in read_jsonl(attempt_path):
            if row.get("status") != "accepted":
                last_rejection[str(row["scenario_id"])] = row
    write_jsonl(
        rejected_path,
        [last_rejection[plan.scenario_id] for plan in unresolved if plan.scenario_id in last_rejection],
    )
    _write_summary(
        path=summary_path,
        pipeline_version=pipeline_version,
        plans=plans,
        accepted=accepted,
        attempt_path=attempt_path,
        raw_directory=raw_directory,
        train_path=train_path,
        complete=not unresolved,
    )
    print(f"[synthesis] training data: {train_path}")
    print(f"[synthesis] quality summary: {summary_path}")
    if active_unresolved and not args.allow_partial:
        raise RuntimeError(
            f"quality gates exhausted for {len(active_unresolved)} active scenarios; "
            f"see {rejected_path}"
        )


if __name__ == "__main__":
    main()
