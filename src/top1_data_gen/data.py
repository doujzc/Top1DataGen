"""Minimal JSONL and Top1 dataset contracts used by synthesis."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROUTING_MODE = "candidate_name_top1"
MEMORIZATION_SOURCE_TYPE = "label_description"
MEMORIZATION_DESCRIPTION_TYPES = (
    "label_term",
    "related_term",
    "concise_definition",
    "extended_definition",
)
MAX_HISTORY_MESSAGES = 16
MAX_HISTORY_CHARACTERS = 12_000
MAX_ASSISTANT_HISTORY_CHARACTERS = 1_200
LATEST_TRUNCATION_MARKER = "\n...[当前用户消息中间内容已截断]...\n"
HISTORY_TRUNCATION_MARKER = "\n...[历史消息中间内容已截断]...\n"
ALLOWED_MESSAGE_ROLES = frozenset({"system", "user", "assistant", "tool"})


class Top1DataError(ValueError):
    """Raised when a synthesis or dataset contract is invalid."""


def sha256_file(path: str | Path) -> str:
    """Return the SHA256 digest of a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a UTF-8 JSONL file whose rows must be objects."""

    source = Path(path)
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            text = raw_line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise Top1DataError(
                    f"invalid JSON at {source}:{line_number}: {exc.msg}"
                ) from exc
            if not isinstance(row, dict):
                raise Top1DataError(
                    f"row at {source}:{line_number} must be an object"
                )
            rows.append(row)
    return rows


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Atomically write a JSON object."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Atomically write JSON objects as UTF-8 JSONL."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _nonempty_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Top1DataError(f"{field} must be a non-empty string")
    return value.strip()


def load_candidate_names(path: str | Path) -> tuple[str, ...]:
    """Load the ordered closed set of legal candidate names."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Top1DataError(f"invalid candidate registry JSON: {source}") from exc
    if not isinstance(payload, dict):
        raise Top1DataError("candidate registry must be a JSON object")
    if payload.get("routing_mode") != ROUTING_MODE:
        raise Top1DataError(
            f"candidate registry routing_mode must be {ROUTING_MODE!r}"
        )
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise Top1DataError(
            "candidate registry must contain a non-empty candidates list"
        )

    names: list[str] = []
    seen: set[str] = set()
    for index, value in enumerate(candidates):
        name = _nonempty_string(value, field=f"candidates[{index}]")
        if name in seen:
            raise Top1DataError(f"duplicate candidate name: {name!r}")
        names.append(name)
        seen.add(name)
    return tuple(names)


def _truncate_middle(content: str, limit: int, marker: str) -> str:
    if limit < 1:
        return ""
    if len(content) <= limit:
        return content
    if limit <= len(marker):
        return content[:limit]
    available = limit - len(marker)
    head_size = (available * 2) // 3
    tail_size = available - head_size
    return content[:head_size] + marker + content[-tail_size:]


def normalize_messages(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str], ...]:
    """Validate messages and retain a bounded conversation ending in user."""

    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        raise Top1DataError("messages must be a sequence")
    if not messages:
        raise Top1DataError("messages cannot be empty")

    normalized: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise Top1DataError(f"messages[{index}] must be an object")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise Top1DataError(f"messages[{index}] role/content must be strings")
        role = role.strip()
        content = content.strip()
        if role not in ALLOWED_MESSAGE_ROLES:
            raise Top1DataError(
                f"messages[{index}] has unsupported role {role!r}"
            )
        if not content:
            raise Top1DataError(f"messages[{index}].content cannot be empty")
        if role != "system":
            normalized.append({"role": role, "content": content})

    if not normalized:
        raise Top1DataError(
            "conversation has no messages after dropping system turns"
        )
    if normalized[-1]["role"] != "user":
        raise Top1DataError("the final non-system message must have role 'user'")

    current = dict(normalized[-1])
    current["content"] = _truncate_middle(
        current["content"], MAX_HISTORY_CHARACTERS, LATEST_TRUNCATION_MARKER
    )
    remaining_characters = MAX_HISTORY_CHARACTERS - len(current["content"])
    recent_history = normalized[:-1][-(MAX_HISTORY_MESSAGES - 1) :]
    kept: list[dict[str, str]] = []
    for original in reversed(recent_history):
        if remaining_characters <= 0:
            break
        message = dict(original)
        limit = remaining_characters
        if message["role"] in {"assistant", "tool"}:
            limit = min(limit, MAX_ASSISTANT_HISTORY_CHARACTERS)
        message["content"] = _truncate_middle(
            message["content"], limit, HISTORY_TRUNCATION_MARKER
        )
        if message["content"]:
            kept.append(message)
            remaining_characters -= len(message["content"])
    return tuple([*reversed(kept), current])


def messages_from_row(row: Mapping[str, Any]) -> tuple[dict[str, str], ...]:
    """Read canonical messages from one generated row."""

    if "messages" not in row:
        raise Top1DataError("training row must contain messages")
    return normalize_messages(row["messages"])


def target_candidate_name(row: Mapping[str, Any]) -> str:
    """Read the canonical candidate-name label from one generated row."""

    return _nonempty_string(
        row.get("target_candidate_name"), field="target_candidate_name"
    )


def validate_training_rows(
    rows: Sequence[Mapping[str, Any]],
    candidate_names: Iterable[str],
    *,
    source: str | Path,
) -> dict[str, Any]:
    """Validate canonical output rows and summarize their supervision."""

    if not rows:
        raise Top1DataError(f"training data is empty: {source}")
    ordered_names = tuple(candidate_names)
    legal_names = set(ordered_names)
    counts: Counter[str] = Counter()
    multi_turn = 0
    for row_number, row in enumerate(rows, start=1):
        try:
            messages = messages_from_row(row)
            name = target_candidate_name(row)
            if name not in legal_names:
                raise Top1DataError(f"unknown target candidate name: {name!r}")
        except Top1DataError as exc:
            raise Top1DataError(f"{source}:{row_number}: {exc}") from exc
        counts[name] += 1
        multi_turn += len(messages) > 1
    return {
        "rows": len(rows),
        "multi_turn_rows": multi_turn,
        "candidate_counts": {
            name: counts[name] for name in ordered_names if counts[name]
        },
    }


def validate_memorization_rows(
    rows: Sequence[Mapping[str, Any]],
    candidate_names: Sequence[str],
    *,
    source: str | Path,
) -> dict[str, Any]:
    """Validate the description-to-candidate taxonomy dataset."""

    report = validate_training_rows(rows, candidate_names, source=source)
    seen_ids: set[str] = set()
    description_counts: Counter[str] = Counter()
    candidate_description_counts: dict[str, Counter[str]] = {
        name: Counter() for name in candidate_names
    }
    for row_number, row in enumerate(rows, start=1):
        try:
            sample_id = _nonempty_string(row.get("id"), field="id")
            if sample_id in seen_ids:
                raise Top1DataError(f"duplicate id: {sample_id!r}")
            seen_ids.add(sample_id)
            source_type = _nonempty_string(row.get("source_type"), field="source_type")
            if source_type != MEMORIZATION_SOURCE_TYPE:
                raise Top1DataError(
                    f"source_type must be {MEMORIZATION_SOURCE_TYPE!r}"
                )
            description_type = _nonempty_string(
                row.get("description_type"), field="description_type"
            )
            if description_type not in MEMORIZATION_DESCRIPTION_TYPES:
                raise Top1DataError(
                    f"unsupported description_type: {description_type!r}"
                )
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) != 1:
                raise Top1DataError(
                    "memorization rows must contain exactly one user message"
                )
            message = messages[0]
            if not isinstance(message, Mapping) or message.get("role") != "user":
                raise Top1DataError(
                    "memorization rows must contain exactly one user message"
                )
            _nonempty_string(message.get("content"), field="messages[0].content")
            target = target_candidate_name(row)
        except Top1DataError as exc:
            raise Top1DataError(f"{source}:{row_number}: {exc}") from exc
        description_counts[description_type] += 1
        candidate_description_counts[target][description_type] += 1

    missing_candidates = [
        name for name in candidate_names if not report["candidate_counts"].get(name)
    ]
    if missing_candidates:
        raise Top1DataError(
            "memorization data must cover every candidate: "
            + ", ".join(missing_candidates)
        )
    return {
        **report,
        "source_type": MEMORIZATION_SOURCE_TYPE,
        "description_type_counts": {
            name: description_counts[name]
            for name in MEMORIZATION_DESCRIPTION_TYPES
        },
        "candidate_description_type_counts": {
            candidate: {
                name: candidate_description_counts[candidate][name]
                for name in MEMORIZATION_DESCRIPTION_TYPES
            }
            for candidate in candidate_names
        },
    }
