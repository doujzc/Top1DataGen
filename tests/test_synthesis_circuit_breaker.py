from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from top1_data_gen.synthesis import DialogueBlueprint, ModelCall
from top1_data_gen.data import Top1DataError, read_jsonl
from top1_data_gen import cli as generator


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
V2_CONFIG = REPOSITORY_ROOT / "configs/top1_synthesis_v2.json"
V2_CANDIDATES = REPOSITORY_ROOT / "configs/top1_candidates_v2.json"
V2_TAXONOMY = REPOSITORY_ROOT / "data_top1/top1_labeldesc_v2.jsonl"


class _ExhaustedClient:
    def __init__(self) -> None:
        self.calls = 0

    def chat_json(self, **_: object) -> None:
        self.calls += 1
        raise TimeoutError("simulated client retries exhausted")


class _SuccessfulClient:
    def __init__(self) -> None:
        self.calls = 0

    def chat_json(self, **_: object) -> ModelCall:
        self.calls += 1
        return ModelCall("{}", {}, "stop", 0.0)


class SynthesisCircuitBreakerTests(unittest.TestCase):
    def test_stage_failure_preserves_raw_while_legacy_default_does_not_abort(
        self,
    ) -> None:
        items = [{"id": "case-a", "attempt": 1}, {"id": "case-b", "attempt": 1}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = root / "strict.jsonl"
            client = _ExhaustedClient()
            with self.assertRaises(generator.StageAPICircuitBreakerError) as raised:
                generator._run_model_batches(
                    stage="reviewer",
                    batches=[items],
                    model="reviewer-model",
                    client=client,
                    max_workers=1,
                    temperature=0.0,
                    max_tokens=100,
                    build_messages=lambda _: [
                        {"role": "user", "content": "audit this batch"}
                    ],
                    item_id=lambda item: str(item["id"]),
                    item_attempt=lambda item: int(item["attempt"]),
                    raw_path=raw_path,
                    abort_on_api_failure=True,
                )

            error = raised.exception
            self.assertEqual(error.stage, "reviewer")
            self.assertEqual(error.failed_scenario_ids, ("case-a", "case-b"))
            self.assertEqual(error.failed_batches, 1)
            self.assertIn("failed_scenarios=2", str(error))
            self.assertIn("sample attempts were not consumed", str(error))
            strict_records = read_jsonl(raw_path)
            self.assertEqual(len(strict_records), 1)
            self.assertEqual(strict_records[0]["status"], "failed")
            self.assertEqual(strict_records[0]["scenario_ids"], ["case-a", "case-b"])

            legacy_path = root / "legacy.jsonl"
            legacy_records = generator._run_model_batches(
                stage="reviewer",
                batches=[items],
                model="reviewer-model",
                client=_ExhaustedClient(),
                max_workers=1,
                temperature=0.0,
                max_tokens=100,
                build_messages=lambda _: [
                    {"role": "user", "content": "audit this batch"}
                ],
                item_id=lambda item: str(item["id"]),
                item_attempt=lambda item: int(item["attempt"]),
                raw_path=legacy_path,
            )
            self.assertEqual(legacy_records[0]["status"], "failed")

    def test_resume_reuses_completed_raw_and_retries_only_failed_batches(self) -> None:
        items = [{"id": "case-a", "attempt": 1}, {"id": "case-b", "attempt": 1}]

        class SuccessThenFailure:
            def __init__(self) -> None:
                self.calls = 0

            def chat_json(self, **_: object) -> ModelCall:
                self.calls += 1
                if self.calls == 2:
                    raise ConnectionError("simulated service reset after retries")
                return ModelCall("{}", {}, "stop", 0.0)

        def run(raw_path: Path, client: object) -> list[dict[str, object]]:
            return generator._run_model_batches(
                stage="labeler",
                batches=[[items[0]], [items[1]]],
                model="labeler-model",
                client=client,
                max_workers=1,
                temperature=0.0,
                max_tokens=100,
                build_messages=lambda batch: [
                    {"role": "user", "content": str(batch[0]["id"])}
                ],
                item_id=lambda item: str(item["id"]),
                item_attempt=lambda item: int(item["attempt"]),
                raw_path=raw_path,
                abort_on_api_failure=True,
            )

        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "labeler.jsonl"
            first_client = SuccessThenFailure()
            with self.assertRaises(generator.StageAPICircuitBreakerError):
                run(raw_path, first_client)
            self.assertEqual(first_client.calls, 2)

            resumed_client = _SuccessfulClient()
            resumed = run(raw_path, resumed_client)
            self.assertEqual(resumed_client.calls, 1)
            self.assertEqual(len(resumed), 2)
            self.assertEqual(sum(bool(row.get("cache_hit")) for row in resumed), 1)
            self.assertEqual(
                [row["status"] for row in read_jsonl(raw_path)],
                ["completed", "failed", "completed"],
            )

    def test_cache_never_crosses_sample_attempts_for_the_same_request_hash(
        self,
    ) -> None:
        messages = [{"role": "user", "content": "same request"}]

        def run(
            raw_path: Path,
            *,
            attempt: int,
            client: _SuccessfulClient,
        ) -> list[dict[str, object]]:
            item = {"id": "case", "attempt": attempt}
            return generator._run_model_batches(
                stage="generation",
                batches=[[item]],
                model="generator-model",
                client=client,
                max_workers=1,
                temperature=0.0,
                max_tokens=100,
                build_messages=lambda _: messages,
                item_id=lambda value: str(value["id"]),
                item_attempt=lambda value: int(value["attempt"]),
                raw_path=raw_path,
                abort_on_api_failure=True,
            )

        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "generation.jsonl"
            first_client = _SuccessfulClient()
            run(raw_path, attempt=1, client=first_client)
            self.assertEqual(first_client.calls, 1)

            second_client = _SuccessfulClient()
            second = run(raw_path, attempt=2, client=second_client)
            self.assertEqual(second_client.calls, 1)
            self.assertFalse(any(row.get("cache_hit") for row in second))
            self.assertEqual(
                [row["sample_attempts"] for row in read_jsonl(raw_path)],
                [{"case": 1}, {"case": 2}],
            )

    def test_legacy_raw_without_attempt_provenance_is_never_reused(self) -> None:
        messages = [{"role": "user", "content": "legacy request"}]
        request_sha256 = generator._request_hash(messages)
        for attempt in (1, 2):
            with self.subTest(attempt=attempt), tempfile.TemporaryDirectory() as directory:
                raw_path = Path(directory) / "legacy.jsonl"
                generator._append_jsonl(
                    raw_path,
                    (
                        {
                            "status": "completed",
                            "stage": "labeler",
                            "model": "labeler-model",
                            "scenario_ids": ["case"],
                            "request_sha256": request_sha256,
                            "content": "{}",
                        },
                    ),
                )
                item = {"id": "case", "attempt": attempt}
                client = _SuccessfulClient()
                records = generator._run_model_batches(
                    stage="labeler",
                    batches=[[item]],
                    model="labeler-model",
                    client=client,
                    max_workers=1,
                    temperature=0.0,
                    max_tokens=100,
                    build_messages=lambda _: messages,
                    item_id=lambda value: str(value["id"]),
                    item_attempt=lambda value: int(value["attempt"]),
                    raw_path=raw_path,
                    abort_on_api_failure=True,
                )
                self.assertEqual(client.calls, 1)
                self.assertFalse(any(row.get("cache_hit") for row in records))
                self.assertEqual(len(read_jsonl(raw_path)), 2)

    def test_v2_audit_api_failure_does_not_commit_or_enter_next_round(self) -> None:
        config = json.loads(V2_CONFIG.read_text(encoding="utf-8"))
        self.assertIs(config["abort_round_on_api_failure"], True)
        blueprint = DialogueBlueprint(
            scenario_id="circuit-breaker-integration",
            phenomenon="progressive_reveal",
            target_candidate_name="StockQuery",
            source_candidate_name=None,
            user_turn_count=2,
            seed=1,
            content_axis="current_price_and_change",
        )
        sample = {
            "scenario_id": blueprint.scenario_id,
            "messages": [
                {"role": "user", "content": "我想查一只股票。"},
                {"role": "assistant", "content": "请问是哪只？"},
                {"role": "user", "content": "查一下贵州茅台今天的价格。"},
            ],
            "attempt": 1,
        }
        judgment = {
            "scenario_id": blueprint.scenario_id,
            "predicted_candidate_name": "StockQuery",
            "quality": {},
        }
        client = _ExhaustedClient()

        def generated_attempt(
            **_: object,
        ) -> tuple[dict[str, dict[str, object]], dict[str, list[str]]]:
            return ({blueprint.scenario_id: dict(sample)}, {blueprint.scenario_id: []})

        def judged_attempt(
            **_: object,
        ) -> tuple[dict[str, dict[str, object]], dict[str, list[str]]]:
            return ({blueprint.scenario_id: dict(judgment)}, {blueprint.scenario_id: []})

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with (
                patch.object(
                    generator,
                    "build_dialogue_blueprints",
                    return_value=[blueprint],
                ),
                patch.object(
                    generator,
                    "load_api_credentials",
                    return_value=("https://example.invalid/v1", "secret"),
                ),
                patch.object(
                    generator,
                    "OpenAICompatibleClient",
                    return_value=client,
                ),
                patch.object(
                    generator,
                    "_audit_existing_directness",
                    side_effect=lambda **kwargs: kwargs["accepted"],
                ),
                patch.object(
                    generator,
                    "_generate_attempt",
                    side_effect=generated_attempt,
                ) as generate,
                patch.object(
                    generator,
                    "_judge_attempt",
                    side_effect=judged_attempt,
                ),
                patch.object(
                    generator,
                    "_dialogue_quality_attempt",
                    return_value=({}, {blueprint.scenario_id: []}),
                ),
                patch.object(generator, "_directness_attempt", return_value=({}, {})),
                patch.object(generator, "_contrast_attempt", return_value=({}, {})),
            ):
                with self.assertRaisesRegex(
                    generator.StageAPICircuitBreakerError,
                    r"stage=plan_fidelity, failed_scenarios=1",
                ):
                    generator.main(
                        [
                            "--config",
                            str(V2_CONFIG),
                            "--candidate-registry",
                            str(V2_CANDIDATES),
                            "--taxonomy-data",
                            str(V2_TAXONOMY),
                            "--output-dir",
                            str(output),
                        ]
                    )

            self.assertEqual(generate.call_count, 1)
            self.assertEqual(client.calls, 1)
            for uncommitted_name in (
                "attempts.jsonl",
                "accepted_records.jsonl",
                "rejected.jsonl",
                "train.jsonl",
                "summary.json",
                "directness_records.jsonl",
            ):
                self.assertFalse((output / uncommitted_name).exists())
            raw_records = read_jsonl(
                output / "raw" / "plan_fidelity_responses.jsonl"
            )
            self.assertEqual(len(raw_records), 1)
            self.assertEqual(raw_records[0]["stage"], "plan_fidelity")
            self.assertEqual(raw_records[0]["status"], "failed")

    def test_circuit_breaker_config_requires_a_boolean(self) -> None:
        self.assertFalse(generator._abort_round_on_api_failure_enabled({}))
        with self.assertRaisesRegex(Top1DataError, "must be a boolean"):
            generator._abort_round_on_api_failure_enabled(
                {"abort_round_on_api_failure": "true"}
            )


if __name__ == "__main__":
    unittest.main()
