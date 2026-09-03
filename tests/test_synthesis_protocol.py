from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from top1_data_gen.synthesis import (
    DIALOGUE_QUALITY_AUDIT_FIELDS,
    QUALITY_FIELDS,
    STRICT_JSON_SCHEMA_PROTOCOL,
    DialogueBlueprint,
    OpenAICompatibleClient,
    contrast_response_schema,
    dialogue_quality_response_schema,
    directness_response_schema,
    generation_response_schema,
    judgment_response_schema,
    load_api_credentials,
    parse_directness_audits,
    parse_json_object,
    plan_fidelity_response_schema,
)
from top1_data_gen.data import Top1DataError
from top1_data_gen.cli import _request_hash


def _completion(
    finish_reason: str,
    content: str,
) -> subprocess.CompletedProcess[bytes]:
    payload = {
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {"content": content},
            }
        ],
        "usage": {"completion_tokens": 12},
    }
    return subprocess.CompletedProcess(
        args=("curl",),
        returncode=0,
        stdout=json.dumps(payload).encode("utf-8"),
        stderr=b"",
    )


def _request_body_from_curl_config(config: bytes) -> dict[str, object]:
    marker = 'data-binary = "@'
    for line in config.decode("utf-8").splitlines():
        if line.startswith(marker) and line.endswith('"'):
            body_path = line[len(marker) : -1]
            return json.loads(Path(body_path).read_text(encoding="utf-8"))
    raise AssertionError("curl config has no request body")


class SynthesisProtocolTests(unittest.TestCase):
    def test_credentials_require_https(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "credentials"
            path.write_text(
                "base_url:http://198.51.100.1:8080/v1\napi_key:secret\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(Top1DataError, "HTTPS"):
                load_api_credentials(path)

            path.write_text(
                "base_url:https://api.example.invalid/v1\napi_key:secret\n",
                encoding="utf-8",
            )
            self.assertEqual(
                load_api_credentials(path),
                ("https://api.example.invalid/v1", "secret"),
            )

    def test_strict_client_retries_length_and_sends_json_schema(self) -> None:
        requests: list[dict[str, object]] = []
        completions = iter(
            (
                _completion("length", '{"ok":'),
                _completion("stop", '{"ok": true}'),
            )
        )

        def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
            del args
            requests.append(_request_body_from_curl_config(kwargs["input"]))
            return next(completions)

        client = OpenAICompatibleClient(
            base_url="https://example.invalid/v1",
            api_key="secret",
            request_attempts=2,
        )
        schema = {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        }
        with (
            patch("top1_data_gen.synthesis.subprocess.run", side_effect=fake_run),
            patch("top1_data_gen.synthesis.time.sleep"),
        ):
            call = client.chat_json(
                model="model",
                messages=[{"role": "user", "content": "test"}],
                temperature=0.0,
                max_tokens=100,
                response_schema=schema,
                require_stop=True,
            )

        self.assertEqual(call.finish_reason, "stop")
        self.assertEqual(len(requests), 2)
        response_format = requests[0]["response_format"]
        self.assertEqual(response_format["type"], "json_schema")
        self.assertTrue(response_format["json_schema"]["strict"])
        self.assertEqual(response_format["json_schema"]["schema"], schema)

    def test_strict_client_never_returns_a_non_stop_completion(self) -> None:
        client = OpenAICompatibleClient(
            base_url="https://example.invalid/v1",
            api_key="secret",
            request_attempts=2,
        )
        with (
            patch(
                "top1_data_gen.synthesis.subprocess.run",
                side_effect=(
                    _completion("length", '{"ok": true}'),
                    _completion("content_filter", '{"ok": true}'),
                ),
            ),
            patch("top1_data_gen.synthesis.time.sleep"),
        ):
            with self.assertRaisesRegex(RuntimeError, "content_filter"):
                client.chat_json(
                    model="model",
                    messages=[{"role": "user", "content": "test"}],
                    temperature=0.0,
                    max_tokens=100,
                    response_schema={"type": "object"},
                    require_stop=True,
                )

    def test_strict_client_retries_fenced_json_but_legacy_stays_compatible(self) -> None:
        client = OpenAICompatibleClient(
            base_url="https://example.invalid/v1",
            api_key="secret",
            request_attempts=2,
        )
        with (
            patch(
                "top1_data_gen.synthesis.subprocess.run",
                side_effect=(
                    _completion("stop", '```json\n{"ok": true}\n```'),
                    _completion("stop", '{"ok": true}'),
                ),
            ) as run,
            patch("top1_data_gen.synthesis.time.sleep"),
        ):
            strict_call = client.chat_json(
                model="model",
                messages=[{"role": "user", "content": "test"}],
                temperature=0.0,
                max_tokens=100,
                response_schema={"type": "object"},
                require_stop=True,
            )
        self.assertEqual(strict_call.content, '{"ok": true}')
        self.assertEqual(run.call_count, 2)

        captured: list[dict[str, object]] = []

        def legacy_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
            del args
            captured.append(_request_body_from_curl_config(kwargs["input"]))
            return _completion("length", '{"ok": true}')

        with patch("top1_data_gen.synthesis.subprocess.run", side_effect=legacy_run):
            legacy_call = client.chat_json(
                model="model",
                messages=[{"role": "user", "content": "test"}],
                temperature=0.0,
                max_tokens=100,
            )
        self.assertEqual(legacy_call.finish_reason, "length")
        self.assertEqual(captured[0]["response_format"], {"type": "json_object"})

    def test_strict_parser_rejects_wrappers_and_extra_fields(self) -> None:
        wrapped = 'before {"value": true} after'
        self.assertEqual(parse_json_object(wrapped), {"value": True})
        with self.assertRaises(Top1DataError):
            parse_json_object(wrapped, strict=True)

        audit = {
            "scenario_id": "case",
            "contains_only_new_request": True,
            "references_previous_exchange": False,
            "uses_transition_or_acknowledgment": False,
            "direct_final_request": True,
            "has_switch_meta_language": False,
            "reason": "direct",
        }
        parsed, errors = parse_directness_audits(
            json.dumps({"audits": [audit], "extra": True}),
            ("case",),
            strict_envelope=True,
        )
        self.assertEqual(parsed, {})
        self.assertTrue(errors["case"])

    def test_strict_request_hash_binds_the_complete_protocol(self) -> None:
        messages = [{"role": "user", "content": "test"}]
        legacy = _request_hash(messages)
        expected_legacy = __import__("hashlib").sha256(
            json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        self.assertEqual(legacy, expected_legacy)

        schema = {"type": "object"}
        base = {
            "stage": "labeler",
            "model": "model-a",
            "temperature": 0.0,
            "max_tokens": 100,
            "response_schema": schema,
            "protocol": STRICT_JSON_SCHEMA_PROTOCOL,
            "require_stop": True,
        }
        strict = _request_hash(messages, **base)
        self.assertNotEqual(strict, legacy)
        variants = (
            {"stage": "reviewer"},
            {"model": "model-b"},
            {"temperature": 0.1},
            {"max_tokens": 101},
            {"response_schema": {"type": "array"}},
            {"protocol": "another-protocol"},
        )
        for changes in variants:
            with self.subTest(changes=changes):
                self.assertNotEqual(strict, _request_hash(messages, **(base | changes)))

    def test_all_stage_schemas_are_exact_dynamic_batch_envelopes(self) -> None:
        ids = ("case-a", "case-b")
        blueprints = (
            DialogueBlueprint("case-a", "single_turn", "A", None, 1, 1),
            DialogueBlueprint("case-b", "progressive_reveal", "B", None, 3, 2),
        )
        schemas = {
            "samples": generation_response_schema(blueprints),
            "judgments": judgment_response_schema(
                ids,
                ("A", "B"),
                quality_fields=QUALITY_FIELDS,
            ),
            "directness": directness_response_schema(ids),
            "dialogue": dialogue_quality_response_schema(ids),
            "contrast": contrast_response_schema(
                ids,
                require_natural_link=True,
            ),
            "plan": plan_fidelity_response_schema(
                ids,
                observed_axis_catalogs={
                    "case-a": {
                        "target_content_axes": ("axis-a",),
                        "source_content_axes": None,
                    },
                    "case-b": {
                        "target_content_axes": ("axis-b",),
                        "source_content_axes": ("source-axis",),
                    },
                },
            ),
        }
        for name, schema in schemas.items():
            with self.subTest(stage=name):
                self.assertFalse(schema["additionalProperties"])
                self.assertEqual(schema["required"], list(schema["properties"]))
                response_field = next(iter(schema["properties"]))
                array = schema["properties"][response_field]
                self.assertEqual(array["minItems"], 2)
                self.assertEqual(array["maxItems"], 2)
                item = array["items"]
                self.assertFalse(item["additionalProperties"])
                self.assertEqual(item["required"], list(item["properties"]))
                self.assertEqual(
                    item["properties"]["scenario_id"]["enum"],
                    list(ids),
                )

        judgment_item = schemas["judgments"]["properties"]["judgments"][
            "items"
        ]
        quality = judgment_item["properties"]["quality"]
        self.assertFalse(quality["additionalProperties"])
        self.assertEqual(set(quality["required"]), set(QUALITY_FIELDS))
        dialogue_item = schemas["dialogue"]["properties"]["audits"]["items"]
        self.assertTrue(set(DIALOGUE_QUALITY_AUDIT_FIELDS) <= set(dialogue_item["required"]))
        plan_item = schemas["plan"]["properties"]["audits"]["items"]
        self.assertEqual(plan_item["properties"]["reason"]["minLength"], 20)


if __name__ == "__main__":
    unittest.main()
