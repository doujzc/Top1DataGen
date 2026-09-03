from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import unittest

from top1_data_gen.synthesis import build_dialogue_blueprints
from top1_data_gen.data import Top1DataError, load_candidate_names


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "configs/top1_synthesis_v2.json"
CANDIDATE_PATH = REPOSITORY_ROOT / "configs/top1_candidates_v2.json"
V1_CANDIDATES = (
    "StockAdvice",
    "StockOther",
    "StockQuery",
    "ProductGeneral",
    "ProductEcommerce",
    "ChitChat",
    "NoAvailable",
)
V1_DEFAULT_PLAN_SHA256 = (
    "0114ecc9bfe4f627a61eafd41d145d817d973e6c9aa6705ebb41a02b46b6c533"
)


class AxisConfusionPlanTests(unittest.TestCase):
    def test_default_v1_plan_hash_is_unchanged(self) -> None:
        payload = [plan.to_dict() for plan in build_dialogue_blueprints(V1_CANDIDATES)]
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

        self.assertEqual(
            hashlib.sha256(serialized).hexdigest(),
            V1_DEFAULT_PLAN_SHA256,
        )

    def test_v2_axis_registry_is_exact_and_every_single_uses_its_whitelist(self) -> None:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        candidates = load_candidate_names(CANDIDATE_PATH)
        registry = config["single_turn_axis_confusions"]

        self.assertEqual(set(registry), set(candidates))
        self.assertEqual(
            sum(len(axis_map) for axis_map in registry.values()),
            101,
        )
        for candidate in candidates:
            self.assertEqual(
                set(registry[candidate]),
                set(config["content_axes"][candidate]),
            )

        kwargs = {
            "target_count": config["target_count"],
            "intent_change_per_pair": config["intent_change_per_pair"],
            "seed": config["seed"],
            "synthesis_version": config["pipeline_version"],
            "single_turn_per_candidate": config["single_turn_per_candidate"],
            "single_turn_axis_confusions": registry,
            "multi_turn_user_counts": config["multi_turn_user_counts"],
            "content_axes": config["content_axes"],
            "content_axis_allowed_phenomena": config[
                "content_axis_allowed_phenomena"
            ],
        }
        plans = build_dialogue_blueprints(candidates, **kwargs)
        repeated = build_dialogue_blueprints(candidates, **kwargs)
        self.assertEqual(
            [plan.to_dict() for plan in plans],
            [plan.to_dict() for plan in repeated],
        )

        grouped: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        singles = [plan for plan in plans if plan.phenomenon == "single_turn"]
        self.assertEqual(len(singles), 450)
        for plan in singles:
            allowed = registry[plan.target_candidate_name][str(plan.content_axis)]
            self.assertIn(plan.contrast_candidate_name, allowed)
            grouped[(plan.target_candidate_name, str(plan.content_axis))][
                str(plan.contrast_candidate_name)
            ] += 1
        for counts in grouped.values():
            self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
        for candidate, axis_map in registry.items():
            for axis, allowed in axis_map.items():
                self.assertEqual(
                    set(grouped[(candidate, axis)]),
                    set(allowed),
                )

        self.assertEqual(
            registry["NoAvailable"][
                "insufficient_or_unintelligible_input_without_anaphora"
            ],
            ["ChitChat"],
        )
        self.assertEqual(
            registry["NoAvailable"][
                "health_diagnosis_prescription_or_emergency"
            ],
            ["MedicalQuestionAnswer"],
        )

    def test_axis_confusion_registry_fails_closed(self) -> None:
        content_axes = {"A": ["a1"], "B": ["b1"]}
        valid = {"A": {"a1": ["B"]}, "B": {"b1": ["A"]}}
        kwargs = {
            "target_count": 2,
            "intent_change_per_pair": 0,
            "single_turn_per_candidate": 1,
            "content_axes": content_axes,
        }

        cases = (
            ({"A": valid["A"]}, "cover exactly the candidate registry"),
            ({**valid, "C": {"c1": ["A"]}}, "unknown candidates"),
            (
                {"A": {"missing": ["B"]}, "B": valid["B"]},
                "cover exactly the configured content axes",
            ),
            (
                {"A": {"a1": []}, "B": valid["B"]},
                "must contain non-empty candidate names",
            ),
            (
                {"A": {"a1": ["B", "B"]}, "B": valid["B"]},
                "contains duplicate candidates",
            ),
            (
                {"A": {"a1": ["A"]}, "B": valid["B"]},
                "unknown or self candidate",
            ),
            (
                {"A": {"a1": ["C"]}, "B": valid["B"]},
                "unknown or self candidate",
            ),
        )
        for configured, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                Top1DataError,
                message,
            ):
                build_dialogue_blueprints(
                    ("A", "B"),
                    **kwargs,
                    single_turn_axis_confusions=configured,
                )


if __name__ == "__main__":
    unittest.main()
