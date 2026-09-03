from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import unittest

from top1_data_gen.synthesis import build_dialogue_blueprints
from top1_data_gen.data import Top1DataError, load_candidate_names


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "configs/top1_synthesis_v2.json"
CANDIDATE_PATH = REPOSITORY_ROOT / "configs/top1_candidates_v2.json"


class AxisPhenomenonPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        cls.candidates = load_candidate_names(CANDIDATE_PATH)
        cls.plans = cls._build_v2_plan()

    @classmethod
    def _build_v2_plan(cls):
        config = cls.config
        return build_dialogue_blueprints(
            cls.candidates,
            target_count=config["target_count"],
            intent_change_per_pair=config["intent_change_per_pair"],
            seed=config["seed"],
            synthesis_version=config["pipeline_version"],
            single_turn_per_candidate=config["single_turn_per_candidate"],
            single_turn_axis_confusions=config["single_turn_axis_confusions"],
            multi_turn_user_counts=config["multi_turn_user_counts"],
            content_axes=config["content_axes"],
            content_axis_allowed_phenomena=config[
                "content_axis_allowed_phenomena"
            ],
        )

    def test_v2_restrictions_preserve_candidate_and_phenomenon_quotas(self) -> None:
        expected_phenomena = Counter(
            {
                "single_turn": 30,
                "intent_change": 42,
                "progressive_reveal": 10,
                "contextual_follow_up": 7,
                "clarification_revision": 5,
                "assistant_distractor": 4,
                "rambling": 2,
            }
        )
        for candidate in self.candidates:
            candidate_plans = [
                plan
                for plan in self.plans
                if plan.target_candidate_name == candidate
            ]
            self.assertEqual(len(candidate_plans), 100)
            self.assertEqual(
                Counter(plan.phenomenon for plan in candidate_plans),
                expected_phenomena,
            )

    def test_every_target_axis_is_covered_and_balanced(self) -> None:
        target_axis_groups = {
            (plan.target_candidate_name, plan.content_axis) for plan in self.plans
        }
        expected_axis_count = sum(
            len(axes) for axes in self.config["content_axes"].values()
        )
        self.assertEqual(len(target_axis_groups), expected_axis_count)
        for candidate in self.candidates:
            counts = Counter(
                plan.content_axis
                for plan in self.plans
                if plan.target_candidate_name == candidate
            )
            self.assertEqual(set(counts), set(self.config["content_axes"][candidate]))
            self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_insufficient_input_axis_is_single_turn_only(self) -> None:
        axis = "insufficient_or_unintelligible_input_without_anaphora"
        target_rows = [plan for plan in self.plans if plan.content_axis == axis]

        no_available_axis_count = len(self.config["content_axes"]["NoAvailable"])
        expected_low = 100 // no_available_axis_count
        self.assertIn(len(target_rows), {expected_low, expected_low + 1})
        self.assertTrue(all(plan.phenomenon == "single_turn" for plan in target_rows))
        self.assertFalse(any(plan.source_content_axis == axis for plan in self.plans))

    def test_atomic_chitchat_axes_avoid_same_intent_multiturn(self) -> None:
        greeting = "current_turn_greeting_or_wakeup"
        farewell = "self_contained_appreciation_or_farewell"
        target_rows = [
            plan for plan in self.plans if plan.content_axis in {greeting, farewell}
        ]

        self.assertEqual({plan.content_axis for plan in target_rows}, {greeting, farewell})
        self.assertTrue(
            all(
                plan.phenomenon in {"single_turn", "intent_change"}
                for plan in target_rows
            )
        )
        self.assertTrue(
            all(
                plan.phenomenon == "single_turn"
                for plan in target_rows
                if plan.content_axis == farewell
            )
        )
        self.assertTrue(any(plan.source_content_axis == greeting for plan in self.plans))
        self.assertFalse(any(plan.source_content_axis == farewell for plan in self.plans))

    def test_axis_assignment_is_deterministic(self) -> None:
        repeated = self._build_v2_plan()
        first = {
            plan.scenario_id: (plan.content_axis, plan.source_content_axis)
            for plan in self.plans
        }
        second = {
            plan.scenario_id: (plan.content_axis, plan.source_content_axis)
            for plan in repeated
        }
        self.assertEqual(first, second)

    def test_invalid_or_infeasible_restrictions_fail_closed(self) -> None:
        base = {
            "target_count": 8,
            "intent_change_per_pair": 1,
            "content_axes": {"A": ["a1", "a2"], "B": ["b1", "b2"]},
        }
        with self.assertRaisesRegex(Top1DataError, "unknown candidates"):
            build_dialogue_blueprints(
                ("A", "B"),
                **base,
                content_axis_allowed_phenomena={"C": {}},
            )
        with self.assertRaisesRegex(Top1DataError, "unknown axes"):
            build_dialogue_blueprints(
                ("A", "B"),
                **base,
                content_axis_allowed_phenomena={"A": {"missing": ["single_turn"]}},
            )
        with self.assertRaisesRegex(Top1DataError, "invalid phenomena"):
            build_dialogue_blueprints(
                ("A", "B"),
                **base,
                content_axis_allowed_phenomena={"A": {"a1": ["unknown"]}},
            )
        with self.assertRaisesRegex(Top1DataError, "does not cover content axes"):
            build_dialogue_blueprints(
                ("A", "B"),
                **base,
                content_axis_allowed_phenomena={"A": {"a1": ["single_turn"]}},
            )


if __name__ == "__main__":
    unittest.main()
