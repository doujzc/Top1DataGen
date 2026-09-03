from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import unittest

from top1_data_gen.synthesis import DialogueBlueprint, build_dialogue_blueprints
from top1_data_gen.data import Top1DataError, load_candidate_names
from top1_data_gen.cli import (
    _pilot_phenomenon_bucket,
    _select_active_plans,
    _select_axis_pilot,
    parse_args,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "configs/top1_synthesis_v2.json"
CANDIDATE_PATH = REPOSITORY_ROOT / "configs/top1_candidates_v2.json"


class AxisPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        candidates = load_candidate_names(CANDIDATE_PATH)
        cls.plans = build_dialogue_blueprints(
            candidates,
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

    def test_axis_pilot_is_deterministic_and_covers_every_target_axis(self) -> None:
        original_ids = [plan.scenario_id for plan in self.plans]
        selected = _select_axis_pilot(self.plans, 1)
        selected_again = _select_axis_pilot(self.plans, 1)
        expected_groups = {
            (plan.target_candidate_name, plan.content_axis) for plan in self.plans
        }

        self.assertEqual(
            [plan.scenario_id for plan in selected],
            [plan.scenario_id for plan in selected_again],
        )
        self.assertEqual(len(selected), len(expected_groups))
        self.assertEqual(
            Counter(
                (plan.target_candidate_name, plan.content_axis) for plan in selected
            ),
            Counter({group: 1 for group in expected_groups}),
        )
        self.assertEqual([plan.scenario_id for plan in self.plans], original_ids)

        buckets_by_target: dict[str, set[str]] = defaultdict(set)
        for plan in selected:
            buckets_by_target[plan.target_candidate_name].add(
                _pilot_phenomenon_bucket(plan)
            )
        self.assertTrue(
            all(
                buckets
                == {"single_turn", "intent_change", "non_switch_multiturn"}
                for buckets in buckets_by_target.values()
            )
        )

    def test_axis_pilot_per_axis_is_an_invocation_only_scope(self) -> None:
        pilot = _select_active_plans(
            self.plans,
            scenario_limit=None,
            axis_pilot_per_axis=2,
        )
        full = _select_active_plans(
            self.plans,
            scenario_limit=None,
            axis_pilot_per_axis=None,
        )
        counts = Counter(
            (plan.target_candidate_name, plan.content_axis) for plan in pilot
        )

        self.assertTrue(counts)
        self.assertEqual(set(counts.values()), {2})
        self.assertEqual(len({plan.scenario_id for plan in pilot}), len(pilot))
        self.assertEqual(
            [plan.scenario_id for plan in full],
            [plan.scenario_id for plan in self.plans],
        )

    def test_axis_pilot_rejects_invalid_or_incompatible_scopes(self) -> None:
        axisless = DialogueBlueprint(
            scenario_id="v1-axisless",
            phenomenon="intent_change",
            target_candidate_name="A",
            source_candidate_name="B",
            user_turn_count=2,
            seed=1,
        )
        with self.assertRaisesRegex(Top1DataError, "content_axis"):
            _select_axis_pilot((axisless,), 1)
        with self.assertRaisesRegex(Top1DataError, "positive integer"):
            _select_axis_pilot(self.plans, 0)
        with self.assertRaisesRegex(Top1DataError, "only 1 are planned"):
            _select_axis_pilot(
                (
                    DialogueBlueprint(
                        scenario_id="one-row",
                        phenomenon="single_turn",
                        target_candidate_name="A",
                        source_candidate_name=None,
                        user_turn_count=1,
                        seed=1,
                        content_axis="axis-a",
                    ),
                ),
                2,
            )
        with self.assertRaisesRegex(Top1DataError, "mutually exclusive"):
            _select_active_plans(
                self.plans,
                scenario_limit=1,
                axis_pilot_per_axis=1,
            )

    def test_cli_scopes_are_mutually_exclusive_and_default_is_unchanged(self) -> None:
        defaults = parse_args(())
        self.assertIsNone(defaults.scenario_limit)
        self.assertIsNone(defaults.axis_pilot_per_axis)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(
                (
                    "--scenario-limit",
                    "10",
                    "--axis-pilot-per-axis",
                    "1",
                )
            )


if __name__ == "__main__":
    unittest.main()
