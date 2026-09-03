from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

from top1_data_gen.synthesis import (
    QUALITY_FIELDS,
    DialogueBlueprint,
    acceptance_reasons,
    build_dialogue_blueprints,
    combine_directness_audits,
    load_taxonomy_descriptions,
    parse_directness_audits,
    parse_generated_samples,
)


CANDIDATES = (
    "StockAdvice",
    "StockOther",
    "StockQuery",
    "ProductGeneral",
    "ProductEcommerce",
    "ChitChat",
    "NoAvailable",
)


class SynthesisTests(unittest.TestCase):
    def test_default_plan_has_800_balanced_rows_and_all_directed_pairs(self) -> None:
        plans = build_dialogue_blueprints(CANDIDATES)

        self.assertEqual(len(plans), 800)
        self.assertEqual(len({plan.scenario_id for plan in plans}), 800)
        self.assertNotIn("contrast_candidate_name", plans[0].to_dict())
        self.assertNotIn("content_axis", plans[0].to_dict())
        targets = Counter(plan.target_candidate_name for plan in plans)
        self.assertLessEqual(max(targets.values()) - min(targets.values()), 1)
        pair_counts = Counter(
            (plan.source_candidate_name, plan.target_candidate_name)
            for plan in plans
            if plan.phenomenon == "intent_change"
        )
        self.assertEqual(len(pair_counts), 42)
        self.assertEqual(set(pair_counts.values()), {10})
        self.assertEqual(
            Counter(plan.phenomenon for plan in plans),
            Counter(
                {
                    "intent_change": 420,
                    "progressive_reveal": 128,
                    "contextual_follow_up": 98,
                    "clarification_revision": 70,
                    "assistant_distractor": 56,
                    "rambling": 28,
                }
            ),
        )

    def test_generation_parser_enforces_alternating_turn_contract(self) -> None:
        plan = DialogueBlueprint(
            scenario_id="case-1",
            phenomenon="contextual_follow_up",
            target_candidate_name="ProductEcommerce",
            source_candidate_name=None,
            user_turn_count=3,
            seed=1,
        )
        content = json.dumps(
            {
                "samples": [
                    {
                        "scenario_id": "case-1",
                        "messages": [
                            {"role": "user", "content": "想买通勤耳机"},
                            {"role": "assistant", "content": "预算和佩戴偏好呢？"},
                            {"role": "user", "content": "五百以内，入耳式"},
                            {"role": "assistant", "content": "需要降噪吗？"},
                            {"role": "user", "content": "要，黑色的呢？"},
                        ],
                    }
                ]
            },
            ensure_ascii=False,
        )

        parsed, errors = parse_generated_samples(
            content,
            {plan.scenario_id: plan},
            CANDIDATES,
        )

        self.assertIn("case-1", parsed)
        self.assertEqual(errors["case-1"], [])

    def test_acceptance_requires_two_labels_and_reviewer_plan_match(self) -> None:
        plan = DialogueBlueprint(
            scenario_id="case-2",
            phenomenon="intent_change",
            target_candidate_name="ProductGeneral",
            source_candidate_name="StockQuery",
            user_turn_count=3,
            seed=2,
        )
        quality = {field: True for field in QUALITY_FIELDS}
        labeler = {
            "predicted_candidate_name": "ProductGeneral",
            "quality": quality,
        }
        reviewer = {
            "predicted_candidate_name": "ProductGeneral",
            "observed_phenomenon": "intent_change",
            "observed_source_candidate_name": "StockQuery",
            "intent_change_is_direct": True,
            "quality": quality,
        }
        directness = {
            "contains_only_new_request": True,
            "references_previous_exchange": False,
            "uses_transition_or_acknowledgment": False,
            "direct_final_request": True,
            "has_switch_meta_language": False,
        }
        self.assertEqual(
            acceptance_reasons(plan, labeler, reviewer, directness),
            [],
        )

        reviewer = dict(reviewer)
        reviewer["predicted_candidate_name"] = "ProductEcommerce"
        self.assertIn(
            "reviewer_label_mismatch",
            acceptance_reasons(plan, labeler, reviewer, directness),
        )

    def test_directness_parser_requires_strict_boolean_contract(self) -> None:
        payload = json.dumps(
            {
                "audits": [
                    {
                        "scenario_id": "case-3",
                        "contains_only_new_request": False,
                        "references_previous_exchange": True,
                        "uses_transition_or_acknowledgment": True,
                        "direct_final_request": False,
                        "has_switch_meta_language": True,
                        "reason": "先结束旧话题再提出新需求",
                    }
                ]
            },
            ensure_ascii=False,
        )
        parsed, errors = parse_directness_audits(payload, ("case-3",))
        self.assertEqual(errors["case-3"], [])
        self.assertFalse(parsed["case-3"]["direct_final_request"])
        self.assertTrue(parsed["case-3"]["references_previous_exchange"])

    def test_directness_consensus_rejects_one_model_disagreement(self) -> None:
        approving = {
            "contains_only_new_request": True,
            "references_previous_exchange": False,
            "uses_transition_or_acknowledgment": False,
            "direct_final_request": True,
            "has_switch_meta_language": False,
            "reason": "只包含新请求",
        }
        rejecting = {
            "contains_only_new_request": False,
            "references_previous_exchange": True,
            "uses_transition_or_acknowledgment": True,
            "direct_final_request": False,
            "has_switch_meta_language": True,
            "reason": "先回应上一轮再切换",
        }

        combined = combine_directness_audits(
            {"reviewer": approving, "crosscheck": rejecting}
        )

        self.assertFalse(combined["contains_only_new_request"])
        self.assertTrue(combined["references_previous_exchange"])
        self.assertTrue(combined["uses_transition_or_acknowledgment"])
        self.assertFalse(combined["direct_final_request"])
        self.assertEqual(len(combined["model_audits"]), 2)

    def test_taxonomy_loader_requires_reviewed_definitions(self) -> None:
        rows = []
        for candidate in CANDIDATES:
            for description_type in ("concise_definition", "extended_definition"):
                rows.append(
                    {
                        "target_candidate_name": candidate,
                        "description_type": description_type,
                        "messages": [
                            {"role": "user", "content": f"{candidate}-{description_type}"}
                        ],
                    }
                )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.jsonl"
            path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            loaded = load_taxonomy_descriptions(path, CANDIDATES)
        self.assertEqual(set(loaded), set(CANDIDATES))
        self.assertEqual(
            loaded["StockQuery"]["concise_definition"],
            "StockQuery-concise_definition",
        )


if __name__ == "__main__":
    unittest.main()
