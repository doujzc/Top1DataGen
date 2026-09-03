from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

from top1_data_gen.cli import (
    _directness_attempt,
    _dialogue_quality_attempt,
    _judge_attempt,
    _plan_fidelity_attempt,
    _plan_fidelity_models,
    _prepare_run,
    _quality_fields,
)
from top1_data_gen.synthesis import (
    DIALOGUE_QUALITY_AUDIT_FIELDS,
    QUALITY_FIELDS,
    STRICT_DIALOGUE_QUALITY_FIELDS,
    DialogueBlueprint,
    ModelCall,
    acceptance_reasons,
    build_dialogue_blueprints,
    combine_contrast_audits,
    combine_dialogue_quality_audits,
    combine_plan_fidelity_audits,
    contrast_messages,
    dialogue_quality_messages,
    directness_messages,
    generation_messages,
    judgment_messages,
    load_taxonomy_descriptions,
    parse_contrast_audits,
    parse_dialogue_quality_audits,
    parse_directness_audits,
    parse_generated_samples,
    parse_judgments,
    parse_plan_fidelity_audits,
    plan_fidelity_messages,
    validate_content_axis_definitions,
    validate_content_axis_priority,
)
from top1_data_gen.data import (
    Top1DataError,
    load_candidate_names,
    read_jsonl,
    validate_memorization_rows,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_PATH = REPOSITORY_ROOT / "configs/top1_candidates_v2.json"
POLICY_PATH = REPOSITORY_ROOT / "configs/top1_decision_policy_v2.json"
SYNTHESIS_CONFIG_PATH = REPOSITORY_ROOT / "configs/top1_synthesis_v2.json"
TAXONOMY_PATH = REPOSITORY_ROOT / "data_top1/top1_labeldesc_v2.jsonl"


class SynthesisV2Tests(unittest.TestCase):
    def test_generation_and_audit_prompts_encode_v2_dialogue_contracts(self) -> None:
        blueprint = DialogueBlueprint(
            scenario_id="prompt-contract",
            phenomenon="intent_change",
            target_candidate_name="ChitChat",
            source_candidate_name="StockQuery",
            user_turn_count=3,
            seed=1,
        )
        generation_system = generation_messages(
            [blueprint],
            "taxonomy",
        )[0]["content"]
        self.assertIn("2 * user_turn_count - 1", generation_system)
        self.assertIn("奇数条是 user、偶数条是 assistant", generation_system)
        self.assertIn("绝对不要生成最后一条 user 之后的 assistant", generation_system)
        self.assertIn("新旧主题可以完全无关", generation_system)
        self.assertIn("不要添加致谢、承接、取消旧任务或宣布换题", generation_system)

        progressive_blueprint = DialogueBlueprint(
            scenario_id="progressive-contract",
            phenomenon="progressive_reveal",
            target_candidate_name="LifeKnowledgeQA",
            source_candidate_name=None,
            user_turn_count=2,
            seed=2,
        )
        progressive_request = generation_messages(
            [progressive_blueprint],
            "taxonomy",
        )[1]["content"]
        self.assertIn("单独阅读也能判明当前目标", progressive_request)
        self.assertIn("否则属于 contextual_follow_up", progressive_request)

        sample = {
            "scenario_id": "prompt-contract",
            "messages": [
                {"role": "user", "content": "查一下今天的收盘价"},
                {"role": "assistant", "content": "你想查哪只股票？"},
                {"role": "user", "content": "最近有点烦，陪我聊聊。"},
            ],
        }
        directness_system = directness_messages([sample])[0]["content"]
        self.assertIn("突然跨到完全无关的领域正是预期", directness_system)
        self.assertIn("不得仅因换题突然、没有过渡", directness_system)
        self.assertIn("问候、告别、情绪分享或兴趣闲聊", directness_system)
        for rejected_behavior in ("回应", "感谢", "取消", "切换元话语", "依赖历史"):
            self.assertIn(rejected_behavior, directness_system)

        quality_system = dialogue_quality_messages([sample])[0]["content"]
        self.assertIn("固有不完整或无意义而判 false", quality_system)
        self.assertIn("缺少先行词", quality_system)
        self.assertIn("合格的直接 IntentChange", quality_system)
        self.assertIn("合格的 assistant_distractor", quality_system)
        self.assertIn("不得仅因这一次偏题而判 false", quality_system)
        self.assertIn("逐条检查每条 assistant 消息", quality_system)
        self.assertIn("看起来合理、常见或可信不构成依据", quality_system)

        judgment_system = judgment_messages(
            [sample],
            "taxonomy",
            candidate_count=15,
            allow_single_turn=True,
            quality_fields=QUALITY_FIELDS + STRICT_DIALOGUE_QUALITY_FIELDS,
        )[0]["content"]
        self.assertIn("突然跨到无关领域正是预期", judgment_system)
        self.assertIn("有意误解、偏题或错误侧重点不得仅因此", judgment_system)
        self.assertIn("问候、告别、情绪分享或兴趣闲聊本身可以构成完整意图", judgment_system)
        self.assertIn("只记录对话自身客观存在的样本质量问题", judgment_system)
        self.assertIn("正确预测为 NoAvailable，本身不是质量问题", judgment_system)
        self.assertIn("只要没有缺失先行词或伪造历史，仍应填 true", judgment_system)
        self.assertIn("逐条检查每条 assistant 消息", judgment_system)
        self.assertIn("看起来合理", judgment_system)
        self.assertIn("若末轮主要靠省略、指代、确认或短追问", judgment_system)

    def test_registry_taxonomy_and_backend_policy_cover_fifteen_candidates(self) -> None:
        candidates = load_candidate_names(CANDIDATE_PATH)
        rows = read_jsonl(TAXONOMY_PATH)
        report = validate_memorization_rows(rows, candidates, source=TAXONOMY_PATH)
        descriptions = load_taxonomy_descriptions(TAXONOMY_PATH, candidates)
        policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
        candidate_to_backend = policy["candidate_to_backend"]

        self.assertEqual(len(candidates), 15)
        self.assertEqual(len(rows), 90)
        for row in rows:
            content = row["messages"][0]["content"]
            self.assertFalse(
                any(candidate in content for candidate in candidates),
                msg=f"candidate name leaked into LabelDesc input: {row['id']}",
            )
        self.assertEqual(report["candidate_counts"], {name: 6 for name in candidates})
        self.assertEqual(set(descriptions), set(candidates))
        self.assertEqual(set(candidate_to_backend), set(candidates))
        self.assertEqual(
            candidate_to_backend["MedicalQuestionAnswer"],
            "讯飞晓医",
        )
        self.assertEqual(
            candidate_to_backend["TravelGuide"],
            "小红书问一问",
        )
        self.assertEqual(
            candidate_to_backend["CivicBoundHelper"],
            "CivicBoundHelper",
        )

    def test_v2_plan_is_balanced_and_covers_all_directed_switches(self) -> None:
        candidates = load_candidate_names(CANDIDATE_PATH)
        config = json.loads(SYNTHESIS_CONFIG_PATH.read_text(encoding="utf-8"))
        plans = build_dialogue_blueprints(
            candidates,
            target_count=config["target_count"],
            intent_change_per_pair=config["intent_change_per_pair"],
            seed=config["seed"],
            synthesis_version=config["pipeline_version"],
            single_turn_per_candidate=config["single_turn_per_candidate"],
            single_turn_axis_confusions=config["single_turn_axis_confusions"],
            multi_turn_user_counts=config["multi_turn_user_counts"],
            content_axes=config["content_axes"],
            content_axis_allowed_phenomena=config.get(
                "content_axis_allowed_phenomena"
            ),
        )

        self.assertEqual(len(plans), 1_500)
        self.assertEqual(len({plan.scenario_id for plan in plans}), 1_500)
        self.assertEqual(
            Counter(plan.target_candidate_name for plan in plans),
            Counter({name: 100 for name in candidates}),
        )
        self.assertEqual(
            Counter(plan.phenomenon for plan in plans),
            Counter(
                {
                    "single_turn": 450,
                    "intent_change": 630,
                    "progressive_reveal": 150,
                    "contextual_follow_up": 105,
                    "clarification_revision": 75,
                    "assistant_distractor": 60,
                    "rambling": 30,
                }
            ),
        )
        pairs = Counter(
            (plan.source_candidate_name, plan.target_candidate_name)
            for plan in plans
            if plan.phenomenon == "intent_change"
        )
        self.assertEqual(len(pairs), 210)
        self.assertEqual(set(pairs.values()), {3})
        singles = [plan for plan in plans if plan.phenomenon == "single_turn"]
        self.assertTrue(all(plan.user_turn_count == 1 for plan in singles))
        self.assertTrue(all(plan.contrast_candidate_name for plan in singles))
        self.assertTrue(all(plan.content_axis for plan in plans))
        self.assertTrue(
            all(
                plan.source_content_axis
                for plan in plans
                if plan.phenomenon == "intent_change"
            )
        )
        self.assertTrue(
            all(
                plan.target_candidate_name != plan.contrast_candidate_name
                for plan in singles
            )
        )

    def test_strict_dialogue_quality_is_opt_in_and_gated_for_both_judges(self) -> None:
        self.assertEqual(_quality_fields({}), QUALITY_FIELDS)
        strict_fields = _quality_fields({"require_strict_dialogue_quality": True})
        self.assertEqual(
            strict_fields,
            QUALITY_FIELDS + STRICT_DIALOGUE_QUALITY_FIELDS,
        )

        prompt = judgment_messages(
            [
                {
                    "scenario_id": "strict-quality",
                    "messages": [{"role": "user", "content": "想了解原理"}],
                }
            ],
            "Candidate：测试定义",
            candidate_count=15,
            allow_single_turn=True,
            quality_fields=strict_fields,
        )
        system_prompt = prompt[0]["content"]
        for field in STRICT_DIALOGUE_QUALITY_FIELDS:
            self.assertIn(field, system_prompt)
        self.assertIn(
            "current_target_identifiable 在这里判断“应选哪个候选”",
            system_prompt,
        )
        schema = json.loads(prompt[1]["content"])["output_schema"]["judgments"][0]
        self.assertEqual(tuple(schema["quality"]), strict_fields)

        incomplete_quality = {field: True for field in strict_fields}
        incomplete_quality.pop("entity_and_facts_consistent")
        parsed, errors = parse_judgments(
            json.dumps(
                {
                    "judgments": [
                        {
                            "scenario_id": "strict-quality",
                            "predicted_candidate_name": "LifeKnowledgeQA",
                            "observed_phenomenon": "progressive_reveal",
                            "observed_source_candidate_name": None,
                            "intent_change_is_direct": True,
                            "quality": incomplete_quality,
                            "issues": [],
                        }
                    ]
                },
                ensure_ascii=False,
            ),
            ("strict-quality",),
            ("LifeKnowledgeQA",),
            quality_fields=strict_fields,
        )
        self.assertNotIn("strict-quality", parsed)
        self.assertIn(
            "quality.entity_and_facts_consistent must be boolean",
            errors["strict-quality"],
        )

        missing_issues_payload = {
            "judgments": [
                {
                    "scenario_id": "strict-quality",
                    "predicted_candidate_name": "LifeKnowledgeQA",
                    "observed_phenomenon": "progressive_reveal",
                    "observed_source_candidate_name": None,
                    "intent_change_is_direct": True,
                    "quality": {field: True for field in strict_fields},
                }
            ]
        }
        parsed, errors = parse_judgments(
            json.dumps(missing_issues_payload, ensure_ascii=False),
            ("strict-quality",),
            ("LifeKnowledgeQA",),
            quality_fields=strict_fields,
            require_issues_field=True,
        )
        self.assertNotIn("strict-quality", parsed)
        self.assertIn("issues field is required", errors["strict-quality"])

        blueprint = DialogueBlueprint(
            scenario_id="strict-quality",
            phenomenon="progressive_reveal",
            target_candidate_name="LifeKnowledgeQA",
            source_candidate_name=None,
            user_turn_count=2,
            seed=1,
            content_axis="daily_physics",
        )
        passing_quality = {field: True for field in strict_fields}
        labeler = {
            "predicted_candidate_name": "LifeKnowledgeQA",
            "quality": passing_quality,
        }
        reviewer = {
            "predicted_candidate_name": "LifeKnowledgeQA",
            "observed_phenomenon": "progressive_reveal",
            "quality": passing_quality,
        }
        self.assertEqual(
            acceptance_reasons(
                blueprint,
                labeler,
                reviewer,
                quality_fields=strict_fields,
            ),
            [],
        )
        for role in ("labeler", "reviewer"):
            failing_quality = dict(passing_quality)
            failing_quality["assistant_no_unsupported_factual_claims"] = False
            failing = dict(labeler if role == "labeler" else reviewer)
            failing["quality"] = failing_quality
            reasons = acceptance_reasons(
                blueprint,
                failing if role == "labeler" else labeler,
                failing if role == "reviewer" else reviewer,
                quality_fields=strict_fields,
            )
            self.assertIn(
                f"{role}_quality_assistant_no_unsupported_factual_claims",
                reasons,
            )

    def test_plan_fidelity_prompt_requires_final_user_axis_evidence(self) -> None:
        prompt = plan_fidelity_messages(
            [
                {
                    "scenario_id": "axis-evidence",
                    "messages": [{"role": "user", "content": "这个呢？"}],
                    "target_candidate_name": "LifeKnowledgeQA",
                    "source_candidate_name": None,
                    "planned_phenomenon": "single_turn",
                    "content_axis": "daily_physics",
                    "source_content_axis": None,
                }
            ],
            "LifeKnowledgeQA：测试定义",
        )[0]["content"]

        self.assertIn("只能依据最后一条 user 消息自身", prompt)
        self.assertIn("不能因为开场、较早历史或 assistant", prompt)
        self.assertIn("single_turn 只能查看唯一一条 user 消息", prompt)
        self.assertIn("这种结构属于 contextual_follow_up", prompt)

    def test_strict_plan_fidelity_uses_two_model_unanimous_consensus(self) -> None:
        config = {
            "reviewer_model": "reviewer",
            "labeler_model": "labeler",
            "plan_auditor_model": "reviewer",
        }
        self.assertEqual(_plan_fidelity_models(config), ("reviewer",))
        self.assertEqual(
            _plan_fidelity_models(
                {**config, "require_strict_dialogue_quality": True}
            ),
            ("reviewer", "labeler"),
        )
        self.assertEqual(
            _plan_fidelity_models(
                {**config, "plan_auditor_models": ["audit-a", "audit-b"]}
            ),
            ("audit-a", "audit-b"),
        )
        with self.assertRaises(Top1DataError):
            _plan_fidelity_models(
                {**config, "plan_auditor_models": ["same", "same"]}
            )

        approving = {
            "target_axis_realized": True,
            "target_candidate_respected": True,
            "source_axis_realized": True,
            "source_candidate_respected": True,
            "phenomenon_realized": True,
            "no_extra_current_goal": True,
            "reason": "计划落实",
        }
        rejecting = dict(approving)
        rejecting["target_axis_realized"] = False
        rejecting["reason"] = "末轮没有落实轴"
        combined = combine_plan_fidelity_audits(
            {"audit-a": approving, "audit-b": rejecting}
        )
        self.assertFalse(combined["target_axis_realized"])
        self.assertTrue(combined["target_candidate_respected"])
        self.assertEqual(len(combined["model_audits"]), 2)

    def test_plan_fidelity_attempt_preserves_each_model_audit(self) -> None:
        class FakeClient:
            def chat_json(
                self,
                *,
                model: str,
                messages: object,
                temperature: float,
                max_tokens: int,
                response_schema: object | None = None,
                require_stop: bool = False,
            ) -> ModelCall:
                del messages, temperature, max_tokens, response_schema, require_stop
                audit = {
                    "scenario_id": "plan-consensus",
                    "target_axis_realized": model == "audit-a",
                    "target_candidate_respected": True,
                    "source_axis_realized": True,
                    "source_candidate_respected": True,
                    "phenomenon_realized": True,
                    "no_extra_current_goal": True,
                    "reason": f"{model} judgment",
                }
                return ModelCall(
                    content=json.dumps({"audits": [audit]}),
                    usage={},
                    finish_reason="stop",
                    elapsed_seconds=0.0,
                )

        sample = {
            "scenario_id": "plan-consensus",
            "attempt": 1,
            "messages": [{"role": "user", "content": "保鲜膜为什么会粘住？"}],
            "target_candidate_name": "LifeKnowledgeQA",
            "source_candidate_name": None,
            "planned_phenomenon": "single_turn",
            "content_axis": "materials_and_chemistry",
            "source_content_axis": None,
        }
        config = {
            "judgment_batch_size": 1,
            "max_workers": 2,
            "judge_temperature": 0.0,
            "judgment_max_tokens": 1000,
        }
        with tempfile.TemporaryDirectory() as directory:
            audits, errors = _plan_fidelity_attempt(
                samples=[sample],
                taxonomy="LifeKnowledgeQA：测试定义",
                models=("audit-a", "audit-b"),
                config=config,
                client=FakeClient(),
                raw_directory=Path(directory),
            )
        self.assertEqual(errors["plan-consensus"], [])
        self.assertFalse(audits["plan-consensus"]["target_axis_realized"])
        self.assertEqual(len(audits["plan-consensus"]["model_audits"]), 2)

    def test_axis_definitions_are_isomorphic_and_injected_into_generation(self) -> None:
        axes = {"Source": ["source_axis"], "Target": ["first_axis", "second_axis"]}
        definitions = {
            "Source": {"source_axis": "历史请求中的来源子场景"},
            "Target": {
                "first_axis": "最终请求中的第一种子场景",
                "second_axis": "最终请求中的第二种子场景",
            },
        }
        normalized = validate_content_axis_definitions(
            axes,
            definitions,
            ("Source", "Target"),
        )
        self.assertEqual(normalized, definitions)
        priority = {
            "Source": ["source_axis"],
            "Target": ["second_axis", "first_axis"],
        }
        self.assertEqual(
            validate_content_axis_priority(
                axes,
                priority,
                ("Source", "Target"),
            ),
            {
                "Source": ("source_axis",),
                "Target": ("second_axis", "first_axis"),
            },
        )
        with self.assertRaises(Top1DataError):
            validate_content_axis_priority(
                axes,
                {"Source": ["source_axis"], "Target": ["first_axis"]},
                ("Source", "Target"),
            )
        with self.assertRaises(Top1DataError):
            validate_content_axis_definitions(
                axes,
                {**definitions, "Target": {"first_axis": "缺一个轴"}},
                ("Source", "Target"),
            )

        blueprint = DialogueBlueprint(
            scenario_id="definition-injection",
            phenomenon="intent_change",
            target_candidate_name="Target",
            source_candidate_name="Source",
            user_turn_count=2,
            seed=1,
            content_axis="second_axis",
            source_content_axis="source_axis",
        )
        legacy_plan = json.loads(
            generation_messages([blueprint], "taxonomy")[1]["content"]
        )["plans"][0]
        self.assertNotIn("content_axis_definition", legacy_plan)
        generation_prompt = generation_messages(
                [blueprint],
                "taxonomy",
                content_axis_definitions=definitions,
            )
        prompt_plan = json.loads(generation_prompt[1]["content"])["plans"][0]
        self.assertEqual(
            prompt_plan["content_axis_definition"],
            "最终请求中的第二种子场景",
        )
        self.assertEqual(
            prompt_plan["source_content_axis_definition"],
            "历史请求中的来源子场景",
        )
        single_blueprint = DialogueBlueprint(
            scenario_id="single-definition-injection",
            phenomenon="single_turn",
            target_candidate_name="Target",
            source_candidate_name=None,
            user_turn_count=1,
            seed=2,
            contrast_candidate_name="Source",
            content_axis="first_axis",
        )
        single_prompt = generation_messages(
            [single_blueprint],
            "taxonomy",
            content_axis_definitions=definitions,
        )
        self.assertIn("固有无意义或信息不足", single_prompt[1]["content"])

    def test_observed_axis_audit_hides_blueprint_axis_and_requires_consensus(self) -> None:
        definitions = {
            "Source": {
                "source_one": "来源内容轴一",
                "source_two": "来源内容轴二",
            },
            "Target": {
                "target_one": "目标内容轴一",
                "target_two": "目标内容轴二",
            },
        }
        sample = {
            "scenario_id": "observed-axis",
            "messages": [
                {"role": "user", "content": "先问来源任务"},
                {"role": "assistant", "content": "请补充"},
                {"role": "user", "content": "现在直接提出目标任务"},
            ],
            "target_candidate_name": "Target",
            "source_candidate_name": "Source",
            "planned_phenomenon": "intent_change",
            "content_axis": "target_two",
            "source_content_axis": "source_two",
        }
        messages = plan_fidelity_messages(
            [sample],
            "taxonomy",
            content_axis_definitions=definitions,
            content_axis_priority={
                "Source": ["source_two", "source_one"],
                "Target": ["target_two", "target_one"],
            },
            require_observed_axis_match=True,
        )
        prompt_sample = json.loads(messages[1]["content"])["samples"][0]
        self.assertNotIn("content_axis", prompt_sample)
        self.assertNotIn("source_content_axis", prompt_sample)
        self.assertEqual(
            prompt_sample["target_content_axis_catalog"], definitions["Target"]
        )
        self.assertEqual(
            prompt_sample["source_content_axis_catalog"], definitions["Source"]
        )
        self.assertEqual(
            prompt_sample["target_content_axis_priority"],
            ["target_two", "target_one"],
        )
        self.assertIn("严格按 target_content_axis_priority", messages[0]["content"])
        self.assertIn("固有无意义、不可理解或信息不足", messages[0]["content"])
        self.assertIn("这种结构属于 contextual_follow_up", messages[0]["content"])

        audit = {
            "scenario_id": "observed-axis",
            "observed_target_content_axis": "target_two",
            "observed_source_content_axis": "source_two",
            "target_axis_unambiguous": True,
            "source_axis_unambiguous": True,
            "target_axis_realized": True,
            "target_candidate_respected": True,
            "source_axis_realized": True,
            "source_candidate_respected": True,
            "phenomenon_realized": True,
            "no_extra_current_goal": True,
            "reason": "最后用户独立提出目标任务且对象动作明确，历史用户也清楚落实来源内容轴。",
        }
        catalogs = {
            "observed-axis": {
                "target_content_axes": tuple(definitions["Target"]),
                "source_content_axes": tuple(definitions["Source"]),
            }
        }
        parsed, errors = parse_plan_fidelity_audits(
            json.dumps({"audits": [audit]}, ensure_ascii=False),
            ("observed-axis",),
            observed_axis_catalogs=catalogs,
        )
        self.assertEqual(errors["observed-axis"], [])
        self.assertEqual(
            parsed["observed-axis"]["observed_target_content_axis"],
            "target_two",
        )

        short_reason = dict(audit)
        short_reason["reason"] = "理由过短"
        parsed, errors = parse_plan_fidelity_audits(
            json.dumps({"audits": [short_reason]}, ensure_ascii=False),
            ("observed-axis",),
            observed_axis_catalogs=catalogs,
        )
        self.assertNotIn("observed-axis", parsed)
        self.assertIn(
            "reason must contain at least 20 characters",
            errors["observed-axis"],
        )

        disagreeing = dict(audit)
        disagreeing["observed_target_content_axis"] = "target_one"
        combined = combine_plan_fidelity_audits(
            {"model-a": audit, "model-b": disagreeing}
        )
        self.assertIsNone(combined["observed_target_content_axis"])
        self.assertFalse(combined["target_axis_unambiguous"])
        self.assertEqual(len(combined["model_audits"]), 2)

        blueprint = DialogueBlueprint(
            scenario_id="observed-axis",
            phenomenon="intent_change",
            target_candidate_name="Target",
            source_candidate_name="Source",
            user_turn_count=2,
            seed=1,
            content_axis="target_two",
            source_content_axis="source_two",
        )
        quality = {field: True for field in QUALITY_FIELDS}
        labeler = {"predicted_candidate_name": "Target", "quality": quality}
        reviewer = {
            "predicted_candidate_name": "Target",
            "observed_phenomenon": "intent_change",
            "observed_source_candidate_name": "Source",
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
        reasons = acceptance_reasons(
            blueprint,
            labeler,
            reviewer,
            directness,
            plan_fidelity=combined,
            require_plan_fidelity=True,
            require_observed_axis_match=True,
        )
        self.assertIn("plan_fidelity_target_axis_unambiguous", reasons)
        self.assertIn("plan_fidelity_target_axis_mismatch", reasons)

    def test_observed_axis_prompt_cannot_distinguish_planned_axis_or_original_id(self) -> None:
        definitions = {
            "Target": {
                "axis_one": "第一种目标内容",
                "axis_two": "第二种目标内容",
            }
        }
        priority = {"Target": ["axis_one", "axis_two"]}
        common = {
            "audit_id": "plan_000001",
            "messages": [{"role": "user", "content": "同一条完整目标请求"}],
            "target_candidate_name": "Target",
            "source_candidate_name": None,
            "planned_phenomenon": "single_turn",
            "source_content_axis": None,
        }
        first = {
            **common,
            "scenario_id": "original_planned_axis_one",
            "content_axis": "axis_one",
        }
        second = {
            **common,
            "scenario_id": "different_planned_axis_two",
            "content_axis": "axis_two",
        }
        first_prompt = plan_fidelity_messages(
            [first],
            "taxonomy",
            content_axis_definitions=definitions,
            content_axis_priority=priority,
            require_observed_axis_match=True,
        )
        second_prompt = plan_fidelity_messages(
            [second],
            "taxonomy",
            content_axis_definitions=definitions,
            content_axis_priority=priority,
            require_observed_axis_match=True,
        )

        self.assertEqual(first_prompt, second_prompt)
        serialized = json.dumps(first_prompt, ensure_ascii=False)
        self.assertIn("plan_000001", serialized)
        self.assertNotIn(first["scenario_id"], serialized)
        self.assertNotIn(second["scenario_id"], serialized)
        self.assertNotIn('"content_axis":', serialized)

    def test_observed_axis_attempt_maps_two_independent_model_outputs(self) -> None:
        seen_samples: list[dict[str, object]] = []

        class FakeClient:
            def chat_json(
                self,
                *,
                model: str,
                messages: object,
                temperature: float,
                max_tokens: int,
                response_schema: object | None = None,
                require_stop: bool = False,
            ) -> ModelCall:
                del temperature, max_tokens, response_schema, require_stop
                request = json.loads(messages[1]["content"])
                request_sample = request["samples"][0]
                seen_samples.append(request_sample)
                audit = {
                    "scenario_id": request_sample["scenario_id"],
                    "observed_target_content_axis": (
                        "axis_one" if model == "model-a" else "axis_two"
                    ),
                    "observed_source_content_axis": None,
                    "target_axis_unambiguous": True,
                    "source_axis_unambiguous": True,
                    "target_axis_realized": True,
                    "target_candidate_respected": True,
                    "source_axis_realized": True,
                    "source_candidate_respected": True,
                    "phenomenon_realized": True,
                    "no_extra_current_goal": True,
                    "reason": "最后一条用户消息自身包含完整动作对象，可从目录中独立选择唯一内容轴。",
                }
                return ModelCall(
                    content=json.dumps({"audits": [audit]}, ensure_ascii=False),
                    usage={},
                    finish_reason="stop",
                    elapsed_seconds=0.0,
                )

        sample = {
            "scenario_id": "observed-integration",
            "attempt": 1,
            "messages": [{"role": "user", "content": "完整目标请求"}],
            "target_candidate_name": "Target",
            "source_candidate_name": None,
            "planned_phenomenon": "single_turn",
            "content_axis": "axis_two",
            "source_content_axis": None,
        }
        config = {
            "require_observed_axis_match": True,
            "content_axis_definitions": {
                "Target": {"axis_one": "第一轴", "axis_two": "第二轴"}
            },
            "content_axis_priority": {"Target": ("axis_two", "axis_one")},
            "judgment_batch_size": 1,
            "max_workers": 2,
            "judge_temperature": 0.0,
            "judgment_max_tokens": 1000,
        }
        with tempfile.TemporaryDirectory() as directory:
            audits, errors = _plan_fidelity_attempt(
                samples=[sample],
                taxonomy="Target 定义",
                models=("model-a", "model-b"),
                config=config,
                client=FakeClient(),
                raw_directory=Path(directory),
            )
        self.assertEqual(errors["observed-integration"], [])
        self.assertIsNone(
            audits["observed-integration"]["observed_target_content_axis"]
        )
        self.assertFalse(
            audits["observed-integration"]["target_axis_unambiguous"]
        )
        self.assertEqual(len(audits["observed-integration"]["model_audits"]), 2)
        self.assertEqual(len(seen_samples), 2)
        for request_sample in seen_samples:
            self.assertEqual(request_sample["scenario_id"], "plan_000001")
            self.assertNotIn(
                "observed-integration",
                json.dumps(request_sample, ensure_ascii=False),
            )
            self.assertNotIn("content_axis", request_sample)
            self.assertNotIn("source_content_axis", request_sample)
            self.assertEqual(
                request_sample["target_content_axis_priority"],
                ["axis_two", "axis_one"],
            )

    def test_dialogue_quality_audit_is_blind_two_model_and_hard_gated(self) -> None:
        prompt = dialogue_quality_messages(
            [
                {
                    "scenario_id": "quality-consensus",
                    "messages": [{"role": "user", "content": "你好"}],
                    "target_candidate_name": "must-not-leak",
                    "planned_phenomenon": "must-not-leak",
                }
            ]
        )
        prompt_sample = json.loads(prompt[1]["content"])["samples"][0]
        self.assertEqual(set(prompt_sample), {"scenario_id", "messages"})
        for field in DIALOGUE_QUALITY_AUDIT_FIELDS:
            self.assertIn(field, prompt[0]["content"])

        dialogue_ids: list[str] = []

        class FakeClient:
            def chat_json(
                self,
                *,
                model: str,
                messages: object,
                temperature: float,
                max_tokens: int,
                response_schema: object | None = None,
                require_stop: bool = False,
            ) -> ModelCall:
                del temperature, max_tokens, response_schema, require_stop
                request = json.loads(messages[1]["content"])
                dialogue_ids.append(request["samples"][0]["scenario_id"])
                audit = {
                    "scenario_id": request["samples"][0]["scenario_id"],
                    **{field: True for field in DIALOGUE_QUALITY_AUDIT_FIELDS},
                    "reason": f"{model} 逐项检查完整对话后未发现质量问题",
                }
                if model == "model-b":
                    audit["temporal_context_consistent"] = False
                return ModelCall(
                    content=json.dumps({"audits": [audit]}, ensure_ascii=False),
                    usage={},
                    finish_reason="stop",
                    elapsed_seconds=0.0,
                )

        sample = {
            "scenario_id": "quality-consensus",
            "attempt": 1,
            "messages": [{"role": "user", "content": "你好"}],
        }
        config = {
            "judgment_batch_size": 1,
            "max_workers": 2,
            "judge_temperature": 0.0,
            "judgment_max_tokens": 1000,
        }
        with tempfile.TemporaryDirectory() as directory:
            audits, errors = _dialogue_quality_attempt(
                samples=[sample],
                models=("model-a", "model-b"),
                config=config,
                client=FakeClient(),
                raw_directory=Path(directory),
            )
        self.assertEqual(errors["quality-consensus"], [])
        combined = audits["quality-consensus"]
        self.assertFalse(combined["temporal_context_consistent"])
        self.assertEqual(len(combined["model_audits"]), 2)
        self.assertEqual(dialogue_ids, ["dialogue_000001"] * 2)

        parsed, parse_errors = parse_dialogue_quality_audits(
            json.dumps(
                {
                    "audits": [
                        {
                            "scenario_id": "quality-consensus",
                            **{
                                field: True
                                for field in DIALOGUE_QUALITY_AUDIT_FIELDS
                            },
                            "reason": "逐项检查通过",
                        }
                    ]
                },
                ensure_ascii=False,
            ),
            ("quality-consensus",),
        )
        self.assertEqual(parse_errors["quality-consensus"], [])
        self.assertIn("quality-consensus", parsed)
        approving = combine_dialogue_quality_audits(
            {"model-a": parsed["quality-consensus"], "model-b": parsed["quality-consensus"]}
        )
        self.assertTrue(approving["natural_dialogue"])

        blueprint = DialogueBlueprint(
            scenario_id="quality-consensus",
            phenomenon="progressive_reveal",
            target_candidate_name="Target",
            source_candidate_name=None,
            user_turn_count=2,
            seed=1,
        )
        quality = {field: True for field in QUALITY_FIELDS}
        labeler = {"predicted_candidate_name": "Target", "quality": quality}
        reviewer = {
            "predicted_candidate_name": "Target",
            "observed_phenomenon": "progressive_reveal",
            "quality": quality,
        }
        self.assertIn(
            "dialogue_quality_temporal_context_consistent",
            acceptance_reasons(
                blueprint,
                labeler,
                reviewer,
                dialogue_quality=combined,
                require_dialogue_quality=True,
            ),
        )

    def test_empty_issues_and_both_blind_plan_observations_are_gated(self) -> None:
        blueprint = DialogueBlueprint(
            scenario_id="blind-contract",
            phenomenon="intent_change",
            target_candidate_name="Target",
            source_candidate_name="Source",
            user_turn_count=2,
            seed=1,
        )
        quality = {field: True for field in QUALITY_FIELDS}
        labeler = {
            "predicted_candidate_name": "Target",
            "observed_phenomenon": "progressive_reveal",
            "observed_source_candidate_name": None,
            "intent_change_is_direct": False,
            "quality": quality,
            "issues": ["现象不匹配"],
        }
        reviewer = {
            "predicted_candidate_name": "Target",
            "observed_phenomenon": "intent_change",
            "observed_source_candidate_name": "Source",
            "intent_change_is_direct": True,
            "quality": quality,
            "issues": [],
        }
        directness = {
            "contains_only_new_request": True,
            "references_previous_exchange": False,
            "uses_transition_or_acknowledgment": False,
            "direct_final_request": True,
            "has_switch_meta_language": False,
        }
        reasons = acceptance_reasons(
            blueprint,
            labeler,
            reviewer,
            directness,
            require_empty_judgment_issues=True,
            require_both_judges_plan_match=True,
        )
        self.assertIn("labeler_issues_not_empty", reasons)
        self.assertIn("labeler_phenomenon_mismatch", reasons)
        self.assertIn("labeler_source_candidate_mismatch", reasons)
        self.assertIn("labeler_intent_change_not_direct", reasons)

    def test_non_intent_blind_source_and_directness_fields_are_strictly_gated(self) -> None:
        blueprint = DialogueBlueprint(
            scenario_id="non-intent-contract",
            phenomenon="progressive_reveal",
            target_candidate_name="Target",
            source_candidate_name=None,
            user_turn_count=2,
            seed=1,
        )
        quality = {field: True for field in QUALITY_FIELDS}
        baseline = {
            "predicted_candidate_name": "Target",
            "observed_phenomenon": "progressive_reveal",
            "observed_source_candidate_name": None,
            "intent_change_is_direct": True,
            "quality": quality,
            "issues": [],
        }
        self.assertEqual(
            acceptance_reasons(
                blueprint,
                dict(baseline),
                dict(baseline),
                require_both_judges_plan_match=True,
            ),
            [],
        )

        for role in ("labeler", "reviewer"):
            with self.subTest(role=role, field="source"):
                labeler = dict(baseline)
                reviewer = dict(baseline)
                target = labeler if role == "labeler" else reviewer
                target["observed_source_candidate_name"] = "Source"
                reasons = acceptance_reasons(
                    blueprint,
                    labeler,
                    reviewer,
                    require_both_judges_plan_match=True,
                )
                self.assertIn(f"{role}_non_intent_source_candidate", reasons)
            with self.subTest(role=role, field="directness"):
                labeler = dict(baseline)
                reviewer = dict(baseline)
                target = labeler if role == "labeler" else reviewer
                target["intent_change_is_direct"] = False
                reasons = acceptance_reasons(
                    blueprint,
                    labeler,
                    reviewer,
                    require_both_judges_plan_match=True,
                )
                self.assertIn(f"{role}_non_intent_directness", reasons)

    def test_strict_blind_and_directness_requests_use_opaque_ids(self) -> None:
        original_id = "pipeline_intent_change_source_to_target_001"
        seen_ids: list[str] = []

        class BlindClient:
            def chat_json(
                self,
                *,
                model: str,
                messages: object,
                temperature: float,
                max_tokens: int,
                response_schema: object | None = None,
                require_stop: bool = False,
            ) -> ModelCall:
                del model, temperature, max_tokens, response_schema, require_stop
                request = json.loads(messages[1]["content"])
                audit_id = request["samples"][0]["scenario_id"]
                seen_ids.append(audit_id)
                quality_schema = request["output_schema"]["judgments"][0][
                    "quality"
                ]
                return ModelCall(
                    content=json.dumps(
                        {
                            "judgments": [
                                {
                                    "scenario_id": audit_id,
                                    "predicted_candidate_name": "Target",
                                    "observed_phenomenon": "intent_change",
                                    "observed_source_candidate_name": "Source",
                                    "intent_change_is_direct": True,
                                    "quality": {
                                        field: True for field in quality_schema
                                    },
                                    "issues": [],
                                }
                            ]
                        },
                        ensure_ascii=False,
                    ),
                    usage={},
                    finish_reason="stop",
                    elapsed_seconds=0.0,
                )

        sample = {
            "scenario_id": original_id,
            "attempt": 1,
            "messages": [{"role": "user", "content": "测试请求"}],
        }
        config = {
            "require_strict_dialogue_quality": True,
            "judgment_batch_size": 1,
            "max_workers": 1,
            "judge_temperature": 0.0,
            "judgment_max_tokens": 1000,
            "single_turn_per_candidate": 1,
        }
        with tempfile.TemporaryDirectory() as directory:
            judgments, errors = _judge_attempt(
                stage="labeler",
                samples=[sample],
                taxonomy="Source 与 Target 的定义",
                candidate_names=("Source", "Target"),
                model="blind-model",
                config=config,
                client=BlindClient(),
                raw_directory=Path(directory),
            )
        self.assertEqual(errors[original_id], [])
        self.assertIn(original_id, judgments)
        self.assertEqual(judgments[original_id]["scenario_id"], original_id)
        self.assertEqual(len(seen_ids), 1)
        self.assertNotEqual(seen_ids[0], original_id)
        for leaked_fragment in ("intent_change", "source", "target"):
            self.assertNotIn(leaked_fragment, seen_ids[0].lower())

        directness_ids: list[str] = []

        class DirectnessClient:
            def chat_json(
                self,
                *,
                model: str,
                messages: object,
                temperature: float,
                max_tokens: int,
                response_schema: object | None = None,
                require_stop: bool = False,
            ) -> ModelCall:
                del model, temperature, max_tokens, response_schema, require_stop
                request = json.loads(messages[1]["content"])
                audit_id = request["samples"][0]["scenario_id"]
                directness_ids.append(audit_id)
                return ModelCall(
                    content=json.dumps(
                        {
                            "audits": [
                                {
                                    "scenario_id": audit_id,
                                    "contains_only_new_request": True,
                                    "references_previous_exchange": False,
                                    "uses_transition_or_acknowledgment": False,
                                    "direct_final_request": True,
                                    "has_switch_meta_language": False,
                                    "reason": "最终消息只表达新的直接请求",
                                }
                            ]
                        },
                        ensure_ascii=False,
                    ),
                    usage={},
                    finish_reason="stop",
                    elapsed_seconds=0.0,
                )

        with tempfile.TemporaryDirectory() as directory:
            audits, errors = _directness_attempt(
                samples=[sample],
                models=("model-a", "model-b"),
                config=config,
                client=DirectnessClient(),
                raw_directory=Path(directory),
            )
        self.assertEqual(errors[original_id], [])
        self.assertIn(original_id, audits)
        self.assertEqual(directness_ids, ["directness_000001"] * 2)
        self.assertNotIn("intent_change", directness_ids[0])

    def test_strict_response_envelopes_fail_closed_for_every_parser(self) -> None:
        expected_ids = ("expected-a", "expected-b")
        quality = {field: True for field in QUALITY_FIELDS}

        def judgment_row(scenario_id: str) -> dict[str, object]:
            return {
                "scenario_id": scenario_id,
                "predicted_candidate_name": "Target",
                "observed_phenomenon": "progressive_reveal",
                "observed_source_candidate_name": None,
                "intent_change_is_direct": True,
                "quality": quality,
                "issues": [],
            }

        def directness_row(scenario_id: str) -> dict[str, object]:
            return {
                "scenario_id": scenario_id,
                "contains_only_new_request": True,
                "references_previous_exchange": False,
                "uses_transition_or_acknowledgment": False,
                "direct_final_request": True,
                "has_switch_meta_language": False,
                "reason": "最终消息只包含直接的新请求",
            }

        def dialogue_row(scenario_id: str) -> dict[str, object]:
            return {
                "scenario_id": scenario_id,
                **{field: True for field in DIALOGUE_QUALITY_AUDIT_FIELDS},
                "reason": "完整对话逐项检查后均符合质量要求",
            }

        def contrast_row(scenario_id: str) -> dict[str, object]:
            return {
                "scenario_id": scenario_id,
                "target_semantics_present": True,
                "contrast_is_plausible": True,
                "target_preferred_over_contrast": True,
                "single_current_goal": True,
                "natural_expression": True,
                "reason": "边界自然且目标唯一",
            }

        def plan_row(scenario_id: str) -> dict[str, object]:
            return {
                "scenario_id": scenario_id,
                "target_axis_realized": True,
                "target_candidate_respected": True,
                "source_axis_realized": True,
                "source_candidate_respected": True,
                "phenomenon_realized": True,
                "no_extra_current_goal": True,
                "reason": "计划完整落实",
            }

        parser_cases = (
            (
                "judgment",
                "judgments",
                judgment_row,
                lambda content, strict: parse_judgments(
                    content,
                    expected_ids,
                    ("Source", "Target"),
                    strict_envelope=strict,
                ),
            ),
            (
                "directness",
                "audits",
                directness_row,
                lambda content, strict: parse_directness_audits(
                    content,
                    expected_ids,
                    strict_envelope=strict,
                ),
            ),
            (
                "dialogue_quality",
                "audits",
                dialogue_row,
                lambda content, strict: parse_dialogue_quality_audits(
                    content,
                    expected_ids,
                    strict_envelope=strict,
                ),
            ),
            (
                "contrast",
                "audits",
                contrast_row,
                lambda content, strict: parse_contrast_audits(
                    content,
                    expected_ids,
                    strict_envelope=strict,
                ),
            ),
            (
                "plan",
                "audits",
                plan_row,
                lambda content, strict: parse_plan_fidelity_audits(
                    content,
                    expected_ids,
                    strict_envelope=strict,
                ),
            ),
        )
        for name, response_field, row_factory, parser in parser_cases:
            valid_rows = [row_factory(value) for value in expected_ids]
            with self.subTest(parser=name, envelope="legacy-compatible"):
                content = json.dumps(
                    {
                        response_field: [
                            *valid_rows,
                            row_factory("unexpected-id"),
                            "not-an-object",
                        ]
                    },
                    ensure_ascii=False,
                )
                parsed, _ = parser(content, False)
                self.assertEqual(set(parsed), set(expected_ids))
            for envelope_name, extra in (
                ("unknown-id", row_factory("unexpected-id")),
                ("non-object", "not-an-object"),
            ):
                with self.subTest(parser=name, envelope=envelope_name):
                    content = json.dumps(
                        {response_field: [*valid_rows, extra]},
                        ensure_ascii=False,
                    )
                    parsed, errors = parser(content, True)
                    self.assertEqual(parsed, {})
                    self.assertTrue(all(errors[value] for value in expected_ids))

        for envelope_name, rows in (
            ("missing", [judgment_row("expected-a")]),
            (
                "duplicate",
                [
                    judgment_row("expected-a"),
                    judgment_row("expected-b"),
                    judgment_row("expected-a"),
                ],
            ),
        ):
            with self.subTest(parser="judgment", envelope=envelope_name):
                parsed, errors = parse_judgments(
                    json.dumps({"judgments": rows}, ensure_ascii=False),
                    expected_ids,
                    ("Source", "Target"),
                    strict_envelope=True,
                )
                self.assertEqual(parsed, {})
                self.assertTrue(all(errors[value] for value in expected_ids))

        assigned = {
            scenario_id: DialogueBlueprint(
                scenario_id=scenario_id,
                phenomenon="single_turn",
                target_candidate_name="Target",
                source_candidate_name=None,
                user_turn_count=1,
                seed=index,
            )
            for index, scenario_id in enumerate(expected_ids, start=1)
        }

        def generated_row(scenario_id: str) -> dict[str, object]:
            return {
                "scenario_id": scenario_id,
                "messages": [{"role": "user", "content": "完整测试请求"}],
                "scenario_summary": "测试摘要",
            }

        valid_generated = [generated_row(value) for value in expected_ids]
        legacy, _ = parse_generated_samples(
            json.dumps(
                {
                    "samples": [
                        *valid_generated,
                        generated_row("unexpected-id"),
                        "not-an-object",
                    ]
                },
                ensure_ascii=False,
            ),
            assigned,
            ("Target",),
        )
        self.assertEqual(set(legacy), set(expected_ids))
        for envelope_name, extra in (
            ("unknown-id", generated_row("unexpected-id")),
            ("non-object", "not-an-object"),
        ):
            with self.subTest(parser="generation", envelope=envelope_name):
                parsed, errors = parse_generated_samples(
                    json.dumps(
                        {"samples": [*valid_generated, extra]},
                        ensure_ascii=False,
                    ),
                    assigned,
                    ("Target",),
                    strict_envelope=True,
                )
                self.assertEqual(parsed, {})
                self.assertTrue(all(errors[value] for value in expected_ids))

    def test_strict_contrast_requires_natural_semantic_link(self) -> None:
        sample = {
            "scenario_id": "natural-link",
            "messages": [{"role": "user", "content": "自然边界请求"}],
            "target_candidate_name": "Target",
            "contrast_candidate_name": "Contrast",
            "content_axis": "insufficient_input",
            "content_axis_definition": "输入本身固有无意义或信息不足",
        }
        prompt = contrast_messages(
            [sample],
            "taxonomy",
            require_natural_link=True,
        )
        self.assertIn("contrast_link_natural", prompt[0]["content"])
        schema = json.loads(prompt[1]["content"])["output_schema"]["audits"][0]
        self.assertIn("contrast_link_natural", schema)
        prompt_sample = json.loads(prompt[1]["content"])["samples"][0]
        self.assertEqual(prompt_sample["content_axis"], "insufficient_input")
        self.assertIn("特殊例外", prompt[0]["content"])

        audit = {
            "scenario_id": "natural-link",
            "target_semantics_present": True,
            "contrast_is_plausible": True,
            "target_preferred_over_contrast": True,
            "single_current_goal": True,
            "natural_expression": True,
            "contrast_link_natural": False,
            "reason": "通过无关背景制造了表面混淆",
        }
        parsed, errors = parse_contrast_audits(
            json.dumps({"audits": [audit]}, ensure_ascii=False),
            ("natural-link",),
            require_natural_link=True,
        )
        self.assertEqual(errors["natural-link"], [])
        combined = combine_contrast_audits(
            {"model-a": parsed["natural-link"], "model-b": parsed["natural-link"]}
        )
        self.assertFalse(combined["contrast_link_natural"])

        blueprint = DialogueBlueprint(
            scenario_id="natural-link",
            phenomenon="single_turn",
            target_candidate_name="Target",
            source_candidate_name=None,
            user_turn_count=1,
            seed=1,
            contrast_candidate_name="Contrast",
        )
        quality = {field: True for field in QUALITY_FIELDS}
        labeler = {"predicted_candidate_name": "Target", "quality": quality}
        reviewer = {
            "predicted_candidate_name": "Target",
            "observed_phenomenon": "single_turn",
            "quality": quality,
        }
        self.assertIn(
            "contrast_contrast_link_natural",
            acceptance_reasons(
                blueprint,
                labeler,
                reviewer,
                contrast=combined,
                require_contrast_link_natural=True,
            ),
        )

    def test_implementation_hash_manifest_is_opt_in_and_resume_sensitive(self) -> None:
        blueprint = DialogueBlueprint(
            scenario_id="manifest-case",
            phenomenon="progressive_reveal",
            target_candidate_name="LifeKnowledgeQA",
            source_candidate_name=None,
            user_turn_count=2,
            seed=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate_path = root / "candidates.json"
            taxonomy_path = root / "taxonomy.jsonl"
            generator_path = root / "generate_top1_multiturn.py"
            synthesis_path = root / "synthesis.py"
            candidate_path.write_text("{}\n", encoding="utf-8")
            taxonomy_path.write_text("{}\n", encoding="utf-8")
            generator_path.write_text("generator-v1\n", encoding="utf-8")
            synthesis_path.write_text("synthesis-v1\n", encoding="utf-8")
            implementation_paths = {
                "src/top1_data_gen/cli.py": generator_path,
                "src/top1_data_gen/synthesis.py": synthesis_path,
            }

            legacy_output = root / "legacy"
            _prepare_run(
                output_directory=legacy_output,
                pipeline_version="legacy",
                config={},
                candidate_path=candidate_path,
                taxonomy_path=taxonomy_path,
                endpoint="https://example.invalid",
                plans=[blueprint],
                taxonomy="taxonomy",
                implementation_paths=implementation_paths,
            )
            legacy_manifest = json.loads(
                (legacy_output / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertNotIn("implementation", legacy_manifest)
            generator_path.write_text("generator-v2\n", encoding="utf-8")
            _prepare_run(
                output_directory=legacy_output,
                pipeline_version="legacy",
                config={},
                candidate_path=candidate_path,
                taxonomy_path=taxonomy_path,
                endpoint="https://example.invalid",
                plans=[blueprint],
                taxonomy="taxonomy",
                implementation_paths=implementation_paths,
            )

            strict_output = root / "strict"
            strict_config = {"record_implementation_hashes": True}
            _prepare_run(
                output_directory=strict_output,
                pipeline_version="strict",
                config=strict_config,
                candidate_path=candidate_path,
                taxonomy_path=taxonomy_path,
                endpoint="https://example.invalid",
                plans=[blueprint],
                taxonomy="taxonomy",
                implementation_paths=implementation_paths,
            )
            strict_manifest = json.loads(
                (strict_output / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                set(strict_manifest["implementation"]),
                set(implementation_paths),
            )
            synthesis_path.write_text("synthesis-v2\n", encoding="utf-8")
            with self.assertRaises(Top1DataError):
                _prepare_run(
                    output_directory=strict_output,
                    pipeline_version="strict",
                    config=strict_config,
                    candidate_path=candidate_path,
                    taxonomy_path=taxonomy_path,
                    endpoint="https://example.invalid",
                    plans=[blueprint],
                    taxonomy="taxonomy",
                    implementation_paths=implementation_paths,
                )

    def test_single_turn_requires_two_model_contrast_consensus(self) -> None:
        blueprint = DialogueBlueprint(
            scenario_id="single-boundary",
            phenomenon="single_turn",
            target_candidate_name="TicketService",
            source_candidate_name=None,
            user_turn_count=1,
            seed=1,
            contrast_candidate_name="TravelGuide",
            content_axis="rail_ticket",
        )
        quality = {field: True for field in QUALITY_FIELDS}
        labeler = {
            "predicted_candidate_name": "TicketService",
            "quality": quality,
        }
        reviewer = {
            "predicted_candidate_name": "TicketService",
            "observed_phenomenon": "single_turn",
            "quality": quality,
        }
        approving = {
            "target_semantics_present": True,
            "contrast_is_plausible": True,
            "target_preferred_over_contrast": True,
            "single_current_goal": True,
            "natural_expression": True,
            "reason": "精确票务与旅行经验构成清晰边界",
        }
        combined = combine_contrast_audits(
            {"reviewer": approving, "crosscheck": approving}
        )

        self.assertIn(
            "missing_contrast_audit",
            acceptance_reasons(blueprint, labeler, reviewer),
        )
        self.assertEqual(
            acceptance_reasons(blueprint, labeler, reviewer, None, combined),
            [],
        )
        self.assertIn(
            "missing_plan_fidelity_audit",
            acceptance_reasons(
                blueprint,
                labeler,
                reviewer,
                None,
                combined,
                require_plan_fidelity=True,
            ),
        )
        plan_fidelity = {
            "target_axis_realized": True,
            "target_candidate_respected": True,
            "source_axis_realized": True,
            "source_candidate_respected": True,
            "phenomenon_realized": True,
            "no_extra_current_goal": True,
        }
        self.assertEqual(
            acceptance_reasons(
                blueprint,
                labeler,
                reviewer,
                None,
                combined,
                plan_fidelity,
                require_plan_fidelity=True,
            ),
            [],
        )

        rejecting = dict(approving)
        rejecting["contrast_is_plausible"] = False
        combined = combine_contrast_audits(
            {"reviewer": approving, "crosscheck": rejecting}
        )
        self.assertIn(
            "contrast_contrast_is_plausible",
            acceptance_reasons(blueprint, labeler, reviewer, None, combined),
        )


if __name__ == "__main__":
    unittest.main()
    contrast_messages,
    dialogue_quality_messages,
    generation_messages,
