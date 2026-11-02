"""Unit tests for provider-neutral LangChain model construction."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import Mock, patch

from pydantic import ValidationError
from langchain_core.messages import AIMessage

from agent.config import LLMConfig, load_config
from agent.llm import (
    ModelAssessmentError,
    ModelAssessmentResponse,
    ModelAssessor,
    ModelInitializationError,
    SkillDocument,
    SkillLoadingError,
    build_assessment_messages,
    create_chat_model,
    load_skills,
)
from agent.state import (
    DecisionAction,
    DecisionSource,
    PhasingAttempt,
    PhasingEvidence,
    PhasingOutcome,
    PhasingStrategy,
    RegionIdentity,
    RegionState,
)


def example_state() -> RegionState:
    baseline = PhasingAttempt(
        attempt_number=1,
        strategy_id="baseline",
        evidence=PhasingEvidence(outcome=PhasingOutcome.FAILED),
    )
    return RegionState(
        sample_id="synthetic-sample",
        region=RegionIdentity(
            gene="Eucgr.A00001",
            target_region="Gene start region",
            chromosome="Chr01",
            start_position=100,
            end_position=120,
        ),
        evidence=baseline.evidence,
        attempt_history=[baseline],
    )


def example_skill() -> SkillDocument:
    return SkillDocument(
        name="synthetic_skill.md",
        content="# Synthetic Skill\n\n**Status:** scaffold only",
    )


def model_assessor(
    response: dict[str, object],
    *,
    retry_strategies: list[PhasingStrategy] | None = None,
) -> tuple[ModelAssessor, Mock, Mock]:
    model = Mock()
    structured_model = Mock()
    structured_model.invoke.return_value = {
        "raw": AIMessage(content=json.dumps(response)),
        "parsed": response,
        "parsing_error": None,
    }
    model.with_structured_output.return_value = structured_model
    assessor = ModelAssessor(
        model=model,
        skills=[example_skill()],
        retry_strategies=retry_strategies or [],
    )
    return assessor, model, structured_model


class ModelFactoryTests(TestCase):
    @patch("agent.llm.init_chat_model")
    def test_configuration_is_passed_to_langchain(self, initializer: Mock) -> None:
        expected_model = Mock()
        initializer.return_value = expected_model
        config = LLMConfig(
            provider="ollama",
            name="qwen2.5:7b",
            temperature=0,
            options={"num_ctx": 4096},
        )

        model = create_chat_model(config)

        self.assertIs(model, expected_model)
        initializer.assert_called_once_with(
            model="qwen2.5:7b",
            model_provider="ollama",
            temperature=0.0,
            num_ctx=4096,
        )

    @patch("agent.llm.init_chat_model")
    def test_hosted_provider_uses_the_same_factory(self, initializer: Mock) -> None:
        initializer.return_value = Mock()
        config = LLMConfig(
            provider="openai",
            name="example-model",
            temperature=0,
            required_environment_variables=["TEST_PROVIDER_API_KEY"],
        )

        with patch.dict(os.environ, {"TEST_PROVIDER_API_KEY": "test-value"}):
            create_chat_model(config)

        initializer.assert_called_once_with(
            model="example-model",
            model_provider="openai",
            temperature=0.0,
        )

    @patch("agent.llm.init_chat_model")
    def test_missing_api_key_is_reported_before_initialization(
        self, initializer: Mock
    ) -> None:
        config = LLMConfig(
            provider="openai",
            name="example-model",
            temperature=0,
            required_environment_variables=["DEFINITELY_MISSING_TEST_API_KEY"],
        )

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(
                ModelInitializationError, "DEFINITELY_MISSING_TEST_API_KEY"
            ):
                create_chat_model(config)

        initializer.assert_not_called()

    @patch("agent.llm.init_chat_model", side_effect=ImportError("missing integration"))
    def test_missing_provider_package_has_clear_error(self, initializer: Mock) -> None:
        config = LLMConfig(
            provider="another-provider",
            name="example-model",
            temperature=0,
        )

        with self.assertRaisesRegex(ModelInitializationError, "not installed"):
            create_chat_model(config)

        initializer.assert_called_once()

    def test_credentials_cannot_be_stored_in_yaml_options(self) -> None:
        with self.assertRaisesRegex(ValidationError, "environment variables"):
            LLMConfig(
                provider="openai",
                name="example-model",
                temperature=0,
                options={"api_key": "must-not-be-stored-here"},
            )


class SkillLoadingTests(TestCase):
    def test_project_skills_load_in_stable_order_without_readme(self) -> None:
        config = load_config()

        skills = load_skills(config.paths.skills_directory)

        self.assertEqual(
            [skill.name for skill in skills],
            [
                "phasing_triage.md",
                "retry_policy.md",
            ],
        )

    def test_skill_without_status_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "unsafe.md"
            path.write_text("# Missing status", encoding="utf-8")

            with self.assertRaises(SkillLoadingError):
                load_skills(temporary_directory)

    def test_missing_skills_directory_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            missing = Path(temporary_directory) / "missing"

            with self.assertRaises(SkillLoadingError):
                load_skills(missing)


class ModelAssessorTests(TestCase):
    def test_prompt_contains_state_skills_and_approved_strategies(self) -> None:
        strategy = PhasingStrategy(
            strategy_id="approved-retry",
            parameters={"example_parameter": 1},
        )

        messages = build_assessment_messages(
            example_state(), [example_skill()], [strategy]
        )

        combined = "\n".join(str(message.content) for message in messages)
        self.assertIn("synthetic-sample", combined)
        self.assertIn("Synthetic Skill", combined)
        self.assertIn("approved-retry", combined)
        self.assertIn("escalate rather than inventing", combined)

    def test_model_acceptance_becomes_llm_assessment(self) -> None:
        assessor, model, structured_model = model_assessor(
            {
                "action": "accept",
                "confidence": 0.8,
                "rationale": "Synthetic structured response.",
                "retry_strategy": None,
            }
        )

        result = assessor.assess(example_state())

        self.assertEqual(result.action, DecisionAction.ACCEPT)
        self.assertEqual(result.source, DecisionSource.LLM)
        model.with_structured_output.assert_called_once_with(
            ModelAssessmentResponse, include_raw=True
        )
        structured_model.invoke.assert_called_once()

    def test_parser_failure_retains_raw_output_and_precise_error(self) -> None:
        assessor, _, structured_model = model_assessor({})
        structured_model.invoke.return_value = {
            "raw": AIMessage(content='{"action":"retry"}'),
            "parsed": None,
            "parsing_error": ValueError("missing retry_strategy"),
        }

        with self.assertRaises(ModelAssessmentError) as caught:
            assessor.assess(example_state())

        detail = caught.exception.audit_detail()
        self.assertEqual(detail["stage"], "parsing")
        self.assertIn("missing retry_strategy", detail["validation_error"])
        self.assertEqual(detail["raw_response"]["content"], '{"action":"retry"}')

    def test_approved_retry_strategy_is_accepted_exactly(self) -> None:
        strategy = PhasingStrategy(
            strategy_id="approved-retry",
            parameters={"example_parameter": 1},
        )
        assessor, _, _ = model_assessor(
            {
                "action": "retry",
                "confidence": 0.7,
                "rationale": "Synthetic approved retry.",
                "retry_strategy": strategy.model_dump(),
            },
            retry_strategies=[strategy],
        )

        result = assessor.assess(example_state())

        self.assertEqual(result.retry_strategy, strategy)

    def test_unapproved_retry_strategy_is_rejected(self) -> None:
        assessor, _, _ = model_assessor(
            {
                "action": "retry",
                "confidence": 0.7,
                "rationale": "Synthetic unapproved retry.",
                "retry_strategy": {
                    "strategy_id": "invented-retry",
                    "parameters": {},
                },
            }
        )

        with self.assertRaisesRegex(ModelAssessmentError, "unapproved"):
            assessor.assess(example_state())

    def test_model_cannot_alter_approved_strategy_parameters(self) -> None:
        approved = PhasingStrategy(
            strategy_id="approved-retry",
            parameters={"example_parameter": 1},
        )
        assessor, _, _ = model_assessor(
            {
                "action": "retry",
                "confidence": 0.7,
                "rationale": "Synthetic altered retry.",
                "retry_strategy": {
                    "strategy_id": "approved-retry",
                    "parameters": {"example_parameter": 999},
                },
            },
            retry_strategies=[approved],
        )

        with self.assertRaisesRegex(ModelAssessmentError, "altered parameters"):
            assessor.assess(example_state())
