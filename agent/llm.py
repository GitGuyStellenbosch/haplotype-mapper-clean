"""Provider-neutral LangChain chat-model construction.

Provider integrations supply a common ``BaseChatModel`` interface, so graph nodes
do not need to know whether inference is local or uses a hosted API.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent.config import AgentConfig, LLMConfig
from agent.nodes import AssessmentResult
from agent.state import DecisionAction, DecisionSource, PhasingStrategy, RegionState


class ModelInitializationError(RuntimeError):
    """Raised when the selected model provider cannot be initialized safely."""


class ModelAssessmentError(RuntimeError):
    """Raised when model-backed assessment cannot produce a safe structured result."""

    def __init__(self, message: str, *, stage: str = "validation",
                 detail: str | None = None, raw_response: object = None) -> None:
        super().__init__(message)
        self.stage = stage
        self.detail = detail
        self.raw_response = raw_response

    def audit_detail(self) -> dict[str, object]:
        """Retain failed model output without changing the safe exception message."""

        raw = self.raw_response
        if isinstance(raw, BaseModel):
            raw = raw.model_dump(mode="json", exclude_none=True)
        return {"stage": self.stage, "validation_error": self.detail,
                "raw_response": raw}


class SkillLoadingError(ValueError):
    """Raised when assessment Skills cannot be loaded safely."""


@dataclass(frozen=True)
class SkillDocument:
    """One named Markdown domain-knowledge document."""

    name: str
    content: str


class ModelAssessmentResponse(BaseModel):
    """Provider-neutral structured response requested from the model."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    action: DecisionAction
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)
    retry_strategy: PhasingStrategy | None = None

    @model_validator(mode="after")
    def retry_has_exactly_one_strategy(self) -> "ModelAssessmentResponse":
        if self.action == DecisionAction.RETRY and self.retry_strategy is None:
            raise ValueError("a retry response requires retry_strategy")
        if self.action != DecisionAction.RETRY and self.retry_strategy is not None:
            raise ValueError("retry_strategy is only valid for a retry response")
        return self


def _check_required_environment(config: LLMConfig) -> None:
    """Confirm credential variables exist without reading or logging their values."""

    missing = [
        variable
        for variable in config.required_environment_variables
        if not os.environ.get(variable)
    ]
    if missing:
        names = ", ".join(missing)
        raise ModelInitializationError(
            f"missing required environment variable(s) for {config.provider}: {names}"
        )


def create_chat_model(config: LLMConfig) -> BaseChatModel:
    """Create the configured model through LangChain's unified initializer.

    The provider's integration package must be installed separately. For example,
    Ollama uses ``langchain-ollama`` and OpenAI uses ``langchain-openai``.
    """

    _check_required_environment(config)
    try:
        return init_chat_model(
            model=config.name,
            model_provider=config.provider,
            temperature=config.temperature,
            **config.options,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        raise ModelInitializationError(
            f"LangChain integration for provider '{config.provider}' is not installed"
        ) from exc
    except Exception as exc:
        raise ModelInitializationError(
            f"could not initialize model '{config.provider}:{config.name}': {exc}"
        ) from exc


def load_skills(skills_directory: str | Path) -> list[SkillDocument]:
    """Load non-README Markdown Skills in stable filename order."""

    directory = Path(skills_directory)
    if not directory.is_dir():
        raise SkillLoadingError(f"Skills directory does not exist: {directory}")

    skill_paths = sorted(
        path
        for path in directory.glob("*.md")
        if path.name.lower() != "readme.md"
    )
    if not skill_paths:
        raise SkillLoadingError(f"no Markdown Skills found in: {directory}")

    documents: list[SkillDocument] = []
    for path in skill_paths:
        try:
            content = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise SkillLoadingError(f"could not read Skill: {path}") from exc
        if not content:
            raise SkillLoadingError(f"Skill is empty: {path}")
        if "**Status:**" not in content:
            raise SkillLoadingError(f"Skill does not declare its validation status: {path}")
        documents.append(SkillDocument(name=path.name, content=content))
    return documents


def build_assessment_messages(
    state: RegionState,
    skills: list[SkillDocument],
    retry_strategies: list[PhasingStrategy],
) -> list[SystemMessage | HumanMessage]:
    """Build a bounded prompt containing evidence, guidance, and allowed actions."""

    skills_text = "\n\n".join(
        f"<skill name={json.dumps(skill.name)}>\n{skill.content}\n</skill>"
        for skill in skills
    )
    strategies_json = json.dumps(
        [strategy.model_dump(mode="json") for strategy in retry_strategies],
        indent=2,
        sort_keys=True,
    )
    state_json = json.dumps(
        state.model_dump(mode="json"),
        indent=2,
        sort_keys=True,
    )

    system_message = SystemMessage(
        content=(
            "You are the constrained assessment component of Haplotype Mapper. "
            "Choose only accept, retry, or escalate. Use only the supplied case "
            "evidence and Skills. Apply active research policy as operational "
            "guidance, without claiming biological correctness or expert approval. "
            "Missing evidence is unknown, never zero. If no policy applies, "
            "escalate rather than inventing a rule. "
            "Retry is permitted only by selecting one listed "
            "strategy exactly, including its parameters. Do not propose commands, "
            "new tools, or unlisted parameters.\n\n"
            f"Skills:\n{skills_text}\n\n"
            f"Allowed retry strategies:\n{strategies_json}"
        )
    )
    human_message = HumanMessage(
        content=f"Assess this normalized case state:\n{state_json}"
    )
    return [system_message, human_message]


class ModelAssessor:
    """Assessor implementation backed by any LangChain chat model."""

    def __init__(
        self,
        *,
        model: BaseChatModel,
        skills: list[SkillDocument],
        retry_strategies: list[PhasingStrategy],
    ) -> None:
        self._skills = list(skills)
        self._retry_strategies = list(retry_strategies)
        self._strategies_by_id = {
            strategy.strategy_id: strategy for strategy in retry_strategies
        }
        if len(self._strategies_by_id) != len(retry_strategies):
            raise ModelAssessmentError("retry strategy IDs must be unique")
        try:
            self._structured_model = model.with_structured_output(
                ModelAssessmentResponse, include_raw=True
            )
        except Exception as exc:
            raise ModelAssessmentError(
                "selected model does not support the required structured output"
            ) from exc

    @classmethod
    def from_config(
        cls,
        *,
        model: BaseChatModel,
        config: AgentConfig,
    ) -> "ModelAssessor":
        return cls(
            model=model,
            skills=load_skills(config.paths.skills_directory),
            retry_strategies=config.pipeline.retry_strategies,
        )

    def assess(self, state: RegionState) -> AssessmentResult:
        messages = build_assessment_messages(
            state,
            self._skills,
            self._retry_strategies,
        )
        try:
            result = self._structured_model.invoke(messages)
        except Exception as exc:
            raise ModelAssessmentError(
                "model invocation failed", stage="invocation",
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc

        raw_response = result.get("raw") if isinstance(result, dict) else None
        try:
            if not isinstance(result, dict) or not {"raw", "parsed", "parsing_error"}.issubset(result):
                raise ValueError("structured model did not return the raw/parsed/error envelope")
            if result["parsing_error"] is not None:
                raise result["parsing_error"]
            response = ModelAssessmentResponse.model_validate(result["parsed"])
        except Exception as exc:
            raise ModelAssessmentError(
                "model did not return a valid structured assessment",
                stage="parsing", detail=f"{type(exc).__name__}: {exc}",
                raw_response=raw_response,
            ) from exc

        retry_strategy = response.retry_strategy
        if retry_strategy is not None:
            approved = self._strategies_by_id.get(retry_strategy.strategy_id)
            if approved is None:
                raise ModelAssessmentError(
                    f"model proposed unapproved retry strategy "
                    f"'{retry_strategy.strategy_id}'", stage="strategy_validation",
                    detail=f"unapproved strategy ID: {retry_strategy.strategy_id}",
                    raw_response=raw_response,
                )
            if retry_strategy != approved:
                raise ModelAssessmentError(
                    f"model altered parameters for retry strategy "
                    f"'{retry_strategy.strategy_id}'", stage="strategy_validation",
                    detail=f"strategy parameters differ from allowlist: {retry_strategy.strategy_id}",
                    raw_response=raw_response,
                )

        return AssessmentResult(
            action=response.action,
            confidence=response.confidence,
            rationale=response.rationale,
            source=DecisionSource.LLM,
            retry_strategy=retry_strategy,
        )


__all__ = [
    "ModelAssessmentError",
    "ModelAssessor",
    "ModelInitializationError",
    "SkillDocument",
    "SkillLoadingError",
    "build_assessment_messages",
    "create_chat_model",
    "load_skills",
]
