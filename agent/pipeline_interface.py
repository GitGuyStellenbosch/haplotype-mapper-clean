"""Boundary between agent orchestration and deterministic phasing tools.

The interface is usable with synthetic evidence today. A real WhatsHap backend can
later implement the same protocol without adding biological judgement to this layer.
"""

from __future__ import annotations

from typing import Mapping, Protocol

from pydantic import BaseModel, ConfigDict, Field

from agent.state import (
    InputFiles,
    PhasingAttempt,
    PhasingEvidence,
    PhasingStrategy,
    RegionIdentity,
    RegionState,
)
from pipeline.preflight import (
    InputValidationReport,
    inspect_real_inputs,
)


class PipelineError(RuntimeError):
    """Base error for deterministic pipeline-interface failures."""


class InputValidationError(PipelineError):
    """Raised when real phasing inputs fail deterministic preflight checks."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("invalid phasing inputs: " + "; ".join(errors))


class StrategyError(PipelineError):
    """Raised when a strategy is unknown or has already been attempted."""


class PipelineModel(BaseModel):
    """Strict common configuration for pipeline-interface records."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class PhasingRequest(PipelineModel):
    """Provider-neutral request sent to a phasing backend."""

    sample_id: str = Field(min_length=1)
    region: RegionIdentity
    inputs: InputFiles
    strategy: PhasingStrategy


class PhasingBackend(Protocol):
    """Contract implemented by synthetic and future real phasing backends."""

    def phase(self, request: PhasingRequest) -> PhasingEvidence:
        """Run one deterministic strategy and return normalized evidence."""


def validate_real_inputs(
    inputs: InputFiles,
    *,
    expected_sample: str | None = None,
    expected_ploidy: int | None = None,
) -> InputValidationReport:
    """Run preflight and raise one pipeline error containing all failures."""

    report = inspect_real_inputs(
        inputs,
        expected_sample=expected_sample,
        expected_ploidy=expected_ploidy,
    )
    if not report.is_valid:
        raise InputValidationError(report.errors)
    return report


class MockPhasingBackend:
    """Deterministic synthetic backend for graph development and testing.

    Results are keyed by strategy ID. They are provided by the caller rather than
    inferred, and therefore carry no biological meaning.
    """

    def __init__(self, results: Mapping[str, PhasingEvidence]) -> None:
        self._results = dict(results)
        self.requests: list[PhasingRequest] = []

    def phase(self, request: PhasingRequest) -> PhasingEvidence:
        try:
            evidence = self._results[request.strategy.strategy_id]
        except KeyError as exc:
            raise StrategyError(
                f"no synthetic result configured for strategy "
                f"'{request.strategy.strategy_id}'"
            ) from exc

        self.requests.append(request)
        return evidence.model_copy(deep=True)


class PipelineInterface:
    """Facade used by graph nodes to request one auditable phasing attempt."""

    def __init__(
        self,
        backend: PhasingBackend,
        *,
        validate_inputs: bool,
        expected_ploidy: int | None = None,
    ) -> None:
        self._backend = backend
        self._validate_inputs = validate_inputs
        self._expected_ploidy = expected_ploidy

    def run(self, state: RegionState, strategy: PhasingStrategy) -> PhasingAttempt:
        """Run one new strategy for a region and normalize it as an attempt record."""

        if strategy.strategy_id in state.strategies_tried:
            raise StrategyError(
                f"strategy '{strategy.strategy_id}' has already been attempted"
            )
        if self._validate_inputs:
            validate_real_inputs(
                state.inputs,
                expected_sample=state.sample_id,
                expected_ploidy=self._expected_ploidy,
            )

        request = PhasingRequest(
            sample_id=state.sample_id,
            region=state.region,
            inputs=state.inputs,
            strategy=strategy,
        )
        evidence = self._backend.phase(request)
        return PhasingAttempt(
            attempt_number=len(state.attempt_history) + 1,
            strategy_id=strategy.strategy_id,
            parameters=strategy.parameters,
            evidence=evidence,
        )


__all__ = [
    "InputValidationError",
    "InputValidationReport",
    "MockPhasingBackend",
    "PhasingBackend",
    "PhasingRequest",
    "PhasingStrategy",
    "PipelineError",
    "PipelineInterface",
    "StrategyError",
    "inspect_real_inputs",
    "validate_real_inputs",
]
