"""Validated shared state passed between Haplotype Mapper graph nodes.

This module describes data; it does not contain triage rules. In particular, it
does not translate Candotti region categories into per-sample agent decisions.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Literal

# Pydantic validates constructed and updated records against these declared types
# and constraints.
from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


ScalarValue = str | int | float | bool | None


def utc_now() -> datetime:
    """Return an aware UTC timestamp for reproducible decision logs."""

    return datetime.now(timezone.utc)


# The closed action set prevents the agent from expanding its own authority.
class DecisionAction(str, Enum):
    """The complete and intentionally closed agent action space."""

    ACCEPT = "accept"
    RETRY = "retry"
    ESCALATE = "escalate"


class DecisionSource(str, Enum):
    """Component responsible for a recorded decision."""

    DETERMINISTIC = "deterministic"
    LLM = "llm"
    HUMAN = "human"

# These outcomes describe pipeline evidence, not agent decisions.
class PhasingOutcome(str, Enum):
    """Normalized outcomes exposed by the deterministic pipeline boundary."""

    NOT_RUN = "not_run"
    PHASED = "phased"
    PARTIALLY_PHASED = "partially_phased"
    UNPHASED = "unphased"
    NO_PHASE_INFORMATIVE_VARIANTS = "no_phase_informative_variants"
    FAILED = "failed"
    UNEXPECTED_HAPLOTYPE_COUNT = "unexpected_haplotype_count"


class StateModel(BaseModel):
    """Common strict configuration for all state records."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=False,
    )

# Region identity uses all five source fields to avoid catalogue key collisions.
class RegionIdentity(StateModel):
    """Collision-free identity for one catalogue target region."""

    gene: str = Field(min_length=1)
    target_region: str = Field(min_length=1)
    chromosome: str = Field(min_length=1)
    start_position: int = Field(ge=1, description="One-based 5' source position")
    end_position: int = Field(ge=1, description="One-based inclusive 3' position")

    @model_validator(mode="after")
    def coordinates_are_ordered(self) -> "RegionIdentity":
        """Reject reversed coordinates while allowing legitimate single-SNP regions."""

        if self.end_position < self.start_position:
            raise ValueError("end_position must be greater than or equal to start_position")
        return self

    @computed_field
    @property
    def region_key(self) -> str:
        """Return the collision-free key established during catalogue exploration."""

        return "|".join(
            (
                self.gene,
                self.target_region,
                self.chromosome,
                str(self.start_position),
                str(self.end_position),
            )
        )


class InputFiles(StateModel):
    """Files needed by a real per-individual phasing attempt.

    Paths are optional during preparation and are not required to exist when the
    state is constructed. The pipeline interface will perform runtime validation.
    """

    bam: Path | None = None
    vcf: Path | None = None
    reference_fasta: Path | None = None
    target_regions_bed: Path | None = None


class PhasingEvidence(StateModel):
    """Normalized evidence produced by one deterministic phasing run."""

    outcome: PhasingOutcome = PhasingOutcome.NOT_RUN
    heterozygous_snp_count: int | None = Field(default=None, ge=0)
    phased_snp_count: int | None = Field(default=None, ge=0)
    unphased_snp_count: int | None = Field(default=None, ge=0)
    block_count: int | None = Field(default=None, ge=0)
    block_n50: int | None = Field(
        default=None,
        ge=0,
        description="WhatsHap phase-block N50 in base pairs",
    )
    haplotype_count: int | None = Field(default=None, ge=0)
    call_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    family_data_available: bool = False
    mendelian_errors: int | None = Field(default=None, ge=0)
    tool_messages: list[str] = Field(default_factory=list)
    additional_metrics: dict[str, ScalarValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def family_fields_are_consistent(self) -> "PhasingEvidence":
        """Prevent Mendelian evidence from appearing without family context."""

        if self.mendelian_errors is not None and not self.family_data_available:
            raise ValueError(
                "mendelian_errors requires family_data_available=True"
            )
        return self


class PhasingAttempt(StateModel):
    """Auditable record of one unique phasing strategy."""

    attempt_number: int = Field(ge=1)
    strategy_id: str = Field(
        min_length=1,
        description="Stable identifier for the tool and exact parameter strategy",
    )
    parameters: dict[str, ScalarValue] = Field(default_factory=dict)
    evidence: PhasingEvidence
    recorded_at: datetime = Field(default_factory=utc_now)


class PhasingStrategy(StateModel):
    """A proposed deterministic tool strategy for an initial run or retry."""

    strategy_id: str = Field(min_length=1)
    parameters: dict[str, ScalarValue] = Field(default_factory=dict)


class DecisionRecord(StateModel):
    """One proposed or terminal triage decision with its provenance."""

    action: DecisionAction
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)
    source: DecisionSource
    created_at: datetime = Field(default_factory=utc_now)


class RegionState(StateModel):
    """Shared case file for one sample at one target region."""

    schema_version: Literal["1.0"] = "1.0"
    sample_id: str = Field(min_length=1)
    region: RegionIdentity
    inputs: InputFiles = Field(default_factory=InputFiles)
    evidence: PhasingEvidence = Field(default_factory=PhasingEvidence)
    attempt_history: list[PhasingAttempt] = Field(default_factory=list)
    proposal_history: list[DecisionRecord] = Field(default_factory=list)
    decision_history: list[DecisionRecord] = Field(default_factory=list)
    decision: DecisionRecord | None = None
    pending_strategy: PhasingStrategy | None = None
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, ScalarValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def attempt_history_is_auditable(self) -> "RegionState":
        """Require sequential attempts and prohibit repeated strategies."""

        expected_numbers = list(range(1, len(self.attempt_history) + 1))
        actual_numbers = [attempt.attempt_number for attempt in self.attempt_history]
        if actual_numbers != expected_numbers:
            raise ValueError("attempt numbers must be sequential and start at 1")

        strategy_ids = [attempt.strategy_id for attempt in self.attempt_history]
        if len(strategy_ids) != len(set(strategy_ids)):
            raise ValueError("a phasing strategy cannot be attempted more than once")

        return self

    @computed_field
    @property
    def retries_used(self) -> int:
        """Count attempts after the initial phasing run."""

        return max(0, len(self.attempt_history) - 1)

    @computed_field
    @property
    def strategies_tried(self) -> list[str]:
        """Expose strategy history for failure-aware routing."""

        return [attempt.strategy_id for attempt in self.attempt_history]


__all__ = [
    "DecisionAction",
    "DecisionRecord",
    "DecisionSource",
    "InputFiles",
    "PhasingAttempt",
    "PhasingEvidence",
    "PhasingOutcome",
    "PhasingStrategy",
    "RegionIdentity",
    "RegionState",
    "ScalarValue",
]
