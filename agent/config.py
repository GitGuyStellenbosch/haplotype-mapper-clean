"""Validated runtime configuration for the Haplotype Mapper agent.

Configuration contains experimental choices, not biological decision logic. Values
can therefore be changed and recorded per run without editing graph nodes.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from agent.state import DecisionAction, PhasingStrategy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG_PATH = PROJECT_ROOT / "config" / "parameters.yaml"


class ConfigurationError(ValueError):
    """Raised when a configuration file cannot be read or validated."""


class PipelineMode(str, Enum):
    """Available implementations behind the deterministic pipeline interface."""

    MOCK = "mock"
    REAL = "real"


class ConfigModel(BaseModel):
    """Base behavior shared by all configuration sections."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class LLMConfig(ConfigModel):
    """Provider-neutral settings for a LangChain chat model."""

    provider: str = Field(min_length=1)
    name: str = Field(min_length=1)
    temperature: float = Field(ge=0.0, le=2.0)
    required_environment_variables: list[str] = Field(default_factory=list)
    options: dict[str, str | int | float | bool | None] = Field(default_factory=dict)

    @model_validator(mode="after")
    def environment_variable_names_are_unique(self) -> "LLMConfig":
        """Validate credential declarations without allowing secrets in YAML."""

        if any(not name.strip() for name in self.required_environment_variables):
            raise ValueError("required environment variable names cannot be blank")
        if len(self.required_environment_variables) != len(
            set(self.required_environment_variables)
        ):
            raise ValueError("required environment variable names must be unique")

        sensitive_fragments = ("api_key", "apikey", "token", "secret", "password")
        unsafe_options = [
            name
            for name in self.options
            if any(fragment in name.lower() for fragment in sensitive_fragments)
        ]
        if unsafe_options:
            raise ValueError(
                "credentials must use environment variables, not model options: "
                + ", ".join(unsafe_options)
            )
        return self


class DecisionConfig(ConfigModel):
    """Settings controlling the closed triage decision space."""

    allowed_actions: list[DecisionAction]
    max_retries: int = Field(
        ge=0,
        description="Project-defined placeholder until empirically justified",
    )
    minimum_confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def action_space_is_complete_and_closed(self) -> "DecisionConfig":
        """Prevent configuration from silently changing the research question."""

        expected = set(DecisionAction)
        actual = set(self.allowed_actions)
        if actual != expected or len(self.allowed_actions) != len(expected):
            raise ValueError(
                "allowed_actions must contain accept, retry, and escalate exactly once"
            )
        return self


class PathConfig(ConfigModel):
    """Project paths used by the agent rather than biological input paths."""

    skills_directory: Path
    output_directory: Path

    def resolved(self, project_root: Path) -> "PathConfig":
        """Resolve relative paths consistently, independent of the current directory."""

        def resolve(path: Path) -> Path:
            return path.resolve() if path.is_absolute() else (project_root / path).resolve()

        return self.model_copy(
            update={
                "skills_directory": resolve(self.skills_directory),
                "output_directory": resolve(self.output_directory),
            }
        )


class PipelineConfig(ConfigModel):
    """Select the synthetic preparation interface or real bioinformatics tools."""

    mode: PipelineMode
    whatshap: "WhatsHapConfig"
    gatk: "GATKConfig"
    retry_strategies: list[PhasingStrategy] = Field(default_factory=list)

    @model_validator(mode="after")
    def retry_strategy_ids_are_unique(self) -> "PipelineConfig":
        strategy_ids = [strategy.strategy_id for strategy in self.retry_strategies]
        if len(strategy_ids) != len(set(strategy_ids)):
            raise ValueError("pipeline retry strategy IDs must be unique")
        return self


class StaticBaselineConfig(ConfigModel):
    """Versioned identity and fallback behavior of the fixed-rule comparator."""

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    fallback_action: DecisionAction = DecisionAction.ESCALATE
    fallback_confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def fallback_is_terminal(self) -> "StaticBaselineConfig":
        if self.fallback_action == DecisionAction.RETRY:
            raise ValueError("static baseline fallback cannot be retry")
        return self


class WhatsHapConfig(ConfigModel):
    """Human-controlled baseline parameters for WhatsHap polyphase."""

    executable: str = Field(min_length=1)
    ploidy: int = Field(ge=1)
    block_cut_sensitivity: int = Field(ge=0, le=5)
    use_reference: bool
    tag: Literal["PS", "HP"]
    mapping_quality: int = Field(ge=0)
    threads: int = Field(ge=1)
    only_snvs: bool
    ignore_read_groups: bool
    include_haploid_sets: bool
    distrust_genotypes: bool
    use_prephasing: bool
    min_overlap: int = Field(ge=1)
    no_mav: bool
    use_supplementary: bool
    supplementary_distance: int = Field(ge=0)


class GATKHaplotypeCallerConfig(ConfigModel):
    """Project-selected GATK HaplotypeCaller settings currently used manually."""

    sample_ploidy: int = Field(ge=1)
    emit_reference_confidence: Literal["GVCF", "BP_RESOLUTION", "NONE"]


class GATKGenotypeGVCFsConfig(ConfigModel):
    """Project-selected GenotypeGVCFs settings, including unresolved ploidy."""

    sample_ploidy: int | None = Field(default=None, ge=1)
    include_non_variant_sites: bool


class GATKHardFilterConfig(ConfigModel):
    """Hard-filter boundaries recovered from the real VCF provenance."""

    minimum_qd: float = Field(ge=0.0)
    minimum_qual: float = Field(ge=0.0)
    maximum_sor: float = Field(ge=0.0)
    maximum_fs: float = Field(ge=0.0)
    minimum_mq: float = Field(ge=0.0)
    minimum_mq_rank_sum: float
    minimum_read_pos_rank_sum: float


class VariantSelectionConfig(ConfigModel):
    """Post-filter bcftools site-selection policy used for the real VCF."""

    executable: str = Field(min_length=1)
    require_pass: bool
    minimum_alleles: int = Field(ge=1)
    maximum_alleles: int = Field(ge=1)
    snps_only: bool

    @model_validator(mode="after")
    def allele_bounds_are_ordered(self) -> "VariantSelectionConfig":
        if self.maximum_alleles < self.minimum_alleles:
            raise ValueError("maximum_alleles must be >= minimum_alleles")
        return self


class GATKConfig(ConfigModel):
    """Variant-calling and filtering parameters used by dry-run plans."""

    executable: str = Field(min_length=1)
    haplotype_caller: GATKHaplotypeCallerConfig
    genotype_gvcfs: GATKGenotypeGVCFsConfig
    hard_filters: GATKHardFilterConfig
    post_filter_selection: VariantSelectionConfig


class AgentLayerConfig(ConfigModel):
    """All human-tuned decision-layer settings."""

    model: LLMConfig
    decisions: DecisionConfig
    static_baseline: StaticBaselineConfig


class ExecutionConfig(ConfigModel):
    """Operational safeguards and progress controls, not biological policy."""

    graph_base_steps: int = Field(ge=1)
    graph_steps_per_retry: int = Field(ge=1)
    real_smoke_progress_every: int = Field(ge=1)
    whatshap_retry_timeout_seconds: int = Field(default=120, ge=1)

    def graph_recursion_limit(self, max_retries: int) -> int:
        return self.graph_base_steps + (max_retries * self.graph_steps_per_retry)


class EvaluationConfig(ConfigModel):
    """Human-selected experimental repetition settings."""

    consistency_repetitions: int = Field(ge=2)


class AnalysisConfig(ConfigModel):
    """Exploratory-analysis thresholds that do not control agent routing."""

    high_frequency_threshold: float = Field(ge=0.0, le=1.0)


class AgentConfig(ConfigModel):
    """Complete validated application configuration."""

    agent: AgentLayerConfig
    paths: PathConfig
    pipeline: PipelineConfig
    execution: ExecutionConfig
    evaluation: EvaluationConfig
    analysis: AnalysisConfig

    @property
    def model(self) -> LLMConfig:
        """Compatibility accessor for the configured model."""

        return self.agent.model

    @property
    def decisions(self) -> DecisionConfig:
        """Compatibility accessor for decision safeguards."""

        return self.agent.decisions

    @property
    def static_baseline(self) -> StaticBaselineConfig:
        """Compatibility accessor for the fixed-rule comparator."""

        return self.agent.static_baseline

    def resolved(self, project_root: Path = PROJECT_ROOT) -> "AgentConfig":
        """Return a copy with project-relative filesystem paths made absolute."""

        return self.model_copy(update={"paths": self.paths.resolved(project_root)})

    def with_pipeline_mode(self, mode: PipelineMode) -> "AgentConfig":
        """Return a run-specific copy without treating execution mode as tuning."""

        return self.model_copy(
            update={"pipeline": self.pipeline.model_copy(update={"mode": mode})}
        )


def _read_yaml(config_path: Path) -> dict[str, Any]:
    """Read a YAML mapping and convert parser/IO failures to one public error type."""

    try:
        with config_path.open(encoding="utf-8") as config_file:
            raw_config = yaml.safe_load(config_file)
    except OSError as exc:
        raise ConfigurationError(
            f"could not read configuration file: {config_path}"
        ) from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(
            f"configuration is not valid YAML: {config_path}"
        ) from exc

    if not isinstance(raw_config, dict):
        raise ConfigurationError("configuration root must be a YAML mapping")
    return raw_config


def load_config(
    config_path: str | Path = EXAMPLE_CONFIG_PATH,
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> AgentConfig:
    """Load, validate, and resolve an agent configuration file.

    Relative configuration file paths and configured directories are interpreted
    from ``project_root``, not from the caller's current working directory.
    """

    root = Path(project_root).resolve()
    path = Path(config_path)
    if not path.is_absolute():
        path = root / path

    raw_config = _read_yaml(path.resolve())
    try:
        config = AgentConfig.model_validate(raw_config)
    except ValidationError as exc:
        raise ConfigurationError(f"configuration validation failed: {exc}") from exc
    return config.resolved(root)


__all__ = [
    "AgentConfig",
    "AgentLayerConfig",
    "AnalysisConfig",
    "ConfigurationError",
    "DecisionConfig",
    "EvaluationConfig",
    "ExecutionConfig",
    "EXAMPLE_CONFIG_PATH",
    "GATKConfig",
    "GATKGenotypeGVCFsConfig",
    "GATKHaplotypeCallerConfig",
    "GATKHardFilterConfig",
    "LLMConfig",
    "PathConfig",
    "PipelineConfig",
    "PipelineMode",
    "PROJECT_ROOT",
    "StaticBaselineConfig",
    "VariantSelectionConfig",
    "WhatsHapConfig",
    "load_config",
]
