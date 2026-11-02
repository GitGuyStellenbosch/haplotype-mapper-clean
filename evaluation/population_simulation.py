"""Frequency-grounded genotype simulation with explicitly non-action labels."""

from __future__ import annotations

import gzip
import hashlib
import heapq
import io
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from math import fsum
from pathlib import Path
from typing import Iterator, Literal, Sequence

from openpyxl import load_workbook
from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent.state import (
    PhasingAttempt,
    PhasingEvidence,
    PhasingOutcome,
    RegionIdentity,
    RegionState,
)


POPULATION_SIMULATION_DATA_STATUS = "simulated_from_population_frequencies"
DEFAULT_SIMULATION_SEED = 20260907
DEFAULT_INDIVIDUAL_COUNT = 10
DEFAULT_PILOT_PER_STRATUM = 5
SOURCE_POPULATION = "E. grandis"
SIMULATION_CONVENTION = "perfect_genotype_observation_v1"
SIMULATION_RECORDED_AT = datetime(2026, 9, 7, tzinfo=timezone.utc)
REGION_COLUMNS = (
    "Gene",
    "Target Region",
    "Chromosome",
    "5' Position",
    "3'Position",
)


class PopulationSimulationError(ValueError):
    """Raised when catalogue data or simulation settings are unsafe."""


class SimulationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class HaplotypeDraw(SimulationModel):
    """One sampled catalogue haplotype copy."""

    haplotype_id: str = Field(min_length=1)
    haplotype_key: str = Field(min_length=1)
    snp_haplotype: str = Field(min_length=1)
    source_frequency: float = Field(gt=0.0, le=1.0)


class SampledGenotypeTruth(SimulationModel):
    """Known genotype label produced directly by weighted sampling."""

    data_status: Literal[
        "simulated_from_population_frequencies"
    ] = POPULATION_SIMULATION_DATA_STATUS
    truth_level: Literal["genotype"] = "genotype"
    action_ground_truth_available: Literal[False] = False
    individual_id: str = Field(min_length=1)
    partition: Literal["train", "development", "held_out"]
    region_key: str = Field(min_length=1)
    catalogue_category: Literal[1, 2, 3]
    source_population: Literal["E. grandis"] = SOURCE_POPULATION
    simulation_seed: int
    haplotype_1: HaplotypeDraw
    haplotype_2: HaplotypeDraw
    zygosity: Literal["homozygous", "heterozygous"]
    heterozygous_snp_count: int = Field(ge=0)

    @model_validator(mode="after")
    def sampled_truth_is_consistent(self) -> "SampledGenotypeTruth":
        observed_count = sum(
            left != right
            for left, right in zip(
                self.haplotype_1.snp_haplotype,
                self.haplotype_2.snp_haplotype,
                strict=True,
            )
        )
        expected_zygosity = "homozygous" if observed_count == 0 else "heterozygous"
        if self.heterozygous_snp_count != observed_count:
            raise ValueError("heterozygous SNP count does not match haplotype strings")
        if self.zygosity != expected_zygosity:
            raise ValueError("zygosity does not match haplotype strings")
        return self


class PopulationSimulationCase(SimulationModel):
    """One generated state plus genotype truth kept outside the model prompt."""

    simulation_schema_version: Literal["1.0"] = "1.0"
    data_status: Literal[
        "simulated_from_population_frequencies"
    ] = POPULATION_SIMULATION_DATA_STATUS
    case_id: str = Field(min_length=1)
    truth: SampledGenotypeTruth
    state: RegionState

    @model_validator(mode="after")
    def state_and_truth_are_consistent(self) -> "PopulationSimulationCase":
        if self.state.sample_id != self.truth.individual_id:
            raise ValueError("state sample_id does not match sampled individual")
        if self.state.region.region_key != self.truth.region_key:
            raise ValueError("state region key does not match sampled truth")
        if self.state.metadata.get("data_status") != self.data_status:
            raise ValueError("state does not carry the simulation data-status tag")
        if (
            self.state.evidence.heterozygous_snp_count
            != self.truth.heterozygous_snp_count
        ):
            raise ValueError("state evidence does not match sampled genotype truth")
        if self.state.evidence.family_data_available:
            raise ValueError("population simulation cannot claim family evidence")
        return self


class ExcludedPopulationRegion(SimulationModel):
    """Catalogue region that cannot be sampled for the selected population."""

    region_key: str = Field(min_length=1)
    source_population: Literal["E. grandis"] = SOURCE_POPULATION
    source_frequency_sum: float = Field(ge=0.0)
    reason: Literal["zero_population_frequency"] = "zero_population_frequency"


class PopulationDatasetManifest(SimulationModel):
    """Compact provenance record for one complete generated dataset."""

    manifest_schema_version: Literal["1.0"] = "1.0"
    data_status: Literal[
        "simulated_from_population_frequencies"
    ] = POPULATION_SIMULATION_DATA_STATUS
    truth_level: Literal["genotype"] = "genotype"
    action_ground_truth_available: Literal[False] = False
    scoring_performed: Literal[False] = False
    source_workbook: str
    source_sheet: str
    source_population: Literal["E. grandis"] = SOURCE_POPULATION
    simulation_convention: Literal[
        "perfect_genotype_observation_v1"
    ] = SIMULATION_CONVENTION
    simulation_seed: int
    individual_count: int = Field(ge=1)
    source_region_count: int = Field(ge=1)
    sampled_region_count: int = Field(ge=1)
    generated_case_count: int = Field(ge=1)
    excluded_regions: list[ExcludedPopulationRegion]
    partition_assignments: dict[str, Literal["train", "development", "held_out"]]
    cases_path: str
    cases_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pilot_cases_path: str
    pilot_case_count: int = Field(ge=0)
    pilot_strata: dict[str, int]
    interpretation_boundary: str = (
        "Known genotype labels only; no accept/retry/escalate action truth or accuracy."
    )


@dataclass(frozen=True)
class CatalogueHaplotype:
    haplotype_id: str
    sequence: str
    frequency: float


@dataclass(frozen=True)
class CatalogueRegion:
    identity: RegionIdentity
    category: int
    snp_count: int
    haplotypes: tuple[CatalogueHaplotype, ...]


def _region_key(values: Sequence[object]) -> str:
    return "|".join(str(value) for value in values)


def load_population_catalogue(
    workbook_path: str | Path,
) -> tuple[list[CatalogueRegion], list[ExcludedPopulationRegion]]:
    """Load and validate the Candotti workbook for E. grandis sampling."""

    path = Path(workbook_path)
    try:
        workbook = load_workbook(path, read_only=True, data_only=True)
    except Exception as exc:
        raise PopulationSimulationError(f"cannot read catalogue workbook: {path}") from exc
    sheet_name = "Supplementary File 1"
    if sheet_name not in workbook.sheetnames:
        raise PopulationSimulationError(f"missing workbook sheet: {sheet_name}")

    rows = workbook[sheet_name].iter_rows(values_only=True)
    header: tuple[object, ...] | None = None
    for row in rows:
        if "Gene" in row:
            header = row
            break
    if header is None:
        raise PopulationSimulationError("could not locate catalogue header row")
    index = {str(name): position for position, name in enumerate(header)}
    required = (*REGION_COLUMNS, "Category", "Haplotype ID", "SNP Haplotype", "Number of SNPs", SOURCE_POPULATION)
    missing = [name for name in required if name not in index]
    if missing:
        raise PopulationSimulationError(
            "catalogue is missing required columns: " + ", ".join(missing)
        )

    grouped: dict[tuple[object, ...], list[CatalogueHaplotype]] = {}
    region_values: dict[tuple[object, ...], tuple[int, int]] = {}
    for row in rows:
        if not row or row[index["Gene"]] is None:
            continue
        key_values = tuple(row[index[name]] for name in REGION_COLUMNS)
        category = int(row[index["Category"]])
        snp_count = int(row[index["Number of SNPs"]])
        sequence = str(row[index["SNP Haplotype"]])
        frequency = float(row[index[SOURCE_POPULATION]])
        if category not in {1, 2, 3}:
            raise PopulationSimulationError(f"invalid category at {_region_key(key_values)}")
        if len(sequence) != snp_count or set(sequence) - set("ACGT"):
            raise PopulationSimulationError(
                f"invalid SNP haplotype at {_region_key(key_values)}"
            )
        if frequency < 0.0 or frequency > 1.0:
            raise PopulationSimulationError(
                f"invalid {SOURCE_POPULATION} frequency at {_region_key(key_values)}"
            )
        constants = region_values.setdefault(key_values, (category, snp_count))
        if constants != (category, snp_count):
            raise PopulationSimulationError(
                f"inconsistent region fields at {_region_key(key_values)}"
            )
        grouped.setdefault(key_values, []).append(
            CatalogueHaplotype(
                haplotype_id=str(row[index["Haplotype ID"]]),
                sequence=sequence,
                frequency=frequency,
            )
        )

    regions: list[CatalogueRegion] = []
    excluded: list[ExcludedPopulationRegion] = []
    for key_values in sorted(grouped, key=lambda values: tuple(map(str, values))):
        category, snp_count = region_values[key_values]
        haplotypes = tuple(
            sorted(grouped[key_values], key=lambda item: (item.haplotype_id, item.sequence))
        )
        if len({item.sequence for item in haplotypes}) != len(haplotypes):
            raise PopulationSimulationError(
                f"duplicate SNP haplotypes within {_region_key(key_values)}"
            )
        frequency_sum = fsum(item.frequency for item in haplotypes)
        region_key = _region_key(key_values)
        if frequency_sum == 0.0:
            excluded.append(
                ExcludedPopulationRegion(
                    region_key=region_key,
                    source_frequency_sum=frequency_sum,
                )
            )
            continue
        identity = RegionIdentity(
            gene=str(key_values[0]),
            target_region=str(key_values[1]),
            chromosome=str(key_values[2]),
            start_position=int(key_values[3]),
            end_position=int(key_values[4]),
        )
        regions.append(
            CatalogueRegion(
                identity=identity,
                category=category,
                snp_count=snp_count,
                haplotypes=haplotypes,
            )
        )
    if not regions:
        raise PopulationSimulationError("catalogue has no sampleable E. grandis regions")
    return regions, excluded


def _hash_integer(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def _weighted_draw(
    region: CatalogueRegion,
    *,
    seed: int,
    individual_id: str,
    copy_number: int,
) -> CatalogueHaplotype:
    total = fsum(item.frequency for item in region.haplotypes)
    unit = _hash_integer(seed, individual_id, region.identity.region_key, copy_number)
    target = (unit / (1 << 256)) * total
    cumulative = 0.0
    for haplotype in region.haplotypes:
        cumulative += haplotype.frequency
        if target < cumulative:
            return haplotype
    return next(item for item in reversed(region.haplotypes) if item.frequency > 0.0)


def assign_partitions(
    individual_ids: Sequence[str], seed: int
) -> dict[str, Literal["train", "development", "held_out"]]:
    """Apply the frozen 60/20/20 individual-level split from Phase 5."""

    if len(individual_ids) < 3:
        raise PopulationSimulationError("at least three individuals are needed for splits")
    ranked = sorted(individual_ids, key=lambda value: _hash_integer(seed, "partition", value))
    train_count = max(1, round(len(ranked) * 0.6))
    development_count = max(1, round(len(ranked) * 0.2))
    if train_count + development_count >= len(ranked):
        train_count = len(ranked) - 2
        development_count = 1
    assignments: dict[str, Literal["train", "development", "held_out"]] = {}
    for index, individual_id in enumerate(ranked):
        if index < train_count:
            partition = "train"
        elif index < train_count + development_count:
            partition = "development"
        else:
            partition = "held_out"
        assignments[individual_id] = partition
    return assignments


def build_population_case(
    region: CatalogueRegion,
    *,
    individual_id: str,
    partition: Literal["train", "development", "held_out"],
    seed: int,
) -> PopulationSimulationCase:
    """Sample two copies and expose them as perfectly observed genotype evidence."""

    first = _weighted_draw(region, seed=seed, individual_id=individual_id, copy_number=1)
    second = _weighted_draw(region, seed=seed, individual_id=individual_id, copy_number=2)
    heterozygous_count = sum(
        left != right for left, right in zip(first.sequence, second.sequence, strict=True)
    )
    zygosity = "homozygous" if heterozygous_count == 0 else "heterozygous"
    total_frequency = fsum(item.frequency for item in region.haplotypes)

    def draw(item: CatalogueHaplotype) -> HaplotypeDraw:
        return HaplotypeDraw(
            haplotype_id=item.haplotype_id,
            haplotype_key=f"{region.identity.region_key}|{item.haplotype_id}",
            snp_haplotype=item.sequence,
            source_frequency=item.frequency / total_frequency,
        )

    evidence = PhasingEvidence(
        outcome=(
            PhasingOutcome.NO_PHASE_INFORMATIVE_VARIANTS
            if heterozygous_count == 0
            else PhasingOutcome.PHASED
        ),
        heterozygous_snp_count=heterozygous_count,
        phased_snp_count=heterozygous_count,
        unphased_snp_count=0,
        block_count=0 if heterozygous_count == 0 else 1,
        block_n50=None,
        haplotype_count=1 if heterozygous_count == 0 else 2,
        call_rate=None,
        family_data_available=False,
        tool_messages=[
            "Population-frequency genotype simulation; no reads, alignment, or WhatsHap run.",
            "Perfect-observation convention: all sampled heterozygous SNP differences are treated as phased.",
            "Block N50 and empirical call rate are unavailable and were not fabricated.",
        ],
        additional_metrics={
            "catalogue_snp_count": region.snp_count,
            "sampled_copy_count": 2,
            "perfect_observation_convention": True,
        },
    )
    attempt = PhasingAttempt(
        attempt_number=1,
        strategy_id=SIMULATION_CONVENTION,
        parameters={
            "source_population": SOURCE_POPULATION,
            "simulation_seed": seed,
            "read_simulation": False,
            "whatshap_executed": False,
        },
        evidence=evidence,
        recorded_at=SIMULATION_RECORDED_AT,
    )
    truth = SampledGenotypeTruth(
        individual_id=individual_id,
        partition=partition,
        region_key=region.identity.region_key,
        catalogue_category=region.category,
        simulation_seed=seed,
        haplotype_1=draw(first),
        haplotype_2=draw(second),
        zygosity=zygosity,
        heterozygous_snp_count=heterozygous_count,
    )
    state = RegionState(
        sample_id=individual_id,
        region=region.identity,
        evidence=evidence,
        attempt_history=[attempt],
        warnings=[
            "Genotype-level population simulation only; action ground truth is unavailable."
        ],
        metadata={
            "data_status": POPULATION_SIMULATION_DATA_STATUS,
            "source_population": SOURCE_POPULATION,
            "simulation_convention": SIMULATION_CONVENTION,
            "simulation_seed": seed,
            "partition": partition,
        },
    )
    return PopulationSimulationCase(
        case_id=f"{individual_id}|{region.identity.region_key}",
        truth=truth,
        state=state,
    )


class _PilotSelector:
    def __init__(self, per_stratum: int, seed: int) -> None:
        self.per_stratum = per_stratum
        self.seed = seed
        self.heaps: dict[
            tuple[int, str], list[tuple[int, str, PopulationSimulationCase]]
        ] = {}

    def consider(self, case: PopulationSimulationCase) -> None:
        if self.per_stratum == 0 or case.truth.partition != "development":
            return
        key = (case.truth.catalogue_category, case.truth.zygosity)
        priority = _hash_integer(self.seed, "pilot", case.case_id)
        item = (-priority, case.case_id, case)
        heap = self.heaps.setdefault(key, [])
        if len(heap) < self.per_stratum:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)

    def selected(self) -> list[PopulationSimulationCase]:
        expected = {
            (category, zygosity)
            for category in (1, 2, 3)
            for zygosity in ("homozygous", "heterozygous")
        }
        missing = [
            f"category_{category}:{zygosity}"
            for category, zygosity in sorted(expected)
            if len(self.heaps.get((category, zygosity), [])) < self.per_stratum
        ]
        if missing:
            raise PopulationSimulationError(
                "not enough development cases for pilot strata: " + ", ".join(missing)
            )
        cases = [item[2] for heap in self.heaps.values() for item in heap]
        return sorted(
            cases,
            key=lambda case: (
                case.truth.catalogue_category,
                case.truth.zygosity,
                case.case_id,
            ),
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: BaseModel | list[BaseModel]) -> None:
    payload = (
        [
            item.model_dump(mode="json", exclude_computed_fields=True)
            for item in value
        ]
        if isinstance(value, list)
        else value.model_dump(mode="json", exclude_computed_fields=True)
    )
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def generate_population_dataset(
    *,
    workbook_path: str | Path,
    output_directory: str | Path,
    seed: int = DEFAULT_SIMULATION_SEED,
    individual_count: int = DEFAULT_INDIVIDUAL_COUNT,
    pilot_per_stratum: int = DEFAULT_PILOT_PER_STRATUM,
) -> PopulationDatasetManifest:
    """Stream the complete reproducible dataset and retain a small pilot manifest."""

    if individual_count < 3:
        raise PopulationSimulationError("individual_count must be at least 3")
    if pilot_per_stratum < 0:
        raise PopulationSimulationError("pilot_per_stratum cannot be negative")
    output = Path(output_directory)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"simulation output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    regions, excluded = load_population_catalogue(workbook_path)
    individual_ids = [
        f"SIM_EGRANDIS_{number:03d}"
        for number in range(1, individual_count + 1)
    ]
    partitions = assign_partitions(individual_ids, seed)
    selector = _PilotSelector(pilot_per_stratum, seed)
    cases_path = output / "cases.jsonl.gz"
    temporary_path = output / "cases.jsonl.gz.tmp"
    case_count = 0
    with temporary_path.open("xb") as raw_handle:
        with gzip.GzipFile(fileobj=raw_handle, mode="wb", mtime=0) as gzip_handle:
            with io.TextIOWrapper(gzip_handle, encoding="utf-8", newline="\n") as text_handle:
                for region in regions:
                    for individual_id in individual_ids:
                        case = build_population_case(
                            region,
                            individual_id=individual_id,
                            partition=partitions[individual_id],
                            seed=seed,
                        )
                        text_handle.write(case.model_dump_json(exclude_computed_fields=True))
                        text_handle.write("\n")
                        selector.consider(case)
                        case_count += 1
    temporary_path.replace(cases_path)

    pilot_cases = selector.selected()
    pilot_path = output / "pilot_cases.json"
    _write_json(pilot_path, pilot_cases)
    _write_json(output / "excluded_regions.json", excluded)
    strata: dict[str, int] = {}
    for case in pilot_cases:
        key = f"category_{case.truth.catalogue_category}:{case.truth.zygosity}"
        strata[key] = strata.get(key, 0) + 1

    manifest = PopulationDatasetManifest(
        source_workbook=str(Path(workbook_path)),
        source_sheet="Supplementary File 1",
        simulation_seed=seed,
        individual_count=individual_count,
        source_region_count=len(regions) + len(excluded),
        sampled_region_count=len(regions),
        generated_case_count=case_count,
        excluded_regions=excluded,
        partition_assignments=partitions,
        cases_path=str(cases_path),
        cases_sha256=_sha256_file(cases_path),
        pilot_cases_path=str(pilot_path),
        pilot_case_count=len(pilot_cases),
        pilot_strata=dict(sorted(strata.items())),
    )
    _write_json(output / "generation_manifest.json", manifest)
    return manifest


def load_population_cases(path: str | Path) -> list[PopulationSimulationCase]:
    """Load the compact JSON pilot-case array and validate every record."""

    try:
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        return [PopulationSimulationCase.model_validate(value) for value in values]
    except Exception as exc:
        raise PopulationSimulationError(f"cannot load simulation cases: {path}") from exc


def iter_population_dataset(path: str | Path) -> Iterator[PopulationSimulationCase]:
    """Stream and validate a complete compressed JSONL dataset."""

    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                yield PopulationSimulationCase.model_validate_json(line)
            except Exception as exc:
                raise PopulationSimulationError(
                    f"invalid simulation case at line {line_number}"
                ) from exc


__all__ = [
    "DEFAULT_INDIVIDUAL_COUNT",
    "DEFAULT_PILOT_PER_STRATUM",
    "DEFAULT_SIMULATION_SEED",
    "POPULATION_SIMULATION_DATA_STATUS",
    "CatalogueHaplotype",
    "CatalogueRegion",
    "ExcludedPopulationRegion",
    "HaplotypeDraw",
    "PopulationDatasetManifest",
    "PopulationSimulationCase",
    "PopulationSimulationError",
    "SampledGenotypeTruth",
    "assign_partitions",
    "build_population_case",
    "generate_population_dataset",
    "iter_population_dataset",
    "load_population_cases",
    "load_population_catalogue",
]
