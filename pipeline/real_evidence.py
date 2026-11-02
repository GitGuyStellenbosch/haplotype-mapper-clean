"""Extract normalized per-region evidence from an existing phased VCF.

This module reads evidence only. It does not run WhatsHap and does not translate
observations into accept/retry/escalate decisions.
"""

from __future__ import annotations

import bisect
import csv
import gzip
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from agent.state import (
    InputFiles,
    PhasingAttempt,
    PhasingEvidence,
    PhasingOutcome,
    RegionIdentity,
    RegionState,
)


REAL_EVIDENCE_DATA_STATUS = "real_evidence_no_ground_truth"


class RealEvidenceError(ValueError):
    """Raised when catalogue or VCF evidence cannot be interpreted safely."""


@dataclass(frozen=True)
class CatalogueRegion:
    """One target region matched exactly between the BED and catalogue CSV."""

    identity: RegionIdentity
    bed_start: int
    bed_end: int
    catalogue_category: str
    catalogue_row: int


@dataclass(frozen=True)
class VariantObservation:
    """Minimal genotype evidence needed for regional SNP counts."""

    position: int
    genotype: str
    called: bool
    heterozygous: bool
    phased: bool
    phase_set: str | None
    ploidy: int | None
    reference_allele: str = ""
    alternate_allele: str = ""


@dataclass(frozen=True)
class ExtractedRegionState:
    """Real state plus catalogue context deliberately excluded from model state."""

    state: RegionState
    catalogue_category: str
    catalogue_row: int


def _open_text(path: Path) -> TextIO:
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, mode="rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def load_target_catalogue(
    bed_path: str | Path,
    region_categories_path: str | Path,
) -> list[CatalogueRegion]:
    """Match zero-based BED rows to one-based inclusive catalogue identities."""

    bed = Path(bed_path)
    categories = Path(region_categories_path)
    try:
        with bed.open(encoding="utf-8") as handle:
            bed_rows = [
                line.rstrip("\n").split("\t")
                for line in handle
                if line.strip() and not line.startswith("#")
            ]
        with categories.open(encoding="utf-8", newline="") as handle:
            csv_rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise RealEvidenceError(f"could not read target catalogue inputs: {exc}") from exc

    if len(bed_rows) != len(csv_rows):
        raise RealEvidenceError(
            "target BED and region catalogue row counts differ: "
            f"{len(bed_rows)} != {len(csv_rows)}"
        )

    required_columns = {
        "Gene",
        "Target Region",
        "Chromosome",
        "5' Position",
        "3'Position",
        "Category",
    }
    if csv_rows and not required_columns.issubset(csv_rows[0]):
        missing = sorted(required_columns - set(csv_rows[0]))
        raise RealEvidenceError(
            "region catalogue is missing required column(s): " + ", ".join(missing)
        )

    regions: list[CatalogueRegion] = []
    seen_keys: set[str] = set()
    for row_number, (bed_fields, row) in enumerate(
        zip(bed_rows, csv_rows), start=2
    ):
        if len(bed_fields) < 3:
            raise RealEvidenceError(f"BED row {row_number - 1} has fewer than 3 fields")
        try:
            chromosome, bed_start_text, bed_end_text = bed_fields[:3]
            bed_start = int(bed_start_text)
            bed_end = int(bed_end_text)
            csv_start = int(row["5' Position"])
            csv_end = int(row["3'Position"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RealEvidenceError(
                f"invalid coordinates at catalogue row {row_number}"
            ) from exc

        if bed_start < 0 or bed_end <= bed_start:
            raise RealEvidenceError(f"invalid BED interval at row {row_number - 1}")
        if (chromosome, bed_start + 1, bed_end) != (
            row["Chromosome"],
            csv_start,
            csv_end,
        ):
            raise RealEvidenceError(
                f"BED/CSV coordinate mismatch at catalogue row {row_number}"
            )

        identity = RegionIdentity(
            gene=row["Gene"],
            target_region=row["Target Region"],
            chromosome=chromosome,
            start_position=csv_start,
            end_position=csv_end,
        )
        if identity.region_key in seen_keys:
            raise RealEvidenceError(
                f"duplicate collision-free region key: {identity.region_key}"
            )
        seen_keys.add(identity.region_key)
        regions.append(
            CatalogueRegion(
                identity=identity,
                bed_start=bed_start,
                bed_end=bed_end,
                catalogue_category=row["Category"],
                catalogue_row=row_number,
            )
        )
    return regions


def _parse_genotype(genotype: str, phase_set: str | None) -> VariantObservation:
    alleles = re.split(r"[|/]", genotype)
    ploidy = len(alleles) if genotype not in {"", "."} else None
    called = bool(alleles) and all(allele not in {"", "."} for allele in alleles)
    heterozygous = called and len(set(alleles)) > 1
    phased = heterozygous and "|" in genotype and "/" not in genotype
    return VariantObservation(
        position=0,
        genotype=genotype,
        called=called,
        heterozygous=heterozygous,
        phased=phased,
        phase_set=phase_set if phase_set not in {None, "", "."} else None,
        ploidy=ploidy,
    )


def read_single_sample_snp_vcf(
    vcf_path: str | Path,
    *,
    expected_sample: str | None = None,
) -> tuple[str, dict[str, list[VariantObservation]]]:
    """Read biallelic SNP genotype observations from a one-sample VCF."""

    path = Path(vcf_path)
    observations: dict[str, list[VariantObservation]] = defaultdict(list)
    sample_name: str | None = None
    try:
        with _open_text(path) as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.startswith("##"):
                    continue
                if line.startswith("#CHROM"):
                    columns = line.rstrip("\n").split("\t")
                    samples = columns[9:]
                    if len(samples) != 1:
                        raise RealEvidenceError(
                            f"real smoke VCF must contain exactly one sample; found {len(samples)}"
                        )
                    sample_name = samples[0]
                    if expected_sample is not None and sample_name != expected_sample:
                        raise RealEvidenceError(
                            f"VCF sample '{sample_name}' does not match expected "
                            f"sample '{expected_sample}'"
                        )
                    continue
                if line.startswith("#"):
                    continue
                if sample_name is None:
                    raise RealEvidenceError("VCF data appeared before the #CHROM header")

                fields = line.rstrip("\n").split("\t")
                if len(fields) < 10:
                    raise RealEvidenceError(f"VCF row {line_number} has fewer than 10 fields")
                chromosome, position_text, _, ref, alt = fields[:5]
                if len(ref) != 1 or len(alt) != 1 or "," in alt:
                    continue
                try:
                    position = int(position_text)
                except ValueError as exc:
                    raise RealEvidenceError(
                        f"invalid VCF position at row {line_number}"
                    ) from exc

                format_names = fields[8].split(":")
                sample_values = fields[9].split(":")
                format_values = dict(zip(format_names, sample_values))
                parsed = _parse_genotype(
                    format_values.get("GT", "."), format_values.get("PS")
                )
                observations[chromosome].append(
                    VariantObservation(
                        position=position,
                        genotype=parsed.genotype,
                        called=parsed.called,
                        heterozygous=parsed.heterozygous,
                        phased=parsed.phased,
                        phase_set=parsed.phase_set,
                        ploidy=parsed.ploidy,
                        reference_allele=ref,
                        alternate_allele=alt,
                    )
                )
    except OSError as exc:
        raise RealEvidenceError(f"could not read VCF: {path}") from exc

    if sample_name is None:
        raise RealEvidenceError("VCF has no #CHROM header")
    for chromosome, values in observations.items():
        if any(left.position > right.position for left, right in zip(values, values[1:])):
            raise RealEvidenceError(f"VCF records are not sorted on contig {chromosome}")
    return sample_name, dict(observations)


def _outcome_for_counts(
    heterozygous: int,
    phased: int,
    unphased: int,
) -> PhasingOutcome:
    if heterozygous == 0:
        return PhasingOutcome.NO_PHASE_INFORMATIVE_VARIANTS
    if phased == heterozygous:
        return PhasingOutcome.PHASED
    if phased > 0 and unphased > 0:
        return PhasingOutcome.PARTIALLY_PHASED
    return PhasingOutcome.UNPHASED


def build_real_region_states(
    *,
    vcf_path: str | Path,
    bam_path: str | Path,
    bed_path: str | Path,
    region_categories_path: str | Path,
    reference_fasta: str | Path | None,
    expected_sample: str | None = None,
) -> list[ExtractedRegionState]:
    """Intersect a phased VCF with catalogue regions and build covered states."""

    catalogue = load_target_catalogue(bed_path, region_categories_path)
    sample_name, by_chromosome = read_single_sample_snp_vcf(
        vcf_path, expected_sample=expected_sample
    )
    positions = {
        chromosome: [observation.position for observation in observations]
        for chromosome, observations in by_chromosome.items()
    }

    extracted: list[ExtractedRegionState] = []
    for region in catalogue:
        chromosome_observations = by_chromosome.get(region.identity.chromosome, [])
        chromosome_positions = positions.get(region.identity.chromosome, [])
        left = bisect.bisect_left(chromosome_positions, region.bed_start + 1)
        right = bisect.bisect_right(chromosome_positions, region.bed_end)
        covered = chromosome_observations[left:right]
        if not covered:
            continue

        total_snps = len(covered)
        called_snps = sum(observation.called for observation in covered)
        heterozygous = sum(observation.heterozygous for observation in covered)
        phased = sum(observation.phased for observation in covered)
        unphased = heterozygous - phased
        homozygous = sum(
            observation.called and not observation.heterozygous
            for observation in covered
        )
        missing = total_snps - called_snps
        phase_sets = {
            observation.phase_set
            for observation in covered
            if observation.phased and observation.phase_set is not None
        }
        ploidies = {
            observation.ploidy
            for observation in covered
            if observation.ploidy is not None
        }
        genotype_ploidy = next(iter(ploidies)) if len(ploidies) == 1 else None

        messages = [
            "Evidence extracted from an existing WhatsHap-phased VCF; the smoke "
            "runner did not re-run WhatsHap."
        ]
        if heterozygous == 0:
            messages.append("Region contains variants but no heterozygous SNPs.")
        if any(
            observation.phased and observation.phase_set is None
            for observation in covered
        ):
            messages.append("At least one phased genotype lacks a PS value.")
        if len(ploidies) > 1:
            messages.append("Inconsistent called genotype ploidy within region.")

        evidence = PhasingEvidence(
            outcome=_outcome_for_counts(heterozygous, phased, unphased),
            heterozygous_snp_count=heterozygous,
            phased_snp_count=phased,
            unphased_snp_count=unphased,
            call_rate=called_snps / total_snps,
            family_data_available=False,
            tool_messages=messages,
            additional_metrics={
                "total_snp_count": total_snps,
                "homozygous_snp_count": homozygous,
                "missing_genotype_snp_count": missing,
                "phase_set_count": len(phase_sets),
                "genotype_ploidy": genotype_ploidy,
            },
        )
        attempt = PhasingAttempt(
            attempt_number=1,
            strategy_id="provided-whatshap-polyphase-output",
            parameters={
                "source_vcf": str(Path(vcf_path)),
                "evidence_extraction": "regional_biallelic_snp_genotypes",
            },
            evidence=evidence,
        )
        state = RegionState(
            sample_id=sample_name,
            region=region.identity,
            inputs=InputFiles(
                bam=Path(bam_path),
                vcf=Path(vcf_path),
                reference_fasta=(
                    Path(reference_fasta) if reference_fasta is not None else None
                ),
                target_regions_bed=Path(bed_path),
            ),
            evidence=evidence,
            attempt_history=[attempt],
            warnings=[
                "Real biological evidence without sample-level ground truth; do not "
                "score or interpret this decision as correctness.",
                "No pedigree data are available for this sample.",
            ],
            metadata={
                "data_status": REAL_EVIDENCE_DATA_STATUS,
                "vcf_sample": sample_name,
                "catalogue_row": region.catalogue_row,
                "catalogue_category_withheld_from_assessor": True,
            },
        )
        extracted.append(
            ExtractedRegionState(
                state=state,
                catalogue_category=region.catalogue_category,
                catalogue_row=region.catalogue_row,
            )
        )
    return extracted


__all__ = [
    "CatalogueRegion",
    "ExtractedRegionState",
    "REAL_EVIDENCE_DATA_STATUS",
    "RealEvidenceError",
    "VariantObservation",
    "build_real_region_states",
    "load_target_catalogue",
    "read_single_sample_snp_vcf",
]
