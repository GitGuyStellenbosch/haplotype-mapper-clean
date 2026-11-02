"""Summarize regional evidence across a real multi-sample manifest without an LLM."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Sequence

from pipeline.batch_preflight import load_batch_manifest
from pipeline.real_evidence import (
    REAL_EVIDENCE_DATA_STATUS,
    ExtractedRegionState,
    build_real_region_states,
)


DIFFICULT_OUTCOMES = {"partially_phased", "unphased", "failed"}


def summarize_states(states: Sequence[ExtractedRegionState]) -> dict[str, object]:
    """Return compact counts for one sample's already-normalized states."""

    outcomes: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    messages: Counter[str] = Counter()
    ploidies: Counter[str] = Counter()
    phase_sets: Counter[str] = Counter()
    call_rate_cases: Counter[str] = Counter()
    call_rates: list[float] = []
    totals: Counter[str] = Counter()
    category_leaks = 0

    for extracted in states:
        state = extracted.state
        evidence = state.evidence
        outcomes[evidence.outcome.value] += 1
        categories[extracted.catalogue_category] += 1
        messages.update(evidence.tool_messages)
        if "catalogue_category" in state.metadata:
            category_leaks += 1
        if evidence.call_rate is not None:
            call_rates.append(evidence.call_rate)
            if evidence.call_rate == 0:
                call_rate_cases["zero"] += 1
            elif evidence.call_rate == 1:
                call_rate_cases["complete"] += 1
            else:
                call_rate_cases["partial"] += 1
        else:
            call_rate_cases["missing"] += 1
        for name, value in (
            ("heterozygous_snps", evidence.heterozygous_snp_count),
            ("phased_snps", evidence.phased_snp_count),
            ("unphased_snps", evidence.unphased_snp_count),
            ("total_snps", evidence.additional_metrics.get("total_snp_count")),
        ):
            if isinstance(value, int):
                totals[name] += value
        ploidy = evidence.additional_metrics.get("genotype_ploidy")
        ploidies[str(ploidy)] += 1
        phase_set_count = evidence.additional_metrics.get("phase_set_count")
        phase_sets[str(phase_set_count)] += 1

    heterozygous = totals["heterozygous_snps"]
    return {
        "covered_regions": len(states),
        "outcomes": dict(sorted(outcomes.items())),
        "catalogue_categories_external_only": dict(sorted(categories.items())),
        "snp_totals": dict(totals),
        "pipe_phased_fraction": (
            totals["phased_snps"] / heterozygous if heterozygous else None
        ),
        "call_rate": {
            "observed_cases": len(call_rates),
            "minimum": min(call_rates) if call_rates else None,
            "mean": mean(call_rates) if call_rates else None,
            "median": median(call_rates) if call_rates else None,
            "maximum": max(call_rates) if call_rates else None,
        },
        "call_rate_case_counts": dict(sorted(call_rate_cases.items())),
        "genotype_ploidy_case_counts": dict(sorted(ploidies.items())),
        "phase_set_count_case_counts": dict(sorted(phase_sets.items())),
        "tool_message_counts": dict(sorted(messages.items())),
        "assessor_category_leaks": category_leaks,
    }


def build_census(
    *,
    manifest_path: str | Path,
    bed_path: str | Path,
    region_categories_path: str | Path,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Extract one sample at a time and retain summaries rather than full states."""

    rows = load_batch_manifest(manifest_path)
    per_sample: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    region_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for row in rows:
        try:
            states = build_real_region_states(
                vcf_path=row["phased_vcf"],
                bam_path=row["bam"],
                bed_path=bed_path,
                region_categories_path=region_categories_path,
                reference_fasta=row["reference_fasta"],
                expected_sample=row["sample_id"],
            )
        except Exception as exc:
            failures.append(
                {
                    "sample_id": row["sample_id"],
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            continue

        summary = summarize_states(states)
        per_sample.append({"sample_id": row["sample_id"], **summary})
        for extracted in states:
            region = region_counts[extracted.state.region.region_key]
            region["covered_samples"] += 1
            region[extracted.state.evidence.outcome.value] += 1

    aggregate_outcomes: Counter[str] = Counter()
    aggregate_snps: Counter[str] = Counter()
    aggregate_categories: Counter[str] = Counter()
    aggregate_ploidies: Counter[str] = Counter()
    aggregate_phase_sets: Counter[str] = Counter()
    aggregate_messages: Counter[str] = Counter()
    aggregate_call_rate_cases: Counter[str] = Counter()
    call_rate_sum = 0.0
    call_rate_count = 0
    call_rate_minima: list[float] = []
    call_rate_maxima: list[float] = []
    total_cases = 0
    category_leaks = 0
    for sample in per_sample:
        total_cases += int(sample["covered_regions"])
        aggregate_outcomes.update(sample["outcomes"])
        aggregate_snps.update(sample["snp_totals"])
        aggregate_categories.update(sample["catalogue_categories_external_only"])
        aggregate_ploidies.update(sample["genotype_ploidy_case_counts"])
        aggregate_phase_sets.update(sample["phase_set_count_case_counts"])
        aggregate_messages.update(sample["tool_message_counts"])
        aggregate_call_rate_cases.update(sample["call_rate_case_counts"])
        call_rate = sample["call_rate"]
        observed = int(call_rate["observed_cases"])
        if observed:
            call_rate_count += observed
            call_rate_sum += float(call_rate["mean"]) * observed
            call_rate_minima.append(float(call_rate["minimum"]))
            call_rate_maxima.append(float(call_rate["maximum"]))
        category_leaks += int(sample["assessor_category_leaks"])

    recurrent = []
    for region_key, counts in region_counts.items():
        difficult = sum(counts[outcome] for outcome in DIFFICULT_OUTCOMES)
        if difficult:
            recurrent.append(
                {
                    "region_key": region_key,
                    "difficult_cases": difficult,
                    **dict(counts),
                }
            )
    recurrent.sort(key=lambda item: (-int(item["difficult_cases"]), str(item["region_key"])))

    phased_fractions = [
        (str(sample["sample_id"]), float(sample["pipe_phased_fraction"]))
        for sample in per_sample
        if sample["pipe_phased_fraction"] is not None
    ]
    sample_mean_call_rates = [
        (str(sample["sample_id"]), float(sample["call_rate"]["mean"]))
        for sample in per_sample
        if sample["call_rate"]["mean"] is not None
    ]
    overall_heterozygous = aggregate_snps["heterozygous_snps"]
    report = {
        "report_schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "batch_id": rows[0]["batch_id"],
        "data_status": REAL_EVIDENCE_DATA_STATUS,
        "model_called": False,
        "action_ground_truth_available": False,
        "scoring_performed": False,
        "inputs": {
            "manifest": str(Path(manifest_path).resolve()),
            "target_regions_bed": str(Path(bed_path).resolve()),
            "region_categories": str(Path(region_categories_path).resolve()),
        },
        "summary": {
            "manifest_samples": len(rows),
            "samples_summarized": len(per_sample),
            "sample_failures": len(failures),
            "sample_region_cases": total_cases,
            "unique_regions_covered": len(region_counts),
            "outcomes": dict(sorted(aggregate_outcomes.items())),
            "snp_totals": dict(aggregate_snps),
            "pipe_phased_fraction": (
                aggregate_snps["phased_snps"] / overall_heterozygous
                if overall_heterozygous
                else None
            ),
            "call_rate": {
                "observed_cases": call_rate_count,
                "minimum": min(call_rate_minima) if call_rate_minima else None,
                "mean": call_rate_sum / call_rate_count if call_rate_count else None,
                "maximum": max(call_rate_maxima) if call_rate_maxima else None,
                "case_counts": dict(sorted(aggregate_call_rate_cases.items())),
            },
            "catalogue_categories_external_only": dict(sorted(aggregate_categories.items())),
            "genotype_ploidy_case_counts": dict(sorted(aggregate_ploidies.items())),
            "phase_set_count_case_counts": dict(sorted(aggregate_phase_sets.items())),
            "tool_message_counts": dict(sorted(aggregate_messages.items())),
            "assessor_category_leaks": category_leaks,
        },
        "sample_metric_extremes": {
            "pipe_phased_fraction": {
                "minimum": min(phased_fractions, key=lambda item: item[1]) if phased_fractions else None,
                "maximum": max(phased_fractions, key=lambda item: item[1]) if phased_fractions else None,
            },
            "mean_call_rate": {
                "minimum": min(sample_mean_call_rates, key=lambda item: item[1]) if sample_mean_call_rates else None,
                "maximum": max(sample_mean_call_rates, key=lambda item: item[1]) if sample_mean_call_rates else None,
            },
        },
        "recurrent_difficult_regions_top_25": recurrent[:25],
        "failures": failures,
        "interpretation_boundary": (
            "Deterministic evidence census only; no agent decisions or biological "
            "scoring. SNP totals are regional observations, so overlapping targets "
            "can count one VCF record more than once. Pipe-phased counts include "
            "singleton phase sets and are not the WhatsHap-stats phased fraction."
        ),
    }
    return report, per_sample


def write_census(
    report: dict[str, object],
    per_sample: Sequence[dict[str, object]],
    output_directory: str | Path,
) -> None:
    """Write one JSON report and a flat sample table."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    columns = (
        "sample_id",
        "covered_regions",
        "pipe_phased_fraction",
        "outcomes",
        "snp_totals",
        "call_rate",
        "call_rate_case_counts",
        "genotype_ploidy_case_counts",
        "phase_set_count_case_counts",
        "assessor_category_leaks",
    )
    with (output / "per_sample.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for sample in per_sample:
            writer.writerow(
                {
                    column: (
                        json.dumps(sample[column], sort_keys=True)
                        if isinstance(sample[column], dict)
                        else sample[column]
                    )
                    for column in columns
                }
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bed", type=Path, required=True)
    parser.add_argument("--region-categories", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report, per_sample = build_census(
        manifest_path=args.manifest,
        bed_path=args.bed,
        region_categories_path=args.region_categories,
    )
    write_census(report, per_sample, args.output_dir)
    summary = report["summary"]
    print(
        f"samples={summary['samples_summarized']}/{summary['manifest_samples']} "
        f"cases={summary['sample_region_cases']} failures={summary['sample_failures']}"
    )
    return 0 if summary["sample_failures"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_census", "summarize_states", "write_census"]
