"""Freeze a small stratified panel from real, unlabeled regional evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from agent.state import PhasingOutcome, RegionState
from pipeline.batch_preflight import load_batch_manifest
from pipeline.real_evidence import (
    REAL_EVIDENCE_DATA_STATUS,
    ExtractedRegionState,
    build_real_region_states,
)


DEFAULT_PANEL_SEED = 20260915


def _call_rate_bin(state: RegionState) -> str:
    rate = state.evidence.call_rate
    if rate is None:
        return "missing"
    if rate == 0:
        return "zero"
    if rate == 1:
        return "complete"
    return "partial"


def _seeded_choice(
    candidates: Sequence[ExtractedRegionState], *, seed: int, slot: str
) -> ExtractedRegionState:
    if not candidates:
        raise ValueError(f"no candidates available for panel slot '{slot}'")

    def rank(item: ExtractedRegionState) -> str:
        case_id = f"{item.state.sample_id}::{item.state.region.region_key}"
        return hashlib.sha256(f"{seed}|{slot}|{case_id}".encode()).hexdigest()

    return min(candidates, key=rank)


def select_sample_cases(
    states: Sequence[ExtractedRegionState], *, sample_index: int, seed: int
) -> list[tuple[str, ExtractedRegionState]]:
    """Select no-information, phased, and difficult cases for one sample."""

    desired_rate = ("zero", "partial", "complete")[sample_index % 3]
    no_information = [
        item
        for item in states
        if item.state.evidence.outcome
        == PhasingOutcome.NO_PHASE_INFORMATIVE_VARIANTS
        and _call_rate_bin(item.state) == desired_rate
    ]
    if not no_information:
        no_information = [
            item
            for item in states
            if item.state.evidence.outcome
            == PhasingOutcome.NO_PHASE_INFORMATIVE_VARIANTS
        ]

    difficult_outcome = (
        PhasingOutcome.PARTIALLY_PHASED
        if sample_index % 2 == 0
        else PhasingOutcome.UNPHASED
    )
    specifications = (
        (f"no_phase_informative:{desired_rate}", no_information),
        (
            "phased",
            [item for item in states if item.state.evidence.outcome == PhasingOutcome.PHASED],
        ),
        (
            difficult_outcome.value,
            [item for item in states if item.state.evidence.outcome == difficult_outcome],
        ),
    )
    return [
        (slot, _seeded_choice(candidates, seed=seed, slot=slot))
        for slot, candidates in specifications
    ]


def build_panel(
    *,
    manifest_path: str | Path,
    bed_path: str | Path,
    region_categories_path: str | Path,
    seed: int = DEFAULT_PANEL_SEED,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Extract and freeze three distinct evidence cases per manifest sample."""

    rows = load_batch_manifest(manifest_path)
    records: list[dict[str, object]] = []
    repeatability_case_ids: list[str] = []
    for sample_index, row in enumerate(rows):
        states = build_real_region_states(
            vcf_path=row["phased_vcf"],
            bam_path=row["bam"],
            bed_path=bed_path,
            region_categories_path=region_categories_path,
            reference_fasta=row["reference_fasta"],
            expected_sample=row["sample_id"],
        )
        selected = select_sample_cases(states, sample_index=sample_index, seed=seed)
        for slot, extracted in selected:
            state = extracted.state
            if state.metadata.get("data_status") != REAL_EVIDENCE_DATA_STATUS:
                raise ValueError("selected state has the wrong data status")
            if "catalogue_category" in state.metadata:
                raise ValueError("catalogue category leaked into assessor state")
            case_id = f"{state.sample_id}::{state.region.region_key}"
            records.append(
                {
                    "case_id": case_id,
                    "selection_slot": slot,
                    "evidence_outcome": state.evidence.outcome.value,
                    "call_rate_bin": _call_rate_bin(state),
                    "catalogue_category_external": extracted.catalogue_category,
                    "catalogue_row": extracted.catalogue_row,
                    "state": state.model_dump(mode="json", exclude_computed_fields=True),
                }
            )
        repeatability_case_ids.append(records[-3 + (sample_index % 3)]["case_id"])

    case_ids = [str(record["case_id"]) for record in records]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("panel contains duplicate sample-region case IDs")
    fingerprint_payload = [
        {
            "case_id": record["case_id"],
            "selection_slot": record["selection_slot"],
            "evidence": record["state"]["evidence"],
        }
        for record in records
    ]
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode()
    ).hexdigest()
    manifest = {
        "panel_schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "batch_id": rows[0]["batch_id"],
        "data_status": REAL_EVIDENCE_DATA_STATUS,
        "action_ground_truth_available": False,
        "scoring_performed": False,
        "selection_seed": seed,
        "selection_method": (
            "Three cases per sample: one no-phase-informative, one phased, and "
            "one alternating partially-phased/unphased case. No-information "
            "call-rate targets rotate zero/partial/complete with deterministic fallback."
        ),
        "case_count": len(records),
        "sample_count": len(rows),
        "counts": {
            "outcome": dict(Counter(str(record["evidence_outcome"]) for record in records)),
            "call_rate_bin": dict(Counter(str(record["call_rate_bin"]) for record in records)),
            "catalogue_category_external": dict(
                Counter(str(record["catalogue_category_external"]) for record in records)
            ),
        },
        "repeatability_case_ids": repeatability_case_ids,
        "content_fingerprint_sha256": fingerprint,
        "interpretation_boundary": (
            "Balanced evidence-coverage panel; not prevalence-weighted and not labeled."
        ),
    }
    return records, manifest


def write_panel(
    records: Sequence[dict[str, object]],
    manifest: dict[str, object],
    output_directory: str | Path,
) -> None:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    (output / "panel.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    (output / "panel_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def load_panel(path: str | Path) -> list[dict[str, object]]:
    """Load the frozen panel and validate every stored RegionState."""

    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    for record in records:
        record["state"] = RegionState.model_validate(record["state"])
    return records


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bed", type=Path, required=True)
    parser.add_argument("--region-categories", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_PANEL_SEED)
    args = parser.parse_args(argv)
    records, panel_manifest = build_panel(
        manifest_path=args.manifest,
        bed_path=args.bed,
        region_categories_path=args.region_categories,
        seed=args.seed,
    )
    write_panel(records, panel_manifest, args.output_dir)
    print(
        f"samples={panel_manifest['sample_count']} cases={panel_manifest['case_count']} "
        f"repeatability={len(panel_manifest['repeatability_case_ids'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_PANEL_SEED",
    "build_panel",
    "load_panel",
    "select_sample_cases",
    "write_panel",
]
