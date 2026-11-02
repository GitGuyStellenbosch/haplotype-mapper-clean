"""Validate a CSV manifest of real one-sample VCF/BAM pairs."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from agent.state import InputFiles
from pipeline.preflight import inspect_real_inputs


REQUIRED_COLUMNS = (
    "batch_id",
    "sample_id",
    "bam",
    "bam_index",
    "phased_vcf",
    "vcf_index",
    "reference_fasta",
    "reference_fai",
    "reference_dictionary",
)
INDEX_COLUMNS = ("bam_index", "vcf_index", "reference_fai", "reference_dictionary")


class BatchManifestError(ValueError):
    """Raised when the manifest itself is incomplete or ambiguous."""


def load_batch_manifest(path: str | Path) -> list[dict[str, str]]:
    """Load a strict manifest and resolve file paths relative to the CSV."""

    manifest = Path(path)
    with manifest.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = set(REQUIRED_COLUMNS) - columns
        if missing:
            raise BatchManifestError(
                "manifest is missing columns: " + ", ".join(sorted(missing))
            )
        rows = list(reader)

    if not rows:
        raise BatchManifestError("manifest contains no samples")
    for row_number, row in enumerate(rows, start=2):
        blank = [column for column in REQUIRED_COLUMNS if not row[column].strip()]
        if blank:
            raise BatchManifestError(
                f"manifest row {row_number} has blank fields: {', '.join(blank)}"
            )
        for column in REQUIRED_COLUMNS[2:]:
            row[column] = str((manifest.parent / row[column]).resolve())

    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise BatchManifestError("manifest sample IDs must be unique")
    batch_ids = {row["batch_id"] for row in rows}
    if len(batch_ids) != 1:
        raise BatchManifestError("manifest must contain exactly one batch ID")
    return rows


def validate_batch_manifest(
    path: str | Path,
    *,
    expected_ploidy: int,
) -> dict[str, object]:
    """Run existing content-aware preflight checks for every manifest row."""

    rows = load_batch_manifest(path)
    sample_results: list[dict[str, object]] = []
    for row in rows:
        inputs = InputFiles(
            bam=Path(row["bam"]),
            vcf=Path(row["phased_vcf"]),
            reference_fasta=Path(row["reference_fasta"]),
        )
        preflight = inspect_real_inputs(
            inputs,
            expected_sample=row["sample_id"],
            expected_ploidy=expected_ploidy,
        )
        errors = list(preflight.errors)
        for column in INDEX_COLUMNS:
            if not Path(row[column]).is_file():
                errors.append(f"declared {column} does not exist: {row[column]}")

        discovered = preflight.discovered_indexes
        expected_discovered = {
            "bam_index": discovered.get("bam"),
            "vcf_index": discovered.get("vcf"),
            "reference_fai": discovered.get("reference_fasta"),
            "reference_dictionary": discovered.get("reference_dictionary"),
        }
        for column, discovered_path in expected_discovered.items():
            if (
                discovered_path is not None
                and Path(row[column]).resolve() != discovered_path.resolve()
            ):
                errors.append(
                    f"declared {column} differs from discovered index: "
                    f"{row[column]} != {discovered_path}"
                )

        sample_results.append(
            {
                "sample_id": row["sample_id"],
                "status": "pass" if not errors else "fail",
                "bam": row["bam"],
                "phased_vcf": row["phased_vcf"],
                "errors": errors,
                "preflight": preflight.model_dump(mode="json"),
            }
        )

    valid = sum(result["status"] == "pass" for result in sample_results)
    return {
        "report_schema_version": "1.0",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(Path(path).resolve()),
        "batch_id": rows[0]["batch_id"],
        "expected_ploidy": expected_ploidy,
        "summary": {
            "samples": len(rows),
            "passed": valid,
            "failed": len(rows) - valid,
        },
        "samples": sample_results,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-ploidy", type=int, default=4)
    args = parser.parse_args(argv)

    report = validate_batch_manifest(
        args.manifest, expected_ploidy=args.expected_ploidy
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary = report["summary"]
    print(
        f"batch={report['batch_id']} samples={summary['samples']} "
        f"passed={summary['passed']} failed={summary['failed']}"
    )
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BatchManifestError",
    "load_batch_manifest",
    "validate_batch_manifest",
]
