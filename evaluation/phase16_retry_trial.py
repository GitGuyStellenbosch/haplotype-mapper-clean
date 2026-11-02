"""Run the frozen, unlabeled Phase 16 paired retry trial; no new retry logic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from agent.config import PipelineMode, load_config
from agent.pipeline_interface import validate_real_inputs
from agent.real_smoke import run_real_evidence_smoke
from agent.state import PhasingOutcome
from pipeline.real_evidence import build_real_region_states


ROOT = Path(__file__).resolve().parents[1]
STUDY_ROOT = Path(os.environ.get("HAPLOTYPE_STUDY_ROOT", ROOT)).resolve()
CONFIG = Path(os.environ.get("HAPLOTYPE_CONFIG", ROOT / "config/parameters.yaml")).resolve()
PANEL = STUDY_ROOT / "evaluation/phase16_panel.json"
OUTPUT = STUDY_ROOT / "outputs/phase16_retry_trial_20261006"
REFERENCE = STUDY_ROOT / "data/reference/Egrandis_v2.fa"
BED = STUDY_ROOT / "data/ground_truth/derived/target_regions.bed"
CATEGORIES = STUDY_ROOT / "data/ground_truth/derived/region_categories.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare():
    rows = json.loads(PANEL.read_text())["cases"]
    samples = list(dict.fromkeys(row["sample"] for row in rows))
    selected = []
    original = {}
    paths = [PANEL, CONFIG, REFERENCE, BED, CATEGORIES]
    for sample in samples:
        directory = STUDY_ROOT / "data/processed/batch20"
        phased = directory / "phased" / f"{sample}.phased.vcf.gz"
        input_vcf = directory / "phased" / f"{sample}.input.vcf.gz"
        bam = directory / f"{sample}.md.bam"
        paths.extend((phased, input_vcf, bam))
        original[sample] = input_vcf
        extracted = build_real_region_states(
            vcf_path=phased, bam_path=bam, bed_path=BED,
            region_categories_path=CATEGORIES, reference_fasta=REFERENCE,
            expected_sample=sample,
        )
        validate_real_inputs(extracted[0].state.inputs, expected_sample=sample,
                             expected_ploidy=4)
        by_region = {item.state.region.region_key: item for item in extracted}
        for row in (row for row in rows if row["sample"] == sample):
            item = by_region[row["region_key"]]
            evidence = item.state.evidence
            if row["stratum"].startswith("fragmented"):
                valid = (evidence.outcome == PhasingOutcome.PHASED
                         and evidence.call_rate == 1
                         and evidence.heterozygous_snp_count >= 2
                         and evidence.additional_metrics.get("phase_set_count", 0) > 1)
            elif row["stratum"] == "partial":
                valid = evidence.outcome == PhasingOutcome.PARTIALLY_PHASED and evidence.call_rate == 1
            else:
                valid = evidence.heterozygous_snp_count == 0 and evidence.call_rate == 1
            if not valid:
                raise ValueError(f"frozen stratum no longer matches: {sample}::{row['region_key']}")
            selected.append(item)
    if len(selected) != len(rows) or len({(x.state.sample_id, x.state.region.region_key) for x in selected}) != len(rows):
        raise ValueError("frozen panel has missing or duplicate cases")
    return selected, original, {(str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)): sha256(path) for path in paths}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    selected, original, before = prepare()
    print(f"preflight passed: {len(selected)} cases, {len(original)} samples", flush=True)
    if args.preflight_only:
        return
    output = args.output.resolve()
    if output.exists() and not args.resume:
        raise FileExistsError(f"refusing existing Phase 16 output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    inputs_path = output / "original_input_sha256.json"
    if inputs_path.exists():
        if json.loads(inputs_path.read_text()) != before:
            raise ValueError("original input hashes changed; refusing resume")
    else:
        inputs_path.write_text(json.dumps(before, indent=2) + "\n")

    enabled = load_config(CONFIG).with_pipeline_mode(PipelineMode.REAL)
    if len(enabled.pipeline.retry_strategies) != 1 or enabled.pipeline.retry_strategies[0].strategy_id != "lower-block-cut-sensitivity-v1":
        raise ValueError("Phase 16 requires the sole frozen phasing-only strategy")
    disabled = enabled.model_copy(deep=True)
    disabled.pipeline.retry_strategies = []
    for name, config, retry_inputs in (("enabled", enabled, original),
                                       ("disabled", disabled, None)):
        print(f"starting {name} arm", flush=True)
        run_real_evidence_smoke(
            config=config, extracted_states=selected, output_directory=output / name,
            retry_inputs=retry_inputs, region_categories_path=CATEGORIES,
            resume=args.resume, progress_every=1, write_summary=False,
        )
    after = {relative: sha256(ROOT / relative) for relative in before}
    if after != before:
        raise RuntimeError("original input hashes changed during trial")
    (output / "original_inputs_unchanged.json").write_text(
        json.dumps({"all_unchanged": True, "sha256": after}, indent=2) + "\n"
    )
    print("both arms complete; original input hashes unchanged", flush=True)


if __name__ == "__main__":
    main()
