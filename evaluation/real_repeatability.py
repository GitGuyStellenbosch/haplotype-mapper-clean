"""Measure repeatability on the frozen, unlabeled real-evidence subset."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from statistics import mean, median
from typing import Sequence

from agent.config import PipelineMode, load_config
from agent.real_smoke import RealEvidenceAuditRecord, run_real_evidence_smoke
from agent.state import RegionState
from evaluation.real_panel import load_panel
from pipeline.real_evidence import ExtractedRegionState


REPETITIONS = 3
RATIONALE_THEMES: dict[str, tuple[str, ...]] = {
    "phasing_completeness": ("phased", "unphased", "partially"),
    "heterozygosity": ("heterozygous", "phaseable"),
    "call_rate_or_genotypes": ("call rate", "call_rate", "genotype"),
    "family_or_mendelian": ("family", "pedigree", "mendelian"),
    "uncertainty": ("uncertain", "insufficient", "ambiguous"),
    "manual_review_or_escalation": ("manual", "review", "escalat"),
}


def _state_sha256(state: RegionState) -> str:
    return hashlib.sha256(
        state.model_dump_json(exclude_computed_fields=True).encode()
    ).hexdigest()


def build_trials(
    panel: Sequence[dict[str, object]],
    repeatability_case_ids: Sequence[str],
) -> tuple[list[ExtractedRegionState], list[dict[str, object]]]:
    """Create three round-major calls without changing the frozen states."""

    if len(repeatability_case_ids) != len(set(repeatability_case_ids)):
        raise ValueError("repeatability case IDs must be unique")
    panel_by_id = {str(item["case_id"]): item for item in panel}
    missing = [case_id for case_id in repeatability_case_ids if case_id not in panel_by_id]
    if missing:
        raise ValueError(f"repeatability case missing from panel: {missing[0]}")

    extracted: list[ExtractedRegionState] = []
    trials: list[dict[str, object]] = []
    for repetition in range(1, REPETITIONS + 1):
        for case_id in repeatability_case_ids:
            item = panel_by_id[case_id]
            state = item["state"]
            if not isinstance(state, RegionState):
                raise TypeError("loaded panel state is not a RegionState")
            extracted.append(
                ExtractedRegionState(
                    state=state,
                    catalogue_category=str(item["catalogue_category_external"]),
                    catalogue_row=int(item["catalogue_row"]),
                )
            )
            trials.append(
                {
                    "trial_index": len(trials) + 1,
                    "repetition": repetition,
                    "case_id": case_id,
                    "evidence_outcome": str(item["evidence_outcome"]),
                    "state_sha256": _state_sha256(state),
                }
            )
    return extracted, trials


def _themes(rationale: str) -> list[str]:
    lowered = rationale.lower()
    return [
        theme
        for theme, terms in RATIONALE_THEMES.items()
        if any(term in lowered for term in terms)
    ]


def build_report(
    records: Sequence[RealEvidenceAuditRecord],
    trials: Sequence[dict[str, object]],
) -> dict[str, object]:
    """Summarize stability only; no correctness metric is created."""

    if len(records) != len(trials):
        raise ValueError("audit and trial counts differ")

    rows: list[dict[str, object]] = []
    by_case: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record, trial in zip(records, trials, strict=True):
        row = {
            **trial,
            "run_id": record.run_id,
            "duration_seconds": record.duration_seconds,
            "execution_status": record.execution_status,
            "action": None,
            "confidence": None,
            "rationale": None,
            "rationale_themes": [],
            "error_type": record.error_type,
            "error_message": record.error_message,
        }
        if record.execution_status == "success" and record.final_state is not None:
            decision = record.final_state.decision_history[0]
            row.update(
                {
                    "action": decision.action.value,
                    "confidence": decision.confidence,
                    "rationale": decision.rationale,
                    "rationale_themes": _themes(decision.rationale),
                }
            )
        rows.append(row)
        by_case[str(trial["case_id"])].append(row)

    case_results: list[dict[str, object]] = []
    agreeing_pairs = comparable_pairs = 0
    all_valid_actions: list[str] = []
    confidence_ranges: list[float] = []
    theme_counts: Counter[str] = Counter()
    for case_id, case_rows in by_case.items():
        actions = [str(row["action"]) for row in case_rows if row["action"] is not None]
        confidences = [
            float(row["confidence"])
            for row in case_rows
            if row["confidence"] is not None
        ]
        rationales = [
            str(row["rationale"])
            for row in case_rows
            if row["rationale"] is not None
        ]
        for left, right in combinations(actions, 2):
            comparable_pairs += 1
            agreeing_pairs += left == right
        all_valid_actions.extend(actions)
        for row in case_rows:
            theme_counts.update(row["rationale_themes"])
        confidence_range = (
            max(confidences) - min(confidences) if confidences else None
        )
        if confidence_range is not None:
            confidence_ranges.append(confidence_range)
        case_results.append(
            {
                "case_id": case_id,
                "evidence_outcome": case_rows[0]["evidence_outcome"],
                "state_sha256": case_rows[0]["state_sha256"],
                "actions": [row["action"] for row in case_rows],
                "strict_three_run_action_agreement": (
                    len(actions) == REPETITIONS and len(set(actions)) == 1
                ),
                "successful_runs": len(actions),
                "errors": REPETITIONS - len(actions),
                "confidence_min": min(confidences) if confidences else None,
                "confidence_max": max(confidences) if confidences else None,
                "confidence_range": confidence_range,
                "exact_rationale_agreement": (
                    len(rationales) == REPETITIONS and len(set(rationales)) == 1
                ),
            }
        )

    action_by_repetition: dict[str, dict[str, int]] = {}
    latency_by_repetition: dict[str, dict[str, float]] = {}
    for repetition in range(1, REPETITIONS + 1):
        subset = [row for row in rows if row["repetition"] == repetition]
        counts = Counter(
            str(row["action"]) if row["action"] is not None else "error"
            for row in subset
        )
        durations = [float(row["duration_seconds"]) for row in subset]
        action_by_repetition[str(repetition)] = dict(counts)
        latency_by_repetition[str(repetition)] = {
            "mean": mean(durations),
            "median": median(durations),
            "total": sum(durations),
        }

    strict_agreement = sum(
        bool(item["strict_three_run_action_agreement"]) for item in case_results
    )
    exact_rationale = sum(
        bool(item["exact_rationale_agreement"]) for item in case_results
    )
    all_durations = [record.duration_seconds for record in records]
    errors = [row for row in rows if row["execution_status"] == "error"]
    return {
        "report_schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_status": "real_evidence_no_ground_truth",
        "action_ground_truth_available": False,
        "scoring_performed": False,
        "cases": len(case_results),
        "repetitions_per_case": REPETITIONS,
        "calls_attempted": len(rows),
        "successful_calls": len(rows) - len(errors),
        "failed_calls": len(errors),
        "strict_three_run_action_agreement": {
            "cases": strict_agreement,
            "total_cases": len(case_results),
            "proportion": strict_agreement / len(case_results),
        },
        "pairwise_action_agreement_among_valid_calls": {
            "agreeing_pairs": agreeing_pairs,
            "comparable_pairs": comparable_pairs,
            "proportion": agreeing_pairs / comparable_pairs if comparable_pairs else None,
        },
        "confidence_variation": {
            "cases_with_nonzero_range": sum(value > 0 for value in confidence_ranges),
            "mean_within_case_range": mean(confidence_ranges) if confidence_ranges else None,
            "maximum_within_case_range": max(confidence_ranges) if confidence_ranges else None,
        },
        "exact_rationale_agreement": {
            "cases": exact_rationale,
            "total_cases": len(case_results),
        },
        "rationale_theme_mentions": dict(sorted(theme_counts.items())),
        "actions_by_repetition": action_by_repetition,
        "errors_by_type": dict(Counter(str(row["error_type"]) for row in errors)),
        "latency_seconds": {
            "mean": mean(all_durations),
            "median": median(all_durations),
            "total": sum(all_durations),
            "by_repetition": latency_by_repetition,
        },
        "phase1_always_accept_pattern_persisted": (
            bool(all_valid_actions)
            and len(all_valid_actions) == len(rows)
            and set(all_valid_actions) == {"accept"}
        ),
        "case_results": sorted(case_results, key=lambda item: str(item["case_id"])),
        "trials": rows,
        "interpretation_boundary": (
            "Repeatability and robustness only; no action labels or biological accuracy."
        ),
    }


def render_summary(report: dict[str, object]) -> str:
    agreement = report["strict_three_run_action_agreement"]
    pairwise = report["pairwise_action_agreement_among_valid_calls"]
    confidence = report["confidence_variation"]
    latency = report["latency_seconds"]
    return "\n".join(
        (
            "Haplotype Mapper Phase 13 repeatability run",
            "Data status: real_evidence_no_ground_truth",
            "Scoring performed: no",
            f"Calls attempted: {report['calls_attempted']}",
            f"Successful calls: {report['successful_calls']}",
            f"Failed calls: {report['failed_calls']}",
            (
                "Strict three-run action agreement: "
                f"{agreement['cases']}/{agreement['total_cases']}"
            ),
            (
                "Valid-call pairwise action agreement: "
                f"{pairwise['agreeing_pairs']}/{pairwise['comparable_pairs']}"
            ),
            (
                "Cases with confidence variation: "
                f"{confidence['cases_with_nonzero_range']}"
            ),
            f"Median call latency: {latency['median']:.3f} seconds",
            (
                "Phase-1 always-accept pattern persisted: "
                f"{str(report['phase1_always_accept_pattern_persisted']).lower()}"
            ),
            "Interpretation: stability only; no biological correctness claim.",
            "",
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--reference-run-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    panel = load_panel(args.panel)
    manifest = json.loads(args.panel_manifest.read_text(encoding="utf-8"))
    reference_plan = json.loads(args.reference_run_plan.read_text(encoding="utf-8"))
    config = load_config(args.config).with_pipeline_mode(PipelineMode.REAL)
    if config.model_dump(mode="json") != reference_plan["configuration"]:
        raise ValueError("current configuration differs from the frozen Phase 12 plan")
    if manifest["content_fingerprint_sha256"] != reference_plan["panel_fingerprint_sha256"]:
        raise ValueError("panel fingerprint differs from the frozen Phase 12 plan")

    extracted, trials = build_trials(panel, manifest["repeatability_case_ids"])
    skills = sorted(config.paths.skills_directory.glob("*.md"))
    skills_payload = b"".join(
        path.name.encode() + b"\0" + path.read_bytes() + b"\0" for path in skills
    )
    plan = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "phase13_real_evidence_repeatability",
        "data_status": "real_evidence_no_ground_truth",
        "action_ground_truth_available": False,
        "scoring_performed": False,
        "panel_fingerprint_sha256": manifest["content_fingerprint_sha256"],
        "repetitions_per_case": REPETITIONS,
        "ordering": "round_major",
        "rationale_theme_terms": {
            theme: list(terms) for theme, terms in RATIONALE_THEMES.items()
        },
        "skills_sha256": hashlib.sha256(skills_payload).hexdigest(),
        "configuration": config.model_dump(mode="json"),
        "trials": trials,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = args.output_dir / "run_plan.json"
    if plan_path.exists() and args.resume:
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        for key in plan.keys() - {"created_at"}:
            if existing[key] != plan[key]:
                raise ValueError(f"resume run plan mismatch: {key}")
        plan = existing
    elif plan_path.exists():
        raise FileExistsError(f"run plan already exists: {plan_path}")
    else:
        plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")

    records = run_real_evidence_smoke(
        config=config,
        extracted_states=extracted,
        output_directory=args.output_dir / "audits",
        resume=args.resume,
        progress_every=1,
        write_summary=False,
    )
    report = build_report(records, trials)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    summary = render_summary(report)
    (args.output_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_report", "build_trials", "render_summary"]
