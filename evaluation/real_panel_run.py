"""Run a frozen real-evidence panel through the existing model graph."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Sequence

from agent.config import PipelineMode, load_config
from agent.real_smoke import RealEvidenceAuditRecord, run_real_evidence_smoke
from evaluation.real_panel import load_panel
from pipeline.real_evidence import ExtractedRegionState


def _case_id(record: RealEvidenceAuditRecord) -> str:
    return f"{record.sample_id}::{record.region_key}"


def choose_rationale_review_ids(panel: Sequence[dict[str, object]]) -> list[str]:
    """Preselect six cases: each outcome plus zero/partial call-rate coverage."""

    chosen: list[str] = []
    groups = [("call_rate", "zero"), ("call_rate", "partial")] + [
        ("outcome", value)
        for value in (
            "no_phase_informative_variants",
            "phased",
            "partially_phased",
            "unphased",
        )
    ]
    for kind, value in groups:
        field = "evidence_outcome" if kind == "outcome" else "call_rate_bin"
        candidates = [
            item
            for item in panel
            if item[field] == value and item["case_id"] not in chosen
        ]
        if not candidates:
            continue
        candidates.sort(
            key=lambda item: hashlib.sha256(
                f"rationale-review|{kind}|{value}|{item['case_id']}".encode()
            ).hexdigest()
        )
        chosen.append(str(candidates[0]["case_id"]))
    return chosen


def build_panel_report(
    records: Sequence[RealEvidenceAuditRecord],
    panel: Sequence[dict[str, object]],
    review_ids: Sequence[str],
) -> dict[str, object]:
    """Aggregate descriptive decisions without creating or scoring labels."""

    panel_by_id = {str(item["case_id"]): item for item in panel}
    actions: Counter[str] = Counter()
    terminal_actions: Counter[str] = Counter()
    by_outcome: dict[str, Counter[str]] = defaultdict(Counter)
    by_sample: dict[str, Counter[str]] = defaultdict(Counter)
    errors_by_outcome: Counter[str] = Counter()
    errors_by_sample: Counter[str] = Counter()
    latencies: list[float] = []
    review: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []

    for record in records:
        case_id = _case_id(record)
        panel_item = panel_by_id[case_id]
        latencies.append(record.duration_seconds)
        if record.execution_status == "error" or record.final_state is None:
            errors_by_outcome[str(panel_item["evidence_outcome"])] += 1
            errors_by_sample[record.sample_id] += 1
            errors.append(
                {
                    "case_id": case_id,
                    "error_type": record.error_type or "unknown",
                    "message": record.error_message or "unknown",
                }
            )
            if case_id in review_ids:
                review.append(
                    {
                        "case_id": case_id,
                        "selection_slot": panel_item["selection_slot"],
                        "evidence_outcome": panel_item["evidence_outcome"],
                        "call_rate_bin": panel_item["call_rate_bin"],
                        "evidence": record.initial_state.evidence.model_dump(mode="json"),
                        "execution_status": "error",
                        "error_type": record.error_type,
                        "error_message": record.error_message,
                    }
                )
            continue
        initial = record.final_state.decision_history[0]
        action = initial.action.value
        actions[action] += 1
        by_outcome[str(panel_item["evidence_outcome"])][action] += 1
        by_sample[record.sample_id][action] += 1
        if record.final_state.decision is not None:
            terminal_actions[record.final_state.decision.action.value] += 1
        if case_id in review_ids:
            review.append(
                {
                    "case_id": case_id,
                    "selection_slot": panel_item["selection_slot"],
                    "evidence_outcome": panel_item["evidence_outcome"],
                    "call_rate_bin": panel_item["call_rate_bin"],
                    "evidence": record.initial_state.evidence.model_dump(mode="json"),
                    "execution_status": "success",
                    "action": action,
                    "confidence": initial.confidence,
                    "rationale": initial.rationale,
                }
            )

    action_names = ("accept", "retry", "escalate")
    normalized = lambda counts: {name: counts[name] for name in action_names}
    return {
        "report_schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_status": "real_evidence_no_ground_truth",
        "action_ground_truth_available": False,
        "scoring_performed": False,
        "accuracy": None,
        "cases": len(panel),
        "processed": len(records),
        "successful": len(records) - len(errors),
        "errors": errors,
        "initial_actions": normalized(actions),
        "terminal_actions": normalized(terminal_actions),
        "actions_by_evidence_outcome": {
            outcome: normalized(by_outcome[outcome])
            for outcome in sorted(
                {str(item["evidence_outcome"]) for item in panel}
            )
        },
        "actions_by_sample": {
            key: normalized(value) for key, value in sorted(by_sample.items())
        },
        "errors_by_evidence_outcome": dict(sorted(errors_by_outcome.items())),
        "errors_by_sample": dict(sorted(errors_by_sample.items())),
        "latency_seconds": {
            "mean": mean(latencies) if latencies else None,
            "median": median(latencies) if latencies else None,
            "total": sum(latencies),
        },
        "preselected_rationale_review": sorted(
            review, key=lambda item: review_ids.index(str(item["case_id"]))
        ),
        "interpretation_boundary": (
            "Descriptive model behavior on real evidence without action ground truth."
        ),
    }


def render_summary(report: dict[str, object]) -> str:
    initial = report["initial_actions"]
    latency = report["latency_seconds"]
    return "\n".join(
        (
            "Haplotype Mapper frozen real-panel run",
            "Data status: real_evidence_no_ground_truth",
            "Scoring performed: no",
            f"Cases processed: {report['processed']}/{report['cases']}",
            f"Successful: {report['successful']}",
            f"Errors: {len(report['errors'])}",
            (
                "Initial actions: "
                f"accept={initial['accept']}, retry={initial['retry']}, "
                f"escalate={initial['escalate']}"
            ),
            f"Median latency: {latency['median']} seconds",
            "Interpretation: descriptive only; no action ground truth or accuracy.",
            "",
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    panel = load_panel(args.panel)
    panel_manifest = json.loads(args.panel_manifest.read_text(encoding="utf-8"))
    review_ids = choose_rationale_review_ids(panel)
    config = load_config(args.config).with_pipeline_mode(PipelineMode.REAL)
    plan = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "panel_fingerprint_sha256": panel_manifest["content_fingerprint_sha256"],
        "panel_cases": len(panel),
        "rationale_review_case_ids": review_ids,
        "configuration": config.model_dump(mode="json"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = args.output_dir / "run_plan.json"
    if plan_path.exists() and args.resume:
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        for key in ("panel_fingerprint_sha256", "panel_cases", "rationale_review_case_ids", "configuration"):
            if existing[key] != plan[key]:
                raise ValueError(f"resume run plan mismatch: {key}")
        plan = existing
    elif plan_path.exists():
        raise FileExistsError(f"run plan already exists: {plan_path}")
    else:
        plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")

    extracted = [
        ExtractedRegionState(
            state=item["state"],
            catalogue_category=str(item["catalogue_category_external"]),
            catalogue_row=int(item["catalogue_row"]),
        )
        for item in panel
    ]
    records = run_real_evidence_smoke(
        config=config,
        extracted_states=extracted,
        output_directory=args.output_dir / "audits",
        resume=args.resume,
        progress_every=1,
        write_summary=False,
    )
    report = build_panel_report(records, panel, review_ids)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "summary.txt").write_text(
        render_summary(report), encoding="utf-8"
    )
    print(render_summary(report))
    return 0 if not report["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_panel_report", "choose_rationale_review_ids", "render_summary"]
