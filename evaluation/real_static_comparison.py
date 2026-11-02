"""Compare a frozen fixed rule with saved, unlabeled real-panel model decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from agent.config import StaticBaselineConfig
from agent.real_smoke import RealEvidenceAuditRecord
from agent.state import DecisionAction, PhasingOutcome, RegionState
from agent.static_baseline import (
    PredicateRule,
    RuleProvenance,
    StaticBaselineAssessor,
    StaticRuleDecision,
)
from evaluation.real_panel import load_panel


PANEL_SHA256 = "4d521940d4f10c04f76bae111fcdf830b3d0679e827b413a8701757a835433c7"
MODEL_PLAN_SHA256 = "c6159f7dfb6e7e310170104eb1b01c078d6a9c0b713d10e897001f61a5ee8fd0"
PANEL_FINGERPRINT = "438428657270c856ca97aaedbd13cb3dcabeab0644300c069a04e4fbd4f4f795"
BASELINE_NAME = "complete-observed-phasing"
BASELINE_VERSION = "1.0.0-provisional"
ACTIONS = tuple(action.value for action in DecisionAction)


def _complete_observed_phasing(state: RegionState) -> bool:
    evidence = state.evidence
    het = evidence.heterozygous_snp_count
    return (
        evidence.outcome == PhasingOutcome.PHASED
        and het is not None
        and het >= 2
        and evidence.phased_snp_count == het
        and evidence.unphased_snp_count == 0
        and evidence.call_rate == 1.0
        and evidence.additional_metrics.get("phase_set_count") == 1
    )


def build_static_assessor() -> StaticBaselineAssessor:
    """Return the Phase 14 rule frozen in its dated methods receipt."""

    return StaticBaselineAssessor(
        definition=StaticBaselineConfig(
            name=BASELINE_NAME,
            version=BASELINE_VERSION,
            fallback_action=DecisionAction.ESCALATE,
            fallback_confidence=1.0,
        ),
        rules=[
            PredicateRule(
                rule_id="complete-observed-phasing",
                provenance=RuleProvenance.PROJECT_DEFINITION,
                predicate=_complete_observed_phasing,
                decision=StaticRuleDecision(
                    action=DecisionAction.ACCEPT,
                    confidence=1.0,
                    rationale="All called SNPs present; at least two heterozygous SNPs are phased in one regional phase set.",
                ),
            )
        ],
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(
    panel: Sequence[dict[str, object]],
    audits: Sequence[RealEvidenceAuditRecord],
) -> dict[str, object]:
    """Require exact initial-state pairing and count only valid model decisions."""

    audit_by_id = {f"{audit.sample_id}::{audit.region_key}": audit for audit in audits}
    panel_ids = [str(item["case_id"]) for item in panel]
    if len(panel_ids) != len(set(panel_ids)) or len(audit_by_id) != len(audits):
        raise ValueError("duplicate case IDs in the panel or model audits")
    if set(panel_ids) != set(audit_by_id):
        raise ValueError("saved model audits do not match the frozen panel")

    assessor = build_static_assessor()
    matrix = {static: {model: 0 for model in ACTIONS} for static in ACTIONS}
    static_counts: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()
    model_errors: Counter[str] = Counter()
    cases: list[dict[str, object]] = []
    disagreements: list[dict[str, object]] = []

    for item in panel:
        case_id = str(item["case_id"])
        state = item["state"]
        if not isinstance(state, RegionState):
            raise ValueError(f"panel state was not validated: {case_id}")
        if state.decision is not None or state.decision_history or state.pending_strategy or len(state.attempt_history) != 1:
            raise ValueError(f"panel case is not an initial state: {case_id}")
        audit = audit_by_id[case_id]
        if audit.initial_state != state:
            raise ValueError(f"model and static initial states differ: {case_id}")
        static = assessor.assess(state)
        static_action = static.action.value
        static_counts[static_action] += 1
        row: dict[str, object] = {
            "case_id": case_id,
            "evidence_outcome": state.evidence.outcome.value,
            "call_rate": state.evidence.call_rate,
            "heterozygous_snp_count": state.evidence.heterozygous_snp_count,
            "phased_snp_count": state.evidence.phased_snp_count,
            "unphased_snp_count": state.evidence.unphased_snp_count,
            "phase_set_count": state.evidence.additional_metrics.get("phase_set_count"),
            "static_action": static_action,
            "static_rationale": static.rationale,
            "model_action": None,
            "model_rationale": None,
            "model_error_type": None,
        }
        if audit.execution_status == "error":
            model_errors[audit.error_type or "unknown"] += 1
            row["model_error_type"] = audit.error_type
        else:
            if audit.final_state is None or len(audit.final_state.decision_history) != 1:
                raise ValueError(f"saved model decision is missing or ambiguous: {case_id}")
            decision = audit.final_state.decision_history[0]
            model_action = decision.action.value
            row["model_action"] = model_action
            row["model_rationale"] = decision.rationale
            model_counts[model_action] += 1
            matrix[static_action][model_action] += 1
            if static_action != model_action:
                disagreements.append(row)
        cases.append(row)

    agreement = sum(matrix[action][action] for action in ACTIONS)
    valid = sum(model_counts.values())
    return {
        "report_schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_status": "real_evidence_no_ground_truth",
        "action_ground_truth_available": False,
        "scoring_performed": False,
        "static_baseline": f"{BASELINE_NAME}@{BASELINE_VERSION}",
        "panel_cases": len(panel),
        "valid_model_decisions": valid,
        "model_failures": dict(sorted(model_errors.items())),
        "static_actions": {action: static_counts[action] for action in ACTIONS},
        "model_actions": {action: model_counts[action] for action in ACTIONS},
        "action_table_static_rows_model_columns": matrix,
        "exact_agreement": {"cases": agreement, "of_valid_model_decisions": valid},
        "disagreement_count": len(disagreements),
        "example_disagreements": disagreements[:6],
        "cases": cases,
        "interpretation_boundary": "Descriptive agreement only; no action accuracy or biological superiority is established.",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--model-plan", type=Path, required=True)
    parser.add_argument("--model-audits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if _sha256(args.panel) != PANEL_SHA256 or _sha256(args.model_plan) != MODEL_PLAN_SHA256:
        raise ValueError("frozen panel or model plan hash mismatch")
    manifest = json.loads(args.panel_manifest.read_text(encoding="utf-8"))
    plan = json.loads(args.model_plan.read_text(encoding="utf-8"))
    if manifest["content_fingerprint_sha256"] != PANEL_FINGERPRINT or plan["panel_fingerprint_sha256"] != PANEL_FINGERPRINT:
        raise ValueError("panel and model plan fingerprints differ")
    if args.output.exists():
        raise FileExistsError(f"refusing to replace static comparison: {args.output}")

    panel = load_panel(args.panel)
    audit_paths = sorted(args.model_audits.glob("region-*.json"))
    audits = [RealEvidenceAuditRecord.model_validate_json(path.read_text(encoding="utf-8")) for path in audit_paths]
    report = compare(panel, audits)
    report["provenance"] = {
        "panel_sha256": PANEL_SHA256,
        "panel_fingerprint_sha256": PANEL_FINGERPRINT,
        "model_plan_sha256": MODEL_PLAN_SHA256,
        "model_audit_file_sha256": {path.name: _sha256(path) for path in audit_paths},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"cases={report['panel_cases']} valid_model={report['valid_model_decisions']} agreement={report['exact_agreement']['cases']} disagreements={report['disagreement_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
