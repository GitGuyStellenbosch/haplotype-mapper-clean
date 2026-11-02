"""Frozen six-case first-pass evidence sensitivity trial; never executes retries."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time
from pathlib import Path

from agent.config import load_config
from agent.llm import ModelAssessmentError, ModelAssessor, create_chat_model
from agent.state import RegionState
from evaluation.real_static_comparison import build_static_assessor
from evaluation.skill_snapshot import load_skill_snapshot, verify_skill_snapshot


ROOT = Path(__file__).resolve().parents[1]
STUDY_ROOT = Path(os.environ.get("HAPLOTYPE_STUDY_ROOT", ROOT)).resolve()
CONFIG = Path(os.environ.get("HAPLOTYPE_CONFIG", ROOT / "config/parameters.yaml")).resolve()
MANIFEST = STUDY_ROOT / "evaluation/phase17_manifest.json"
OUTPUT = STUDY_ROOT / "outputs/phase17_sensitivity_20261006"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state_hash(state: RegionState) -> str:
    payload = state.model_dump(mode="json", exclude_computed_fields=True)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def source_state(spec: dict) -> dict:
    path = STUDY_ROOT / spec["source"]
    if sha256(path) != spec["source_sha256"]:
        raise ValueError(f"source hash changed: {path}")
    if path.suffix == ".jsonl":
        matching = [json.loads(line)["state"] for line in path.read_text().splitlines()
                    if json.loads(line)["case_id"] == spec["case_id"]]
        if len(matching) != 1:
            raise ValueError(f"expected one source case: {spec['case_id']}")
        return matching[0]
    audit = json.loads(path.read_text())
    if f"{audit['sample_id']}::{audit['region_key']}" != spec["case_id"]:
        raise ValueError(f"source case ID differs: {path}")
    return audit["initial_state"]


def evidence_value(evidence: dict, field: str):
    if field == "tool_messages":
        return evidence[field]
    if field == "additional_metrics.phase_set_count":
        return evidence["additional_metrics"]["phase_set_count"]
    raise ValueError(f"unsupported frozen field: {field}")


def set_evidence_value(evidence: dict, field: str, value) -> None:
    if field == "tool_messages":
        evidence[field] = value
    elif field == "additional_metrics.phase_set_count":
        evidence["additional_metrics"]["phase_set_count"] = value
    else:
        raise ValueError(f"unsupported frozen field: {field}")


def validate_state(raw: dict) -> RegionState:
    state = RegionState.model_validate(raw)
    if len(state.attempt_history) != 1 or state.decision is not None or state.decision_history:
        raise ValueError("Phase 17 requires an undecided initial state")
    if state.evidence != state.attempt_history[0].evidence:
        raise ValueError("current and attempt evidence differ")
    ev = state.evidence
    if ev.heterozygous_snp_count != ev.phased_snp_count + ev.unphased_snp_count:
        raise ValueError("heterozygous SNP arithmetic differs")
    ps = ev.additional_metrics.get("phase_set_count")
    if not isinstance(ps, int) or ps < 0 or ps > ev.phased_snp_count:
        raise ValueError("impossible phase-set count")
    return state


def prepare(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to replace Phase 17 output: {output}")
    manifest = json.loads(MANIFEST.read_text())
    cases = []
    for spec in manifest["pairs"]:
        original_raw = source_state(spec)
        original = validate_state(original_raw)
        if f"{original.sample_id}::{original.region.region_key}" != spec["case_id"]:
            raise ValueError(f"identity mismatch: {spec['id']}")
        if evidence_value(original_raw["evidence"], spec["field"]) != spec["original"]:
            raise ValueError(f"original field mismatch: {spec['id']}")
        altered_raw = copy.deepcopy(original_raw)
        for evidence in (altered_raw["evidence"], altered_raw["attempt_history"][0]["evidence"]):
            set_evidence_value(evidence, spec["field"], spec["replacement"])
        altered = validate_state(altered_raw)
        cases.extend({"pair_id": spec["id"], "kind": kind,
                      "state": state.model_dump(mode="json", exclude_computed_fields=True),
                      "state_sha256": state_hash(state)}
                     for kind, state in (("original", original), ("altered", altered)))
    _, skill_sha256 = load_skill_snapshot(load_config(CONFIG).paths.skills_directory)
    plan = {"manifest_sha256": sha256(MANIFEST), "config_sha256": sha256(CONFIG),
            "skill_sha256": skill_sha256,
            "cases": cases, "data_status": "real_evidence_counterfactual_no_ground_truth",
            "method": "direct assessor proposals; no graph, WhatsHap, or GATK"}
    output.mkdir(parents=True)
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(f"prepared {len(cases)} frozen assessor inputs at {output}", flush=True)


def run(output: Path) -> None:
    plan = json.loads((output / "plan.json").read_text())
    if plan["manifest_sha256"] != sha256(MANIFEST) or plan["config_sha256"] != sha256(CONFIG):
        raise ValueError("manifest or config changed after preparation")
    config = load_config(CONFIG)
    skills = verify_skill_snapshot(config.paths.skills_directory, plan.get("skill_sha256"))
    model = ModelAssessor(model=create_chat_model(config.agent.model), skills=skills,
                          retry_strategies=config.pipeline.retry_strategies)
    static = build_static_assessor()
    for index, item in enumerate(plan["cases"], start=1):
        path = output / f"case-{index:02d}-{item['pair_id']}-{item['kind']}.json"
        if path.exists():
            raise FileExistsError(f"refusing to overwrite result: {path}")
        state = validate_state(item["state"])
        if state_hash(state) != item["state_sha256"]:
            raise ValueError(f"state hash changed: {item['pair_id']} {item['kind']}")
        static_result = static.assess(state)
        started = time.monotonic()
        try:
            response = model.assess(state)
            model_result = {"status": "success", "assessment": response.model_dump(mode="json")}
        except ModelAssessmentError as exc:
            model_result = {"status": "error", "error_type": type(exc).__name__, "error": str(exc)}
        record = {"pair_id": item["pair_id"], "kind": item["kind"],
                  "state_sha256": item["state_sha256"],
                  "static": static_result.model_dump(mode="json"),
                  "model": model_result, "model_duration_seconds": round(time.monotonic() - started, 3),
                  "interpretation": "proposal sensitivity only; no biological action truth"}
        with path.open("x") as target:
            json.dump(record, target, indent=2)
            target.write("\n")
        action = model_result.get("assessment", {}).get("action", "ERROR")
        print(f"{index}/{len(plan['cases'])} {item['pair_id']} {item['kind']}: "
              f"static={static_result.action.value} model={action} "
              f"seconds={record['model_duration_seconds']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.output)
    else:
        run(args.output)


if __name__ == "__main__":
    main()
