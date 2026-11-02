"""Freeze and run one bounded, unlabeled final reliability study."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

from agent.config import PipelineMode, load_config
from agent.llm import ModelAssessmentError, ModelAssessor, create_chat_model
from agent.nodes import guard_acceptance
from agent.real_smoke import RealEvidenceAuditRecord, run_real_evidence_smoke
from agent.state import PhasingOutcome, RegionState
from evaluation.real_panel import load_panel
from evaluation.real_static_comparison import build_static_assessor
from evaluation.skill_snapshot import load_skill_snapshot, verify_skill_snapshot
from pipeline.batch_preflight import load_batch_manifest
from pipeline.real_evidence import ExtractedRegionState, build_real_region_states


ROOT = Path(__file__).resolve().parents[1]
STUDY_ROOT = Path(os.environ.get("HAPLOTYPE_STUDY_ROOT", ROOT)).resolve()
CONFIG = Path(os.environ.get("HAPLOTYPE_CONFIG", ROOT / "config/parameters.yaml")).resolve()
OUTPUT = STUDY_ROOT / "outputs/phase18_final_study_20261007"
BATCH = STUDY_ROOT / "config/batch20_manifest.csv"
BED = STUDY_ROOT / "data/ground_truth/derived/target_regions.bed"
CATEGORIES = STUDY_ROOT / "data/ground_truth/derived/region_categories.csv"
SEED = "20261007"
MAX_MODEL_CALLS = 120
MAX_SECONDS = 8 * 60 * 60
# Original DEV_CASE identity, frozen without distributing a sample/case record.
DEVELOPMENT_CASE_SHA256 = "52bfd259082d1a54647dbe007e6358a16a465c8477a75c3e1a60f9d7646b1c46"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_hash(state: RegionState) -> str:
    payload = state.model_dump(mode="json", exclude_computed_fields=True)
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def case_id(item: ExtractedRegionState) -> str:
    return f"{item.state.sample_id}::{item.state.region.region_key}"


def development_case_id() -> str:
    identity = os.environ.get("HAPLOTYPE_DEVELOPMENT_CASE_ID", "")
    if hashlib.sha256(identity.encode()).hexdigest() != DEVELOPMENT_CASE_SHA256:
        raise ValueError("HAPLOTYPE_DEVELOPMENT_CASE_ID must match the original historical exclusion")
    return identity


def excluded_ids() -> set[str]:
    development_case = development_case_id()
    old = {str(row["case_id"]) for row in load_panel(
        STUDY_ROOT / "outputs/batch20_real_panel_seed_20260915/panel.jsonl")}
    old.update(f"{row['sample']}::{row['region_key']}" for row in json.loads(
        (STUDY_ROOT / "evaluation/phase16_panel.json").read_text())["cases"])
    old.update(str(row["case_id"]) for row in json.loads(
        (STUDY_ROOT / "evaluation/phase17_manifest.json").read_text())["pairs"])
    old.add(development_case)
    return old


def choose(items: list[ExtractedRegionState], slot: str) -> ExtractedRegionState:
    if not items:
        raise ValueError(f"no fresh candidates for {slot}")
    return min(items, key=lambda item: hashlib.sha256(
        f"{SEED}|{slot}|{case_id(item)}".encode()).hexdigest())


def select_panel() -> tuple[list[dict], list[dict], dict[str, object]]:
    rows = load_batch_manifest(BATCH)
    if len(rows) != 20:
        raise ValueError("final study requires the frozen 20-sample batch")
    excluded = excluded_ids()
    selected: list[dict] = []
    e2e_extra: dict | None = None
    for index, row in enumerate(rows):
        states = build_real_region_states(
            vcf_path=row["phased_vcf"], bam_path=row["bam"], bed_path=BED,
            region_categories_path=CATEGORIES, reference_fasta=row["reference_fasta"],
            expected_sample=row["sample_id"],
        )
        fresh = [item for item in states if case_id(item) not in excluded]
        no_info = [item for item in fresh if item.state.evidence.outcome ==
                   PhasingOutcome.NO_PHASE_INFORMATIVE_VARIANTS]
        rate_target = (0.0, 0.5, 1.0)[index % 3]
        if rate_target == 0.5:
            desired = [item for item in no_info if item.state.evidence.call_rate is not None
                       and 0 < item.state.evidence.call_rate < 1]
        else:
            desired = [item for item in no_info if item.state.evidence.call_rate == rate_target]
        no_info = desired or no_info
        clean = [item for item in fresh if item.state.evidence.outcome == PhasingOutcome.PHASED
                 and item.state.evidence.call_rate == 1
                 and (item.state.evidence.heterozygous_snp_count or 0) >= 2
                 and item.state.evidence.additional_metrics.get("phase_set_count") == 1]
        fragmented = [item for item in fresh if item.state.evidence.outcome == PhasingOutcome.PHASED
                      and item.state.evidence.call_rate == 1
                      and (item.state.evidence.heterozygous_snp_count or 0) >= 2
                      and isinstance(item.state.evidence.additional_metrics.get("phase_set_count"), int)
                      and item.state.evidence.additional_metrics["phase_set_count"] > 1]
        phased_slot = "clean_one_ps" if index % 2 == 0 else "fragmented_multi_ps"
        phased_pool = clean if index % 2 == 0 else fragmented
        difficult_slot = "partially_phased" if index % 2 == 0 else "unphased"
        difficult_pool = [item for item in fresh if item.state.evidence.outcome.value == difficult_slot
                          and item.state.evidence.call_rate == 1]
        for slot, pool in (("no_phase_informative", no_info),
                           (phased_slot, phased_pool), (difficult_slot, difficult_pool)):
            item = choose(pool, slot)
            state = item.state
            if state.metadata.get("data_status") != "real_evidence_no_ground_truth":
                raise ValueError("selected state has incorrect data status")
            selected.append({"case_id": case_id(item), "slot": slot,
                             "candidate_pool_size": len(pool),
                             "no_info_call_rate_target": (("zero", "partial", "complete")[index % 3]
                                                          if slot == "no_phase_informative" else None),
                             "catalogue_category_external": item.catalogue_category,
                             "catalogue_row": item.catalogue_row,
                             "state": state.model_dump(mode="json", exclude_computed_fields=True),
                             "state_sha256": state_hash(state)})
        if index == 1:
            first_fragment = selected[-2]["case_id"]
            other = choose([item for item in fragmented if case_id(item) != first_fragment],
                           "fragmented_multi_ps_e2e_reuse")
            e2e_extra = {"case_id": case_id(other), "slot": "fragmented_multi_ps_e2e_reuse",
                         "catalogue_category_external": other.catalogue_category,
                         "catalogue_row": other.catalogue_row,
                         "state": other.state.model_dump(mode="json", exclude_computed_fields=True),
                         "state_sha256": state_hash(other.state)}
    if len(selected) != 60 or len({row["case_id"] for row in selected}) != 60:
        raise ValueError("panel must contain 60 unique cases")
    repeat_ids = [row["case_id"] for slot in (
        "no_phase_informative", "clean_one_ps", "fragmented_multi_ps",
        "partially_phased", "unphased") for row in sorted(
            (row for row in selected if row["slot"] == slot),
            key=lambda row: hashlib.sha256(f"{SEED}|repeat|{row['case_id']}".encode()).hexdigest())[:2]]
    e2e_ids = [next(row["case_id"] for row in selected if row["slot"] == slot)
               for slot in ("clean_one_ps", "fragmented_multi_ps",
                            "partially_phased", "unphased", "no_phase_informative")]
    e2e_ids.append(next(row["case_id"] for row in selected
                        if row["slot"] == "fragmented_multi_ps" and
                        row["case_id"].split("::")[0] != e2e_ids[1].split("::")[0]))
    if e2e_extra is None or e2e_extra["case_id"] in {row["case_id"] for row in selected}:
        raise ValueError("end-to-end cache-reuse case is missing or duplicated")
    e2e_ids.append(e2e_extra["case_id"])
    tasks = [{"case_id": row["case_id"], "slot": row["slot"], "repetition": 1,
              "state_sha256": row["state_sha256"]} for row in selected]
    by_id = {row["case_id"]: row for row in selected}
    tasks.extend({"case_id": case, "slot": by_id[case]["slot"], "repetition": repetition,
                  "state_sha256": by_id[case]["state_sha256"]}
                 for repetition in (2, 3) for case in repeat_ids)
    return selected, tasks, {"repeat_ids": repeat_ids, "e2e_ids": e2e_ids,
                             "e2e_extra": e2e_extra}


def input_paths() -> list[Path]:
    rows = load_batch_manifest(BATCH)
    paths = [CONFIG, BATCH, BED, CATEGORIES]
    paths.extend(Path(row[column]) for row in rows for column in (
        "bam", "bam_index", "phased_vcf", "vcf_index", "reference_fasta",
        "reference_fai", "reference_dictionary"))
    for row in rows:
        source = STUDY_ROOT / f"data/processed/batch20/phased/{row['sample_id']}.input.vcf.gz"
        paths.extend((source, Path(str(source) + ".tbi")))
    return sorted(set(paths))


def prepare(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to replace final study: {output}")
    selected, tasks, subsets = select_panel()
    if len(tasks) + 3 + 2 * len(subsets["e2e_ids"]) > MAX_MODEL_CALLS:
        raise ValueError("predeclared upper model-call bound exceeds 120")
    _, skill_sha256 = load_skill_snapshot(load_config(CONFIG).paths.skills_directory)
    fingerprints = {(str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)): sha256(path) for path in input_paths()}
    output.mkdir(parents=True)
    panel = output / "panel.jsonl"
    with panel.open("x") as target:
        for row in selected:
            target.write(json.dumps(row, sort_keys=True) + "\n")
    plan = {"study": "phase18_final_reliability_20261007",
            "data_status": "real_evidence_no_ground_truth", "ground_truth_available": False,
            "scoring_performed": False, "selection_seed": SEED,
            "selection": "20 samples x [one no-info; alternating clean/multi-PS; alternating partial/unphased]; fixed hash rank within stratum; prior development/evaluation IDs excluded",
            "analysis": {"validity": "valid structured response / all direct calls, by slot",
                         "repeatability": "three identical-state calls for ten selected cases; errors included",
                         "guard": "compare raw proposal to existing acceptance guard on same state",
                         "static": "Phase14 assessor on same initial states; no inference",
                         "e2e": "six panel cases plus one fresh same-sample fragmented case through real graph; retry requests, tool calls, cache reuse, terminal actions"},
            "model_call_cap_including_diagnostics": MAX_MODEL_CALLS,
            "execution_seconds_cap": MAX_SECONDS,
            "prior_diagnostic_calls": 3,
            "expected_direct_calls": len(tasks), "e2e_case_bound": len(subsets["e2e_ids"]),
            "panel_sha256": sha256(panel), "input_sha256": fingerprints,
            "skill_sha256": skill_sha256,
            "development_case_sha256": DEVELOPMENT_CASE_SHA256,
            "tasks": tasks, **subsets}
    with (output / "plan.json").open("x") as target:
        json.dump(plan, target, indent=2)
        target.write("\n")
    print(f"frozen {len(selected)} fresh cases, {len(subsets['repeat_ids'])} repeats "
          f"and {len(subsets['e2e_ids'])} end-to-end cases at {output}", flush=True)


def verify_plan(output: Path) -> tuple[dict, dict[str, dict]]:
    development_case_id()
    plan = json.loads((output / "plan.json").read_text())
    if plan.get("development_case_sha256") != DEVELOPMENT_CASE_SHA256:
        raise ValueError("historical development exclusion not frozen; prepare a new plan")
    verify_skill_snapshot(load_config(CONFIG).paths.skills_directory, plan.get("skill_sha256"))
    if sha256(output / "panel.jsonl") != plan["panel_sha256"]:
        raise ValueError("frozen panel changed")
    for relative, expected in plan["input_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"frozen input changed: {relative}")
    rows = [json.loads(line) for line in (output / "panel.jsonl").read_text().splitlines()]
    by_id = {row["case_id"]: row for row in rows}
    if len(rows) != 60 or len(by_id) != 60:
        raise ValueError("frozen panel has missing or duplicate cases")
    for row in rows:
        if state_hash(RegionState.model_validate(row["state"])) != row["state_sha256"]:
            raise ValueError(f"frozen state changed: {row['case_id']}")
    extra = plan["e2e_extra"]
    if extra["case_id"] in by_id or state_hash(RegionState.model_validate(extra["state"])) != extra["state_sha256"]:
        raise ValueError("frozen extra end-to-end case changed")
    by_id[extra["case_id"]] = extra
    return plan, by_id


def run(output: Path, *, resume: bool) -> None:
    with (output / "owner.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another process owns the final study") from exc
        plan, by_id = verify_plan(output)
        existing = list(output.glob("assessment-*.json"))
        if existing and not resume:
            raise FileExistsError("results exist; use --resume after checking the plan")
        config = load_config(CONFIG).with_pipeline_mode(PipelineMode.REAL)
        skills = verify_skill_snapshot(config.paths.skills_directory, plan.get("skill_sha256"))
        assessor = ModelAssessor(model=create_chat_model(config.model), skills=skills,
                                 retry_strategies=config.pipeline.retry_strategies)
        static = build_static_assessor()
        completed_seconds = 0.0
        for index, task in enumerate(plan["tasks"], start=1):
            path = output / f"assessment-{index:03d}.json"
            if path.exists():
                saved = json.loads(path.read_text())
                if saved["task"] != task:
                    raise ValueError(f"saved result does not match frozen task {index}")
                completed_seconds += saved["model_duration_seconds"]
                continue
            if completed_seconds >= MAX_SECONDS:
                print("eight-hour study execution budget reached", flush=True)
                return
            state = RegionState.model_validate(by_id[task["case_id"]]["state"])
            static_result = static.assess(state)
            started = time.monotonic()
            try:
                proposal = assessor.assess(state)
                guarded = guard_acceptance(state, proposal, config)
                model_result = {"status": "success", "proposal": proposal.model_dump(mode="json"),
                                "guarded_proposal": guarded.model_dump(mode="json"),
                                "guard_intervened": guarded is not proposal}
            except ModelAssessmentError as exc:
                model_result = {"status": "error", "error_type": type(exc).__name__,
                                "error_message": str(exc), **exc.audit_detail()}
            duration = round(time.monotonic() - started, 3)
            record = {"task": task, "static": static_result.model_dump(mode="json"),
                      "model": model_result, "model_duration_seconds": duration,
                      "interpretation": "assessor proposal and guard projection only; no action truth"}
            with path.open("x") as target:
                json.dump(record, target, indent=2)
                target.write("\n")
            completed_seconds += duration
            action = model_result.get("proposal", {}).get("action", "ERROR")
            print(f"direct {index}/{len(plan['tasks'])}: {task['slot']} {action} "
                  f"{duration}s", flush=True)
        if completed_seconds >= MAX_SECONDS:
            return
        e2e_dir = output / "e2e"
        if e2e_dir.exists() and any(e2e_dir.iterdir()) and not resume:
            raise FileExistsError("end-to-end output exists; use --resume")
        manifest_rows = {row["sample_id"]: row for row in load_batch_manifest(BATCH)}
        selected = [by_id[case] for case in plan["e2e_ids"]]
        extracted = [ExtractedRegionState(
            state=RegionState.model_validate(row["state"]),
            catalogue_category=row["catalogue_category_external"],
            catalogue_row=row["catalogue_row"]) for row in selected]
        retry_inputs = {row["sample_id"]: STUDY_ROOT /
                        f"data/processed/batch20/phased/{row['sample_id']}.input.vcf.gz"
                        for row in manifest_rows.values() if row["sample_id"] in
                        {item.state.sample_id for item in extracted}}
        run_real_evidence_smoke(
            config=config, extracted_states=extracted, output_directory=e2e_dir,
            resume=resume and e2e_dir.exists(), retry_inputs=retry_inputs,
            region_categories_path=CATEGORIES, write_summary=False, progress_every=1,
        )
        print("direct and end-to-end runs complete", flush=True)


def report(output: Path) -> None:
    plan, _ = verify_plan(output)
    assessments = [json.loads(path.read_text()) for path in sorted(output.glob("assessment-*.json"))]
    if len(assessments) != len(plan["tasks"]):
        raise ValueError("direct assessment outputs are incomplete")
    audits = [RealEvidenceAuditRecord.model_validate_json(path.read_text())
              for path in sorted((output / "e2e").glob("region-*.json"))]
    if len(audits) != len(plan["e2e_ids"]):
        raise ValueError("end-to-end outputs are incomplete")
    by_slot = {}
    for slot in sorted({record["task"]["slot"] for record in assessments}):
        group = [record for record in assessments if record["task"]["slot"] == slot]
        by_slot[slot] = {"calls": len(group), "valid": sum(x["model"]["status"] == "success" for x in group),
                         "errors": sum(x["model"]["status"] == "error" for x in group),
                         "actions": dict(Counter(x["model"]["proposal"]["action"] for x in group
                                                 if x["model"]["status"] == "success")),
                         "guard_interventions": sum(x["model"].get("guard_intervened", False) for x in group)}
    initial = [record for record in assessments if record["task"]["repetition"] == 1]
    valid_initial = [record for record in initial if record["model"]["status"] == "success"]
    repeatability = []
    for case in plan["repeat_ids"]:
        rows = sorted((record for record in assessments if record["task"]["case_id"] == case),
                      key=lambda record: record["task"]["repetition"])
        sequence = [row["model"].get("proposal", {}).get("action", "ERROR") for row in rows]
        repeatability.append({"case_id": case, "slot": rows[0]["task"]["slot"],
                              "actions": sequence, "all_same_valid": len(set(sequence)) == 1 and
                              "ERROR" not in sequence})
    e2e_actions = Counter()
    e2e_sources = Counter()
    regional_retries = 0
    for audit in audits:
        if audit.execution_status == "success" and audit.final_state is not None:
            e2e_actions[audit.final_state.decision.action.value] += 1
            e2e_sources[audit.final_state.decision.source.value] += 1
            regional_retries += max(0, len(audit.final_state.attempt_history) - 1)
    tool_calls = len(list((output / "e2e/retries").glob("*/attempt.json")))
    summary = {"data_status": "real_evidence_no_ground_truth", "scoring_performed": False,
               "direct_calls": len(assessments), "initial_cases": len(initial), "by_slot": by_slot,
               "initial_static_model_agreement": {"agree": sum(
                   row["static"]["action"] == row["model"]["proposal"]["action"]
                   for row in valid_initial), "valid": len(valid_initial)},
               "repeatability": repeatability,
               "guard_interventions_initial": sum(x["model"].get("guard_intervened", False) for x in initial),
               "model_duration_seconds": round(sum(x["model_duration_seconds"] for x in assessments), 3),
               "end_to_end": {"cases": len(audits),
                              "success": sum(a.execution_status == "success" for a in audits),
                              "errors": sum(a.execution_status == "error" for a in audits),
                              "terminal_actions": dict(e2e_actions), "terminal_sources": dict(e2e_sources),
                              "regional_retry_attempts": regional_retries,
                              "sample_wide_whatshap_calls": tool_calls,
                              "cache_reuse_count": max(0, regional_retries - tool_calls)},
               "interpretation": "Unlabeled operational reliability only; no biological accuracy or improvement."}
    path = output / "report.json"
    with path.open("x") as target:
        json.dump(summary, target, indent=2)
        target.write("\n")
    print(f"report saved: {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "report"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.output)
    elif args.mode == "run":
        run(args.output, resume=args.resume)
    else:
        report(args.output)


if __name__ == "__main__":
    main()
