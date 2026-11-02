"""Small, fixed raw-response diagnostic for the final-study preparation."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from agent.config import load_config
from agent.llm import ModelAssessmentError, ModelAssessor, create_chat_model, load_skills
from agent.state import RegionState


ROOT = Path(__file__).resolve().parents[1]
# Ordered identities of the original SOURCES/CLEAN_ID, without saved case records.
EXPECTED_CASE_SHA256 = (
    "f0c920b4d9cf156b00ff871ca6ccf6c131d5a4c710d0bee2b06fd0c3712cfd91",
    "e75fa7895356b8d295bdda22142b69666c842f5f176cf76f3b18272327442258",
    "7be22a800665833d2947e4e07c6f3c3bebe459075af5dfc08fe1950b56f74779",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/parameters.yaml")
    parser.add_argument("--sources", type=Path, required=True,
                        help="JSON list of three {name, path, case_id (for JSONL)} source specifications.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    sources = json.loads(args.sources.read_text())
    if not isinstance(sources, list) or len(sources) != 3:
        raise ValueError("the fixed diagnostic requires exactly three source cases")
    if output.exists():
        raise FileExistsError(f"refusing to replace diagnostic outputs: {output}")
    config_path = args.config.resolve()
    config = load_config(config_path)
    skills = load_skills(config.paths.skills_directory)
    cases = []
    for spec in sources:
        name = spec["name"]
        path = (args.sources.resolve().parent / spec["path"]).resolve()
        if path.suffix == ".jsonl":
            matches = [json.loads(line) for line in path.read_text().splitlines()
                       if json.loads(line)["case_id"] == spec["case_id"]]
            if len(matches) != 1:
                raise ValueError("clean diagnostic case missing or duplicated")
            raw_state = matches[0]["state"]
        else:
            raw_state = json.loads(path.read_text())["initial_state"]
        state = RegionState.model_validate(raw_state)
        identity = f"{state.sample_id}::{state.region.region_key}"
        if spec.get("case_id", identity) != identity:
            raise ValueError("supplied diagnostic case identity differs from source state")
        cases.append({"name": name, "source": str(path), "source_sha256": sha256(path),
                      "case_id": f"{state.sample_id}::{state.region.region_key}",
                      "state": state.model_dump(mode="json", exclude_computed_fields=True)})
    identities = [case["case_id"] for case in cases]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate diagnostic case identities")
    if tuple(hashlib.sha256(identity.encode()).hexdigest() for identity in identities) != EXPECTED_CASE_SHA256:
        raise ValueError("diagnostic case identities/order must match the original historical selection")
    plan = {"purpose": "diagnose prior structured-output failures before final study",
            "config_sha256": sha256(config_path),
            "skill_sha256": {skill.name: sha256(config.paths.skills_directory / skill.name)
                             for skill in skills}, "cases": cases}
    output.mkdir(parents=True)
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    assessor = ModelAssessor(model=create_chat_model(config.model), skills=skills,
                             retry_strategies=config.pipeline.retry_strategies)
    for index, case in enumerate(cases, start=1):
        started = time.monotonic()
        try:
            result = assessor.assess(RegionState.model_validate(case["state"]))
            assessment = {"status": "success", "proposal": result.model_dump(mode="json")}
        except ModelAssessmentError as exc:
            assessment = {"status": "error", "error_type": type(exc).__name__,
                          "error_message": str(exc), **exc.audit_detail()}
        record = {"name": case["name"], "case_id": case["case_id"],
                  "duration_seconds": round(time.monotonic() - started, 3),
                  "assessment": assessment}
        with (output / f"case-{index:02d}.json").open("x") as target:
            json.dump(record, target, indent=2)
            target.write("\n")
        print(f"{index}/{len(cases)} {case['name']}: {assessment['status']} "
              f"{record['duration_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
