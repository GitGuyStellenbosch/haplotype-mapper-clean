"""Research retry execution, persistence, eligibility, and reassessment guards."""

import gzip
import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from agent.config import PipelineMode, load_config
from agent.graph import build_agent_graph
from agent.nodes import AssessmentResult, make_assess_node
from agent.pipeline_interface import PhasingRequest, PipelineError, PipelineInterface
from agent.real_retry import WhatsHapRetryBackend, _genotypes
from agent.real_smoke import RealSmokeError, run_real_evidence_smoke
from agent.state import (
    DecisionAction, DecisionSource, InputFiles, PhasingAttempt, PhasingEvidence,
    PhasingOutcome, PhasingStrategy, RegionIdentity, RegionState,
)
from pipeline.real_evidence import ExtractedRegionState


def vcf(*, changed=False, phased=False):
    genotype = "0/0/0/1" if changed else "0/0/1/1"
    if phased:
        genotype = genotype.replace("/", "|")
    return ("##fileformat=VCFv4.2\n"
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE1\n"
            f"Chr01\t10\t.\tA\tG\t50\tPASS\t.\tGT\t{genotype}\n"
            "Chr01\t20\t.\tC\tT\t50\tPASS\t.\tGT\t0/0/1/1\n")


class ResearchRetryTests(TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "input.vcf"
        self.source.write_text(vcf())
        for name in ["sample.bam", "reference.fa", "targets.bed", "categories.csv"]:
            (self.root / name).touch()
        self.config = load_config().with_pipeline_mode(PipelineMode.REAL)
        self.strategies = self.config.pipeline.retry_strategies
        self.state = RegionState(
            sample_id="SAMPLE1",
            region=RegionIdentity(gene="GeneA", target_region="start", chromosome="Chr01",
                                  start_position=10, end_position=20),
            inputs=InputFiles(bam=self.root / "sample.bam", vcf=self.source,
                              reference_fasta=self.root / "reference.fa",
                              target_regions_bed=self.root / "targets.bed"),
            evidence=PhasingEvidence(outcome=PhasingOutcome.PHASED,
                heterozygous_snp_count=2, phased_snp_count=2, unphased_snp_count=0,
                call_rate=1, additional_metrics={"phase_set_count": 2}),
        )
        self.state.attempt_history = [PhasingAttempt(attempt_number=1, strategy_id="baseline",
                                                     evidence=self.state.evidence)]

    def backend(self):
        return WhatsHapRetryBackend(whatshap=self.config.pipeline.whatshap,
            strategies=self.strategies, input_vcfs={"SAMPLE1": self.source},
            region_categories_path=self.root / "categories.csv",
            output_directory=self.root / "retries", timeout_seconds=120)

    def request(self, strategy=None):
        return PhasingRequest(sample_id=self.state.sample_id, region=self.state.region,
                              inputs=self.state.inputs, strategy=strategy or self.strategies[0])

    def output_state(self):
        state = self.state.model_copy(deep=True)
        state.evidence.additional_metrics["phase_set_count"] = 1
        return ExtractedRegionState(state=state, catalogue_category="1", catalogue_row=1)

    def fake_run(self, argv, **kwargs):
        output = Path(argv[argv.index("--output") + 1])
        with gzip.open(output, "wt") as handle:
            handle.write(vcf(changed="--distrust-genotypes" in argv, phased=True))
        return SimpleNamespace(returncode=0, stdout="test output", stderr="")

    def test_phasing_strategy_executes_once_and_resume_reuses_verified_artifact(self):
        backend = self.backend()
        with patch("agent.real_retry.validate_real_inputs"), patch(
            "agent.real_retry.build_real_region_states", return_value=[self.output_state()]
        ), patch("agent.real_retry.subprocess.run", side_effect=self.fake_run) as run:
            first = backend.phase(self.request())
            self.assertEqual(first.additional_metrics["retry_genotype_change_count"], 0)
            backend.phase(self.request())
            # A new backend after process restart reconstructs the same cache.
            self.backend().phase(self.request())
            self.assertEqual(run.call_count, 1)
            self.assertEqual(run.call_args.kwargs["timeout"], 120)
        self.assertEqual(self.source.read_text(), vcf())
        self.assertEqual(len(list((self.root / "retries").glob("*/attempt.json"))), 1)

    def test_different_regions_share_artifact_without_mutating_case_records(self):
        backend = self.backend()
        second = self.state.model_copy(deep=True)
        second.region.gene = "GeneB"
        extracted = [self.output_state(), ExtractedRegionState(state=second, catalogue_category="2", catalogue_row=2)]
        with patch("agent.real_retry.validate_real_inputs"), patch(
            "agent.real_retry.build_real_region_states", return_value=extracted
        ), patch("agent.real_retry.subprocess.run", side_effect=self.fake_run) as run:
            backend.phase(self.request())
            request = self.request().model_copy(update={"region": second.region})
            returned = backend.phase(request)
            returned.additional_metrics["phase_set_count"] = 99
            self.assertEqual(backend.phase(request).additional_metrics["phase_set_count"], 2)
            self.assertEqual(run.call_count, 1)
        self.assertEqual(second.evidence.additional_metrics["phase_set_count"], 2)

    def test_rejects_unlisted_altered_and_wrongly_typed_parameters(self):
        backend = self.backend()
        for strategy in [PhasingStrategy(strategy_id="invented"),
            PhasingStrategy(strategy_id=self.strategies[0].strategy_id, parameters={"block_cut_sensitivity": 3}),
            PhasingStrategy(strategy_id=self.strategies[0].strategy_id, parameters={"block_cut_sensitivity": 2.0}),
            PhasingStrategy(strategy_id="distrust-genotypes-v1", parameters={"distrust_genotypes": True})]:
            with self.subTest(strategy=strategy), self.assertRaises(PipelineError):
                backend.phase(self.request(strategy))

        with self.assertRaisesRegex(ValueError, "unsupported retry strategy"):
            WhatsHapRetryBackend(whatshap=self.config.pipeline.whatshap,
                strategies=[PhasingStrategy(strategy_id="distrust-genotypes-v1",
                                            parameters={"distrust_genotypes": True})],
                input_vcfs={"SAMPLE1": self.source},
                region_categories_path=self.root / "categories.csv",
                output_directory=self.root / "retries", timeout_seconds=120)

    def test_distrusting_input_genotypes_is_rejected_at_backend_initialization(self):
        whatshap = self.config.pipeline.whatshap.model_copy(update={"distrust_genotypes": True})
        with self.assertRaisesRegex(ValueError, "trusted input genotypes"):
            WhatsHapRetryBackend(whatshap=whatshap, strategies=self.strategies,
                input_vcfs={"SAMPLE1": self.source},
                region_categories_path=self.root / "categories.csv",
                output_directory=self.root / "retries", timeout_seconds=120)

    def test_modified_output_and_modified_input_are_rejected(self):
        backend = self.backend()
        with patch("agent.real_retry.validate_real_inputs"), patch(
            "agent.real_retry.build_real_region_states", return_value=[self.output_state()]
        ), patch("agent.real_retry.subprocess.run", side_effect=self.fake_run):
            evidence = backend.phase(self.request())
            output = Path(evidence.additional_metrics["retry_output_vcf"])
            saved = output.read_bytes()
            output.write_bytes(saved + b"changed")
            with self.assertRaisesRegex(PipelineError, "artifact changed"):
                backend.phase(self.request())
            output.write_bytes(saved)
            self.source.write_text(vcf() + "\n")
            with self.assertRaisesRegex(PipelineError, "inputs changed"):
                backend.phase(self.request())

    def test_timeout_is_recorded_and_never_reexecuted(self):
        backend = self.backend()
        with patch("agent.real_retry.validate_real_inputs"), patch(
            "agent.real_retry.subprocess.run", side_effect=subprocess.TimeoutExpired("whatshap", 120)
        ) as run:
            for _ in range(2):
                with self.assertRaisesRegex(PipelineError, "TimeoutExpired"):
                    backend.phase(self.request())
            self.assertEqual(run.call_count, 1)
        audit = json.loads(next((self.root / "retries").glob("*/attempt.json")).read_text())
        self.assertEqual(audit["status"], "error")
        self.assertFalse(audit["scientifically_validated"])

    def test_phased_retry_source_is_rejected(self):
        self.source.write_text(vcf(phased=True))
        with patch("agent.real_retry.validate_real_inputs"), self.assertRaisesRegex(
            PipelineError, "original unphased"
        ):
            self.backend().phase(self.request())

    def test_duplicate_variants_preserve_multiplicity_and_reject_dosage_changes(self):
        original = vcf()
        row = original.splitlines(keepends=True)[2]
        duplicate = row.replace("0/0/1/1", "0/0/0/1")
        other_allele = row.replace("\tG\t", "\tT\t")
        self.source.write_text(original.replace(row, row + duplicate + other_allele))
        before = _genotypes(self.source, "SAMPLE1")
        output = self.root / "alternative.vcf"
        output.write_text(original.replace(row, duplicate + other_allele + row))
        self.assertEqual(before, _genotypes(output, "SAMPLE1"))
        output.write_text(original.replace(row, duplicate + other_allele + duplicate))
        with self.assertRaisesRegex(PipelineError, "allele dosage"):
            self.backend()._extract(output, before, self.request(), self.strategies[0])

    def test_audit_rejects_dropped_duplicates_and_changed_variant_alleles(self):
        original = vcf()
        row = original.splitlines(keepends=True)[2]
        self.source.write_text(original.replace(row, row + row))
        before = _genotypes(self.source, "SAMPLE1")
        output = self.root / "alternative.vcf"
        for content in [original, original.replace(row, row + row.replace("\tG\t", "\tT\t"))]:
            output.write_text(content)
            with self.assertRaisesRegex(PipelineError, "records or alleles"):
                self.backend()._extract(output, before, self.request(), self.strategies[0])

    def retry_assessor(self, strategy):
        assessor = Mock()
        assessor.assess.return_value = AssessmentResult(action=DecisionAction.RETRY,
            confidence=0.5, rationale="synthetic eligibility check", source=DecisionSource.LLM,
            retry_strategy=strategy)
        return assessor

    def test_real_retry_eligibility_does_not_infer_missing_evidence(self):
        for strategy in self.strategies:
            state = self.state.model_copy(deep=True)
            state.evidence.additional_metrics.clear()
            result = make_assess_node(self.retry_assessor(strategy), self.config)(state)
            self.assertEqual(result["decision"].action, DecisionAction.ESCALATE)
        self.state.evidence.additional_metrics["genotype_read_disagreement"] = True
        disallowed = PhasingStrategy(strategy_id="distrust-genotypes-v1",
                                     parameters={"distrust_genotypes": True})
        result = make_assess_node(self.retry_assessor(disallowed), self.config)(self.state)
        self.assertEqual(result["decision"].action, DecisionAction.ESCALATE)

    def test_clear_real_fragmentation_uses_recorded_policy_without_model(self):
        state = self.state.model_copy(deep=True)
        state.metadata["data_status"] = "real_evidence_no_ground_truth"
        assessor = Mock()
        assessor.assess.side_effect = AssertionError("clear policy case should not call model")
        assess = make_assess_node(assessor, self.config)

        first = assess(state)
        self.assertEqual(first["decision"].action, DecisionAction.RETRY)
        self.assertEqual(first["decision"].source, DecisionSource.DETERMINISTIC)
        self.assertEqual(first["pending_strategy"], self.strategies[0])

        changed = state.model_copy(deep=True)
        changed.evidence = state.evidence.model_copy(deep=True)
        changed.evidence.additional_metrics.update(
            phase_set_count=1, retry_policy_version="research-retries-20261006-v2",
            retry_genotype_change_count=0,
        )
        changed.attempt_history.append(PhasingAttempt(
            attempt_number=2, strategy_id=self.strategies[0].strategy_id,
            evidence=changed.evidence,
        ))
        second = assess(changed)
        self.assertEqual(second["decision"].action, DecisionAction.ACCEPT)
        self.assertEqual(second["decision"].source, DecisionSource.DETERMINISTIC)
        self.assertEqual(assessor.assess.call_count, 0)

        changed.warnings.append("Unresolved input problem")
        with self.assertRaisesRegex(AssertionError, "should not call model"):
            assess(changed)

    def test_graph_reassesses_phasing_only_retry(self):
        backend = self.backend()
        state = self.state.model_copy(deep=True)
        assessor = self.retry_assessor(self.strategies[0])
        accept = AssessmentResult(action=DecisionAction.ACCEPT, confidence=0.5,
            rationale="synthetic reassessment", source=DecisionSource.LLM)
        assessor.assess.side_effect = [assessor.assess.return_value, accept]
        graph = build_agent_graph(config=self.config, assessor=assessor,
            pipeline=PipelineInterface(backend, validate_inputs=False))
        with patch("agent.real_retry.validate_real_inputs"), patch(
            "agent.real_retry.build_real_region_states", return_value=[self.output_state()]
        ), patch("agent.real_retry.subprocess.run", side_effect=self.fake_run):
            final = RegionState.model_validate(graph.invoke(state))
        self.assertEqual(len(final.attempt_history), 2)
        self.assertEqual(final.decision.action, DecisionAction.ACCEPT)
        self.assertEqual(assessor.assess.call_count, 2)

    def test_backend_rejects_genotype_change_even_under_phasing_only_strategy(self):
        def changed_run(argv, **kwargs):
            output = Path(argv[argv.index("--output") + 1])
            with gzip.open(output, "wt") as handle:
                handle.write(vcf(changed=True, phased=True))
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch("agent.real_retry.validate_real_inputs"), patch(
            "agent.real_retry.subprocess.run", side_effect=changed_run
        ), self.assertRaisesRegex(PipelineError, "phasing-only retry changed allele dosage"):
            self.backend().phase(self.request())
        audit = json.loads(next((self.root / "retries").glob("*/attempt.json")).read_text())
        self.assertEqual(audit["status"], "error")

    def test_unchanged_retry_stops_before_model_without_treating_unknown_as_zero(self):
        current = self.state.evidence.model_copy(deep=True)
        current.additional_metrics.update(retry_policy_version="research-retries-20261006-v2",
                                          retry_genotype_change_count=0)
        self.state.attempt_history.append(PhasingAttempt(attempt_number=2,
            strategy_id=self.strategies[0].strategy_id, evidence=current))
        self.state.evidence = current
        assessor = Mock()
        result = make_assess_node(assessor, self.config)(self.state)
        self.assertEqual(result["decision"].action, DecisionAction.ESCALATE)
        assessor.assess.assert_not_called()
        self.state.attempt_history[0].evidence.call_rate = None
        assessor.assess.return_value = AssessmentResult(action=DecisionAction.ESCALATE,
            confidence=0.5, rationale="missing evidence", source=DecisionSource.LLM)
        make_assess_node(assessor, self.config)(self.state)
        assessor.assess.assert_called_once()

    def test_runner_requires_original_inputs_before_model_initialization(self):
        with patch("agent.real_smoke.create_chat_model") as create:
            with self.assertRaisesRegex(RealSmokeError, "explicit original"):
                run_real_evidence_smoke(config=self.config, extracted_states=[self.output_state()],
                                       output_directory=self.root / "run")
            create.assert_not_called()

    def test_runner_snapshots_skills_runs_retry_and_rejects_changed_resume(self):
        model = Mock()
        structured = model.with_structured_output.return_value
        structured.invoke.side_effect = [
            {"raw": None, "parsing_error": None,
             "parsed": {"action": "retry", "confidence": 0.5,
                        "rationale": "synthetic fragmentation",
                        "retry_strategy": self.strategies[0].model_dump()}},
            {"raw": None, "parsing_error": None,
             "parsed": {"action": "accept", "confidence": 0.5,
                        "rationale": "synthetic reassessment",
                        "retry_strategy": None}},
        ]
        output = self.root / "run"
        arguments = dict(config=self.config, extracted_states=[ExtractedRegionState(
            state=self.state, catalogue_category="1", catalogue_row=1)],
            output_directory=output, retry_inputs={"SAMPLE1": self.source},
            region_categories_path=self.root / "categories.csv", write_summary=False)
        with patch("agent.real_smoke.create_chat_model", return_value=model), patch(
            "agent.real_smoke.validate_real_inputs"
        ), patch("agent.pipeline_interface.validate_real_inputs"), patch(
            "agent.real_retry.validate_real_inputs"
        ), patch("agent.real_retry.build_real_region_states", return_value=[self.output_state()]), patch(
            "agent.real_retry.subprocess.run", side_effect=self.fake_run
        ) as run:
            records = run_real_evidence_smoke(**arguments)
            self.assertEqual(records[0].execution_status, "success")
            self.assertEqual(len(records[0].final_state.attempt_history), 2)
            resumed = run_real_evidence_smoke(**arguments, resume=True)
            self.assertEqual(resumed[0].run_id, records[0].run_id)
            self.assertEqual(structured.invoke.call_count, 2)
            self.assertEqual(run.call_count, 1)
        plan = json.loads((output / "assessment_plan.json").read_text())
        self.assertEqual(set(plan["skills"]), {"phasing_triage.md", "retry_policy.md"})
        self.assertTrue(all(len(s["sha256"]) == 64 for s in plan["skills"].values()))
        changed = self.config.model_copy(deep=True)
        changed.execution.whatshap_retry_timeout_seconds += 1
        with patch("agent.real_smoke.create_chat_model") as create, self.assertRaisesRegex(
            RealSmokeError, "assessment plan changed"
        ):
            run_real_evidence_smoke(**{**arguments, "config": changed}, resume=True)
        create.assert_not_called()
