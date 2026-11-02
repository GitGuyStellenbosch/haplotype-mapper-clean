"""Tests for the isolated, unvalidated real-retry mechanics path."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from agent.config import load_config
from agent.pipeline_interface import PhasingRequest, PipelineError
from agent.state import (
    InputFiles,
    PhasingAttempt,
    PhasingEvidence,
    PhasingOutcome,
    RegionIdentity,
    RegionState,
)
from evaluation.real_retry_mechanics import (
    ARTIFACT_POLICY,
    EXECUTION_SCOPE,
    IsolatedWhatsHapMechanicsBackend,
    UNVALIDATED_STATUS,
    load_mechanics_strategy,
)
from pipeline.real_evidence import ExtractedRegionState


MECHANICS_CONFIG = Path("config/unvalidated_real_retry_mechanics.json")


def example_state() -> RegionState:
    evidence = PhasingEvidence(
        outcome=PhasingOutcome.UNPHASED,
        heterozygous_snp_count=1,
        phased_snp_count=0,
        unphased_snp_count=1,
    )
    return RegionState(
        sample_id="sample-1",
        region=RegionIdentity(
            gene="gene-1",
            target_region="target-1",
            chromosome="Chr01",
            start_position=10,
            end_position=20,
        ),
        inputs=InputFiles(
            bam=Path("sample.bam"),
            vcf=Path("sample.input.vcf.gz"),
            reference_fasta=Path("reference.fa"),
            target_regions_bed=Path("targets.bed"),
        ),
        evidence=evidence,
        attempt_history=[
            PhasingAttempt(
                attempt_number=1,
                strategy_id="provided-whatshap-polyphase-output",
                evidence=evidence,
            )
        ],
    )


class RealRetryMechanicsTests(TestCase):
    def test_isolated_config_is_explicitly_unvalidated(self) -> None:
        raw, strategy = load_mechanics_strategy(MECHANICS_CONFIG)

        self.assertEqual(raw["validation_status"], UNVALIDATED_STATUS)
        self.assertFalse(raw["scientifically_approved"])
        self.assertFalse(raw["eligible_for_real_evaluation"])
        self.assertEqual(raw["execution_scope"], EXECUTION_SCOPE)
        self.assertEqual(raw["artifact_policy"], ARTIFACT_POLICY)
        self.assertEqual(strategy.parameters["block_cut_sensitivity"], 2)

    def test_active_research_config_does_not_promote_mechanics_strategy(self) -> None:
        _, mechanics_strategy = load_mechanics_strategy(MECHANICS_CONFIG)
        active = load_config().pipeline.retry_strategies
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0].strategy_id, "lower-block-cut-sensitivity-v1")
        self.assertNotIn(mechanics_strategy, active)
        self.assertFalse(any("mechanics" in s.strategy_id for s in active))

    def test_loader_rejects_scientific_approval(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            raw = json.loads(MECHANICS_CONFIG.read_text(encoding="utf-8"))
            raw["scientifically_approved"] = True
            path.write_text(json.dumps(raw), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "scientifically unapproved"):
                load_mechanics_strategy(path)

    def test_backend_reuses_command_builder_and_normalizes_output(self) -> None:
        _, strategy = load_mechanics_strategy(MECHANICS_CONFIG)
        state = example_state()
        extracted_state = example_state()
        extracted_state.evidence = PhasingEvidence(
            outcome=PhasingOutcome.PHASED,
            heterozygous_snp_count=1,
            phased_snp_count=1,
            unphased_snp_count=0,
            tool_messages=[
                "Evidence extracted from an existing WhatsHap-phased VCF; the "
                "smoke runner did not re-run WhatsHap."
            ],
        )
        with TemporaryDirectory() as directory:
            output = Path(directory) / "unvalidated-mechanics.vcf.gz"
            backend = IsolatedWhatsHapMechanicsBackend(
                whatshap=load_config().pipeline.whatshap,
                strategy=strategy,
                output_vcf=output,
                region_categories_path="categories.csv",
            )

            def fake_run(argv: list[str], **_: object) -> object:
                output.write_text("placeholder", encoding="utf-8")
                return type(
                    "Completed",
                    (),
                    {"returncode": 0, "stdout": "", "stderr": ""},
                )()

            request = PhasingRequest(
                sample_id=state.sample_id,
                region=state.region,
                inputs=state.inputs,
                strategy=strategy,
            )
            with patch(
                "evaluation.real_retry_mechanics.subprocess.run",
                side_effect=fake_run,
            ), patch(
                "evaluation.real_retry_mechanics.build_real_region_states",
                return_value=[
                    ExtractedRegionState(
                        state=extracted_state,
                        catalogue_category="2",
                        catalogue_row=2,
                    )
                ],
            ):
                evidence = backend.phase(request)

        assert backend.executed_argv is not None
        self.assertEqual(
            backend.executed_argv[
                backend.executed_argv.index("--block-cut-sensitivity") + 1
            ],
            "2",
        )
        self.assertEqual(evidence.outcome, PhasingOutcome.PHASED)
        self.assertEqual(
            evidence.additional_metrics["mechanics_retry_validation_status"],
            UNVALIDATED_STATUS,
        )
        self.assertEqual(
            evidence.additional_metrics["mechanics_retry_execution_scope"],
            EXECUTION_SCOPE,
        )
        self.assertEqual(
            evidence.additional_metrics["mechanics_retry_artifact_policy"],
            ARTIFACT_POLICY,
        )
        self.assertIn("unvalidated mechanics-test", evidence.tool_messages[0])

    def test_backend_runs_sample_strategy_once_and_reuses_cached_regions(self) -> None:
        _, strategy = load_mechanics_strategy(MECHANICS_CONFIG)
        first = example_state()
        second = example_state()
        second.region = RegionIdentity(
            gene="gene-2",
            target_region="target-2",
            chromosome="Chr01",
            start_position=30,
            end_position=40,
        )
        second.evidence = PhasingEvidence(
            outcome=PhasingOutcome.PHASED,
            heterozygous_snp_count=1,
            phased_snp_count=1,
            unphased_snp_count=0,
        )

        with TemporaryDirectory() as directory:
            output = Path(directory) / "sample-wide-retry.vcf.gz"
            backend = IsolatedWhatsHapMechanicsBackend(
                whatshap=load_config().pipeline.whatshap,
                strategy=strategy,
                output_vcf=output,
                region_categories_path="categories.csv",
            )

            def fake_run(argv: list[str], **_: object) -> object:
                output.write_text("immutable artifact", encoding="utf-8")
                return type(
                    "Completed", (), {"returncode": 0, "stdout": "", "stderr": ""}
                )()

            first_request = PhasingRequest(
                sample_id=first.sample_id,
                region=first.region,
                inputs=first.inputs,
                strategy=strategy,
            )
            second_request = first_request.model_copy(
                update={"region": second.region}
            )
            extracted = [
                ExtractedRegionState(
                    state=state,
                    catalogue_category="2",
                    catalogue_row=index,
                )
                for index, state in enumerate((first, second), start=2)
            ]
            with patch(
                "evaluation.real_retry_mechanics.subprocess.run",
                side_effect=fake_run,
            ) as run_mock, patch(
                "evaluation.real_retry_mechanics.build_real_region_states",
                return_value=extracted,
            ) as extraction_mock:
                first_result = backend.phase(first_request)
                second_result = backend.phase(second_request)

            self.assertEqual(first_result.outcome, PhasingOutcome.UNPHASED)
            self.assertEqual(second_result.outcome, PhasingOutcome.PHASED)
            self.assertEqual(backend.execution_count, 1)
            run_mock.assert_called_once()
            extraction_mock.assert_called_once()

    def test_backend_rejects_mutated_cached_artifact(self) -> None:
        _, strategy = load_mechanics_strategy(MECHANICS_CONFIG)
        state = example_state()
        request = PhasingRequest(
            sample_id=state.sample_id,
            region=state.region,
            inputs=state.inputs,
            strategy=strategy,
        )
        with TemporaryDirectory() as directory:
            output = Path(directory) / "sample-wide-retry.vcf.gz"
            backend = IsolatedWhatsHapMechanicsBackend(
                whatshap=load_config().pipeline.whatshap,
                strategy=strategy,
                output_vcf=output,
                region_categories_path="categories.csv",
            )

            def fake_run(argv: list[str], **_: object) -> object:
                output.write_text("original artifact", encoding="utf-8")
                return type(
                    "Completed", (), {"returncode": 0, "stdout": "", "stderr": ""}
                )()

            with patch(
                "evaluation.real_retry_mechanics.subprocess.run",
                side_effect=fake_run,
            ), patch(
                "evaluation.real_retry_mechanics.build_real_region_states",
                return_value=[
                    ExtractedRegionState(
                        state=state,
                        catalogue_category="2",
                        catalogue_row=2,
                    )
                ],
            ):
                backend.phase(request)
                output.write_text("changed artifact", encoding="utf-8")
                with self.assertRaisesRegex(PipelineError, "changed after creation"):
                    backend.phase(request)


if __name__ == "__main__":
    raise SystemExit(TestCase)
