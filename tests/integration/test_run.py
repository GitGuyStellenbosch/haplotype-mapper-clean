"""Integration tests for the synthetic CLI workflow and audit records."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from agent.config import load_config
from agent.run import RunRecord, run_synthetic_case, write_run_record
from agent.state import DecisionAction, PhasingOutcome


class SyntheticRunTests(TestCase):
    def test_scripted_acceptance_produces_labelled_record(self) -> None:
        record = run_synthetic_case(
            config=load_config(),
            scenario="accept",
            assessor_kind="scripted",
        )

        self.assertFalse(record.scientific_interpretation_allowed)
        self.assertEqual(record.data_mode, "synthetic")
        self.assertEqual(record.final_state.decision.action, DecisionAction.ACCEPT)
        self.assertIsNone(record.model_name)

    def test_scripted_retry_success_records_both_attempts(self) -> None:
        record = run_synthetic_case(
            config=load_config(),
            scenario="retry-success",
            assessor_kind="scripted",
        )

        self.assertEqual(len(record.final_state.attempt_history), 2)
        self.assertEqual(len(record.final_state.decision_history), 2)
        self.assertEqual(
            record.final_state.decision_history[0].action,
            DecisionAction.RETRY,
        )
        self.assertEqual(record.final_state.evidence.outcome, PhasingOutcome.PHASED)
        self.assertEqual(record.final_state.decision.action, DecisionAction.ACCEPT)

    def test_retry_exhaustion_records_deterministic_escalation(self) -> None:
        record = run_synthetic_case(
            config=load_config(),
            scenario="retry-exhausted",
            assessor_kind="scripted",
        )

        self.assertEqual(record.final_state.decision.action, DecisionAction.ESCALATE)
        self.assertTrue(record.final_state.warnings)

    def test_writer_does_not_silently_overwrite(self) -> None:
        record = run_synthetic_case(
            config=load_config(),
            scenario="escalate",
            assessor_kind="scripted",
        )
        with TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "record.json"

            write_run_record(record, output)
            with self.assertRaises(FileExistsError):
                write_run_record(record, output)

            written = write_run_record(record, output, overwrite=True)
            self.assertEqual(written, output)
            self.assertIn('"data_mode": "synthetic"', output.read_text())
            reloaded = RunRecord.model_validate_json(output.read_text())
            self.assertEqual(reloaded.run_id, record.run_id)
