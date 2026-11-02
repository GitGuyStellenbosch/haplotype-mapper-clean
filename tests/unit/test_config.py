"""Unit tests for loading and validating agent configuration."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
import yaml

from agent.config import (
    EXAMPLE_CONFIG_PATH,
    ConfigurationError,
    PipelineMode,
    load_config,
)
from agent.state import DecisionAction


VALID_CONFIG = EXAMPLE_CONFIG_PATH.read_text(encoding="utf-8")


class ConfigurationTests(TestCase):
    def write_config(self, directory: Path, content: str) -> Path:
        path = directory / "agent.yaml"
        path.write_text(content, encoding="utf-8")
        return path

    def test_example_configuration_loads(self) -> None:
        config = load_config()

        self.assertEqual(config.pipeline.mode, PipelineMode.MOCK)
        self.assertEqual(config.decisions.max_retries, 2)
        self.assertEqual(set(config.decisions.allowed_actions), set(DecisionAction))
        self.assertEqual(config.pipeline.whatshap.block_cut_sensitivity, 4)
        self.assertEqual(
            config.pipeline.gatk.hard_filters.model_dump(),
            {
                "minimum_qd": 2.0,
                "minimum_qual": 30.0,
                "maximum_sor": 3.0,
                "maximum_fs": 60.0,
                "minimum_mq": 40.0,
                "minimum_mq_rank_sum": -12.5,
                "minimum_read_pos_rank_sum": -8.0,
            },
        )
        self.assertEqual(
            config.execution.graph_recursion_limit(config.decisions.max_retries),
            14,
        )

    def test_relative_paths_resolve_from_project_root(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = self.write_config(root, VALID_CONFIG)

            config = load_config(config_path, project_root=root)

            self.assertEqual(config.paths.skills_directory, root / "agent" / "skills")
            self.assertEqual(config.paths.output_directory, root / "outputs")

    def test_unknown_configuration_fields_are_rejected(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = self.write_config(
                root,
                VALID_CONFIG.replace(
                    "    temperature: 0",
                    "    temperature: 0\n    typo: true",
                ),
            )

            with self.assertRaises(ConfigurationError):
                load_config(config_path, project_root=root)

    def test_action_space_cannot_be_changed(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = self.write_config(
                root,
                VALID_CONFIG.replace(
                    "allowed_actions: [accept, retry, escalate]",
                    "allowed_actions: [accept, retry]",
                ),
            )

            with self.assertRaises(ConfigurationError):
                load_config(config_path, project_root=root)

    def test_negative_retry_limit_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = self.write_config(
                root,
                VALID_CONFIG.replace("max_retries: 2", "max_retries: -1"),
            )

            with self.assertRaises(ConfigurationError):
                load_config(config_path, project_root=root)

    def test_negative_nonnegative_gatk_threshold_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = self.write_config(
                root,
                VALID_CONFIG.replace("minimum_qd: 2.0", "minimum_qd: -0.1"),
            )

            with self.assertRaises(ConfigurationError):
                load_config(config_path, project_root=root)

    def test_missing_file_has_clear_configuration_error(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            missing_path = Path(temporary_directory) / "missing.yaml"

            with self.assertRaises(ConfigurationError):
                load_config(missing_path, project_root=temporary_directory)

    def test_duplicate_retry_strategy_ids_are_rejected(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw = yaml.safe_load(VALID_CONFIG)
            raw["pipeline"]["retry_strategies"] = [
                {"strategy_id": "duplicate"}, {"strategy_id": "duplicate"}
            ]
            content = yaml.safe_dump(raw)
            config_path = self.write_config(root, content)

            with self.assertRaises(ConfigurationError):
                load_config(config_path, project_root=root)
