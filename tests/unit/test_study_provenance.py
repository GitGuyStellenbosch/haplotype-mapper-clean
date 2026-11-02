"""Artificial regression checks for frozen study inputs; no model/tool calls."""

import hashlib
import io
import json
import os
import sys
from contextlib import ExitStack, contextmanager, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from agent.config import load_config
from evaluation import phase17_sensitivity as sensitivity
from evaluation import phase18_diagnostics as diagnostics
from evaluation import phase18_final_study as final_study
from tests.unit.test_nodes import example_state


class ModelBoundary(Exception):
    """Stop before constructing any model in positive validation checks."""


def identity_hash(identity):
    return hashlib.sha256(identity.encode()).hexdigest()


class StudyProvenanceTests(TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.skills = self.root / "skills"
        self.skills.mkdir()
        self.reset_skills()
        self.config = load_config()
        self.config = self.config.model_copy(update={
            "paths": self.config.paths.model_copy(update={"skills_directory": self.skills})
        })
        self.config_path = self.root / "config.yaml"
        self.config_path.write_text("artificial configuration fingerprint\n")
        self.development_id = "ARTIFICIAL_DEVELOPMENT::artificial-region"

    def reset_skills(self):
        for path in self.skills.iterdir():
            path.unlink()
        for name in ("alpha.md", "beta.md"):
            (self.skills / name).write_text("**Status:** artificial test\n\nGuidance.\n")
        (self.skills / "README.md").write_text("Not an assessment Skill.\n")

    def panel_row(self, index):
        state = example_state().model_copy(update={"sample_id": f"ARTIFICIAL_{index}"})
        return {
            "case_id": f"{state.sample_id}::{state.region.region_key}",
            "state": state.model_dump(mode="json", exclude_computed_fields=True),
            "state_sha256": final_study.state_hash(state),
        }

    @contextmanager
    def prepared(self, module):
        output = self.root / module.__name__.split(".")[-1]
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps({"pairs": []}))
        with ExitStack() as stack:
            stack.enter_context(patch.object(module, "CONFIG", self.config_path))
            stack.enter_context(patch.object(module, "load_config", return_value=self.config))
            if module is sensitivity:
                stack.enter_context(patch.object(module, "MANIFEST", manifest))
            else:
                stack.enter_context(patch.object(module, "DEVELOPMENT_CASE_SHA256",
                                                identity_hash(self.development_id)))
                stack.enter_context(patch.dict(os.environ, {
                    "HAPLOTYPE_DEVELOPMENT_CASE_ID": self.development_id,
                }))
                rows = [self.panel_row(index) for index in range(60)]
                subsets = {"repeat_ids": [], "e2e_ids": [], "e2e_extra": self.panel_row(60)}
                stack.enter_context(patch.object(module, "select_panel", return_value=(rows, [], subsets)))
                stack.enter_context(patch.object(module, "input_paths", return_value=[self.config_path]))
            with redirect_stdout(io.StringIO()):
                module.prepare(output)
            yield output

    def test_unchanged_configured_skills_reach_model_boundary(self):
        for module in (sensitivity, final_study):
            with self.subTest(runner=module.__name__), self.prepared(module) as output:
                plan = json.loads((output / "plan.json").read_text())
                self.assertEqual(set(plan["skill_sha256"]), {"alpha.md", "beta.md"})
                with patch.object(module, "create_chat_model", side_effect=ModelBoundary) as create:
                    with self.assertRaises(ModelBoundary):
                        if module is sensitivity:
                            module.run(output)
                        else:
                            module.run(output, resume=False)
                    create.assert_called_once()

    def test_skill_content_and_file_set_changes_reject_before_model(self):
        for module in (sensitivity, final_study):
            with self.prepared(module) as output:
                for mutation in ("edit", "whitespace", "add", "remove", "rename"):
                    with self.subTest(runner=module.__name__, mutation=mutation):
                        self.reset_skills()
                        path = self.skills / "alpha.md"
                        if mutation == "edit":
                            path.write_text("**Status:** artificial test\nChanged guidance.\n")
                        elif mutation == "whitespace":
                            path.write_text(path.read_text() + "\n")
                        elif mutation == "add":
                            (self.skills / "extra.md").write_text(path.read_text())
                        elif mutation == "remove":
                            path.unlink()
                        else:
                            path.rename(self.skills / "renamed.md")
                        with patch.object(module, "create_chat_model") as create:
                            with self.assertRaisesRegex(ValueError, "Skill snapshot"):
                                if module is sensitivity:
                                    module.run(output)
                                else:
                                    module.run(output, resume=False)
                            create.assert_not_called()
                self.reset_skills()

    def test_missing_skill_snapshot_rejects_before_model(self):
        for module in (sensitivity, final_study):
            with self.subTest(runner=module.__name__), self.prepared(module) as output:
                path = output / "plan.json"
                plan = json.loads(path.read_text())
                del plan["skill_sha256"]
                path.write_text(json.dumps(plan))
                with patch.object(module, "create_chat_model") as create:
                    with self.assertRaisesRegex(ValueError, "Skill snapshot"):
                        if module is sensitivity:
                            module.run(output)
                        else:
                            module.run(output, resume=False)
                    create.assert_not_called()

    def test_wrong_development_identity_rejects_before_model(self):
        with self.prepared(final_study) as output:
            for identity in ("", "NOT_A_CASE_ID", "OTHER::artificial-region"):
                with self.subTest(identity=identity), patch.dict(os.environ, {
                    "HAPLOTYPE_DEVELOPMENT_CASE_ID": identity,
                }), patch.object(final_study, "create_chat_model") as create:
                    with self.assertRaisesRegex(ValueError, "original historical exclusion"):
                        final_study.excluded_ids()
                    with self.assertRaisesRegex(ValueError, "original historical exclusion"):
                        final_study.run(output, resume=False)
                    create.assert_not_called()

    def test_development_exclusion_is_retained_and_plan_hash_is_required(self):
        with self.prepared(final_study) as output:
            study = self.root / "study"
            (study / "evaluation").mkdir(parents=True)
            (study / "evaluation/phase16_panel.json").write_text('{"cases": []}')
            (study / "evaluation/phase17_manifest.json").write_text('{"pairs": []}')
            with patch.object(final_study, "STUDY_ROOT", study), patch.object(
                final_study, "load_panel", return_value=[]
            ):
                self.assertEqual(final_study.excluded_ids(), {self.development_id})
            path = output / "plan.json"
            plan = json.loads(path.read_text())
            del plan["development_case_sha256"]
            path.write_text(json.dumps(plan))
            with patch.object(final_study, "create_chat_model") as create:
                with self.assertRaisesRegex(ValueError, "development exclusion not frozen"):
                    final_study.run(output, resume=False)
                create.assert_not_called()

    def diagnostic_sources(self):
        sources = []
        for index in range(3):
            row = self.panel_row(index)
            path = self.root / f"case-{index}.json"
            path.write_text(json.dumps({"initial_state": row["state"]}))
            sources.append({"name": f"artificial-{index}", "path": str(path),
                            "case_id": row["case_id"]})
        return sources

    def run_diagnostic(self, sources):
        specs = self.root / "sources.json"
        specs.write_text(json.dumps(sources))
        with patch.object(sys, "argv", ["diagnostics", "--sources", str(specs),
                                       "--output-dir", str(self.root / "diagnostics")]), patch.object(
            diagnostics, "load_config", return_value=self.config
        ):
            diagnostics.main()

    def test_correct_diagnostic_identities_reach_model_boundary(self):
        sources = self.diagnostic_sources()
        expected = tuple(identity_hash(spec["case_id"]) for spec in sources)
        with patch.object(diagnostics, "EXPECTED_CASE_SHA256", expected), patch.object(
            diagnostics, "create_chat_model", side_effect=ModelBoundary
        ) as create:
            with self.assertRaises(ModelBoundary):
                self.run_diagnostic(sources)
            create.assert_called_once()

    def test_duplicate_incorrect_and_reordered_diagnostics_reject_before_model(self):
        sources = self.diagnostic_sources()
        expected = tuple(identity_hash(spec["case_id"]) for spec in sources)
        for kind in ("duplicate", "incorrect", "reordered", "supplied_identity"):
            with self.subTest(kind=kind):
                selected = [dict(spec) for spec in sources]
                if kind == "duplicate":
                    selected[1] = dict(selected[0])
                elif kind == "incorrect":
                    row = self.panel_row(99)
                    path = self.root / "incorrect.json"
                    path.write_text(json.dumps({"initial_state": row["state"]}))
                    selected[1].update(path=str(path), case_id=row["case_id"])
                elif kind == "reordered":
                    selected[0], selected[1] = selected[1], selected[0]
                else:
                    selected[1]["case_id"] = "INCORRECT::identity"
                with patch.object(diagnostics, "EXPECTED_CASE_SHA256", expected), patch.object(
                    diagnostics, "create_chat_model"
                ) as create:
                    with self.assertRaisesRegex(ValueError, "diagnostic case identit"):
                        self.run_diagnostic(selected)
                    create.assert_not_called()
